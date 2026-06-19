"""Tests for streaming functionality."""

from unittest.mock import MagicMock, patch
import pytest

from harness.agent import AgentHarness


class MockStreamChunk:
    """Mock a streaming chunk from the API."""
    def __init__(self, content=None, tool_calls=None, usage=None, reasoning=None):
        self.choices = []
        choice = MagicMock()
        delta = MagicMock()
        delta.content = content
        delta.tool_calls = tool_calls
        delta.reasoning_content = reasoning
        delta.thinking = None
        delta.thought = None
        choice.delta = delta
        choice.finish_reason = "stop" if not tool_calls else "tool_calls"
        self.choices = [choice]
        self.usage = usage


def _make_tool_call_chunk(index, tc_id, name, arguments, content=None):
    """Build a stream chunk carrying a tool-call delta."""
    choice = MagicMock()
    delta = MagicMock()
    delta.content = content
    delta.reasoning_content = None
    delta.thinking = None
    delta.thought = None
    func = MagicMock()
    func.name = name
    func.arguments = arguments
    tc_delta = MagicMock()
    tc_delta.index = index
    tc_delta.id = tc_id
    tc_delta.function = func
    delta.tool_calls = [tc_delta]
    choice.delta = delta
    choice.finish_reason = "tool_calls"
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = None
    return chunk


class TestStreamingParsing:
    """Test the _process_stream method."""

    def setup_method(self):
        self.agent = AgentHarness(model="test-model", api_key="sk-test")
    
    def test_simple_text_stream(self):
        """Test streaming plain text response."""
        chunks = [
            MockStreamChunk(content="Hello"),
            MockStreamChunk(content=" world"),
            MockStreamChunk(content="!"),
        ]
        
        callback_events = []
        def callback(event):
            callback_events.append(event)
        
        text, reasoning, tool_calls, usage = self.agent._process_stream(iter(chunks), callback)
        
        assert text == "Hello world!"
        assert reasoning is None
        assert tool_calls == []
        
        # Check that text_delta events were emitted
        deltas = [e for e in callback_events if e["type"] == "text_delta"]
        assert len(deltas) == 3
        assert deltas[0]["content"] == "Hello"
        assert deltas[1]["content"] == " world"
        assert deltas[2]["content"] == "!"
    
    def test_stream_with_reasoning(self):
        """Test streaming with reasoning content."""
        chunks = [
            MockStreamChunk(reasoning="Thinking"),
            MockStreamChunk(reasoning=" step"),
            MockStreamChunk(content="Final answer"),
        ]
        
        callback_events = []
        def callback(event):
            callback_events.append(event)
        
        text, reasoning, tool_calls, usage = self.agent._process_stream(iter(chunks), callback)
        
        assert text == "Final answer"
        assert reasoning == "Thinking step"
        # Reasoning is streamed live as thinking_delta events during the stream.
        thinking_deltas = [e for e in callback_events if e["type"] == "thinking_delta"]
        assert [e["content"] for e in thinking_deltas] == ["Thinking", " step"]
        assert any(e["type"] == "thinking_end" for e in callback_events)
    
    def test_stream_empty_response(self):
        """Test streaming with no content."""
        chunks = []

        text, reasoning, tool_calls, usage = self.agent._process_stream(iter(chunks))

        assert text is None
        assert reasoning is None
        assert tool_calls == []
        assert usage is None

    def test_stream_with_tool_call(self):
        """Test streaming a response that contains a tool call."""
        chunks = [
            _make_tool_call_chunk(0, "call_1", "read", None),
            _make_tool_call_chunk(0, None, None, '{"path": "/tmp/x.txt"}'),
        ]

        text, reasoning, tool_calls, usage = self.agent._process_stream(iter(chunks))

        assert text is None
        assert tool_calls == [
            {"id": "call_1", "name": "read", "arguments": {"path": "/tmp/x.txt"}}
        ]

    def test_stream_with_multiple_tool_calls(self):
        """Test streaming multiple tool calls in a single response."""
        chunks = [
            _make_tool_call_chunk(0, "c1", "read", '{"path": "/a.txt"}'),
            _make_tool_call_chunk(1, "c2", "bash", '{"command": "ls"}'),
        ]

        text, reasoning, tool_calls, usage = self.agent._process_stream(iter(chunks))

        assert len(tool_calls) == 2
        assert tool_calls[0] == {"id": "c1", "name": "read", "arguments": {"path": "/a.txt"}}
        assert tool_calls[1] == {"id": "c2", "name": "bash", "arguments": {"command": "ls"}}

    def test_stream_malformed_tool_arguments(self):
        """Malformed JSON tool arguments fall back to an empty dict."""
        chunks = [
            _make_tool_call_chunk(0, "call_1", "read", "not json"),
        ]

        text, reasoning, tool_calls, usage = self.agent._process_stream(iter(chunks))

        assert tool_calls[0]["arguments"] == {}

    def test_stream_extracts_usage(self):
        """Test that usage information from the stream is returned."""
        usage_mock = MagicMock()
        usage_mock.prompt_tokens = 42
        usage_mock.completion_tokens = 7
        usage_mock.prompt_tokens_details = None

        chunk = MagicMock()
        choice = MagicMock()
        delta = MagicMock()
        delta.content = "Hi"
        delta.tool_calls = None
        delta.reasoning_content = None
        choice.delta = delta
        choice.finish_reason = "stop"
        chunk.choices = [choice]
        chunk.usage = usage_mock

        text, reasoning, tool_calls, usage = self.agent._process_stream(iter([chunk]))

        assert usage == {
            "prompt_tokens": 42,
            "completion_tokens": 7,
            "prompt_tokens_details": {},
        }

    def test_stream_reasoning_via_thinking_field(self):
        """Test that reasoning can arrive in the thinking field."""
        chunk = MagicMock()
        choice = MagicMock()
        delta = MagicMock()
        delta.content = "Answer"
        delta.tool_calls = None
        delta.reasoning_content = None
        delta.thinking = "Step one"
        delta.thought = None
        choice.delta = delta
        choice.finish_reason = "stop"
        chunk.choices = [choice]
        chunk.usage = None

        text, reasoning, tool_calls, usage = self.agent._process_stream(iter([chunk]))

        assert text == "Answer"
        assert reasoning == "Step one"


class TestStreamingRun:
    """Test AgentHarness.run() in streaming mode."""

    @patch("harness.agent.OpenAI")
    def test_streaming_text_response_emits_text_end(self, mock_openai):
        """A simple streaming text response emits text_delta and text_end."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        def make_stream():
            for word in ["Hello", " world", "!"]:
                chunk = MagicMock()
                choice = MagicMock()
                delta = MagicMock()
                delta.content = word
                delta.tool_calls = None
                delta.reasoning_content = None
                choice.delta = delta
                choice.finish_reason = "stop"
                chunk.choices = [choice]
                chunk.usage = None
                yield chunk

        mock_client.chat.completions.create.return_value = make_stream()

        agent = AgentHarness(model="test-model", api_key="sk-test")
        events = []
        result = agent.run("Hi", callback=events.append, stream=True)

        assert result == "Hello world!"
        deltas = [e for e in events if e["type"] == "text_delta"]
        assert [e["content"] for e in deltas] == ["Hello", " world", "!"]
        assert events[-1] == {"type": "text_end", "content": "Hello world!"}

    @patch("harness.agent.OpenAI")
    def test_streaming_text_response_emits_tokens_event(self, mock_openai):
        """Token usage from the final stream chunk is reported."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        usage_mock = MagicMock()
        usage_mock.prompt_tokens = 10
        usage_mock.completion_tokens = 5
        usage_mock.prompt_tokens_details = None

        def make_stream():
            chunk = MagicMock()
            choice = MagicMock()
            delta = MagicMock()
            delta.content = "Done"
            delta.tool_calls = None
            delta.reasoning_content = None
            choice.delta = delta
            choice.finish_reason = "stop"
            chunk.choices = [choice]
            chunk.usage = usage_mock
            yield chunk

        mock_client.chat.completions.create.return_value = make_stream()

        agent = AgentHarness(model="test-model", api_key="sk-test")
        events = []
        agent.run("Hi", callback=events.append, stream=True)

        token_events = [e for e in events if e["type"] == "tokens"]
        assert len(token_events) == 1
        assert token_events[0]["input_tokens"] == 10
        assert token_events[0]["output_tokens"] == 5

    @patch("harness.agent.OpenAI")
    def test_streaming_passes_include_usage_option(self, mock_openai):
        """Streaming requests must request usage in the final chunk.

        OpenAI-spec endpoints only send chunk.usage when stream_options.
        include_usage is True; otherwise every chunk's usage is None and the
        token bar shows zeros for the whole session.
        """
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        def make_stream():
            chunk = MagicMock()
            choice = MagicMock()
            delta = MagicMock()
            delta.content = "Hi"
            delta.tool_calls = None
            delta.reasoning_content = None
            choice.delta = delta
            choice.finish_reason = "stop"
            chunk.choices = [choice]
            chunk.usage = None
            yield chunk

        mock_client.chat.completions.create.return_value = make_stream()

        agent = AgentHarness(model="test-model", api_key="sk-test")
        agent.run("Hi", stream=True)

        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs.get("stream") is True
        assert kwargs.get("stream_options") == {"include_usage": True}

    @patch("harness.agent.OpenAI")
    def test_streaming_with_reasoning_emits_thinking_deltas(self, mock_openai):
        """A streaming response with reasoning emits thinking_delta events."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        def make_stream():
            for chunk_data in [
                {"reasoning": "Let me think", "content": None},
                {"reasoning": " step by step", "content": None},
                {"reasoning": None, "content": "Answer"},
            ]:
                chunk = MagicMock()
                choice = MagicMock()
                delta = MagicMock()
                delta.content = chunk_data["content"]
                delta.tool_calls = None
                delta.reasoning_content = chunk_data["reasoning"]
                delta.thinking = None
                delta.thought = None
                choice.delta = delta
                choice.finish_reason = "stop"
                chunk.choices = [choice]
                chunk.usage = None
                yield chunk

        mock_client.chat.completions.create.return_value = make_stream()

        agent = AgentHarness(model="test-model", api_key="sk-test")
        events = []
        result = agent.run("Hi", callback=events.append, stream=True)

        assert result == "Answer"
        thinking_deltas = [e for e in events if e["type"] == "thinking_delta"]
        assert [e["content"] for e in thinking_deltas] == ["Let me think", " step by step"]
        assert any(e["type"] == "thinking_end" for e in events)
        # No full thinking block event should be emitted in streaming mode.
        assert not any(e["type"] == "thinking" for e in events)

    @patch("harness.agent.OpenAI")
    def test_streaming_tool_call_loop(self, mock_openai):
        """Streaming responses with tool calls execute tools and continue."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        def first_stream():
            yield _make_tool_call_chunk(0, "call_1", "bash", '{"command": "echo hello"}')

        def second_stream():
            chunk = MagicMock()
            choice = MagicMock()
            delta = MagicMock()
            delta.content = "All done"
            delta.tool_calls = None
            delta.reasoning_content = None
            choice.delta = delta
            choice.finish_reason = "stop"
            chunk.choices = [choice]
            chunk.usage = None
            yield chunk

        mock_client.chat.completions.create.side_effect = [first_stream(), second_stream()]

        agent = AgentHarness(model="test-model", api_key="sk-test")
        events = []
        result = agent.run("Say hello", callback=events.append, stream=True)

        assert result == "All done"
        assert any(e["type"] == "tool_call" and e["name"] == "bash" for e in events)
        assert any(e["type"] == "tool_result" for e in events)

    @patch("harness.agent.OpenAI")
    def test_streaming_text_plus_tool_call_emits_text_end(self, mock_openai):
        """A turn with both text and tool_calls must emit text_end for the text.

        Without this, the CLI's response buffer accumulates intermediate text
        that gets concatenated with the next turn's text, showing
        "Let me check...All done" instead of two separate blocks.
        """
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        def first_stream():
            # Text delta first, then tool call
            chunk1 = MagicMock()
            choice1 = MagicMock()
            delta1 = MagicMock()
            delta1.content = "Let me check..."
            delta1.tool_calls = None
            delta1.reasoning_content = None
            choice1.delta = delta1
            choice1.finish_reason = None
            chunk1.choices = [choice1]
            chunk1.usage = None
            yield chunk1

            yield _make_tool_call_chunk(0, "call_1", "bash", '{"command": "echo hi"}')

        def second_stream():
            chunk = MagicMock()
            choice = MagicMock()
            delta = MagicMock()
            delta.content = "All done"
            delta.tool_calls = None
            delta.reasoning_content = None
            choice.delta = delta
            choice.finish_reason = "stop"
            chunk.choices = [choice]
            chunk.usage = None
            yield chunk

        mock_client.chat.completions.create.side_effect = [first_stream(), second_stream()]

        agent = AgentHarness(model="test-model", api_key="sk-test")
        events = []
        result = agent.run("Check and report", callback=events.append, stream=True)

        assert result == "All done"
        # Intermediate text_end should be emitted for the first turn's text.
        text_ends = [e for e in events if e["type"] == "text_end"]
        assert len(text_ends) == 2
        assert text_ends[0]["content"] == "Let me check..."
        assert text_ends[1]["content"] == "All done"
        # A turn_start event should be emitted before the second turn.
        turn_starts = [e for e in events if e["type"] == "turn_start"]
        assert len(turn_starts) == 1


class TestStreamingCLI:
    """Test CLI streaming output."""

    @patch("harness.cli.AgentHarness")
    @patch("harness.cli.input")
    def test_stream_flag_enabled(self, mock_input, mock_harness):
        """--stream explicitly enables streaming (no longer a no-op)."""
        from harness.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--stream"])
        assert args.stream is True

    @patch("harness.cli.AgentHarness")
    @patch("harness.cli.input")
    def test_no_stream_flag(self, mock_input, mock_harness):
        """--no-stream disables streaming."""
        from harness.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--no-stream"])
        assert args.stream is False
    
    def test_stream_command_toggles(self):
        """Test /stream command toggles streaming mode."""
        from harness.cli import main
        
        mock_agent = MagicMock()
        mock_agent.run.return_value = "Response"
        mock_agent.reasoning_effort = "high"
        
        with patch("harness.cli.AgentHarness", return_value=mock_agent):
            with patch("harness.cli.input", side_effect=["/stream", "/exit"]):
                with patch("builtins.print") as mock_print:
                    main(["--api-key", "sk-test"])
                    
                    # Check that streaming was toggled
                    print_calls = [str(call) for call in mock_print.call_args_list]
                    assert any("Streaming disabled" in call for call in print_calls)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
