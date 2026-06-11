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
        # Reasoning is accumulated silently during the stream — a single
        # thinking event is emitted by run() after _process_stream returns.
        # _process_stream itself no longer emits thinking_delta events.
    
    def test_stream_empty_response(self):
        """Test streaming with no content."""
        chunks = []
        
        text, reasoning, tool_calls, usage = self.agent._process_stream(iter(chunks))
        
        assert text is None
        assert reasoning is None
        assert tool_calls == []
        assert usage is None


class TestStreamingCLI:
    """Test CLI streaming output."""
    
    @patch("harness.cli.AgentHarness")
    @patch("harness.cli.input")
    def test_stream_flag_enabled(self, mock_input, mock_harness):
        """Test that --stream flag enables streaming."""
        from harness.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--stream"])
        assert args.stream is True
        assert args.no_stream is False
    
    @patch("harness.cli.AgentHarness")
    @patch("harness.cli.input")
    def test_no_stream_flag(self, mock_input, mock_harness):
        """Test that --no-stream flag disables streaming."""
        from harness.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--no-stream"])
        assert args.no_stream is True
    
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
