"""Tests for the AgentHarness class."""

import json
from unittest.mock import patch, MagicMock

import pytest

from harness.agent import AgentHarness
from harness.tools import TOOL_DEFINITIONS


class TestAgentHarnessInit:
    def test_accepts_api_key_and_base_url(self):
        agent = AgentHarness(
            model="deepseek-v4-pro",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
        )
        assert agent.model == "deepseek-v4-pro"
        assert agent.client.api_key == "sk-test"
        assert str(agent.client.base_url).rstrip("/") == "https://api.openai.com/v1"

    def test_defaults_to_openai_env(self):
        agent = AgentHarness(model="gpt-3.5-turbo")
        assert agent.model == "gpt-3.5-turbo"

    def test_accepts_system_prompt(self):
        custom_prompt = "You are a helpful assistant."
        agent = AgentHarness(model="deepseek-v4-pro", system_prompt=custom_prompt)
        assert agent.system_prompt == custom_prompt

    def test_has_tool_registry(self):
        agent = AgentHarness(model="deepseek-v4-pro")
        tools = agent.tool_registry.get_definitions()
        assert len(tools) == 4

    def test_accepts_working_dir(self):
        agent = AgentHarness(model="deepseek-v4-pro", working_dir="/tmp/foo")
        assert agent.tool_registry.working_dir == "/tmp/foo"

    def test_accepts_reasoning_effort(self):
        agent = AgentHarness(model="deepseek-v4-pro", reasoning_effort="max")
        assert agent.reasoning_effort == "max"

    def test_reasoning_effort_none_by_default(self):
        agent = AgentHarness(model="deepseek-v4-pro")
        assert agent.reasoning_effort is None

    def test_max_turns_default_is_1000(self):
        agent = AgentHarness(model="deepseek-v4-pro")
        assert agent.max_turns == 1000

    def test_context_window_default_is_1000000(self):
        agent = AgentHarness(model="deepseek-v4-pro")
        assert agent.context_window == 1000000

    def test_max_turns_explicit_override(self):
        agent = AgentHarness(model="deepseek-v4-pro", max_turns=5)
        assert agent.max_turns == 5

    def test_context_window_explicit_override(self):
        agent = AgentHarness(model="deepseek-v4-pro", context_window=256000)
        assert agent.context_window == 256000


class TestCustomContext:
    def test_get_and_set_custom_context(self):
        agent = AgentHarness(model="deepseek-v4-pro")
        assert agent.get_custom_context() is None
        agent.set_custom_context("Extra info")
        assert agent.get_custom_context() == "Extra info"

    def test_set_custom_context_clears_with_empty_string(self):
        agent = AgentHarness(model="deepseek-v4-pro")
        agent.set_custom_context("Extra info")
        assert agent.get_custom_context() == "Extra info"
        agent.set_custom_context("")
        assert agent.get_custom_context() is None

    def test_build_system_content_includes_context(self):
        agent = AgentHarness(model="deepseek-v4-pro", system_prompt="Be helpful.")
        agent.set_custom_context("Project: harness")
        content = agent._build_system_content()
        assert "Be helpful." in content
        assert "Project: harness" in content
        assert "Additional Context" in content

    def test_set_custom_context_updates_existing_system_message(self):
        agent = AgentHarness(model="deepseek-v4-pro")
        agent.run = lambda *a, **k: None  # avoid API calls
        # Simulate an existing conversation
        agent.messages = agent._build_initial_messages("Hello")
        original_content = agent.messages[0]["content"]
        assert "Additional Context" not in original_content

        agent.set_custom_context("New context")
        assert "New context" in agent.messages[0]["content"]
        assert agent.messages[0]["role"] == "system"

    def test_clear_history_keeps_custom_context(self):
        agent = AgentHarness(model="deepseek-v4-pro")
        agent.set_custom_context("Keep me")
        agent.messages = [{"role": "user", "content": "old"}]
        agent.clear_history()
        assert agent.messages == []
        assert agent.get_custom_context() == "Keep me"


class TestMessageBuilding:
    def test_builds_initial_messages_with_system_prompt(self):
        agent = AgentHarness(model="deepseek-v4-pro", system_prompt="Be helpful.")
        messages = agent._build_initial_messages("Do something.")
        assert messages[0]["role"] == "system"
        assert messages[0]["content"].startswith("Be helpful.\nWorking directory:")
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Do something."

    def test_default_system_prompt_includes_tool_instructions(self):
        agent = AgentHarness(model="deepseek-v4-pro")
        msg = agent._build_initial_messages("test")[0]["content"]
        assert "read" in msg.lower()
        assert "write" in msg.lower()
        assert "edit" in msg.lower()
        assert "bash" in msg.lower()

    def test_user_message_injected(self):
        agent = AgentHarness(model="deepseek-v4-pro")
        messages = agent._build_initial_messages("Hello world!")
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Hello world!"


class TestToolCallResponseParsing:
    def test_parses_single_tool_call(self):
        agent = AgentHarness(model="deepseek-v4-pro")
        response = MagicMock()
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "read"
        tool_call.function.arguments = '{"path": "/tmp/test.txt"}'
        response.choices = [MagicMock()]
        response.choices[0].message.tool_calls = [tool_call]
        response.choices[0].message.content = None

        text, reasoning, parsed = agent._parse_response(response)
        assert text is None
        assert reasoning is None
        assert len(parsed) == 1
        assert parsed[0]["id"] == "call_1"
        assert parsed[0]["name"] == "read"
        assert parsed[0]["arguments"] == {"path": "/tmp/test.txt"}

    def test_parses_text_response(self):
        agent = AgentHarness(model="deepseek-v4-pro")
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.tool_calls = None
        response.choices[0].message.content = "Hello there!"

        text, reasoning, tool_calls = agent._parse_response(response)
        assert text == "Hello there!"
        assert reasoning is None
        assert tool_calls == []

    def test_parses_multiple_tool_calls(self):
        agent = AgentHarness(model="deepseek-v4-pro")
        response = MagicMock()
        tc1 = MagicMock()
        tc1.id = "call_1"
        tc1.function.name = "read"
        tc1.function.arguments = '{"path": "/a.txt"}'
        tc2 = MagicMock()
        tc2.id = "call_2"
        tc2.function.name = "bash"
        tc2.function.arguments = '{"command": "ls"}'
        response.choices = [MagicMock()]
        response.choices[0].message.tool_calls = [tc1, tc2]
        response.choices[0].message.content = None

        text, reasoning, parsed = agent._parse_response(response)
        assert text is None
        assert reasoning is None
        assert len(parsed) == 2
        assert parsed[0]["name"] == "read"
        assert parsed[1]["name"] == "bash"

    def test_parses_malformed_json_gracefully(self):
        agent = AgentHarness(model="deepseek-v4-pro")
        response = MagicMock()
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "read"
        tc.function.arguments = "{bad json}"
        response.choices = [MagicMock()]
        response.choices[0].message.tool_calls = [tc]
        response.choices[0].message.content = None

        text, reasoning, parsed = agent._parse_response(response)
        assert text is None
        assert reasoning is None
        assert parsed[0]["arguments"] == {}

    def test_parses_reasoning_content(self):
        """When the message has reasoning_content, it is extracted."""
        agent = AgentHarness(model="deepseek-v4-pro")
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.tool_calls = None
        response.choices[0].message.content = "The answer is 42."
        response.choices[0].message.reasoning_content = "Let me think about this..."

        text, reasoning, tool_calls = agent._parse_response(response)
        assert text == "The answer is 42."
        assert reasoning == "Let me think about this..."
        assert tool_calls == []

    def test_reasoning_content_ignores_non_string(self):
        """Non-string reasoning_content (e.g. None, dict) is treated as absent."""
        agent = AgentHarness(model="deepseek-v4-pro")
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.tool_calls = None
        response.choices[0].message.content = "OK"
        del response.choices[0].message.reasoning_content  # remove MagicMock auto-attr
        response.choices[0].message.reasoning_content = None

        text, reasoning, tool_calls = agent._parse_response(response)
        assert text == "OK"
        assert reasoning is None

    def test_parses_thinking_field(self):
        """Some providers return reasoning in a 'thinking' field."""
        agent = AgentHarness(model="deepseek-v4-pro")
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.tool_calls = None
        response.choices[0].message.content = "The answer is 42."
        del response.choices[0].message.reasoning_content  # remove MagicMock auto-attr
        response.choices[0].message.thinking = "Let me think about this..."

        text, reasoning, tool_calls = agent._parse_response(response)
        assert text == "The answer is 42."
        assert reasoning == "Let me think about this..."
        assert tool_calls == []

    def test_parses_thought_field(self):
        """Some providers return reasoning in a 'thought' field."""
        agent = AgentHarness(model="deepseek-v4-pro")
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.tool_calls = None
        response.choices[0].message.content = "The answer is 42."
        del response.choices[0].message.reasoning_content  # remove MagicMock auto-attr
        response.choices[0].message.thought = "Let me think about this..."

        text, reasoning, tool_calls = agent._parse_response(response)
        assert text == "The answer is 42."
        assert reasoning == "Let me think about this..."
        assert tool_calls == []

    def test_parses_reasoning_from_model_extra(self):
        """Some providers return reasoning inside model_extra dict."""
        agent = AgentHarness(model="deepseek-v4-pro")
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.tool_calls = None
        response.choices[0].message.content = "The answer is 42."
        del response.choices[0].message.reasoning_content  # remove MagicMock auto-attr
        response.choices[0].message.model_extra = {"reasoning": "Let me think about this..."}

        text, reasoning, tool_calls = agent._parse_response(response)
        assert text == "The answer is 42."
        assert reasoning == "Let me think about this..."
        assert tool_calls == []

    def test_parses_thinking_from_model_extra(self):
        """Some providers return reasoning inside model_extra as 'thinking'."""
        agent = AgentHarness(model="deepseek-v4-pro")
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.tool_calls = None
        response.choices[0].message.content = "The answer is 42."
        del response.choices[0].message.reasoning_content  # remove MagicMock auto-attr
        response.choices[0].message.model_extra = {"thinking": "Let me think about this..."}

        text, reasoning, tool_calls = agent._parse_response(response)
        assert text == "The answer is 42."
        assert reasoning == "Let me think about this..."
        assert tool_calls == []

    def test_reasoning_content_takes_precedence_over_thinking(self):
        """reasoning_content should be preferred over thinking/thought fields."""
        agent = AgentHarness(model="deepseek-v4-pro")
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.tool_calls = None
        response.choices[0].message.content = "The answer is 42."
        response.choices[0].message.reasoning_content = "Primary reasoning"
        response.choices[0].message.thinking = "Secondary thinking"

        text, reasoning, tool_calls = agent._parse_response(response)
        assert reasoning == "Primary reasoning"


class TestRunIntegration:
    @patch("harness.agent.OpenAI")
    def test_simple_text_response(self, mock_openai):
        """When the model returns text without tool calls, run() returns it."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.content = "I'll help you with that."
        mock_client.chat.completions.create.return_value = mock_response

        agent = AgentHarness(model="deepseek-v4-pro", api_key="sk-test")
        result = agent.run("Write a poem.")

        assert result == "I'll help you with that."

    @patch("harness.agent.OpenAI")
    def test_passes_reasoning_effort_to_api(self, mock_openai):
        """When reasoning_effort is set, it is passed to the API call."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.content = "Done."
        mock_client.chat.completions.create.return_value = mock_response

        agent = AgentHarness(model="deepseek-v4-pro", api_key="sk-test", reasoning_effort="max")
        agent.run("Hello")

        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("reasoning_effort") == "max"

    @patch("harness.agent.OpenAI")
    def test_does_not_pass_reasoning_effort_when_none(self, mock_openai):
        """When reasoning_effort is not set, it is NOT passed to the API call."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.content = "Done."
        mock_client.chat.completions.create.return_value = mock_response

        agent = AgentHarness(model="deepseek-v4-pro", api_key="sk-test")
        agent.run("Hello")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "reasoning_effort" not in call_kwargs

    @patch("harness.agent.OpenAI")
    def test_tool_calling_loop(self, mock_openai):
        """When the model makes tool calls, run() executes them and continues."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # First response: tool call to read
        resp1 = MagicMock()
        tc1 = MagicMock()
        tc1.id = "call_1"
        tc1.function.name = "bash"
        tc1.function.arguments = '{"command": "echo hello"}'
        resp1.choices = [MagicMock()]
        resp1.choices[0].message.tool_calls = [tc1]
        resp1.choices[0].message.content = None

        # Second response: final text
        resp2 = MagicMock()
        resp2.choices = [MagicMock()]
        resp2.choices[0].message.tool_calls = None
        resp2.choices[0].message.content = "Command executed: hello"

        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        agent = AgentHarness(model="deepseek-v4-pro", api_key="sk-test")
        result = agent.run("Run echo hello")

        assert result == "Command executed: hello"
        assert mock_client.chat.completions.create.call_count == 2

    @patch("harness.agent.OpenAI")
    def test_max_turns_limit(self, mock_openai):
        """The agent stops after max_turns even if model keeps calling tools."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Always respond with a tool call
        def make_response(*args, **kwargs):
            resp = MagicMock()
            tc = MagicMock()
            tc.id = "call_1"
            tc.function.name = "bash"
            tc.function.arguments = '{"command": "echo loop"}'
            resp.choices = [MagicMock()]
            resp.choices[0].message.tool_calls = [tc]
            resp.choices[0].message.content = None
            return resp

        mock_client.chat.completions.create.side_effect = make_response

        agent = AgentHarness(model="deepseek-v4-pro", api_key="sk-test", max_turns=3)
        result = agent.run("Loop forever")

        assert "Max turns" in result
        assert mock_client.chat.completions.create.call_count == 3


class TestConversationHistory:
    @patch("harness.agent.OpenAI")
    def test_run_appends_to_history(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.tool_calls = None
        resp.choices[0].message.content = "Done."
        mock_client.chat.completions.create.return_value = resp

        agent = AgentHarness(model="deepseek-v4-pro", api_key="sk-test")
        agent.run("Task 1")

        assert len(agent.messages) >= 2  # system + user + assistant
        roles = [m["role"] for m in agent.messages]
        assert "assistant" in roles

    def test_clear_history(self):
        agent = AgentHarness(model="deepseek-v4-pro")
        agent.messages = [{"role": "user", "content": "old"}]
        agent.clear_history()
        assert agent.messages == []

    @patch("harness.agent.OpenAI")
    def test_preserves_history_across_runs(self, mock_openai):
        """Second run() should include messages from the first run."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        resp1 = MagicMock()
        resp1.choices = [MagicMock()]
        resp1.choices[0].message.tool_calls = None
        resp1.choices[0].message.content = "First response."

        resp2 = MagicMock()
        resp2.choices = [MagicMock()]
        resp2.choices[0].message.tool_calls = None
        resp2.choices[0].message.content = "Second response."

        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        agent = AgentHarness(model="deepseek-v4-pro", api_key="sk-test")
        agent.run("First prompt")
        agent.run("Second prompt")

        # The second API call should have the full history from the first run:
        # system, user(First prompt), assistant(First response), user(Second prompt)
        # (call_args stores a reference, so the second assistant msg is also present)
        second_call_messages = mock_client.chat.completions.create.call_args_list[1][1]["messages"]
        roles = [m["role"] for m in second_call_messages]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles
        # Verify the first run's messages are present in the history
        contents = [m["content"] for m in second_call_messages]
        assert "First prompt" in contents
        assert "First response." in contents
        assert "Second prompt" in contents


class TestProgressCallback:
    """The run() method accepts an optional callback for progress events."""

    @patch("harness.agent.OpenAI")
    def test_text_event_on_simple_response(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.tool_calls = None
        resp.choices[0].message.content = "Here is the answer."
        mock_client.chat.completions.create.return_value = resp

        agent = AgentHarness(model="deepseek-v4-pro", api_key="sk-test")
        events = []
        result = agent.run("Hello", callback=events.append)

        assert result == "Here is the answer."
        # tokens event fires before text
        assert len(events) == 2
        assert events[0]["type"] == "tokens"
        assert events[1] == {"type": "text", "content": "Here is the answer."}

    @patch("harness.agent.OpenAI")
    def test_tool_call_and_result_events(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # First: tool call
        resp1 = MagicMock()
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "read"
        tc.function.arguments = '{"path": "/tmp/x.txt"}'
        resp1.choices = [MagicMock()]
        resp1.choices[0].message.tool_calls = [tc]
        resp1.choices[0].message.content = "Let me read the file."

        # Second: final text
        resp2 = MagicMock()
        resp2.choices = [MagicMock()]
        resp2.choices[0].message.tool_calls = None
        resp2.choices[0].message.content = "File contents: hello"

        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        agent = AgentHarness(model="deepseek-v4-pro", api_key="sk-test")
        events = []
        result = agent.run("Read /tmp/x.txt", callback=events.append)

        assert result == "File contents: hello"
        # tokens + tool_call + tool_result + tokens + text = 5
        assert len(events) == 5
        assert events[0]["type"] == "tokens"
        assert events[1] == {
            "type": "tool_call",
            "name": "read",
            "arguments": {"path": "/tmp/x.txt"},
        }
        assert events[2]["type"] == "tool_result"
        assert events[2]["name"] == "read"
        assert "Error" in events[2]["result"]  # file doesn't exist
        assert events[3]["type"] == "tokens"
        assert events[4] == {"type": "text", "content": "File contents: hello"}

    @patch("harness.agent.OpenAI")
    def test_multiple_tool_calls_in_one_turn(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        resp1 = MagicMock()
        tc1 = MagicMock()
        tc1.id = "c1"
        tc1.function.name = "read"
        tc1.function.arguments = '{"path": "/a.txt"}'
        tc2 = MagicMock()
        tc2.id = "c2"
        tc2.function.name = "bash"
        tc2.function.arguments = '{"command": "ls"}'
        resp1.choices = [MagicMock()]
        resp1.choices[0].message.tool_calls = [tc1, tc2]
        resp1.choices[0].message.content = None

        resp2 = MagicMock()
        resp2.choices = [MagicMock()]
        resp2.choices[0].message.tool_calls = None
        resp2.choices[0].message.content = "Done."
        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        agent = AgentHarness(model="deepseek-v4-pro", api_key="sk-test")
        events = []
        agent.run("Read and list", callback=events.append)

        event_types = [e["type"] for e in events]
        assert event_types == [
            "tokens",
            "tool_call", "tool_result",  # read
            "tool_call", "tool_result",  # bash
            "tokens",
            "text",                       # final
        ]

    @patch("harness.agent.OpenAI")
    def test_callback_not_required(self, mock_openai):
        """When no callback is provided, run() still works normally."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.tool_calls = None
        resp.choices[0].message.content = "OK"
        mock_client.chat.completions.create.return_value = resp

        agent = AgentHarness(model="deepseek-v4-pro", api_key="sk-test")
        result = agent.run("Hi")  # no callback
        assert result == "OK"

    @patch("harness.agent.OpenAI")
    def test_thinking_event_with_reasoning_content(self, mock_openai):
        """When the response includes reasoning_content, a thinking event is emitted."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.tool_calls = None
        resp.choices[0].message.content = "The answer is 42."
        resp.choices[0].message.reasoning_content = "Step 1: analyze. Step 2: conclude."
        mock_client.chat.completions.create.return_value = resp

        agent = AgentHarness(model="deepseek-v4-pro", api_key="sk-test")
        events = []
        result = agent.run("What is the answer?", callback=events.append)

        assert result == "The answer is 42."
        # tokens fires before thinking, which fires before text
        assert len(events) == 3
        assert events[0]["type"] == "tokens"
        assert events[1] == {
            "type": "thinking",
            "content": "Step 1: analyze. Step 2: conclude.",
        }
        assert events[2] == {"type": "text", "content": "The answer is 42."}


class TestTokenEvent:
    """After each API call, the agent emits a 'tokens' event with usage stats."""

    @patch("harness.agent.OpenAI")
    def test_emits_tokens_event_after_each_api_call(self, mock_openai):
        """A 'tokens' event is emitted after every successful API call."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.tool_calls = None
        resp.choices[0].message.content = "OK"
        resp.usage = MagicMock()
        resp.usage.prompt_tokens = 100
        resp.usage.completion_tokens = 50
        resp.usage.total_tokens = 150
        resp.usage.prompt_tokens_details = None
        mock_client.chat.completions.create.return_value = resp

        agent = AgentHarness(model="deepseek-v4-pro", api_key="sk-test")
        events = []
        result = agent.run("Hello", callback=events.append)

        assert result == "OK"
        token_events = [e for e in events if e["type"] == "tokens"]
        assert len(token_events) == 1
        te = token_events[0]
        assert te["input_tokens"] == 100
        assert te["output_tokens"] == 50
        assert te["total_tokens"] == 150
        assert te["turn_input"] == 100
        assert te["turn_output"] == 50
        assert te["cached_tokens"] == 0

    @patch("harness.agent.OpenAI")
    def test_tokens_event_cumulative_across_turns(self, mock_openai):
        """Token counts accumulate cumulatively across multiple API calls."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # First response: tool call
        resp1 = MagicMock()
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "bash"
        tc.function.arguments = '{"command": "echo hello"}'
        resp1.choices = [MagicMock()]
        resp1.choices[0].message.tool_calls = [tc]
        resp1.choices[0].message.content = None
        resp1.usage = MagicMock()
        resp1.usage.prompt_tokens = 50
        resp1.usage.completion_tokens = 30
        resp1.usage.total_tokens = 80
        resp1.usage.prompt_tokens_details = None

        # Second response: final text
        resp2 = MagicMock()
        resp2.choices = [MagicMock()]
        resp2.choices[0].message.tool_calls = None
        resp2.choices[0].message.content = "Done."
        resp2.usage = MagicMock()
        resp2.usage.prompt_tokens = 60
        resp2.usage.completion_tokens = 20
        resp2.usage.total_tokens = 80
        resp2.usage.prompt_tokens_details = None

        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        agent = AgentHarness(model="deepseek-v4-pro", api_key="sk-test")
        events = []
        result = agent.run("Do something", callback=events.append)

        assert result == "Done."
        token_events = [e for e in events if e["type"] == "tokens"]
        assert len(token_events) == 2

        # First turn
        assert token_events[0]["input_tokens"] == 50
        assert token_events[0]["output_tokens"] == 30
        assert token_events[0]["turn_input"] == 50
        assert token_events[0]["turn_output"] == 30

        # Second turn — cumulative
        assert token_events[1]["input_tokens"] == 110  # 50 + 60
        assert token_events[1]["output_tokens"] == 50  # 30 + 20
        assert token_events[1]["turn_input"] == 60
        assert token_events[1]["turn_output"] == 20

    @patch("harness.agent.OpenAI")
    def test_tokens_event_includes_cache_info_when_present(self, mock_openai):
        """When prompt_tokens_details has cached_tokens, it is included."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.tool_calls = None
        resp.choices[0].message.content = "OK"
        resp.usage = MagicMock()
        resp.usage.prompt_tokens = 200
        resp.usage.completion_tokens = 80
        resp.usage.total_tokens = 280
        prompt_details = MagicMock()
        prompt_details.cached_tokens = 150
        resp.usage.prompt_tokens_details = prompt_details
        mock_client.chat.completions.create.return_value = resp

        agent = AgentHarness(model="deepseek-v4-pro", api_key="sk-test")
        events = []
        result = agent.run("Hello", callback=events.append)

        assert result == "OK"
        token_events = [e for e in events if e["type"] == "tokens"]
        assert len(token_events) == 1
        te = token_events[0]
        assert te["cached_tokens"] == 150
        assert te["turn_cached"] == 150

    @patch("harness.agent.OpenAI")
    def test_tokens_event_cache_accumulates_across_turns(self, mock_openai):
        """Cache hits accumulate cumulatively across API calls."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        resp1 = MagicMock()
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "bash"
        tc.function.arguments = '{"command": "echo hello"}'
        resp1.choices = [MagicMock()]
        resp1.choices[0].message.tool_calls = [tc]
        resp1.choices[0].message.content = None
        resp1.usage = MagicMock()
        resp1.usage.prompt_tokens = 100
        resp1.usage.completion_tokens = 50
        resp1.usage.total_tokens = 150
        pd1 = MagicMock()
        pd1.cached_tokens = 80
        resp1.usage.prompt_tokens_details = pd1

        resp2 = MagicMock()
        resp2.choices = [MagicMock()]
        resp2.choices[0].message.tool_calls = None
        resp2.choices[0].message.content = "Done."
        resp2.usage = MagicMock()
        resp2.usage.prompt_tokens = 120
        resp2.usage.completion_tokens = 30
        resp2.usage.total_tokens = 150
        pd2 = MagicMock()
        pd2.cached_tokens = 60
        resp2.usage.prompt_tokens_details = pd2

        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        agent = AgentHarness(model="deepseek-v4-pro", api_key="sk-test")
        events = []
        result = agent.run("Do something", callback=events.append)

        assert result == "Done."
        token_events = [e for e in events if e["type"] == "tokens"]
        assert len(token_events) == 2
        assert token_events[0]["cached_tokens"] == 80
        assert token_events[1]["cached_tokens"] == 140  # 80 + 60
        assert token_events[0]["turn_cached"] == 80
        assert token_events[1]["turn_cached"] == 60

    @patch("harness.agent.OpenAI")
    def test_tokens_event_without_usage_is_graceful(self, mock_openai):
        """When response has no usage attribute, tokens event still emits with zeros."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.tool_calls = None
        resp.choices[0].message.content = "OK"
        del resp.usage  # no usage at all
        mock_client.chat.completions.create.return_value = resp

        agent = AgentHarness(model="deepseek-v4-pro", api_key="sk-test")
        events = []
        result = agent.run("Hello", callback=events.append)

        assert result == "OK"
        token_events = [e for e in events if e["type"] == "tokens"]
        assert len(token_events) == 1
        te = token_events[0]
        assert te["input_tokens"] == 0
        assert te["output_tokens"] == 0
        assert te["cached_tokens"] == 0

    @patch("harness.agent.OpenAI")
    def test_tokens_event_clear_history_resets_counters(self, mock_openai):
        """After clear_history(), token counters reset to zero."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.tool_calls = None
        resp.choices[0].message.content = "OK"
        resp.usage = MagicMock()
        resp.usage.prompt_tokens = 100
        resp.usage.completion_tokens = 50
        resp.usage.total_tokens = 150
        resp.usage.prompt_tokens_details = None
        mock_client.chat.completions.create.return_value = resp

        agent = AgentHarness(model="deepseek-v4-pro", api_key="sk-test")

        # First run
        events1 = []
        agent.run("First", callback=events1.append)
        te1 = [e for e in events1 if e["type"] == "tokens"][0]
        assert te1["input_tokens"] == 100

        # Clear and run again
        agent.clear_history()
        events2 = []
        agent.run("Second", callback=events2.append)
        te2 = [e for e in events2 if e["type"] == "tokens"][0]
        assert te2["input_tokens"] == 100  # fresh counter, not 200

    @patch("harness.agent.OpenAI")
    def test_callback_not_required_for_tokens_event(self, mock_openai):
        """When no callback is provided, token tracking still works internally."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.tool_calls = None
        resp.choices[0].message.content = "OK"
        resp.usage = MagicMock()
        resp.usage.prompt_tokens = 30
        resp.usage.completion_tokens = 20
        resp.usage.total_tokens = 50
        resp.usage.prompt_tokens_details = None
        mock_client.chat.completions.create.return_value = resp

        agent = AgentHarness(model="deepseek-v4-pro", api_key="sk-test")
        result = agent.run("Hello")  # no callback
        assert result == "OK"
        assert agent.input_tokens == 30
        assert agent.output_tokens == 20

    @patch("harness.agent.OpenAI")
    def test_tokens_event_includes_model_and_reasoning_effort(self, mock_openai):
        """The tokens event includes the model and reasoning_effort from the agent."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.tool_calls = None
        resp.choices[0].message.content = "OK"
        resp.usage = MagicMock()
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        resp.usage.total_tokens = 15
        resp.usage.prompt_tokens_details = None
        mock_client.chat.completions.create.return_value = resp

        agent = AgentHarness(
            model="deepseek-v4-pro",
            api_key="sk-test",
            reasoning_effort="high",
        )
        events = []
        result = agent.run("Hello", callback=events.append)

        assert result == "OK"
        token_events = [e for e in events if e["type"] == "tokens"]
        assert len(token_events) == 1
        te = token_events[0]
        assert te["model"] == "deepseek-v4-pro"
        assert te["reasoning_effort"] == "high"

    @patch("harness.agent.OpenAI")
    def test_tokens_event_reasoning_effort_none_when_not_set(self, mock_openai):
        """When reasoning_effort is None, the token event includes None."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.tool_calls = None
        resp.choices[0].message.content = "OK"
        resp.usage = MagicMock()
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        resp.usage.total_tokens = 15
        resp.usage.prompt_tokens_details = None
        mock_client.chat.completions.create.return_value = resp

        agent = AgentHarness(model="deepseek-v4-pro", api_key="sk-test")
        events = []
        result = agent.run("Hello", callback=events.append)

        assert result == "OK"
        token_events = [e for e in events if e["type"] == "tokens"]
        assert len(token_events) == 1
        te = token_events[0]
        assert te["model"] == "deepseek-v4-pro"
        assert te["reasoning_effort"] is None

    @patch("harness.agent.OpenAI")
    def test_tokens_event_order_before_text(self, mock_openai):
        """The tokens event fires before the final text event."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.tool_calls = None
        resp.choices[0].message.content = "Final answer."
        resp.usage = MagicMock()
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        resp.usage.total_tokens = 15
        resp.usage.prompt_tokens_details = None
        mock_client.chat.completions.create.return_value = resp

        agent = AgentHarness(model="deepseek-v4-pro", api_key="sk-test")
        events = []
        result = agent.run("Hello", callback=events.append)

        assert result == "Final answer."
        assert events[0]["type"] == "tokens"
        assert events[1]["type"] == "text"
