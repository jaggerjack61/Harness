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
    def test_empty_choices_does_not_crash(self, mock_openai):
        """An empty choices array (e.g. content_filter) must not crash run()."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        resp = MagicMock()
        resp.choices = []
        mock_client.chat.completions.create.return_value = resp

        agent = AgentHarness(model="test-model", api_key="sk-test")
        result = agent.run("Hello")

        assert result == ""

    @patch("harness.agent.time.sleep")
    @patch("harness.agent.OpenAI")
    def test_retries_on_rate_limit_error(self, mock_openai, mock_sleep):
        """429 errors should be retried, not abort the turn."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        class FakeAPIError(Exception):
            def __init__(self, status_code):
                self.status_code = status_code
                super().__init__(f"HTTP {status_code}")

        success_resp = MagicMock()
        success_resp.choices = [MagicMock()]
        success_resp.choices[0].message.tool_calls = None
        success_resp.choices[0].message.content = "Success!"

        mock_client.chat.completions.create.side_effect = [
            FakeAPIError(429), success_resp
        ]

        agent = AgentHarness(model="test-model", api_key="sk-test")
        result = agent.run("Hello")

        assert result == "Success!"
        assert mock_client.chat.completions.create.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("harness.agent.time.sleep")
    @patch("harness.agent.OpenAI")
    def test_retries_on_server_error(self, mock_openai, mock_sleep):
        """500 errors should be retried."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        class FakeAPIError(Exception):
            def __init__(self, status_code):
                self.status_code = status_code
                super().__init__(f"HTTP {status_code}")

        success_resp = MagicMock()
        success_resp.choices = [MagicMock()]
        success_resp.choices[0].message.tool_calls = None
        success_resp.choices[0].message.content = "OK"

        mock_client.chat.completions.create.side_effect = [
            FakeAPIError(503), success_resp
        ]

        agent = AgentHarness(model="test-model", api_key="sk-test")
        result = agent.run("Hello")

        assert result == "OK"
        assert mock_client.chat.completions.create.call_count == 2

    @patch("harness.agent.time.sleep")
    @patch("harness.agent.OpenAI")
    def test_does_not_retry_on_client_error(self, mock_openai, mock_sleep):
        """400 errors should NOT be retried — they are not transient."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        class FakeAPIError(Exception):
            def __init__(self, status_code):
                self.status_code = status_code
                super().__init__(f"HTTP {status_code}")

        mock_client.chat.completions.create.side_effect = FakeAPIError(400)

        agent = AgentHarness(model="test-model", api_key="sk-test")
        with pytest.raises(Exception):
            agent.run("Hello")
        assert mock_client.chat.completions.create.call_count == 1
        assert mock_sleep.call_count == 0

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

    @patch("harness.agent.OpenAI")
    def test_finish_reason_length_emits_warning(self, mock_openai):
        """finish_reason='length' should emit a warning callback event."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.tool_calls = None
        resp.choices[0].message.content = "Truncated text..."
        resp.choices[0].finish_reason = "length"
        mock_client.chat.completions.create.return_value = resp

        agent = AgentHarness(model="test-model", api_key="sk-test")
        events = []
        agent.run("Write a long essay", callback=events.append)

        warnings = [e for e in events if e["type"] == "finish_reason"]
        assert len(warnings) == 1
        assert warnings[0]["reason"] == "length"

    @patch("harness.agent.OpenAI")
    def test_finish_reason_content_filter_emits_warning(self, mock_openai):
        """finish_reason='content_filter' should emit a warning callback event."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.tool_calls = None
        resp.choices[0].message.content = ""
        resp.choices[0].finish_reason = "content_filter"
        mock_client.chat.completions.create.return_value = resp

        agent = AgentHarness(model="test-model", api_key="sk-test")
        events = []
        agent.run("Something flagged", callback=events.append)

        warnings = [e for e in events if e["type"] == "finish_reason"]
        assert len(warnings) == 1
        assert warnings[0]["reason"] == "content_filter"

    @patch("harness.agent.OpenAI")
    def test_finish_reason_stop_no_warning(self, mock_openai):
        """finish_reason='stop' should NOT emit a warning callback event."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.tool_calls = None
        resp.choices[0].message.content = "Complete answer."
        resp.choices[0].finish_reason = "stop"
        mock_client.chat.completions.create.return_value = resp

        agent = AgentHarness(model="test-model", api_key="sk-test")
        events = []
        agent.run("Hello", callback=events.append)

        warnings = [e for e in events if e["type"] == "finish_reason"]
        assert len(warnings) == 0


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
        # tokens + tool_call + tool_result + text_end(intermediate) + turn_start + tokens + text = 7
        assert len(events) == 7
        assert events[0]["type"] == "tokens"
        assert events[1] == {
            "type": "tool_call",
            "name": "read",
            "arguments": {"path": "/tmp/x.txt"},
        }
        assert events[2]["type"] == "tool_result"
        assert events[2]["name"] == "read"
        assert "Error" in events[2]["result"]  # file doesn't exist
        assert events[3] == {"type": "text_end", "content": "Let me read the file."}
        assert events[4]["type"] == "turn_start"
        assert events[5]["type"] == "tokens"
        assert events[6] == {"type": "text", "content": "File contents: hello"}

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
            "turn_start",                 # before the second turn
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

    @patch("harness.agent.OpenAI")
    def test_reasoning_not_stored_in_api_bound_messages(self, mock_openai):
        """reasoning_content must NOT be stored in messages sent back to the API.

        Strict OpenAI-compatible providers 400 on the unknown field, and
        re-sent CoT bloats every subsequent request.
        """
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        resp1 = MagicMock()
        resp1.choices = [MagicMock()]
        resp1.choices[0].message.tool_calls = None
        resp1.choices[0].message.content = "Answer."
        resp1.choices[0].message.reasoning_content = "My reasoning."

        resp2 = MagicMock()
        resp2.choices = [MagicMock()]
        resp2.choices[0].message.tool_calls = None
        resp2.choices[0].message.content = "Follow-up answer."

        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        agent = AgentHarness(model="test-model", api_key="sk-test")
        agent.run("First question")
        agent.run("Second question")

        # Check the messages sent on the second API call.
        second_call_messages = mock_client.chat.completions.create.call_args_list[1][1]["messages"]
        for msg in second_call_messages:
            assert "reasoning_content" not in msg, (
                f"reasoning_content found in {msg['role']} message sent to API"
            )


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


class TestTrackAndEmitTokens:
    """The shared token-tracking helper accumulates and emits one event."""

    def test_accumulates_and_emits_event(self):
        agent = AgentHarness(model="deepseek-v4-pro", api_key="sk-test",
                              reasoning_effort="high", context_window=256000)
        agent.input_tokens = 100
        agent.output_tokens = 20
        agent.cached_tokens = 5
        events = []
        agent._track_and_emit_tokens(50, 10, 3, callback=events.append)
        assert agent.input_tokens == 150
        assert agent.output_tokens == 30
        assert agent.cached_tokens == 8
        assert len(events) == 1
        e = events[0]
        assert e["type"] == "tokens"
        assert e["input_tokens"] == 150
        assert e["output_tokens"] == 30
        assert e["total_tokens"] == 180
        assert e["cached_tokens"] == 8
        assert e["turn_input"] == 50
        assert e["turn_output"] == 10
        assert e["turn_cached"] == 3
        assert e["model"] == "deepseek-v4-pro"
        assert e["reasoning_effort"] == "high"
        assert e["context_window"] == 256000

    def test_no_callback_still_accumulates(self):
        agent = AgentHarness(model="m", api_key="sk-test")
        agent._track_and_emit_tokens(40, 5, 0, callback=None)
        assert agent.input_tokens == 40
        assert agent.output_tokens == 5
        assert agent.cached_tokens == 0


class TestExtractReasoningFields:
    """Shared helper extracts reasoning from deltas and messages alike."""

    def _obj(self, **kw):
        m = MagicMock()
        m.reasoning_content = kw.get("reasoning_content", None)
        m.thinking = kw.get("thinking", None)
        m.thought = kw.get("thought", None)
        m.model_extra = kw.get("model_extra", None)
        return m

    def test_extracts_direct_reasoning_content(self):
        from harness.agent import _extract_reasoning_fields
        assert _extract_reasoning_fields(self._obj(reasoning_content="hello")) == "hello"

    def test_extracts_thinking_attr(self):
        from harness.agent import _extract_reasoning_fields
        assert _extract_reasoning_fields(self._obj(thinking="step")) == "step"

    def test_extracts_thought_attr(self):
        from harness.agent import _extract_reasoning_fields
        assert _extract_reasoning_fields(self._obj(thought="idea")) == "idea"

    def test_extracts_from_model_extra_reasoning(self):
        from harness.agent import _extract_reasoning_fields
        assert _extract_reasoning_fields(self._obj(model_extra={"reasoning": "deep"})) == "deep"

    def test_extracts_from_model_extra_thinking(self):
        from harness.agent import _extract_reasoning_fields
        assert _extract_reasoning_fields(self._obj(model_extra={"thinking": "x"})) == "x"

    def test_extracts_from_model_extra_thought(self):
        from harness.agent import _extract_reasoning_fields
        assert _extract_reasoning_fields(self._obj(model_extra={"thought": "y"})) == "y"

    def test_returns_none_when_absent(self):
        from harness.agent import _extract_reasoning_fields
        assert _extract_reasoning_fields(self._obj()) is None

    def test_precedence_reasoning_content_over_thinking(self):
        from harness.agent import _extract_reasoning_fields
        obj = self._obj(reasoning_content="primary", thinking="secondary")
        assert _extract_reasoning_fields(obj) == "primary"

    def test_ignores_non_string_values(self):
        from harness.agent import _extract_reasoning_fields
        obj = self._obj(thinking={"not": "a string"})
        assert _extract_reasoning_fields(obj) is None

    def test_ignores_empty_string(self):
        from harness.agent import _extract_reasoning_fields
        obj = self._obj(reasoning_content="")
        assert _extract_reasoning_fields(obj) is None


class TestContextWindowManagement:
    """Tests for context-window management with summarization (B2)."""

    @patch("harness.agent.OpenAI")
    def test_trims_history_when_approaching_context_window(self, mock_openai):
        """When history nears the context window, old turns are summarized."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        agent = AgentHarness(model="test-model", api_key="sk-test", context_window=1000)
        # Pre-populate history with many messages.
        agent.messages = [{"role": "system", "content": "System prompt"}]
        for i in range(20):
            agent.messages.append({"role": "user", "content": f"Question {i} " * 100})
            agent.messages.append({"role": "assistant", "content": f"Answer {i} " * 100})
        original_count = len(agent.messages)
        # Simulate that the last API call reported prompt_tokens near the limit.
        agent._last_prompt_tokens = 900

        # Mock: first call = summary, second call = final response.
        summary_resp = MagicMock()
        summary_resp.choices = [MagicMock()]
        summary_resp.choices[0].message.content = "Summary of earlier conversation"
        summary_resp.choices[0].message.tool_calls = None
        summary_resp.usage = MagicMock()
        summary_resp.usage.prompt_tokens = 100
        summary_resp.usage.completion_tokens = 50
        summary_resp.usage.prompt_tokens_details = None

        final_resp = MagicMock()
        final_resp.choices = [MagicMock()]
        final_resp.choices[0].message.content = "Final answer"
        final_resp.choices[0].message.tool_calls = None
        final_resp.usage = MagicMock()
        final_resp.usage.prompt_tokens = 200
        final_resp.usage.completion_tokens = 50
        final_resp.usage.prompt_tokens_details = None

        mock_client.chat.completions.create.side_effect = [summary_resp, final_resp]

        result = agent.run("New question")

        assert result == "Final answer"
        # History should be shorter than before.
        assert len(agent.messages) < original_count
        # A summary system message should be present.
        summaries = [
            m for m in agent.messages
            if m.get("role") == "system" and "Summary" in m.get("content", "")
        ]
        assert len(summaries) == 1
        assert "Summary of earlier conversation" in summaries[0]["content"]

    @patch("harness.agent.OpenAI")
    def test_does_not_trim_when_history_is_small(self, mock_openai):
        """No trimming or summarization when history is well within the window."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        agent = AgentHarness(model="test-model", api_key="sk-test", context_window=100000)
        agent.messages = [{"role": "system", "content": "System"}]
        agent.messages.append({"role": "user", "content": "Hi"})
        agent.messages.append({"role": "assistant", "content": "Hello"})
        agent._last_prompt_tokens = 50

        final_resp = MagicMock()
        final_resp.choices = [MagicMock()]
        final_resp.choices[0].message.content = "Response"
        final_resp.choices[0].message.tool_calls = None
        final_resp.usage = MagicMock()
        final_resp.usage.prompt_tokens = 50
        final_resp.usage.completion_tokens = 10
        final_resp.usage.prompt_tokens_details = None

        mock_client.chat.completions.create.return_value = final_resp

        agent.run("Follow-up")

        # Only one API call (no summary call).
        assert mock_client.chat.completions.create.call_count == 1
        # No summary message added.
        summaries = [
            m for m in agent.messages
            if m.get("role") == "system" and "Summary" in m.get("content", "")
        ]
        assert len(summaries) == 0

    @patch("harness.agent.OpenAI")
    def test_trim_preserves_system_prompt_and_recent_turns(self, mock_openai):
        """Trimming keeps the original system prompt and the most recent turns."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        agent = AgentHarness(model="test-model", api_key="sk-test", context_window=1000)
        system_msg = {"role": "system", "content": "System prompt"}
        agent.messages = [system_msg]
        for i in range(10):
            agent.messages.append({"role": "user", "content": f"Q{i} " * 100})
            agent.messages.append({"role": "assistant", "content": f"A{i} " * 100})
        agent._last_prompt_tokens = 900

        summary_resp = MagicMock()
        summary_resp.choices = [MagicMock()]
        summary_resp.choices[0].message.content = "Summary text"
        summary_resp.choices[0].message.tool_calls = None
        summary_resp.usage = MagicMock()
        summary_resp.usage.prompt_tokens = 100
        summary_resp.usage.completion_tokens = 50
        summary_resp.usage.prompt_tokens_details = None

        final_resp = MagicMock()
        final_resp.choices = [MagicMock()]
        final_resp.choices[0].message.content = "Done"
        final_resp.choices[0].message.tool_calls = None
        final_resp.usage = MagicMock()
        final_resp.usage.prompt_tokens = 200
        final_resp.usage.completion_tokens = 50
        final_resp.usage.prompt_tokens_details = None

        mock_client.chat.completions.create.side_effect = [summary_resp, final_resp]

        agent.run("New question")

        # System prompt is preserved as the first message.
        assert agent.messages[0] is system_msg or agent.messages[0]["content"] == "System prompt"
        # The new user question is present in the kept history.
        user_msgs = [m for m in agent.messages if m.get("role") == "user"]
        assert any("New question" in m["content"] for m in user_msgs)
