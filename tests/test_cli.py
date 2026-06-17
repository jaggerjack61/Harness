"""Tests for the interactive CLI."""

import sys
from io import StringIO
from unittest.mock import patch, MagicMock

import pytest

from prompt_toolkit.keys import Keys

from harness.cli import (
    _build_parser,
    _interactive_select,
    _on_event,
    _prompt_model_selection,
    _prompt_reasoning_selection,
    REASONING_OPTIONS,
)


class TestArgumentParser:
    def test_default_model(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.model == "deepseek-v4-pro"

    def test_custom_model(self):
        parser = _build_parser()
        args = parser.parse_args(["--model", "gpt-3.5-turbo"])
        assert args.model == "gpt-3.5-turbo"

    def test_short_flags(self):
        parser = _build_parser()
        args = parser.parse_args(["-m", "claude-3", "-d", "/tmp", "-k", "sk-abc"])
        assert args.model == "claude-3"
        assert args.dir == "/tmp"
        assert args.api_key == "sk-abc"

    def test_max_turns_default(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.max_turns == 25

    def test_max_turns_custom(self):
        parser = _build_parser()
        args = parser.parse_args(["--max-turns", "10"])
        assert args.max_turns == 10

    def test_reasoning_effort_default_is_high(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.reasoning_effort == "high"

    def test_reasoning_effort_xhigh_allowed(self):
        parser = _build_parser()
        args = parser.parse_args(["--reasoning-effort", "xhigh"])
        assert args.reasoning_effort == "xhigh"

    def test_reasoning_effort_low_allowed(self):
        parser = _build_parser()
        args = parser.parse_args(["--reasoning-effort", "low"])
        assert args.reasoning_effort == "low"


class TestOnEvent:
    def setup_method(self):
        """Reset module-level state before each test."""
        import harness.cli as cli
        cli._thinking_streaming = False

    def test_tool_call_event_prints(self, capsys):
        _on_event({
            "type": "tool_call",
            "name": "read",
            "arguments": {"path": "/tmp/foo.txt"},
        })
        captured = capsys.readouterr()
        assert "read" in captured.out
        assert "foo.txt" in captured.out

    def test_tool_result_event_prints(self, capsys):
        _on_event({
            "type": "tool_result",
            "name": "bash",
            "result": "hello world",
        })
        captured = capsys.readouterr()
        assert "bash" in captured.out or "hello world" in captured.out

    def test_text_event_is_silent(self, capsys):
        _on_event({
            "type": "text",
            "content": "Some final answer.",
        })
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_thinking_event_prints_in_gray(self, capsys):
        _on_event({
            "type": "thinking",
            "content": "Let me reason through this step by step...",
        })
        captured = capsys.readouterr()
        assert "reason" in captured.out
        # Fallback path (no _console) uses plain text without ANSI codes
        assert "\033[90m" not in captured.out

    def test_thinking_delta_event_prints_content(self, capsys):
        """thinking_delta should output content (with Rich dim style if console available)."""
        _on_event({
            "type": "thinking_delta",
            "content": "reasoning chunk",
        })
        captured = capsys.readouterr()
        assert "reasoning chunk" in captured.out
        assert "\033[90m" not in captured.out  # no raw ANSI codes

    def test_text_delta_event_is_silent(self, capsys):
        """text_delta should not print anything — full text rendered at text_end."""
        _on_event({
            "type": "text_delta",
            "content": "some streaming text",
        })
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_text_end_renders_markdown(self, capsys):
        """text_end should render full content as markdown."""
        import harness.cli as cli
        cli._no_markdown = False
        cli._console = None  # force creation of a new console
        _on_event({
            "type": "text_end",
            "content": "**bold** and `code`",
        })
        captured = capsys.readouterr()
        # The markdown rendered output should contain the bold text
        assert "bold" in captured.out
        assert "code" in captured.out

    def test_text_end_plain_text_when_no_markdown(self, capsys):
        """text_end should print plain text when _no_markdown is True."""
        import harness.cli as cli
        cli._no_markdown = True
        _on_event({
            "type": "text_end",
            "content": "**bold** and `code`",
        })
        captured = capsys.readouterr()
        assert "**bold**" in captured.out  # raw markdown, not rendered
        assert "`code`" in captured.out

    def test_text_end_empty_content_does_not_crash(self, capsys):
        """text_end with empty content should not crash."""
        import harness.cli as cli
        cli._no_markdown = False
        cli._console = None
        _on_event({
            "type": "text_end",
            "content": "",
        })
        captured = capsys.readouterr()
        # Should not crash; should output newlines at minimum
        assert captured.out is not None


class TestToolResultTruncation:
    """GUI should only show the first 5 lines of tool results."""

    def test_short_output_shown_fully(self, capsys):
        """5 lines or fewer should be displayed completely."""
        result = "line1\nline2\nline3\nline4\nline5"
        _on_event({
            "type": "tool_result",
            "name": "bash",
            "result": result,
        })
        captured = capsys.readouterr()
        assert "line1" in captured.out
        assert "line5" in captured.out
        assert "truncat" not in captured.out.lower()

    def test_long_output_truncated_to_first_5_lines(self, capsys):
        """More than 5 lines: only first 5 lines shown + truncation notice."""
        lines = [f"line{i}" for i in range(1, 11)]  # 10 lines
        result = "\n".join(lines)
        _on_event({
            "type": "tool_result",
            "name": "bash",
            "result": result,
        })
        captured = capsys.readouterr()
        # First 5 lines should be present
        assert "line1" in captured.out
        assert "line2" in captured.out
        assert "line3" in captured.out
        assert "line4" in captured.out
        assert "line5" in captured.out
        # Lines 6+ should NOT be present
        assert "line6" not in captured.out
        assert "line7" not in captured.out
        assert "line10" not in captured.out
        # Should show a truncation notice
        assert "truncat" in captured.out.lower()
        # Should mention how many lines were hidden
        assert "5" in captured.out

    def test_exactly_6_lines_shows_first_5_with_notice(self, capsys):
        """Boundary: exactly 6 lines triggers truncation of 1 line."""
        lines = [f"line{i}" for i in range(1, 7)]  # 6 lines
        result = "\n".join(lines)
        _on_event({
            "type": "tool_result",
            "name": "bash",
            "result": result,
        })
        captured = capsys.readouterr()
        assert "line5" in captured.out
        assert "line6" not in captured.out
        assert "truncat" in captured.out.lower()
        assert "1" in captured.out  # 1 line hidden

    def test_single_line_output_shown_fully(self, capsys):
        """A single-line result should pass through unchanged."""
        _on_event({
            "type": "tool_result",
            "name": "bash",
            "result": "just one line",
        })
        captured = capsys.readouterr()
        assert "just one line" in captured.out
        assert "truncat" not in captured.out.lower()

    def test_empty_output_shown_as_is(self, capsys):
        """Empty result should still show header without crash."""
        _on_event({
            "type": "tool_result",
            "name": "bash",
            "result": "",
        })
        captured = capsys.readouterr()
        assert "result" in captured.out.lower()
        assert "truncat" not in captured.out.lower()

    def test_truncation_notice_shows_total_line_count(self, capsys):
        """The notice should tell the user total lines and how many hidden."""
        lines = [f"line{i}" for i in range(1, 21)]  # 20 lines
        result = "\n".join(lines)
        _on_event({
            "type": "tool_result",
            "name": "bash",
            "result": result,
        })
        captured = capsys.readouterr()
        assert "15" in captured.out  # 20 - 5 = 15 hidden
        assert "20" in captured.out  # total lines


class TestAgentResultNotTruncated:
    """Full tool results must reach the LLM; truncation is display-only."""

    @patch("harness.agent.OpenAI")
    def test_full_bash_output_sent_to_model(self, mock_openai):
        """The tool result appended to messages for the LLM is NOT truncated."""
        from harness.agent import AgentHarness

        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Generate a known 20-line result to inject via mock
        full_output = "\n".join(f"line{i}" for i in range(1, 21))

        # First response: tool call
        resp1 = MagicMock()
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "bash"
        tc.function.arguments = '{"command": "generate-lines"}'
        resp1.choices = [MagicMock()]
        resp1.choices[0].message.tool_calls = [tc]
        resp1.choices[0].message.content = None
        resp1.usage = MagicMock()
        resp1.usage.prompt_tokens = 10
        resp1.usage.completion_tokens = 10

        # Second response: final text
        resp2 = MagicMock()
        resp2.choices = [MagicMock()]
        resp2.choices[0].message.tool_calls = None
        resp2.choices[0].message.content = "Done."
        resp2.usage = MagicMock()
        resp2.usage.prompt_tokens = 10
        resp2.usage.completion_tokens = 10

        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        agent = AgentHarness(model="deepseek-v4-pro", api_key="sk-test")
        # Mock the tool registry to return known multi-line output
        agent.tool_registry.execute = MagicMock(return_value=full_output)
        agent.run("run a command generating 20 lines")

        # Find the tool message in history
        tool_messages = [m for m in agent.messages if m["role"] == "tool"]
        assert len(tool_messages) == 1
        result_content = tool_messages[0]["content"]
        # The full output must be preserved in the message to the model
        assert result_content == full_output


class TestCliMain:
    @patch("harness.cli.AgentHarness")
    @patch("harness.cli.input")
    def test_runs_without_api_key_arg(self, mock_input, mock_harness):
        """Agent should start successfully without providing --api-key (uses hardcoded credentials)."""
        mock_agent = MagicMock()
        mock_harness.return_value = mock_agent
        mock_input.side_effect = ["/exit"]

        from harness.cli import main
        with patch.dict("os.environ", {}, clear=True):
            main([])

        mock_harness.assert_called_once()

    @patch("harness.cli.AgentHarness")
    @patch("harness.cli.input")
    def test_exit_command_breaks_loop(self, mock_input, mock_harness):
        """Typing /exit should exit the loop."""
        mock_agent = MagicMock()
        mock_harness.return_value = mock_agent
        mock_input.side_effect = ["/exit"]

        from harness.cli import main
        main(["--api-key", "sk-test"])
        # Should exit cleanly without calling agent.run()
        mock_agent.run.assert_not_called()

    @patch("harness.cli.AgentHarness")
    @patch("harness.cli.input")
    def test_clear_command_resets_history(self, mock_input, mock_harness):
        """Typing /clear should reset agent history."""
        mock_agent = MagicMock()
        mock_harness.return_value = mock_agent
        mock_input.side_effect = ["/clear", "/exit"]

        from harness.cli import main
        main(["--api-key", "sk-test"])
        mock_agent.clear_history.assert_called_once()

    @patch("harness.cli.AgentHarness")
    @patch("harness.cli.input")
    def test_user_prompt_runs_agent_with_callback(self, mock_input, mock_harness):
        """A normal prompt should call agent.run() with a callback."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = "Hello back!"
        mock_agent.input_tokens = 42
        mock_agent.output_tokens = 13
        mock_harness.return_value = mock_agent
        mock_input.side_effect = ["Say hi", "/exit"]

        from harness.cli import main
        main(["--api-key", "sk-test"])

        mock_agent.run.assert_called_once()
        call_args = mock_agent.run.call_args
        assert call_args[0][0] == "Say hi"  # positional arg: prompt
        assert callable(call_args[1]["callback"])  # keyword arg: callback

    @patch("harness.cli.AgentHarness")
    @patch("harness.cli.input")
    def test_empty_input_is_skipped(self, mock_input, mock_harness):
        """Empty input should be skipped, not passed to the agent."""
        mock_agent = MagicMock()
        mock_harness.return_value = mock_agent
        mock_input.side_effect = ["", "   ", "/exit"]

        from harness.cli import main
        main(["--api-key", "sk-test"])

        mock_agent.run.assert_not_called()

    @patch("harness.cli.AgentHarness")
    @patch("harness.cli.input")
    def test_reasoning_command_changes_effort(self, mock_input, mock_harness):
        """Typing /reasoning should change the agent's reasoning_effort."""
        mock_agent = MagicMock()
        mock_agent.reasoning_effort = "high"
        mock_harness.return_value = mock_agent
        mock_input.side_effect = ["/reasoning", "2", "/exit"]

        from harness.cli import main
        main(["--api-key", "sk-test"])
        assert mock_agent.reasoning_effort == "medium"

    @patch("harness.cli.AgentHarness")
    @patch("harness.cli.input")
    def test_reasoning_command_cancel_does_not_change(self, mock_input, mock_harness):
        """Cancelling /reasoning should keep the current effort."""
        mock_agent = MagicMock()
        mock_agent.reasoning_effort = "high"
        mock_harness.return_value = mock_agent
        mock_input.side_effect = ["/reasoning", "", "/exit"]

        from harness.cli import main
        main(["--api-key", "sk-test"])
        assert mock_agent.reasoning_effort == "high"

    @patch("harness.cli.AgentHarness")
    @patch("harness.cli.input")
    def test_context_command_sets_context(self, mock_input, mock_harness):
        """Typing /context should set the agent's custom context."""
        mock_agent = MagicMock()
        mock_agent.get_custom_context.return_value = "Some context"
        mock_harness.return_value = mock_agent
        mock_input.side_effect = ["/context", "Some context", ".", "/exit"]

        from harness.cli import main
        main(["--api-key", "sk-test"])
        mock_agent.set_custom_context.assert_called_once_with("Some context")

    @patch("harness.cli.AgentHarness")
    @patch("harness.cli.input")
    def test_context_clear_command_clears_context(self, mock_input, mock_harness):
        """Typing /context clear should clear the custom context."""
        mock_agent = MagicMock()
        mock_harness.return_value = mock_agent
        mock_input.side_effect = ["/context clear", "/exit"]

        from harness.cli import main
        main(["--api-key", "sk-test"])
        mock_agent.set_custom_context.assert_called_once_with(None)

    @patch("harness.cli.AgentHarness")
    @patch("harness.cli.input")
    def test_context_show_command_prints_context(self, mock_input, mock_harness, capsys):
        """Typing /context show should display the current custom context."""
        mock_agent = MagicMock()
        mock_agent.get_custom_context.return_value = "Current context"
        mock_harness.return_value = mock_agent
        mock_input.side_effect = ["/context show", "/exit"]

        from harness.cli import main
        main(["--api-key", "sk-test"])
        captured = capsys.readouterr()
        assert "Current context" in captured.out


class TestTokensEventDisplay:
    """Live token events should print stats in a compact format."""

    def test_tokens_event_without_cache(self, capsys):
        _on_event({
            "type": "tokens",
            "input_tokens": 150,
            "output_tokens": 50,
            "total_tokens": 200,
            "cached_tokens": 0,
            "turn_input": 150,
            "turn_output": 50,
            "turn_cached": 0,
        })
        captured = capsys.readouterr()
        assert "150" in captured.out
        assert "50" in captured.out
        assert "200" in captured.out

    def test_tokens_event_with_cache(self, capsys):
        _on_event({
            "type": "tokens",
            "input_tokens": 500,
            "output_tokens": 100,
            "total_tokens": 600,
            "cached_tokens": 300,
            "turn_input": 200,
            "turn_output": 100,
            "turn_cached": 150,
        })
        captured = capsys.readouterr()
        assert "500" in captured.out
        assert "100" in captured.out
        assert "300" in captured.out  # cache hit info
        assert "50.0%" in captured.out  # cache hit rate: 300/600 = 50%

    def test_tokens_event_zero_total(self, capsys):
        """Should not crash with zero token counts."""
        _on_event({
            "type": "tokens",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "turn_input": 0,
            "turn_output": 0,
            "turn_cached": 0,
        })
        captured = capsys.readouterr()
        assert "0" in captured.out


class TestBuildTokenText:
    """_build_token_text should include model and reasoning_effort when present."""

    def test_includes_model(self):
        from harness.cli import _build_token_text
        event = {
            "type": "tokens",
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "cached_tokens": 0,
            "turn_input": 100,
            "turn_output": 50,
            "turn_cached": 0,
            "model": "deepseek-v4-pro",
            "reasoning_effort": "high",
        }
        text = _build_token_text(event)
        rendered = text.plain
        assert "deepseek-v4-pro" in rendered

    def test_includes_reasoning_effort(self):
        from harness.cli import _build_token_text
        event = {
            "type": "tokens",
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "cached_tokens": 0,
            "turn_input": 100,
            "turn_output": 50,
            "turn_cached": 0,
            "model": "gpt-4o",
            "reasoning_effort": "xhigh",
        }
        text = _build_token_text(event)
        rendered = text.plain
        assert "xhigh" in rendered

    def test_reasoning_effort_none_not_displayed(self):
        from harness.cli import _build_token_text
        event = {
            "type": "tokens",
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "cached_tokens": 0,
            "turn_input": 100,
            "turn_output": 50,
            "turn_cached": 0,
            "model": "gpt-4o",
            "reasoning_effort": None,
        }
        text = _build_token_text(event)
        rendered = text.plain
        assert "gpt-4o" in rendered
        assert "reasoning" not in rendered.lower()

    def test_backward_compatible_without_model_field(self):
        """Old events without model/reasoning_effort still render fine."""
        from harness.cli import _build_token_text
        event = {
            "type": "tokens",
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "cached_tokens": 0,
            "turn_input": 100,
            "turn_output": 50,
            "turn_cached": 0,
        }
        text = _build_token_text(event)
        rendered = text.plain
        # Should still show the usual stats without errors
        assert "100" in rendered
        assert "50" in rendered

    def test_token_text_does_not_wrap(self):
        """Status bar text should stay on a single line even when long."""
        from harness.cli import _build_token_text
        event = {
            "type": "tokens",
            "input_tokens": 36103,
            "output_tokens": 674,
            "total_tokens": 36777,
            "cached_tokens": 20096,
            "turn_input": 15901,
            "turn_output": 393,
            "turn_cached": 0,
            "context_window": 1000000,
            "model": "mimo-v2.5-pro",
            "reasoning_effort": "high",
        }
        text = _build_token_text(event)
        assert text.no_wrap is True


def _make_fake_application(key_sequence):
    """Build a fake prompt_toolkit Application that replays a key sequence."""
    class FakeApp:
        def __init__(self, **kwargs):
            self.kb = kwargs.get("key_bindings")
            self.invalidate = MagicMock()
            self.exit = MagicMock()

        def run(self):
            event = MagicMock()
            event.app = self
            for key in key_sequence:
                bindings = self.kb.get_bindings_for_keys((key,))
                assert bindings, f"No binding registered for {key}"
                bindings[0].handler(event)

    return FakeApp


class TestInteractiveSelect:
    def test_select_second_option(self):
        fake_app = _make_fake_application([Keys.Down, Keys.Enter])
        with patch("harness.cli.Application", fake_app):
            selected = _interactive_select(["gpt-4", "gpt-3.5-turbo", "claude-3"], "gpt-4", "Models")
        assert selected == "gpt-3.5-turbo"

    def test_cancel_with_escape_returns_none(self):
        fake_app = _make_fake_application([Keys.Escape])
        with patch("harness.cli.Application", fake_app):
            selected = _interactive_select(["gpt-4", "gpt-3.5-turbo"], "gpt-4", "Models")
        assert selected is None

    def test_up_wraps_to_last_option(self):
        fake_app = _make_fake_application([Keys.Up, Keys.Enter])
        with patch("harness.cli.Application", fake_app):
            selected = _interactive_select(["gpt-4", "gpt-3.5-turbo", "claude-3"], "gpt-4", "Models")
        assert selected == "claude-3"


class TestModelPrompt:
    @patch("builtins.input", return_value="1")
    def test_numeric_fallback_valid_selection(self, mock_input):
        models = ["gpt-4", "gpt-3.5-turbo", "claude-3"]
        selected = _prompt_model_selection(models, "claude-3")
        assert selected == "gpt-4"

    @patch("builtins.input", return_value="")
    def test_numeric_fallback_cancel_returns_none(self, mock_input):
        models = ["gpt-4", "gpt-3.5-turbo"]
        selected = _prompt_model_selection(models, "gpt-4")
        assert selected is None

    @patch("builtins.input", return_value="invalid")
    def test_numeric_fallback_invalid_input_returns_none(self, mock_input):
        models = ["gpt-4", "gpt-3.5-turbo"]
        selected = _prompt_model_selection(models, "gpt-4")
        assert selected is None

    @patch("harness.cli._interactive_select")
    def test_interactive_path_uses_inline_selector(self, mock_select):
        models = ["gpt-4", "gpt-3.5-turbo", "claude-3"]
        mock_select.return_value = "claude-3"

        with patch.object(sys.stdin, "isatty", return_value=True):
            selected = _prompt_model_selection(models, "gpt-4")

        assert selected == "claude-3"
        mock_select.assert_called_once_with(models, "gpt-4", title="📋 Available models")


class TestReasoningPrompt:
    @patch("builtins.input", return_value="1")
    def test_valid_selection(self, mock_input):
        selected = _prompt_reasoning_selection("medium")
        assert selected == "low"

    @patch("builtins.input", return_value="")
    def test_cancel_returns_none(self, mock_input):
        selected = _prompt_reasoning_selection("high")
        assert selected is None

    @patch("builtins.input", return_value="invalid")
    def test_invalid_input_returns_none(self, mock_input):
        selected = _prompt_reasoning_selection("high")
        assert selected is None
