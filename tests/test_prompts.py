"""Tests for the unified prompt selection module."""

from harness import prompts


class TestPromptSelectionNumeric:
    """Numeric fallback prompt should work for any list of options."""

    def test_valid_selection_returns_option(self):
        from unittest.mock import patch
        with patch("builtins.input", return_value="2"):
            result = prompts.prompt_selection_numeric(
                ["alpha", "beta", "gamma"], "alpha", title="Choose:"
            )
        assert result == "beta"

    def test_empty_input_returns_none(self):
        from unittest.mock import patch
        with patch("builtins.input", return_value=""):
            result = prompts.prompt_selection_numeric(["a", "b"], "a")
        assert result is None

    def test_invalid_number_returns_none(self):
        from unittest.mock import patch
        with patch("builtins.input", return_value="99"):
            result = prompts.prompt_selection_numeric(["a", "b"], "a")
        assert result is None

    def test_non_numeric_input_returns_none(self):
        from unittest.mock import patch
        with patch("builtins.input", return_value="abc"):
            result = prompts.prompt_selection_numeric(["a", "b"], "a")
        assert result is None

    def test_empty_options_returns_none(self):
        result = prompts.prompt_selection_numeric([], "x")
        assert result is None

    def test_current_option_marked(self, capsys):
        from unittest.mock import patch
        with patch("builtins.input", return_value="1"):
            prompts.prompt_selection_numeric(["alpha", "beta"], "alpha")
        out = capsys.readouterr().out
        assert "alpha" in out
        assert "Current" in out


class TestPromptSelection:
    """Unified prompt_selection should handle TTY and non-TTY paths."""

    def test_non_tty_falls_back_to_numeric(self):
        from unittest.mock import patch
        options = ["a", "b", "c"]
        with patch("harness.prompts.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = False
            with patch("builtins.input", return_value="2"):
                result = prompts.prompt_selection(options, "a", title="Pick:")
        assert result == "b"

    def test_empty_options_returns_none(self, capsys):
        result = prompts.prompt_selection([], "x")
        assert result is None
        out = capsys.readouterr().out
        assert "No options" in out
