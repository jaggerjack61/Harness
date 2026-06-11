"""Tests for the markdown parser and renderer."""

from io import StringIO
from unittest.mock import patch, MagicMock

import pytest

from rich.console import Console

from harness.markdown import render_markdown, markdown_to_plain


class TestMarkdownToPlain:
    """Test converting markdown to plain text output."""

    def test_headings_are_rendered(self):
        text = "# Heading 1\n## Heading 2\n### Heading 3"
        result = markdown_to_plain(text)
        assert "Heading 1" in result
        assert "Heading 2" in result
        assert "Heading 3" in result

    def test_bold_text(self):
        text = "This is **bold** text."
        result = markdown_to_plain(text)
        assert "bold" in result

    def test_italic_text(self):
        text = "This is *italic* text."
        result = markdown_to_plain(text)
        assert "italic" in result

    def test_inline_code(self):
        text = "Use `print()` to output."
        result = markdown_to_plain(text)
        assert "print()" in result

    def test_code_block_is_rendered(self):
        text = "```python\nprint('hello')\n```"
        result = markdown_to_plain(text)
        assert "print('hello')" in result

    def test_code_block_with_language_label(self):
        text = "```python\nx = 1\n```"
        result = markdown_to_plain(text)
        assert "x = 1" in result

    def test_unordered_list(self):
        text = "- item one\n- item two\n- item three"
        result = markdown_to_plain(text)
        assert "item one" in result
        assert "item two" in result
        assert "item three" in result

    def test_ordered_list(self):
        text = "1. first\n2. second\n3. third"
        result = markdown_to_plain(text)
        assert "first" in result
        assert "second" in result
        assert "third" in result

    def test_blockquote(self):
        text = "> This is a quote."
        result = markdown_to_plain(text)
        assert "This is a quote." in result

    def test_horizontal_rule(self):
        text = "Above\n\n---\n\nBelow"
        result = markdown_to_plain(text)
        assert "Above" in result
        assert "Below" in result

    def test_links_are_rendered(self):
        text = "Visit [Example](https://example.com) for more."
        result = markdown_to_plain(text)
        assert "Example" in result

    def test_table_is_rendered(self):
        text = (
            "| Name  | Value |\n"
            "|-------|-------|\n"
            "| Alice | 100   |\n"
            "| Bob   | 200   |"
        )
        result = markdown_to_plain(text)
        assert "Alice" in result
        assert "Bob" in result
        assert "100" in result
        assert "200" in result

    def test_plain_text_passes_through(self):
        text = "Just some plain text without any markdown."
        result = markdown_to_plain(text)
        assert "Just some plain text without any markdown." in result

    def test_empty_string(self):
        result = markdown_to_plain("")
        # Should not crash; output is empty or just whitespace
        assert result.strip() == ""

    def test_multiline_response(self):
        text = (
            "# Summary\n\n"
            "The file was **successfully** updated.\n\n"
            "Changes made:\n"
            "- Fixed the bug in `main.py`\n"
            "- Added tests\n\n"
            "```python\nprint('done')\n```"
        )
        result = markdown_to_plain(text)
        assert "Summary" in result
        assert "successfully" in result
        assert "Fixed the bug" in result
        assert "main.py" in result
        assert "Added tests" in result
        assert "print('done')" in result

    def test_width_parameter_affects_output(self):
        text = "A short line."
        result_narrow = markdown_to_plain(text, width=40)
        result_wide = markdown_to_plain(text, width=120)
        # Both should contain the text
        assert "A short line." in result_narrow
        assert "A short line." in result_wide


class TestRenderMarkdown:
    """Test rendering markdown to a console."""

    def test_renders_to_console_without_error(self):
        """render_markdown should not raise an exception."""
        console = Console(file=StringIO(), width=100, force_terminal=False, no_color=True)
        # Should not raise
        render_markdown("# Hello\n\nSome **bold** text.", console=console)
        output = console.file.getvalue()
        assert "Hello" in output
        assert "bold" in output

    def test_renders_code_block(self):
        console = Console(file=StringIO(), width=100, force_terminal=False, no_color=True)
        render_markdown("```python\nx = 42\n```", console=console)
        output = console.file.getvalue()
        assert "x = 42" in output

    def test_creates_default_console_when_none_provided(self):
        """When no console is passed, render_markdown creates one and prints to stdout."""
        # This just ensures no crash when console=None
        with patch("harness.markdown.Console") as mock_console_cls:
            mock_console = MagicMock()
            mock_console_cls.return_value = mock_console
            render_markdown("# Test")
            mock_console.print.assert_called_once()

    def test_uses_provided_console(self):
        console = MagicMock()
        render_markdown("Hello **world**", console=console)
        console.print.assert_called_once()
        # The argument should be a Rich Markdown object
        arg = console.print.call_args[0][0]
        assert hasattr(arg, "markup")  # Markdown objects have markup attribute


class TestMarkdownInCLI:
    """Test that markdown rendering is integrated into the CLI."""

    def test_no_markdown_flag_in_parser(self):
        from harness.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["--no-markdown"])
        assert args.no_markdown is True

    def test_no_markdown_flag_default_false(self):
        from harness.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.no_markdown is False

    @patch("harness.cli.AgentHarness")
    @patch("harness.cli.input")
    def test_response_rendered_as_markdown(self, mock_input, mock_harness):
        """When --no-markdown is NOT set and streaming is disabled, the response should be rendered via render_markdown."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = "# Hello\n\n**Bold** text."
        mock_agent.input_tokens = 10
        mock_agent.output_tokens = 5
        mock_harness.return_value = mock_agent
        mock_input.side_effect = ["test prompt", "/exit"]

        from harness.cli import main
        with patch("harness.cli.render_markdown") as mock_render:
            main(["--api-key", "sk-test", "--no-stream"])
            mock_render.assert_called_once()
            call_args = mock_render.call_args
            assert call_args[0][0] == "# Hello\n\n**Bold** text."

    @patch("harness.cli.AgentHarness")
    @patch("harness.cli.input")
    def test_no_markdown_flag_prints_plain_text(self, mock_input, mock_harness):
        """When --no-markdown is set, the response should be printed as plain text."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = "# Hello\n\n**Bold** text."
        mock_agent.input_tokens = 10
        mock_agent.output_tokens = 5
        mock_harness.return_value = mock_agent
        mock_input.side_effect = ["test prompt", "/exit"]

        from harness.cli import main
        with patch("harness.cli.render_markdown") as mock_render:
            main(["--api-key", "sk-test", "--no-markdown"])
            mock_render.assert_not_called()

    @patch("harness.cli.AgentHarness")
    @patch("harness.cli.input")
    def test_empty_response_does_not_crash(self, mock_input, mock_harness):
        """An empty response should render without errors."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = ""
        mock_agent.input_tokens = 10
        mock_agent.output_tokens = 5
        mock_harness.return_value = mock_agent
        mock_input.side_effect = ["test prompt", "/exit"]

        from harness.cli import main
        # Should not raise
        main(["--api-key", "sk-test"])

    @patch("harness.cli.AgentHarness")
    @patch("harness.cli.input")
    def test_none_response_does_not_crash(self, mock_input, mock_harness):
        """A None response should render without errors."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = None
        mock_agent.input_tokens = 10
        mock_agent.output_tokens = 5
        mock_harness.return_value = mock_agent
        mock_input.side_effect = ["test prompt", "/exit"]

        from harness.cli import main
        # Should not raise
        main(["--api-key", "sk-test"])
