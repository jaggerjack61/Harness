"""Markdown parser and renderer for agent responses using Rich.

Provides terminal-friendly rendering of Markdown with syntax-highlighted
code blocks, formatted headings, lists, tables, and more.
"""

from io import StringIO
from typing import Optional

from rich.console import Console
from rich.errors import MarkupError
from rich.markdown import Markdown
from rich.markup import escape as rich_escape
from rich.style import Style
from rich.theme import Theme


# Custom theme that avoids purple/magenta — uses neutral grays and blues instead.
_MARKDOWN_THEME = Theme({
    "markdown.code": Style(color="cyan", bgcolor="grey23", bold=False),
    "markdown.code_block": Style(bgcolor="grey23"),
    "markdown.inline_code": Style(color="cyan", bgcolor="grey23"),
    "markdown.link": Style(color="blue", underline=True),
    "markdown.h1": Style(bold=True, color="bright_white"),
    "markdown.h2": Style(bold=True, color="bright_white"),
    "markdown.h3": Style(bold=True, color="white"),
    "markdown.h4": Style(bold=True, color="white"),
    "markdown.h5": Style(bold=True),
    "markdown.h6": Style(bold=True, dim=True),
    "markdown.block_quote": Style(color="bright_black", italic=True),
    "markdown.hr": Style(color="bright_black"),
    "markdown.list_bullet": Style(color="cyan"),
    "markdown.list_number": Style(color="cyan"),
    "markdown.table": Style(),
    "markdown.table_header": Style(bold=True),
})


def _make_console(width: Optional[int] = None, no_color: bool = False) -> Console:
    """Create a Rich Console with the custom markdown theme.

    When ``width`` is ``None`` (the default), Rich auto-detects the terminal
    width. A fixed width can still be passed for testing or plain-text
    conversion.
    """
    return Console(
        file=StringIO() if no_color else None,
        width=width,
        force_terminal=False if no_color else None,
        no_color=no_color,
        theme=_MARKDOWN_THEME,
    )


def render_markdown(text: str, console: Optional[Console] = None) -> None:
    """Render markdown text to the terminal using Rich.

    Supports headings, bold/italic, code blocks with syntax highlighting,
    inline code, lists, blockquotes, tables, links, and horizontal rules.

    Uses a custom theme that avoids purple/magenta — code is cyan on dark
    gray, links are blue, and headings are white.

    Falls back to escaped plain text if Rich's markup parser fails
    (e.g., due to unmatched [...] patterns in the text).

    Args:
        text: The markdown text to render.
        console: Optional Rich Console instance. If None, a new one is created
                 with the custom markdown theme.
    """
    if console is None:
        console = _make_console()

    try:
        md = Markdown(text)
        console.print(md)
    except MarkupError:
        # The text contains patterns that Rich's markup parser can't handle
        # (e.g., unmatched closing tags like [/Additional Context]).
        # Fall back to rendering as plain escaped text.
        console.print(rich_escape(text))


def markdown_to_plain(text: str, width: int = 100) -> str:
    """Convert markdown text to rendered plain text (useful for testing).

    Renders the markdown through Rich's Markdown renderer and captures
    the output as a string, preserving formatting like indentation and
    alignment but without ANSI escape codes.

    Falls back to escaped text if Rich's markup parser fails.

    Args:
        text: The markdown text to convert.
        width: Terminal width for rendering.

    Returns:
        Rendered text representation of the markdown.
    """
    buf = StringIO()
    console = Console(file=buf, width=width, no_color=True, force_terminal=False, theme=_MARKDOWN_THEME)
    try:
        md = Markdown(text)
        console.print(md)
    except MarkupError:
        console.print(rich_escape(text))
    value = buf.getvalue()
    buf.close()
    return value
