"""Unified prompt selection for the CLI.

Collapses the previous four near-duplicate functions
(``_prompt_model_selection`` / ``_prompt_model_selection_numeric`` /
``_prompt_reasoning_selection`` / ``_prompt_reasoning_selection_numeric``)
into a single pair: ``prompt_selection`` and ``prompt_selection_numeric``.
"""

import sys
from typing import List, Optional

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl


def prompt_selection_numeric(
    options: List[str],
    current: str,
    title: str = "Options",
    current_label: str = "Current",
) -> Optional[str]:
    """Numeric fallback: print options, read a number, return the choice.

    Returns None on empty input, invalid number, non-numeric input, or
    out-of-range selection.
    """
    if not options:
        return None

    print(f"\n{title}:")
    print("─" * 50)
    for i, opt in enumerate(options, 1):
        marker = " → " if opt == current else "   "
        print(f"{marker}{i:2d}. {opt}")
    print("─" * 50)
    print(f"{current_label}: {current}")
    print()

    try:
        choice = input(f"Select number (or press Enter to cancel): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return None
    if not choice:
        return None
    try:
        idx = int(choice)
    except ValueError:
        print("❌ Invalid input. Please enter a number.")
        return None
    if 1 <= idx <= len(options):
        return options[idx - 1]
    print(f"❌ Invalid selection. Please enter 1-{len(options)}.")
    return None


def prompt_selection(
    options: List[str],
    current: str,
    title: str = "Options",
    current_label: str = "Current",
) -> Optional[str]:
    """Show a list and let the user pick one (arrow keys or numeric fallback)."""
    if not options:
        print("No options available.")
        return None

    if not sys.stdin.isatty():
        return prompt_selection_numeric(options, current, title, current_label)

    try:
        return _interactive_select(options, current, title, current_label)
    except Exception as e:
        print(f"❌ Interactive selection failed ({e}); falling back to numeric input.")
        return prompt_selection_numeric(options, current, title, current_label)


def _interactive_select(
    options: List[str],
    current: str,
    title: str,
    current_label: str,
) -> Optional[str]:
    """Arrow-key inline selector using prompt_toolkit."""
    if not options:
        return None

    print(f"\n{title}:")
    print("─" * 50)
    print(f"{current_label}: {current}")
    print("Use ↑/↓ to navigate, Enter to select, Esc to cancel.\n")

    index = options.index(current) if current in options else 0
    result: List[Optional[str]] = [None]

    kb = KeyBindings()

    @kb.add("up")
    def _up(event) -> None:
        nonlocal index
        index = (index - 1) % len(options)
        event.app.invalidate()

    @kb.add("down")
    def _down(event) -> None:
        nonlocal index
        index = (index + 1) % len(options)
        event.app.invalidate()

    @kb.add("enter")
    def _enter(event) -> None:
        result[0] = options[index]
        event.app.exit()

    @kb.add("escape")
    @kb.add("c-c")
    def _cancel(event) -> None:
        event.app.exit()

    def _get_text():
        fragments = []
        for i, opt in enumerate(options):
            pointer = "> " if i == index else "  "
            style = "reverse bold" if i == index else ""
            fragments.append((style, f"{pointer}{opt}\n"))
        return fragments

    control = FormattedTextControl(_get_text)
    layout = Layout(Window(control, height=len(options), wrap_lines=False))
    app = Application(
        layout=layout,
        key_bindings=kb,
        full_screen=False,
        erase_when_done=False,
        mouse_support=True,
    )
    app.run()
    return result[0]
