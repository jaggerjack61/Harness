"""Interactive CLI for the Nasa Level Genius Agent.

Usage:
    python -m harness.cli [--model MODEL] [--api-key KEY] [--base-url URL] [--dir DIR]
"""

import argparse
import json
import os
import sys
from typing import List, Optional

import httpx
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from rich.console import Console
from rich.live import Live
from rich.markup import escape as rich_escape
from rich.text import Text

from harness.markdown import render_markdown, _make_console

from harness.agent import AgentHarness

# Module-level Rich console and live display for the token status bar
_console: Optional[Console] = None
_live: Optional[Live] = None
_no_markdown: bool = False  # set by main() from --no-markdown flag
_thinking_streaming: bool = False  # track whether we're in a thinking stream


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interactive Nasa Level Genius Agent — chat with an AI that can read, write, edit, and run commands.",
    )
    p.add_argument(
        "--model", "-m",
        default=os.environ.get("HARNESS_MODEL", "deepseek-v4-pro"),
        help="Model name (default: deepseek-v4-pro or $HARNESS_MODEL)",
    )
    p.add_argument(
        "--api-key", "-k",
        default=os.environ.get("OPENAI_API_KEY"),
        help="API key (default: $OPENAI_API_KEY)",
    )
    p.add_argument(
        "--base-url", "-u",
        default=os.environ.get("HARNESS_BASE_URL", "https://api.openai.com/v1"),
        help="Base URL for the API",
    )
    p.add_argument(
        "--dir", "-d",
        default=os.getcwd(),
        help="Working directory for bash commands (default: current dir)",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=25,
        help="Maximum tool-calling turns (default: 25)",
    )
    p.add_argument(
        "--system-prompt",
        default=None,
        help="Custom system prompt (or use HARNESS_PROMPT env var)",
    )
    p.add_argument(
        "--reasoning-effort",
        default="high",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Reasoning effort level (for models that support it, e.g. o-series). Default: high",
    )
    p.add_argument(
        "--context-window",
        type=int,
        default=256000,
        help="Model context window size in tokens (default: 256000 for 256K).",
    )
    p.add_argument(
        "--no-markdown",
        action="store_true",
        default=False,
        help="Disable markdown rendering for agent responses (print plain text instead).",
    )
    p.add_argument(
        "--stream",
        action="store_true",
        default=True,
        help="Enable streaming output (default: enabled).",
    )
    p.add_argument(
        "--no-stream",
        action="store_true",
        default=False,
        help="Disable streaming output (wait for complete response).",
    )
    return p


def _on_event(event: dict) -> None:
    """Handle a progress event from the agent — print live status."""
    global _thinking_streaming
    etype = event["type"]

    BLUE = "\033[34m"
    GREEN = "\033[32m"
    GRAY = "\033[90m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    RESET = "\033[0m"

    if etype == "tokens":
        _print_tokens(event)

    elif etype == "thinking":
        content = event.get("content", "")
        # Print thinking in dim style via Rich (cross-platform safe).
        # Escape content to prevent Rich from interpreting any [...] patterns
        # in the model's reasoning as markup tags (e.g., [/Additional Context]).
        _thinking_streaming = False
        if _console is not None:
            _console.print(f"\n  🧠 [dim]{rich_escape(content)}[/dim]")
        else:
            print(f"\n  🧠 {content}")

    elif etype == "thinking_delta":
        # Stream thinking content live as it arrives.
        # Use Rich markup for cross-platform coloring — raw ANSI codes
        # show up as garbage on some terminals (ESC char gets stripped).
        # Escape content to prevent Rich from interpreting any [...] patterns
        # in the model's reasoning as markup tags.
        content = event.get('content', '')
        if not _thinking_streaming:
            _thinking_streaming = True
            if _console is not None:
                _console.print(f"\n  🧠 [dim]{rich_escape(content)}[/dim]", end="")
            else:
                print(f"\n  🧠 {content}", end="", flush=True)
        else:
            if _console is not None:
                _console.print(f"[dim]{rich_escape(content)}[/dim]", end="")
            else:
                print(content, end="", flush=True)

    elif etype == "tool_call":
        name = event["name"]
        args = event.get("arguments", {})

        def _print_truncated(lines, prefix, color):
            if len(lines) > 5:
                for line in lines[:5]:
                    print(f"{prefix}{color}{line}{RESET}", flush=True)
                print(f"     └─ {color}... truncated: {len(lines) - 5} of {len(lines)} lines hidden (full content sent to agent){RESET}", flush=True)
            else:
                for line in lines:
                    print(f"{prefix}{color}{line}{RESET}", flush=True)

        if name == "bash":
            cmd = args.get("command", "")
            print(f"\n  🔧 {BLUE}{cmd}{RESET}", flush=True)
        elif name == "write":
            path = args.get("path", "")
            content = args.get("content", "")
            lines = content.splitlines()
            print(f"\n  🔧 {GREEN}write(path={repr(path)}){RESET}", flush=True)
            _print_truncated(lines, "     │ ", GREEN)
        elif name == "edit":
            path = args.get("path", "")
            edits = args.get("edits", [])
            print(f"\n  🔧 {GREEN}edit(path={repr(path)}) — {len(edits)} edit(s){RESET}", flush=True)
            for i, edit in enumerate(edits):
                old_text = edit.get("oldText", "")
                new_text = edit.get("newText", "")
                print(f"     ├─ Edit {i+1}:", flush=True)
                if old_text:
                    print(f"     │ {GREEN}oldText ({len(old_text.splitlines())} lines):{RESET}", flush=True)
                    _print_truncated(old_text.splitlines(), "     │ ", GREEN)
                if new_text:
                    print(f"     │ {GREEN}newText ({len(new_text.splitlines())} lines):{RESET}", flush=True)
                    _print_truncated(new_text.splitlines(), "     │ ", GREEN)
        else:
            args_str = ", ".join(
                f"{k}={repr(v)}" for k, v in args.items()
            )
            print(f"\n  🔧 {name}({args_str})", flush=True)

    elif etype == "tool_result":
        name = event["name"]
        result = event.get("result", "")
        color = GREEN if name in ("write", "edit") else ""
        lines = result.splitlines()
        print(f"     ├─ result:", flush=True)
        if len(lines) > 5:
            hidden = len(lines) - 5
            total = len(lines)
            shown = lines[:5]
            indented = "\n".join(f"     │ {color}{line}{RESET}" for line in shown)
            print(indented, flush=True)
            print(f"     └─ {color}... truncated: {hidden} of {total} lines hidden (full output received by agent){RESET}", flush=True)
        else:
            indented = "\n".join(f"     │ {color}{line}{RESET}" for line in lines)
            print(indented, flush=True)

    elif etype == "text":
        # Don't print text events here — the final answer is printed by main()
        pass

    elif etype == "text_delta":
        # Don't print raw text deltas — we render full markdown at text_end.
        # (Plain text output defeats markdown rendering.)
        pass

    elif etype == "text_end":
        # Streaming complete — render full response as markdown
        _thinking_streaming = False  # end of response, reset thinking state
        content = event.get("content", "")
        if _no_markdown:
            print(f"\n{content}\n")
        else:
            print()
            render_markdown(content or "", console=_console)
            print()


def _build_token_text(event: dict) -> Text:
    """Build a Rich Text renderable for the live token status bar."""
    input_tk = event.get("input_tokens", 0)
    output_tk = event.get("output_tokens", 0)
    total_tk = event.get("total_tokens", 0)
    cached_tk = event.get("cached_tokens", 0)
    turn_input = event.get("turn_input", 0)
    turn_output = event.get("turn_output", 0)
    context_window = event.get("context_window")
    model = event.get("model")
    reasoning_effort = event.get("reasoning_effort")

    parts: List[str] = []
    styles: List[str] = []

    parts.append("📊 ")
    styles.append("bold")

    if model:
        parts.append(f"{model}  ")
        styles.append("bold magenta")

    parts.append(f"In:{input_tk:,}  ")
    styles.append("cyan")

    parts.append(f"Out:{output_tk:,}  ")
    styles.append("green")

    parts.append(f"Tot:{total_tk:,}  ")
    styles.append("bold")

    if context_window:
        ctx_used_pct = (turn_input / context_window) * 100 if context_window > 0 else 0
        ctx_style = "green" if ctx_used_pct < 50 else "yellow" if ctx_used_pct < 80 else "red"
        parts.append(f"Ctx:{ctx_used_pct:.1f}%  ")
        styles.append(ctx_style)

    if cached_tk > 0:
        cache_rate = (cached_tk / total_tk * 100) if total_tk > 0 else 0
        parts.append(f"Cache:{cached_tk:,} ({cache_rate:.1f}%)  ")
        styles.append("yellow")

    if reasoning_effort:
        parts.append(f"🧠 {reasoning_effort}  ")
        styles.append("dim")

    parts.append(f"[+{turn_input:,}/{turn_output:,}]")
    styles.append("dim")

    text = Text(no_wrap=True)
    for part, style in zip(parts, styles):
        text.append(part, style=style)
    return text


def _print_tokens(event: dict) -> None:
    """Update the live token status bar via Rich, or print directly if no live display."""
    global _live, _console
    text = _build_token_text(event)
    if _live is not None:
        _live.update(text)
    else:
        # Fallback for tests or when live display is not active
        if _console is None:
            _console = Console()
        _console.print(text)


def _clear_status_bar() -> None:
    """Stop the live display and clear the status bar."""
    global _live
    if _live is not None:
        _live.stop()
        _live = None


def _fetch_models() -> List[str]:
    """Fetch available models from the OpenCode API."""
    url = "https://opencode.ai/zen/go/v1/models"
    try:
        resp = httpx.get(url, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        models = [m["id"] for m in data.get("data", [])]
        return sorted(models)
    except Exception as e:
        print(f"❌ Failed to fetch models: {e}")
        return []


def _read_multiline_context() -> Optional[str]:
    """Read multiline context input from the user. Terminate with '.' on a line by itself."""
    print("\n📝 Enter custom context (end with '.' on a line by itself):")
    lines = []
    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return None
        if line.strip() == ".":
            break
        lines.append(line)
    if not lines:
        print("No text entered — context unchanged.")
        return None
    return "\n".join(lines)


def _prompt_model_selection_numeric(models: List[str], current_model: str) -> Optional[str]:
    """Numeric fallback for selecting a model when interactive input isn't available."""
    print("\n📋 Available models:")
    print("─" * 50)
    for i, model in enumerate(models, 1):
        marker = " → " if model == current_model else "   "
        print(f"{marker}{i:2d}. {model}")
    print("─" * 50)
    print(f"Current model: {current_model}")
    print()

    try:
        choice = input("Select model number (or press Enter to cancel): ").strip()
        if not choice:
            return None
        idx = int(choice)
        if 1 <= idx <= len(models):
            return models[idx - 1]
        else:
            print(f"❌ Invalid selection. Please enter 1-{len(models)}.")
            return None
    except ValueError:
        print("❌ Invalid input. Please enter a number.")
        return None


def _interactive_select(options: List[str], current: str, title: str) -> Optional[str]:
    """Show a lightweight inline list selector using the arrow keys.

    Renders at the current cursor position without clearing the screen, so the
    existing terminal contents stay visible. Returns the selected option or None.
    """
    if not options:
        return None

    print(f"\n{title}:")
    print("─" * 50)
    print(f"Current model: {current}")
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
    layout = Layout(
        Window(control, height=len(options), wrap_lines=False)
    )
    app = Application(
        layout=layout,
        key_bindings=kb,
        full_screen=False,
        erase_when_done=False,
        mouse_support=True,
    )
    app.run()
    return result[0]


def _prompt_model_selection(models: List[str], current_model: str) -> Optional[str]:
    """Display models and let the user pick one using arrow keys + Enter.

    Falls back to numeric input when stdin is not an interactive terminal.
    """
    if not models:
        print("No models available.")
        return None

    if not sys.stdin.isatty():
        return _prompt_model_selection_numeric(models, current_model)

    try:
        return _interactive_select(models, current_model, title="📋 Available models")
    except Exception as e:
        print(f"❌ Interactive selection failed ({e}); falling back to numeric input.")
        return _prompt_model_selection_numeric(models, current_model)


REASONING_OPTIONS = ["low", "medium", "high", "xhigh", "max"]


def _prompt_reasoning_selection(current_effort: Optional[str]) -> Optional[str]:
    """Display reasoning options and prompt user to select one. Returns selected or None."""
    print("\n🧠 Reasoning effort:")
    print("─" * 30)
    for i, level in enumerate(REASONING_OPTIONS, 1):
        marker = " → " if level == (current_effort or "high") else "   "
        print(f"{marker}{i}. {level}")
    print("─" * 30)
    print(f"Current: {current_effort or 'high'}")
    print()

    try:
        choice = input("Select level number (or press Enter to cancel): ").strip()
        if not choice:
            return None
        idx = int(choice)
        if 1 <= idx <= len(REASONING_OPTIONS):
            return REASONING_OPTIONS[idx - 1]
        else:
            print(f"❌ Invalid selection. Please enter 1-{len(REASONING_OPTIONS)}.")
            return None
    except ValueError:
        print("❌ Invalid input. Please enter a number.")
        return None


def _build_welcome_box(model: str, reasoning_effort: str, context_window: int, working_dir: str, streaming: bool = True) -> str:
    """Build the welcome box with dynamic width based on content."""
    ctx_str = f"{context_window:,} tokens"
    cwd_display = working_dir
    stream_str = "on" if streaming else "off"

    # Collect all content lines (without borders) to calculate max width
    lines = [
        "🤖 Nasa Level Genius Agent",
        f"Model:           {model}",
        f"Reasoning:       {reasoning_effort}",
        f"Context window:  {ctx_str}",
        f"Streaming:       {stream_str}",
        f"CWD:             {cwd_display}",
        "─" * 40,
        "Commands:  /exit  /clear  /models  /reasoning",
        "           /stream  /context  /context show  /context clear",
    ]

    # Calculate box width: max content length + padding (2 left + 2 right)
    max_len = max(len(line) for line in lines)
    inner_w = max_len + 4  # 2 spaces padding on each side

    # Build the box
    result = []
    result.append(f"╔{'═' * inner_w}╗")
    for line in lines:
        # Pad each line to inner_w (2 left pad + content + right pad)
        padded = f"  {line}"
        padded = f"{padded:<{inner_w}s}"
        result.append(f"║{padded}║")
    result.append(f"╚{'═' * inner_w}╝")
    return "\n".join(result)


def main(argv: Optional[list] = None) -> None:
    global _console, _live, _no_markdown
    args = _build_parser().parse_args(argv)
    _no_markdown = args.no_markdown

    # Determine streaming mode
    use_stream = args.stream and not args.no_stream

    system_prompt = args.system_prompt or os.environ.get("HARNESS_PROMPT")

    agent = AgentHarness(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        working_dir=args.dir,
        system_prompt=system_prompt,
        max_turns=args.max_turns,
        reasoning_effort=args.reasoning_effort,
        context_window=args.context_window,
    )

    current_model = args.model

    print(_build_welcome_box(args.model, args.reasoning_effort, args.context_window, args.dir, use_stream))
    print()

    # Pre-fetch models on launch so /models is instant
    models = _fetch_models()
    if models:
        print(f"📦 {len(models)} models loaded. Use /models to switch.\n")
    else:
        print("⚠️ Could not pre-fetch models — /models will retry on demand.\n")

    while True:
        try:
            user_input = input("▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            _clear_status_bar()
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "/exit":
            _clear_status_bar()
            print("Goodbye!")
            break

        if user_input.lower() == "/clear":
            agent.clear_history()
            # Clear the screen (cross-platform)
            os.system('cls' if os.name == 'nt' else 'clear')
            # Redisplay the welcome banner
            print(_build_welcome_box(current_model, agent.reasoning_effort, args.context_window, args.dir, use_stream))
            print()
            if models:
                print(f"📦 {len(models)} models loaded. Use /models to switch.\n")
            else:
                print("⚠️ Could not pre-fetch models — /models will retry on demand.\n")
            print("🔄 History cleared.\n")
            continue

        if user_input.lower() == "/stream":
            use_stream = not use_stream
            status = "enabled" if use_stream else "disabled"
            print(f"✅ Streaming {status}.")
            continue

        if user_input.lower() == "/models":
            if not models:
                print("\nFetching models...")
                models = _fetch_models()
            selected = _prompt_model_selection(models, current_model)
            if selected and selected != current_model:
                current_model = selected
                agent.model = selected
                print(f"✅ Model changed to: {selected}")
            continue

        if user_input.lower() == "/reasoning":
            selected = _prompt_reasoning_selection(agent.reasoning_effort)
            if selected and selected != agent.reasoning_effort:
                agent.reasoning_effort = selected
                print(f"✅ Reasoning effort set to: {selected}")
            continue

        if user_input.lower() == "/context":
            ctx = _read_multiline_context()
            if ctx is not None:
                agent.set_custom_context(ctx)
                print(f"✅ Custom context set ({len(ctx.splitlines())} lines).")
            continue

        if user_input.lower() == "/context clear":
            agent.set_custom_context(None)
            print("✅ Custom context cleared.")
            continue

        if user_input.lower() == "/context show":
            ctx = agent.get_custom_context()
            if ctx:
                print(f"\n📋 Current custom context ({len(ctx.splitlines())} lines):\n{'─'*40}")
                print(ctx)
                print("─" * 40)
            else:
                print("No custom context set. Use /context to add one.")
            continue

        # Run the agent with live progress
        print(flush=True)
        _console = _make_console()

        if use_stream:
            # Streaming mode: print prefix, then text arrives via callback
            _live = Live(Text(), console=_console, refresh_per_second=4, transient=False)
            _live.start()
            print("🤖 ", end="", flush=True)
            try:
                _thinking_streaming = False  # reset thinking state for new response
                response = agent.run(user_input, callback=_on_event, stream=True)
            except Exception as e:
                _clear_status_bar()
                print(f"\n❌ Error: {e}")
                continue

            # Clear status bar (text already printed by callback)
            _clear_status_bar()
        else:
            # Non-streaming mode: wait for complete response
            _live = Live(Text(), console=_console, refresh_per_second=4, transient=False)
            _live.start()
            try:
                _thinking_streaming = False  # reset thinking state for new response
                response = agent.run(user_input, callback=_on_event)
            except Exception as e:
                _clear_status_bar()
                print(f"\n❌ Error: {e}")
                continue

            # Clear status bar and print final response
            _clear_status_bar()
            print("\n🤖 ", end="")
            if args.no_markdown:
                print(f"{response}\n")
            else:
                render_markdown(response or "", console=_console)
                print()


if __name__ == "__main__":
    main()
