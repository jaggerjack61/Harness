# 🤖 Nasa Level Genius Agent

A lightweight AI agent framework that connects to any OpenAI-compatible chat model and gives it the ability to read, write, edit files, and execute shell commands — all through an interactive CLI.

## Features

- **OpenAI-compatible** — works with OpenAI, DeepSeek, local models, or any API that implements the chat completions + tool-calling spec.
- **Four built-in tools**: `read`, `write`, `edit`, and `bash` — giving the model full access to the filesystem and shell.
- **Reasoning / chain-of-thought support** — displays model reasoning (e.g. `reasoning_content` from DeepSeek or o-series models) as dimmed "thinking" output.
- **Markdown rendering** — agent responses are rendered as rich Markdown in the terminal with syntax-highlighted code blocks (cyan on dark gray), formatted headings, lists, tables, and more.
- **Streaming** — responses stream token-by-token; the final answer is then rendered as formatted Markdown.
- **Configurable reasoning effort** — pass `--reasoning-effort low|medium|high|xhigh|max` for models that support it.
- **Live progress events** — see tool calls, tool results, and thinking blocks in real time as the agent works.
- **Conversation history** — the agent remembers context across turns; `/clear` to reset.
- **Automatic context management** — history is trimmed/summarized when the conversation approaches the context window limit, so long sessions don't crash.
- **Retry with exponential backoff** — transient API errors (429 rate limits, 5xx server errors, timeouts) are retried automatically up to 3 times.
- **Real-time token tracking** — live status bar showing cumulative input/output/cache token counts and context window usage.
- **Cross-platform** — uses PowerShell (pwsh 7+ preferred) on Windows, bash on Unix/macOS. Shell can be overridden via `HARNESS_SHELL`.
- **Output safety** — tool outputs above 1,000 lines (or 100 KB) are dropped and the agent is asked to retry with line-limiting commands (`head`, `Select-Object -First`, etc.).

## Requirements

- Python ≥ 3.10
- An API key for an OpenAI-compatible service

## Installation

### From source

```bash
git clone https://github.com/jaggerjack61/Harness.git
cd harness
pip install .
```

For development (includes pytest):

```bash
pip install -e ".[dev]"
```

### PowerShell launcher (Windows)

Copy the example files and fill in your credentials:

```bash
cp harness.ps1.example harness.ps1
cp .env.example .env
# Edit .env — replace YOUR_API_KEY and HARNESS_BASE_URL with your credentials
```

Then add the harness directory to your `PATH` so you can run `harness` from anywhere.

> **⚠️ `.env` and `harness.ps1` are gitignored** — never commit your API key.
>
> The launcher loads credentials from `.env` at runtime, so your key never needs to sit inside `harness.ps1` itself.

## Quick Start

Set your API key and run:

```bash
export OPENAI_API_KEY="sk-..."
harness
```

Or pass credentials directly:

```bash
harness --model deepseek-v4-pro --api-key sk-... --base-url https://api.deepseek.com/v1
```

You can also run as a module:

```bash
python -m harness
```

Then just type a prompt and watch the agent work!

```
╔═══════════════════════════════════════════════════════════════╗
║  🤖 Nasa Level Genius Agent                                   ║
║  Model:           deepseek-v4-pro                             ║
║  Reasoning:       high                                        ║
║  Context window:  1,000,000 tokens                            ║
║  Streaming:       on                                          ║
║  CWD:             my-project                                  ║
║  ────────────────────────────────────────                     ║
║  Commands:  /exit  /clear  /models  /reasoning                ║
║             /stream  /context  /context show  /context clear  ║
╚═══════════════════════════════════════════════════════════════╝

📦 42 models loaded. Use /models to switch.

▸ Read pyproject.toml and tell me the version

🤖

  🧠 The user wants to read pyproject.toml and extract the version number.

  🔧 read(path='pyproject.toml')
     ├─ result:
     │ [project]
     │ name = "harness"
     │ version = "0.1.0"
     │ ...

🤖 The project version is **0.1.0**.

📊 deepseek-v4-pro  In:5,000  Out:200  Tot:5,200  Ctx:0.5%  🧠 high  [+5,000/200]
```

## CLI Options

| Flag | Short | Default | Description |
|---|---|---|---|
| `--model` | `-m` | `deepseek-v4-pro` (or `$HARNESS_MODEL`) | Model name |
| `--api-key` | `-k` | `$OPENAI_API_KEY` | API key |
| `--base-url` | `-u` | `https://api.openai.com/v1` | API base URL |
| `--dir` | `-d` | Current directory | Working directory for shell commands |
| `--max-turns` | | `1000` (or `$HARNESS_MAX_TURNS`) | Maximum tool-calling turns per prompt |
| `--system-prompt` | | Built-in default | Custom system prompt (or `$HARNESS_PROMPT`) |
| `--reasoning-effort` | | `high` | `low`, `medium`, `high`, `xhigh`, or `max` |
| `--context-window` | | `1000000` (or `$HARNESS_CONTEXT_WINDOW`) | Model context window size in tokens |
| `--stream` | | `true` (default) | Enable streaming output |
| `--no-stream` | | `false` | Disable streaming (wait for complete response) |
| `--no-markdown` | | `false` | Disable Markdown rendering (print plain text) |
| `--shell` | | Auto-detected | Override shell used by `bash` tool (env: `HARNESS_SHELL`) |

## Slash Commands

| Command | Description |
|---|---|
| `/exit` | Quit the harness |
| `/clear` | Reset conversation history and token counters (keeps custom context) |
| `/models` | Browse and select from available models fetched from the configured API (`--base-url`/models) |
| `/reasoning` | Interactively change the reasoning effort level |
| `/stream` | Toggle streaming mode on/off |
| `/context` | Enter custom context text (multiline, end with `.` on its own line) |
| `/context show` | Display current custom context |
| `/context clear` | Remove custom context |

## Tools

The agent has access to four tools:

| Tool | Description |
|---|---|
| **read** | Read file contents with optional `offset` and `limit` for large files |
| **write** | Create or overwrite a file (auto-creates parent directories) |
| **edit** | Apply precise find-and-replace edits to a file — multiple edits can be applied in a single call |
| **bash** | Execute a shell command (PowerShell on Windows, bash elsewhere) with a 60-second timeout |

### Tool output limits

Any tool that returns more than **1,000 lines** of output is **dropped** before reaching the model. The agent is told the output was discarded and asked to retry using line-limiting commands such as `head -n <N>`, `tail -n <N>`, or `Select-Object -First <N>`. This keeps the model context window from being filled by a single verbose command.

## Environment Variables

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Default API key |
| `HARNESS_MODEL` | Default model name |
| `HARNESS_BASE_URL` | Default API base URL |
| `HARNESS_PROMPT` | Default system prompt |
| `HARNESS_MAX_TURNS` | Default `--max-turns` value |
| `HARNESS_CONTEXT_WINDOW` | Default `--context-window` value |
| `HARNESS_SHELL` | Override shell executable for `bash` tool (default: auto-detected `pwsh`/`powershell`/`bash`) |

## Markdown Rendering

Agent responses are automatically rendered as Markdown in the terminal using a custom theme:

- **Syntax-highlighted code blocks** — language-aware highlighting, cyan on dark gray background
- **Inline code** — cyan on dark gray
- **Headings** — bold white
- **Links** — blue underlined
- **Lists, tables, blockquotes, horizontal rules** — fully rendered

To disable markdown rendering, use the `--no-markdown` flag:

```bash
harness --no-markdown
```

### Programmatic Markdown Rendering

You can also use the markdown module directly:

```python
from harness.markdown import render_markdown, markdown_to_plain

# Render to terminal
render_markdown("# Hello\n\n**Bold** and `code`.")

# Convert to plain text (useful for testing)
plain = markdown_to_plain("# Hello\n\n**Bold** text.")
print(plain)
```

## Streaming vs Non-Streaming

By default, responses **stream** token-by-token for a live feel, and the final answer is then rendered as formatted Markdown. Thinking/reasoning blocks appear as a single dimmed block before tool calls.

To disable streaming:

```bash
harness --no-stream
```

Toggle streaming on/off during a session with the `/stream` slash command.

## Programmatic Usage

You can also use `AgentHarness` as a library:

```python
from harness import AgentHarness

agent = AgentHarness(
    model="deepseek-v4-pro",
    api_key="sk-...",
    base_url="https://api.deepseek.com/v1",
    working_dir="/path/to/project",
    reasoning_effort="high",
)

# Simple usage
response = agent.run("List all Python files in this directory.")
print(response)

# With a progress callback
def on_event(event):
    if event["type"] == "tool_call":
        print(f"  Calling {event['name']}...")
    elif event["type"] == "thinking":
        print(f"  Thinking: {event['content'][:80]}...")
    elif event["type"] == "tokens":
        print(f"  Tokens — In: {event['input_tokens']}, Out: {event['output_tokens']}")

response = agent.run("Refactor main.py", callback=on_event)

# Token usage after a run
print(f"Input: {agent.input_tokens}, Output: {agent.output_tokens}")
```

### Callback event types

| Event | Fields | Description |
|---|---|---|
| `thinking` | `content` | Complete model reasoning / chain-of-thought (non-streaming) |
| `thinking_delta` | `content` | Incremental reasoning chunk (streaming only) |
| `thinking_end` | *(none)* | End of a streaming reasoning block — emitted when the model finishes thinking and starts producing tool calls or text |
| `text` | `content` | Final model text response (non-streaming) |
| `text_delta` | `content` | Incremental text token (streaming only; rendered via text_end) |
| `text_end` | `content` | Complete response text (streaming only; triggers markdown rendering) |
| `tool_call` | `name`, `arguments` | A tool is about to be executed |
| `tool_result` | `name`, `result` | A tool has returned its result |
| `turn_start` | *(none)* | A new agent turn is starting (after a tool-calling turn) |
| `finish_reason` | `reason`, `content` | Non-standard finish reason (truncation, content filter, etc.) |
| `history_trimmed` | `summarized` | Conversation history was trimmed/summarized to stay within context window |
| `tokens` | `input_tokens`, `output_tokens`, `total_tokens`, `cached_tokens`, `turn_input`, `turn_output`, `turn_cached`, `context_window`, `model`, `reasoning_effort` | Live token usage after each API call |

### Typed event dataclasses

Events are also available as typed dataclasses via `harness.events`:

```python
from harness.events import TokensEvent, ToolCallEvent, from_dict

# Parse a dict event into a typed dataclass
event = from_dict({"type": "tool_call", "name": "read", "arguments": {"path": "main.py"}})
assert isinstance(event, ToolCallEvent)

# Events have a to_dict() method for serialization
assert event.to_dict() == {"type": "tool_call", "name": "read", "arguments": {"path": "main.py"}}
```

All event types: `TokensEvent`, `ThinkingEvent`, `ThinkingDeltaEvent`, `ThinkingEndEvent`, `TextEvent`, `TextDeltaEvent`, `TextEndEvent`, `ToolCallEvent`, `ToolResultEvent`, `TurnStartEvent`, `FinishReasonEvent`, `HistoryTrimmedEvent`.

## Running Tests

```bash
pytest
```

## Project Structure

```
harness/
├── harness/
│   ├── __init__.py      # Package exports
│   ├── __main__.py      # python -m harness entry point
│   ├── agent.py         # AgentHarness — core agent loop with streaming, tool calling, retry, and context trimming
│   ├── cli.py           # Interactive CLI with argument parsing and live event display
│   ├── constants.py     # Centralized defaults, limits, and tuning knobs (DRY)
│   ├── display.py       # LiveDisplay and ResponseBuffer for streaming output
│   ├── events.py        # Typed event dataclasses and Callback protocol
│   ├── markdown.py      # Markdown parser and renderer using Rich (custom theme)
│   ├── prompts.py       # Unified interactive/numeric selection prompts
│   └── tools.py         # Tool definitions and implementations
├── tests/
│   ├── test_agent.py    # Tests for agent logic, message parsing, and callbacks
│   ├── test_cli.py      # Tests for CLI argument parsing and event display
│   ├── test_constants.py # Tests for constants module
│   ├── test_display.py  # Tests for LiveDisplay and ResponseBuffer
│   ├── test_events.py   # Tests for event dataclasses and from_dict
│   ├── test_markdown.py # Tests for markdown rendering
│   ├── test_prompts.py  # Tests for selection prompt helpers
│   ├── test_resolve_path.py # Tests for path resolution
│   ├── test_streaming.py # Tests for streaming response processing
│   └── test_tools.py    # Tests for file and shell tool implementations
├── harness.ps1.example  # Template PowerShell launcher (copy to harness.ps1)
├── pyproject.toml
├── .gitignore
└── README.md
```

## Trust & Safety

> **⚠️ Harness runs with full user privileges.** It is a research and development
> tool, not a sandboxed environment. Read this section before using it on
> important systems.

### What Harness can do

Harness gives an AI model the ability to:

- **Read any file** the current user can read (no path confinement)
- **Write, create, or overwrite any file** the current user can write
- **Execute arbitrary shell commands** (PowerShell on Windows, bash on Unix)
  with a 60-second timeout
- **Make network calls** to the configured `--base-url`

There is no allowlist, no sandbox, no confirmation prompt, and no
command blocklist.

### Implications

- **Prompt injection is a real risk.** If the model reads an untrusted file
  (e.g., a web page, a README, an email) that contains instructions, those
  instructions may be followed — including running destructive shell commands.
  Treat any content the model reads as if it were code.
- **The model is not trusted.** It may produce output that, if executed
  blindly, could delete files, exfiltrate data, or compromise your system.
  Review tool calls and shell commands before they execute in production.
- **There is no undo.** `write` and `edit` overwrite files immediately.
  `bash` runs whatever the model decides. There is no dry-run mode.

### Recommended practices

- **Run in a dedicated working directory** (`--dir`) that contains only
  the project you want the model to work on. Avoid pointing Harness at
  your home directory, system folders, or shared drives.
- **Use version control.** Commit your work before starting a session so
  you can review and revert changes.
- **Use a dedicated API key** with spending limits. Avoid using a key
  with access to production systems or sensitive data.
- **Review the welcome box** at session start to confirm the model,
  reasoning effort, and working directory are correct.
- **Interrupt with Ctrl-C** if the model starts doing something unexpected.
  The agent loop will exit cleanly.
- **Do not point Harness at secrets.** API keys, credentials, and
  `.env` files in the working directory can be read by the model.

### Threat model

Harness assumes the user is a trusted developer and the model is
untrusted-but-curious. It is suitable for:

- Local development and prototyping
- Automated refactoring in a controlled codebase
- Research and experimentation

It is **not** suitable for:

- Running against production data or systems
- Multi-tenant or shared environments
- Scenarios where the model output is not human-supervised

If you need stronger guarantees, consider running the agent in a
container, VM, or restricted shell that limits what it can access.

## License

MIT
