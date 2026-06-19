"""Centralized constants for the Harness AI agent.

All magic numbers, default values, and configuration constants live here
so they have a single authoritative location (DRY). Other modules import
from here instead of inlining literals.
"""

# ── Agent defaults ──────────────────────────────────────────────────────────

DEFAULT_MAX_TURNS: int = 1000
DEFAULT_CONTEXT_WINDOW: int = 1000000
DEFAULT_MODEL: str = "deepseek-v4-pro"
DEFAULT_BASE_URL: str = "https://api.openai.com/v1"
DEFAULT_REASONING_EFFORT: str = "high"

# ── Reasoning effort ────────────────────────────────────────────────────────

REASONING_OPTIONS: list = ["low", "medium", "high", "xhigh", "max"]
NONSTANDARD_REASONING_EFFORTS: set = {"xhigh", "max"}

# ── Retry configuration ─────────────────────────────────────────────────────

MAX_RETRIES: int = 3
RETRY_BASE_DELAY: float = 1.0  # seconds; actual = base * 2^attempt + jitter

# ── Context-window management ───────────────────────────────────────────────

CONTEXT_WINDOW_TRIM_THRESHOLD: float = 0.8  # trim at 80% of context window
RECENT_TURNS_TO_KEEP: int = 6
SUMMARY_MAX_TOKENS: int = 500

# ── Tool output limits ──────────────────────────────────────────────────────

MAX_OUTPUT_LINES: int = 1000
MAX_OUTPUT_BYTES: int = 100_000  # 100 KB guard for single-line giants
BASH_TIMEOUT: int = 60  # seconds

# ── Model fetch ─────────────────────────────────────────────────────────────

FETCH_TIMEOUT: float = 10.0  # seconds for /models endpoint

# ── CLI display ─────────────────────────────────────────────────────────────

DISPLAY_TRUNCATION_LIMIT: int = 5  # lines shown for tool calls/results
LIVE_REFRESH_STREAMING: int = 12  # Hz
LIVE_REFRESH_NON_STREAMING: int = 4  # Hz

SPINNER_FRAMES: list = [
    "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
]
