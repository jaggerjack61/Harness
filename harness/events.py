"""Typed event dataclasses and Callback protocol for agent progress events.

The agent emits events as either typed dataclasses (preferred) or plain dicts
(backward compat). All events have a ``type`` field and a ``to_dict()`` method
that produces the same dict shape the CLI expects.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable


# ── Event dataclasses ───────────────────────────────────────────────────────


@dataclass
class TokensEvent:
    """Live token usage after an API call."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int
    turn_input: int
    turn_output: int
    turn_cached: int
    context_window: int
    model: str
    reasoning_effort: Optional[str]
    type: str = "tokens"

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "turn_input": self.turn_input,
            "turn_output": self.turn_output,
            "turn_cached": self.turn_cached,
            "context_window": self.context_window,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
        }


@dataclass
class ThinkingEvent:
    """Full reasoning block (non-streaming only)."""
    content: str
    type: str = "thinking"

    def to_dict(self) -> dict:
        return {"type": self.type, "content": self.content}


@dataclass
class ThinkingDeltaEvent:
    """Incremental reasoning chunk (streaming only)."""
    content: str
    type: str = "thinking_delta"

    def to_dict(self) -> dict:
        return {"type": self.type, "content": self.content}


@dataclass
class ThinkingEndEvent:
    """End of a streaming reasoning block."""
    type: str = "thinking_end"

    def to_dict(self) -> dict:
        return {"type": self.type}


@dataclass
class TextEvent:
    """Final model text response (non-streaming)."""
    content: str
    type: str = "text"

    def to_dict(self) -> dict:
        return {"type": self.type, "content": self.content}


@dataclass
class TextDeltaEvent:
    """Incremental text token (streaming)."""
    content: str
    type: str = "text_delta"

    def to_dict(self) -> dict:
        return {"type": self.type, "content": self.content}


@dataclass
class TextEndEvent:
    """Complete response text (streaming)."""
    content: str
    type: str = "text_end"

    def to_dict(self) -> dict:
        return {"type": self.type, "content": self.content}


@dataclass
class ToolCallEvent:
    """A tool is about to be executed."""
    name: str
    arguments: Dict[str, Any]
    type: str = "tool_call"

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass
class ToolResultEvent:
    """A tool has returned its result."""
    name: str
    result: str
    type: str = "tool_result"

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "name": self.name,
            "result": self.result,
        }


@dataclass
class TurnStartEvent:
    """A new agent turn is starting (after a tool-calling turn)."""
    type: str = "turn_start"

    def to_dict(self) -> dict:
        return {"type": self.type}


@dataclass
class FinishReasonEvent:
    """Non-standard finish reason (truncation, content filter, etc)."""
    reason: str
    content: str
    type: str = "finish_reason"

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "reason": self.reason,
            "content": self.content,
        }


@dataclass
class HistoryTrimmedEvent:
    """Conversation history was trimmed/summarized."""
    summarized: int
    type: str = "history_trimmed"

    def to_dict(self) -> dict:
        return {"type": self.type, "summarized": self.summarized}


# ── Registry & conversion ───────────────────────────────────────────────────

_EVENT_TYPES = {
    "tokens": TokensEvent,
    "thinking": ThinkingEvent,
    "thinking_delta": ThinkingDeltaEvent,
    "thinking_end": ThinkingEndEvent,
    "text": TextEvent,
    "text_delta": TextDeltaEvent,
    "text_end": TextEndEvent,
    "tool_call": ToolCallEvent,
    "tool_result": ToolResultEvent,
    "turn_start": TurnStartEvent,
    "finish_reason": FinishReasonEvent,
    "history_trimmed": HistoryTrimmedEvent,
}


def from_dict(d: dict) -> Any:
    """Parse a dict event into a typed dataclass if possible.

    Returns the original dict if the type is unknown.
    """
    etype = d.get("type")
    cls = _EVENT_TYPES.get(etype)
    if cls is None:
        return d
    kwargs = {k: v for k, v in d.items() if k != "type"}
    return cls(**kwargs)


# ── Callback protocol ───────────────────────────────────────────────────────


@runtime_checkable
class Callback(Protocol):
    """Protocol for agent progress callbacks.

    Accepts either a typed event dataclass or a plain dict.
    """
    def __call__(self, event: Any) -> None:
        ...
