"""Tests for typed event dataclasses and Callback protocol."""

import pytest
from harness import events


class TestEventDataclasses:
    """Verify event dataclasses have the right fields and types."""

    def test_tokens_event_fields(self):
        e = events.TokensEvent(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            cached_tokens=2,
            turn_input=10,
            turn_output=5,
            turn_cached=2,
            context_window=1000,
            model="test",
            reasoning_effort="high",
        )
        assert e.type == "tokens"
        assert e.input_tokens == 10

    def test_thinking_event_fields(self):
        e = events.ThinkingEvent(content="reasoning")
        assert e.type == "thinking"
        assert e.content == "reasoning"

    def test_thinking_delta_fields(self):
        e = events.ThinkingDeltaEvent(content="reasoning chunk")
        assert e.type == "thinking_delta"

    def test_thinking_end_fields(self):
        e = events.ThinkingEndEvent()
        assert e.type == "thinking_end"

    def test_text_event_fields(self):
        e = events.TextEvent(content="answer")
        assert e.type == "text"
        assert e.content == "answer"

    def test_text_delta_fields(self):
        e = events.TextDeltaEvent(content="tok")
        assert e.type == "text_delta"

    def test_text_end_fields(self):
        e = events.TextEndEvent(content="full answer")
        assert e.type == "text_end"
        assert e.content == "full answer"

    def test_tool_call_event_fields(self):
        e = events.ToolCallEvent(name="read", arguments={"path": "x.txt"})
        assert e.type == "tool_call"
        assert e.name == "read"
        assert e.arguments == {"path": "x.txt"}

    def test_tool_result_event_fields(self):
        e = events.ToolResultEvent(name="read", result="contents")
        assert e.type == "tool_result"

    def test_turn_start_fields(self):
        e = events.TurnStartEvent()
        assert e.type == "turn_start"

    def test_finish_reason_fields(self):
        e = events.FinishReasonEvent(reason="length", content="truncated")
        assert e.type == "finish_reason"
        assert e.reason == "length"

    def test_history_trimmed_fields(self):
        e = events.HistoryTrimmedEvent(summarized=5)
        assert e.type == "history_trimmed"
        assert e.summarized == 5

    def test_to_dict_conversion(self):
        """Dataclasses should convert to the same dict shape callbacks expect."""
        e = events.ToolCallEvent(name="bash", arguments={"command": "ls"})
        d = e.to_dict()
        assert d == {"type": "tool_call", "name": "bash", "arguments": {"command": "ls"}}

    def test_from_dict_creates_typed_event(self):
        """from_dict should parse a dict back into a typed event."""
        d = {"type": "text_delta", "content": "hello"}
        e = events.from_dict(d)
        assert isinstance(e, events.TextDeltaEvent)
        assert e.content == "hello"


class TestCallbackProtocol:
    """Verify the Callback protocol works as expected."""

    def test_callback_accepts_dict(self):
        """Plain dict events should still work (backward compat)."""
        received = []

        def cb(event):
            received.append(event)

        cb({"type": "text", "content": "hi"})
        assert received[0] == {"type": "text", "content": "hi"}

    def test_callback_accepts_typed_event(self):
        """Typed event dataclasses should be accepted by callbacks."""
        received = []

        def cb(event):
            received.append(event)

        e = events.TextEvent(content="hi")
        cb(e)
        assert received[0].type == "text"

    def test_callback_protocol_is_runtime_checkable(self):
        """Functions and callables should satisfy the Callback protocol."""
        from harness.events import Callback

        def my_cb(event):
            pass

        assert isinstance(my_cb, Callback)
