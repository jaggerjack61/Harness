"""Tests for the LiveDisplay class."""

from harness.display import LiveDisplay, ResponseBuffer


class TestResponseBuffer:
    """Tests for the response buffer (moved from cli.py)."""

    def test_initial_state(self):
        buf = ResponseBuffer()
        assert buf.text == ""
        assert buf.partial == ""
        assert buf.complete == ""

    def test_append_and_partial(self):
        buf = ResponseBuffer()
        buf.append("hello ")
        buf.append("world")
        assert buf.complete == ""
        assert buf.partial == "hello world"
        assert buf.text == "hello world"

    def test_complete_line_folded_on_newline(self):
        buf = ResponseBuffer()
        buf.append("## Heading\n")
        assert buf.complete == "## Heading\n"
        assert buf.partial == ""

    def test_hybrid_complete_plus_partial(self):
        buf = ResponseBuffer()
        buf.append("## Heading\npartial ")
        buf.append("text")
        assert buf.complete == "## Heading\n"
        assert buf.partial == "partial text"

    def test_set_complete_finalizes(self):
        buf = ResponseBuffer()
        buf.append("partial")
        buf.set_complete("final answer\n")
        assert buf.complete == "final answer\n"
        assert buf.partial == ""
        assert buf.text == "final answer\n"

    def test_reset_clears_state(self):
        buf = ResponseBuffer()
        buf.append("x\ny")
        buf.reset()
        assert buf.complete == ""
        assert buf.partial == ""
        assert buf.text == ""

    def test_partial_is_cached_between_appends(self):
        buf = ResponseBuffer()
        buf.append("x" * 200)
        buf.append("y" * 200)
        first = buf.partial
        second = buf.partial
        assert first is second
        buf.append("z" * 200)
        third = buf.partial
        assert third is not first
        assert third == "x" * 200 + "y" * 200 + "z" * 200

    def test_compact_scales_linearly_with_newlines(self):
        import time
        buf = ResponseBuffer()

        def time_n(n):
            buf.reset()
            t = time.perf_counter()
            for _ in range(n):
                buf.append("a")
            buf.append("a\n")
            return time.perf_counter() - t

        t_small = time_n(2000)
        t_large = time_n(8000)
        ratio = t_large / t_small if t_small > 0 else 0
        assert ratio < 8, f"compact scaled super-linearly: ratio {ratio:.1f}"


class TestLiveDisplay:
    """LiveDisplay encapsulates the streaming response + token bar state."""

    def test_initial_state(self):
        display = LiveDisplay()
        assert display.streamed_any is False
        assert display.response_buffer.text == ""
        assert display.agent_active is False

    def test_reset_for_new_turn(self):
        display = LiveDisplay()
        display.streamed_any = True
        display.response_buffer.append("test")
        display.last_token_event = {"type": "tokens"}
        display.reset_for_new_turn()
        assert display.streamed_any is False
        assert display.response_buffer.text == ""
        assert display.last_token_event is None
        assert display.agent_active is True

    def test_on_event_accepts_dict(self):
        display = LiveDisplay()
        # Should not raise — the callback accepts dicts.
        display.on_event({"type": "text_delta", "content": "hi"})

    def test_two_displays_are_independent(self):
        d1 = LiveDisplay()
        d2 = LiveDisplay()
        d1.response_buffer.append("d1")
        d2.response_buffer.append("d2")
        assert d1.response_buffer.text == "d1"
        assert d2.response_buffer.text == "d2"
