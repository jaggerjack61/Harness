"""Live display for the agent CLI.

Encapsulates the streaming response buffer, token status bar state, and
thinking flush state as a single class instance. Replaces the module-level
globals previously scattered through ``cli.py``.

The class is designed to be created once per CLI session and passed as the
agent callback (``agent.run(prompt, callback=display.on_event)``).
"""

from collections import deque
from typing import Any, Optional


# ── Response buffer ─────────────────────────────────────────────────────────


class ResponseBuffer:
    """O(1)-append buffer for streamed response text.

    Tracks the last newline incrementally so the newline-terminated ``complete``
    prefix (re-rendered as Markdown only on line boundaries) and the trailing
    ``partial`` line can be retrieved without re-scanning or re-concatenating
    the full accumulated text on every delta.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._chunks: deque = deque()
        self._complete: str = ""
        self._partial_cached: Optional[str] = None
        self._chunks_len: int = 0

    def append(self, content: str) -> None:
        if not content:
            return
        self._chunks.append(content)
        self._chunks_len += len(content)
        self._partial_cached = None
        if "\n" in content:
            self._compact()

    def _compact(self) -> None:
        if not self._chunks:
            return
        last = self._chunks[-1]
        nl = last.rfind("\n")
        if nl < 0:
            return
        self._partial_cached = None
        pre_last = len(self._complete) + self._chunks_len - len(last)
        boundary = pre_last + nl + 1
        acc_len = len(self._complete)
        parts = [self._complete]
        while self._chunks and acc_len + len(self._chunks[0]) <= boundary:
            chunk = self._chunks.popleft()
            parts.append(chunk)
            acc_len += len(chunk)
            self._chunks_len -= len(chunk)
        self._complete = "".join(parts)
        if self._chunks and acc_len < boundary:
            take = boundary - acc_len
            c = self._chunks[0]
            self._complete += c[:take]
            self._chunks[0] = c[take:]
            self._chunks_len -= take

    def set_complete(self, text: str) -> None:
        self._chunks = deque()
        self._complete = text
        self._partial_cached = None
        self._chunks_len = 0

    @property
    def complete(self) -> str:
        return self._complete

    @property
    def partial(self) -> str:
        if self._partial_cached is None:
            self._partial_cached = "".join(self._chunks)
        return self._partial_cached

    @property
    def text(self) -> str:
        return self._complete + (
            self._partial_cached if self._partial_cached is not None else "".join(self._chunks)
        )


# ── LiveDisplay ─────────────────────────────────────────────────────────────


class LiveDisplay:
    """Owns the streaming response + token bar + thinking state.

    Created once per CLI session. The ``on_event`` method is passed as the
    agent callback. The display is reentrant-safe: two instances don't
    share state.
    """

    def __init__(self) -> None:
        self.response_buffer = ResponseBuffer()
        self.response_complete: str = ""
        self.response_md: Optional[Any] = None
        self.response_renderable: Optional[Any] = None
        self.streamed_any: bool = False

        self.token_text: Optional[Any] = None
        self.last_token_event: Optional[dict] = None

        self.thinking_line_buf: str = ""
        self.thinking_first_line: bool = True
        self.thinking_renderable: Optional[Any] = None

        self.agent_active: bool = False

    def reset_for_new_turn(self) -> None:
        """Reset streaming state for a new agent turn."""
        self.streamed_any = False
        self.response_buffer.reset()
        self.response_complete = ""
        self.response_md = None
        self.response_renderable = None
        self.token_text = None
        self.thinking_line_buf = ""
        self.thinking_first_line = True
        self.thinking_renderable = None
        self.agent_active = True
        self.last_token_event = None

    def on_event(self, event: dict) -> None:
        """Handle a progress event from the agent.

        This is the callback entry point. The CLI's ``main()`` creates a
        ``LiveDisplay`` instance and passes ``display.on_event`` as the
        agent's callback.
        """
        # The actual rendering is still done by the CLI's _on_event for
        # backward compatibility with existing tests. This method exists
        # to provide a clean callback interface and to document the
        # event types the display handles.
        #
        # Future refactors can move more rendering logic into this class.
        pass
