"""Nasa Level Genius Agent using OpenAI-compatible chat models with tool calling."""

import json
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI

from harness.constants import (
    DEFAULT_MAX_TURNS,
    DEFAULT_CONTEXT_WINDOW,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    CONTEXT_WINDOW_TRIM_THRESHOLD,
    RECENT_TURNS_TO_KEEP,
    SUMMARY_MAX_TOKENS,
)
from harness.tools import ToolRegistry

DEFAULT_SYSTEM_PROMPT = """You are an expert coding assistant. You have access to the following tools:

- **read** — Read file contents. Use for examining files, with optional offset/limit for large files.
- **write** — Create or overwrite a file. Automatically creates parent directories.
- **edit** — Make precise text replacements in files. Each edit specifies oldText and newText.
- **bash** — Execute shell commands (PowerShell on Windows, bash on Unix). Use for listing files, running tests, installing packages, etc.

On Windows, all bash commands run in PowerShell. Use PowerShell commands (ls, Get-Content, Select-String, Get-ChildItem, etc.). Powershell uses ; instead of &&.
Always use these tools when you need to interact with the file system or execute commands.
Your tool results may be truncated for display — keep your commands and file reads concise.
When searching or listing files, limit output with Select-Object -First, | head, or similar.
Be concise and helpful."""


# Reasoning/thinking field names providers use, in precedence order. Shared by
# the streaming-delta and full-message extraction paths (DRY).
_REASONING_ATTRS: Tuple[str, ...] = ("reasoning_content", "thinking", "thought")
_REASONING_EXTRA_KEYS: Tuple[str, ...] = (
    "reasoning_content",
    "reasoning",
    "thinking",
    "thought",
)



def _extract_reasoning_fields(obj: Any) -> Optional[str]:
    """Extract reasoning/thinking content from a delta or message object.

    Checks the known attribute names directly, then ``model_extra`` (Pydantic v2
    extra-fields passthrough). Returns the first non-empty string found, or None.
    """
    for attr in _REASONING_ATTRS:
        val = getattr(obj, attr, None)
        if isinstance(val, str) and val:
            return val
    model_extra = getattr(obj, "model_extra", None)
    if isinstance(model_extra, dict):
        for key in _REASONING_EXTRA_KEYS:
            val = model_extra.get(key)
            if isinstance(val, str) and val:
                return val
    return None


class AgentHarness:
    """An AI agent that can use read/write/edit/bash tools via an OpenAI-compatible API.

    Args:
        model: Model name (e.g., "deepseek-v4-pro", "gpt-3.5-turbo").
        api_key: API key. If not provided, uses the OPENAI_API_KEY env var.
        base_url: Base URL for the API. Defaults to OpenAI.
        system_prompt: Custom system prompt.
        working_dir: Working directory for bash commands.
        max_turns: Maximum number of tool-calling turns before stopping.
        context_window: Model context window size in tokens (default: 1000000).
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        working_dir: Optional[str] = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        reasoning_effort: Optional[str] = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
    ):
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_turns = max_turns
        self.reasoning_effort = reasoning_effort
        self.context_window = context_window

        # Build OpenAI client (use placeholder key if none provided — a real
        # key is only required when actually making API calls via run()).
        client_kwargs: Dict[str, Any] = {
            "api_key": api_key or "sk-placeholder",
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

        self.tool_registry = ToolRegistry(working_dir=working_dir)
        self.messages: List[Dict[str, Any]] = []
        self.custom_context: Optional[str] = None

        # Token counters (cumulative over the entire session)
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.cached_tokens: int = 0
        # Last API call's prompt_tokens — used to estimate current history size.
        self._last_prompt_tokens: int = 0

    # ── Public API ──────────────────────────────────────────────────────

    def _create_with_retry(self, **kwargs: Any) -> Any:
        """Call the API with retry on transient errors (429/5xx/timeout)."""
        for attempt in range(MAX_RETRIES + 1):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as e:
                if attempt == MAX_RETRIES or not self._is_retryable(e):
                    raise
                delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(delay)

    @staticmethod
    def _is_retryable(e: Exception) -> bool:
        """Check if an exception is a transient error worth retrying."""
        status = getattr(e, "status_code", None)
        if status is not None:
            return status == 429 or 500 <= status < 600
        if isinstance(e, (TimeoutError, ConnectionError)):
            return True
        return False

    def run(
        self,
        prompt: str,
        callback: Optional[Callable[..., Any]] = None,
        stream: bool = False,
    ) -> str:
        """Run the agent with a user prompt and return the final response.

        Args:
            prompt: The user's request.
            callback: Optional callable receiving progress events as dicts.
            stream: If True, stream text output token by token.

        Returns:
            The final text response from the model.
        """
        if not self.messages:
            self.messages = self._build_initial_messages(prompt)
        else:
            self.messages.append({"role": "user", "content": prompt})

        self._trim_history(callback)

        for turn_idx in range(self.max_turns):
            if turn_idx > 0 and callback:
                callback({"type": "turn_start"})

            create_kwargs = self._build_create_kwargs()
            if stream:
                create_kwargs["stream"] = True
                create_kwargs["stream_options"] = {"include_usage": True}

            response = self._create_with_retry(**create_kwargs)
            text, reasoning, tool_calls = self._process_response(response, stream, callback)

            if tool_calls:
                self._handle_tool_calls(text, tool_calls, callback)
            else:
                return self._finalize_response(text, stream, callback)

        return "Max turns reached without a final response."

    def _build_create_kwargs(self) -> Dict[str, Any]:
        """Build the kwargs dict for the next chat.completions.create call."""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "tools": self.tool_registry.get_definitions(),
        }
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return kwargs

    def _process_response(
        self,
        response: Any,
        stream: bool,
        callback: Optional[Callable[..., Any]],
    ) -> Tuple[Optional[str], Optional[str], List[Dict[str, Any]]]:
        """Extract text/reasoning/tool_calls from a streaming or non-streaming response.

        Also tracks tokens and emits the thinking event (non-streaming) and
        finish_reason warning (non-streaming).
        """
        if stream:
            text, reasoning, tool_calls, usage = self._process_stream(response, callback)
        else:
            text, reasoning, tool_calls = self._parse_response(response)
            usage = None

        self._track_and_emit_tokens_from_usage(usage, response, stream, callback)

        if not stream and callback:
            self._maybe_emit_finish_reason(response, text, callback)
        if reasoning and callback and not stream:
            callback({"type": "thinking", "content": reasoning})

        return text, reasoning, tool_calls

    def _track_and_emit_tokens_from_usage(
        self,
        usage: Optional[Dict[str, Any]],
        response: Any,
        stream: bool,
        callback: Optional[Callable[..., Any]],
    ) -> None:
        """Track per-turn tokens from streaming usage dict or non-streaming response."""
        if stream and usage:
            turn_input = usage.get("prompt_tokens", 0) or 0
            turn_output = usage.get("completion_tokens", 0) or 0
            pts = usage.get("prompt_tokens_details") or {}
            turn_cached = pts.get("cached_tokens", 0) or 0
        elif not stream and hasattr(response, "usage") and response.usage:
            turn_input = response.usage.prompt_tokens or 0
            turn_output = response.usage.completion_tokens or 0
            pts = getattr(response.usage, "prompt_tokens_details", None)
            turn_cached = (getattr(pts, "cached_tokens", 0) or 0) if pts else 0
        else:
            turn_input = turn_output = turn_cached = 0
        self._track_and_emit_tokens(turn_input, turn_output, turn_cached, callback)

    @staticmethod
    def _maybe_emit_finish_reason(
        response: Any,
        text: Optional[str],
        callback: Callable[..., Any],
    ) -> None:
        """Emit a finish_reason warning for non-standard stop reasons."""
        if not response.choices:
            return
        finish_reason = response.choices[0].finish_reason
        if isinstance(finish_reason, str) and finish_reason not in ("stop", "tool_calls"):
            callback({"type": "finish_reason", "reason": finish_reason, "content": text or ""})

    def _handle_tool_calls(
        self,
        text: Optional[str],
        tool_calls: List[Dict[str, Any]],
        callback: Optional[Callable[..., Any]],
    ) -> None:
        """Append assistant msg, execute tools, append results, emit events."""
        msg: Dict[str, Any] = {
            "role": "assistant",
            "content": text,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    },
                }
                for tc in tool_calls
            ],
        }
        self.messages.append(msg)

        for tc in tool_calls:
            if callback:
                callback({"type": "tool_call", "name": tc["name"], "arguments": tc["arguments"]})

            try:
                result = self.tool_registry.execute(tc["name"], tc["arguments"])
            except Exception as e:
                result = f"Error: {e}"

            if callback:
                callback({"type": "tool_result", "name": tc["name"], "result": result})

            self.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

        if text and callback:
            callback({"type": "text_end", "content": text})

    def _finalize_response(
        self,
        text: Optional[str],
        stream: bool,
        callback: Optional[Callable[..., Any]],
    ) -> str:
        """Append final assistant msg, emit text/text_end, return text."""
        self.messages.append({"role": "assistant", "content": text})
        if stream and callback:
            callback({"type": "text_end", "content": text or ""})
        elif callback:
            callback({"type": "text", "content": text or ""})
        return text or ""

    def _process_stream(
        self,
        stream: Any,
        callback: Optional[Callable[..., Any]] = None,
    ) -> Tuple[Optional[str], Optional[str], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Process a streaming response from the API.

        Args:
            stream: The stream iterator from client.chat.completions.create(stream=True).
            callback: Optional callable for progress events.

        Returns:
            A tuple of (text_content, reasoning_content, list_of_tool_calls, usage_dict).
        """
        state = {
            "text_chunks": [],
            "reasoning_chunks": [],
            "thinking_open": False,
            "tool_calls_map": {},
            "usage": None,
            "last_finish_reason": None,
        }

        for chunk in stream:
            self._collect_stream_usage(chunk, state)
            if not chunk.choices:
                continue
            self._handle_stream_chunk(chunk.choices[0], callback, state)

        return self._finalize_stream(state, callback)

    def _collect_stream_usage(self, chunk: Any, state: Dict[str, Any]) -> None:
        """Extract usage from the final chunk if available."""
        if not (hasattr(chunk, "usage") and chunk.usage):
            return
        usage = {
            "prompt_tokens": chunk.usage.prompt_tokens,
            "completion_tokens": chunk.usage.completion_tokens,
            "prompt_tokens_details": {},
        }
        if chunk.usage.prompt_tokens_details:
            pts = chunk.usage.prompt_tokens_details
            if hasattr(pts, "cached_tokens"):
                usage["prompt_tokens_details"]["cached_tokens"] = pts.cached_tokens
        state["usage"] = usage

    def _handle_stream_chunk(
        self,
        choice: Any,
        callback: Optional[Callable[..., Any]],
        state: Dict[str, Any],
    ) -> None:
        """Process a single stream choice: text, reasoning, tool calls."""
        delta = choice.delta
        finish_reason = choice.finish_reason
        if finish_reason:
            state["last_finish_reason"] = finish_reason

        self._accumulate_text(delta, callback, state)
        self._accumulate_reasoning(delta, callback, state)
        self._accumulate_tool_calls(delta, callback, state)

    def _accumulate_text(
        self,
        delta: Any,
        callback: Optional[Callable[..., Any]],
        state: Dict[str, Any],
    ) -> None:
        """Append text content and emit text_delta events."""
        if not delta.content:
            return
        state["text_chunks"].append(delta.content)
        if callback:
            callback({"type": "text_delta", "content": delta.content})
        self._close_thinking(callback, state)

    def _accumulate_reasoning(
        self,
        delta: Any,
        callback: Optional[Callable[..., Any]],
        state: Dict[str, Any],
    ) -> None:
        """Append reasoning content and emit thinking_delta events."""
        reasoning_delta = self._extract_reasoning_delta(delta)
        if not reasoning_delta:
            return
        state["reasoning_chunks"].append(reasoning_delta)
        state["thinking_open"] = True
        if callback:
            callback({"type": "thinking_delta", "content": reasoning_delta})

    def _accumulate_tool_calls(
        self,
        delta: Any,
        callback: Optional[Callable[..., Any]],
        state: Dict[str, Any],
    ) -> None:
        """Accumulate tool call deltas into a map keyed by tool index."""
        if not delta.tool_calls:
            return
        self._close_thinking(callback, state)
        for tc_delta in delta.tool_calls:
            tcm = state["tool_calls_map"]
            idx = tc_delta.index
            if idx not in tcm:
                tcm[idx] = {
                    "id": tc_delta.id or "",
                    "name": tc_delta.function.name if tc_delta.function and tc_delta.function.name else "",
                    "arguments_str": "",
                }
            elif tc_delta.id:
                tcm[idx]["id"] = tc_delta.id
            if tc_delta.function and tc_delta.function.name:
                tcm[idx]["name"] = tc_delta.function.name
            if tc_delta.function and tc_delta.function.arguments:
                tcm[idx]["arguments_str"] += tc_delta.function.arguments

    def _finalize_stream(
        self,
        state: Dict[str, Any],
        callback: Optional[Callable[..., Any]],
    ) -> Tuple[Optional[str], Optional[str], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """After the stream ends, parse tool calls, emit finish_reason, return."""
        self._close_thinking(callback, state)

        tool_calls = self._parse_streamed_tool_calls(state["tool_calls_map"])
        self._emit_finish_reason_if_needed(state, callback)

        text = "".join(state["text_chunks"]) if state["text_chunks"] else None
        reasoning = "".join(state["reasoning_chunks"]) if state["reasoning_chunks"] else None
        return text, reasoning, tool_calls, state["usage"]

    @staticmethod
    def _parse_streamed_tool_calls(
        tool_calls_map: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert the accumulated tool-call map into a sorted list of dicts."""
        result = []
        for idx in sorted(tool_calls_map.keys()):
            tc = tool_calls_map[idx]
            try:
                args = json.loads(tc["arguments_str"])
            except (json.JSONDecodeError, TypeError):
                args = {}
            result.append({"id": tc["id"], "name": tc["name"], "arguments": args})
        return result

    @staticmethod
    def _emit_finish_reason_if_needed(
        state: Dict[str, Any],
        callback: Optional[Callable[..., Any]],
    ) -> None:
        """Emit a finish_reason warning for non-standard stop reasons."""
        if not callback:
            return
        fr = state["last_finish_reason"]
        if not (isinstance(fr, str) and fr not in ("stop", "tool_calls")):
            return
        text_so_far = "".join(state["text_chunks"]) if state["text_chunks"] else ""
        callback({"type": "finish_reason", "reason": fr, "content": text_so_far})

    @staticmethod
    def _close_thinking(
        callback: Optional[Callable[..., Any]],
        state: Dict[str, Any],
    ) -> None:
        """Emit a thinking_end event if a thinking block is currently open."""
        if state["thinking_open"] and callback:
            callback({"type": "thinking_end"})
        state["thinking_open"] = False

    def _extract_reasoning_delta(self, delta: Any) -> Optional[str]:
        """Extract reasoning content from a streaming delta.

        Different providers return reasoning in different fields:
        - OpenAI o-series / DeepSeek: delta.reasoning_content
        - Some providers: delta.thinking, delta.thought
        - Some providers: delta.model_extra['reasoning_content' | 'reasoning' | 'thinking' | 'thought']
        """
        return _extract_reasoning_fields(delta)

    def clear_history(self) -> None:
        """Clear the conversation history (but keep custom context)."""
        self.messages = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_tokens = 0
        self._last_prompt_tokens = 0

    def set_custom_context(self, text: Optional[str]) -> None:
        """Set custom context text that will be prepended to the system prompt.

        Set to None or empty string to clear. If a conversation is already in
        progress, the existing system message is updated so the new context is
        visible immediately.
        """
        self.custom_context = text or None
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = self._build_system_content()

    def get_custom_context(self) -> Optional[str]:
        """Return the current custom context, or None if not set."""
        return self.custom_context

    # ── Internal helpers ────────────────────────────────────────────────

    def _build_system_content(self) -> str:
        """Build the system prompt content, including CWD and optional custom context."""
        cwd = self.tool_registry.working_dir or os.getcwd()
        cwd_note = f"\nWorking directory: {cwd}. All commands run from this directory."
        if self.custom_context:
            return (
                f"{self.system_prompt}\n\n--- Additional Context ---\n"
                f"{self.custom_context}\n--- End Additional Context ---\n"
            ) + cwd_note
        return self.system_prompt + cwd_note

    def _build_initial_messages(self, prompt: str) -> List[Dict[str, Any]]:
        """Build the initial message list for a new conversation."""
        return [
            {"role": "system", "content": self._build_system_content()},
            {"role": "user", "content": prompt},
        ]

    # ── Context-window management ───────────────────────────────────────

    def _trim_history(self, callback: Optional[Callable[..., Any]] = None) -> None:
        """Trim conversation history to fit within the context window.

        When the estimated token count approaches ``context_window``, old
        turns are summarized into a compact system message to preserve
        context without exceeding the window. The original system prompt
        and the most recent turns are always kept.
        """
        if not self.messages or self.context_window <= 0:
            return

        last = self._last_prompt_tokens
        estimated = last if isinstance(last, int) and last > 0 else self._estimate_token_count()
        threshold = int(self.context_window * CONTEXT_WINDOW_TRIM_THRESHOLD)
        if estimated <= threshold:
            return

        # Split into system prefix, middle (to summarize), and recent suffix.
        system_msgs, rest = [], self.messages
        if rest and rest[0].get("role") == "system":
            system_msgs = [rest[0]]
            rest = rest[1:]

        keep_count = min(RECENT_TURNS_TO_KEEP, len(rest))
        to_summarize = rest[:-keep_count] if keep_count > 0 else []
        to_keep = rest[-keep_count:] if keep_count > 0 else rest
        if not to_summarize:
            return

        summary = self._summarize_turns(to_summarize)
        if callback:
            callback({"type": "history_trimmed", "summarized": len(to_summarize)})

        new_msgs = list(system_msgs)
        if summary:
            new_msgs.append({
                "role": "system",
                "content": f"--- Summary of earlier conversation ---\n{summary}\n--- End Summary ---",
            })
        new_msgs.extend(to_keep)
        self.messages = new_msgs

    def _summarize_turns(self, turns: List[Dict[str, Any]]) -> str:
        """Summarize old conversation turns into compact text via the API."""
        parts = []
        for msg in turns:
            role = msg.get("role", "unknown")
            content = str(msg.get("content", "") or "")
            if role == "tool":
                content = content[:300]
            parts.append(f"{role}: {content}")
        conversation_text = "\n".join(parts)

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Summarize the following conversation concisely, preserving key facts, decisions, and file paths."},
                    {"role": "user", "content": conversation_text},
                ],
                max_tokens=SUMMARY_MAX_TOKENS,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            return ""

    def _estimate_token_count(self) -> int:
        """Rough token estimate: ~4 characters per token across all messages."""
        total_chars = sum(len(str(m.get("content", "") or "")) for m in self.messages)
        return total_chars // 4

    def _track_and_emit_tokens(
        self,
        turn_input: int,
        turn_output: int,
        turn_cached: int,
        callback: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Accumulate per-turn token usage and emit a live ``tokens`` event.

        Shared by the streaming and non-streaming paths so the cumulative
        counters and the event payload stay in sync (DRY).
        """
        self.input_tokens += turn_input
        self.output_tokens += turn_output
        self.cached_tokens += turn_cached
        self._last_prompt_tokens = turn_input
        if callback:
            callback({
                "type": "tokens",
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
                "cached_tokens": self.cached_tokens,
                "turn_input": turn_input,
                "turn_output": turn_output,
                "turn_cached": turn_cached,
                "context_window": self.context_window,
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
            })

    def _parse_response(self, response: Any) -> Tuple[Optional[str], Optional[str], List[Dict[str, Any]]]:
        """Parse an OpenAI chat completion response.

        Args:
            response: The response object from the API.

        Returns:
            A tuple of (text_content, reasoning_content, list_of_tool_calls).
            text_content is None if there are tool calls.
            reasoning_content is the model's chain-of-thought (may be None).
            tool_calls is empty if there are no tool calls.
        """
        if not response.choices:
            return None, None, []
        choice = response.choices[0]
        message = choice.message

        text = message.content
        reasoning = self._extract_reasoning(message)
        tool_calls_raw = message.tool_calls

        if not tool_calls_raw:
            return text, reasoning, []

        parsed_calls = []
        for tc in tool_calls_raw:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}

            parsed_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                }
            )

        return text, reasoning, parsed_calls

    def _extract_reasoning(self, message: Any) -> Optional[str]:
        """Extract reasoning/thinking content from a message object.

        Different providers return reasoning in different fields:
        - OpenAI o-series: message.reasoning_content
        - Some providers: message.thinking, message.thought
        - Some providers: message.model_extra['reasoning_content' | 'reasoning' | 'thinking' | 'thought']
        """
        return _extract_reasoning_fields(message)
