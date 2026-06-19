"""Nasa Level Genius Agent using OpenAI-compatible chat models with tool calling."""

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI

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
        max_turns: int = 1000,
        reasoning_effort: Optional[str] = None,
        context_window: int = 1000000,
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

    # ── Public API ──────────────────────────────────────────────────────

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
                Event types:
                - {"type": "tokens", "input_tokens": N, "output_tokens": N, ...}  (live token usage after each API call)
                - {"type": "thinking", "content": "..."}  (reasoning/chain-of-thought, non-streaming)
                - {"type": "thinking_delta", "content": "..."}  (incremental reasoning, streaming only)
                - {"type": "thinking_end"}  (end of a streaming reasoning block)
                - {"type": "text", "content": "..."}  (model text response, non-streaming)
                - {"type": "text_delta", "content": "..."}  (incremental text, streaming only)
                - {"type": "text_end", "content": "..."}  (final complete text, streaming only)
                - {"type": "tool_call", "name": "...", "arguments": {...}}
                - {"type": "tool_result", "name": "...", "result": "..."}
            stream: If True, stream text output token by token.

        Returns:
            The final text response from the model.
        """
        if not self.messages:
            self.messages = self._build_initial_messages(prompt)
        else:
            self.messages.append({"role": "user", "content": prompt})

        for _ in range(self.max_turns):
            create_kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": self.messages,
                "tools": self.tool_registry.get_definitions(),
            }
            if self.reasoning_effort is not None:
                create_kwargs["reasoning_effort"] = self.reasoning_effort

            if stream:
                # Streaming mode
                create_kwargs["stream"] = True
                response_stream = self.client.chat.completions.create(**create_kwargs)
                text, reasoning, tool_calls, usage = self._process_stream(response_stream, callback)

                # Track token usage per-turn and emit live stats event.
                turn_input = turn_output = turn_cached = 0
                if usage:
                    turn_input = usage.get('prompt_tokens', 0) or 0
                    turn_output = usage.get('completion_tokens', 0) or 0
                    pts = usage.get('prompt_tokens_details') or {}
                    turn_cached = pts.get('cached_tokens', 0) or 0
                self._track_and_emit_tokens(turn_input, turn_output, turn_cached, callback)
            else:
                # Non-streaming mode
                response = self.client.chat.completions.create(**create_kwargs)

                # Track token usage per-turn and emit live stats event.
                turn_input = turn_output = turn_cached = 0
                if hasattr(response, 'usage') and response.usage:
                    turn_input = response.usage.prompt_tokens or 0
                    turn_output = response.usage.completion_tokens or 0
                    if response.usage.prompt_tokens_details:
                        pts = response.usage.prompt_tokens_details
                        if hasattr(pts, 'cached_tokens') and pts.cached_tokens:
                            turn_cached = pts.cached_tokens
                self._track_and_emit_tokens(turn_input, turn_output, turn_cached, callback)

                text, reasoning, tool_calls = self._parse_response(response)

            # Emit thinking/reasoning if present (before tool calls or text).
            # In streaming mode, _process_stream already emitted thinking_delta
            # events, so we only emit the full thinking block here for the
            # non-streaming path.
            if reasoning and callback and not stream:
                callback({
                    "type": "thinking",
                    "content": reasoning,
                })

            if tool_calls:
                # Add assistant message with tool calls (preserve reasoning for history)
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
                if reasoning:
                    msg["reasoning_content"] = reasoning
                self.messages.append(msg)

                # Execute each tool, fire events, and add results
                for tc in tool_calls:
                    if callback:
                        callback({
                            "type": "tool_call",
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        })

                    try:
                        result = self.tool_registry.execute(
                            tc["name"], tc["arguments"]
                        )
                    except Exception as e:
                        result = f"Error: {e}"

                    if callback:
                        callback({
                            "type": "tool_result",
                            "name": tc["name"],
                            "result": result,
                        })

                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result,
                        }
                    )
            else:
                # Final text response (preserve reasoning for history)
                msg = {"role": "assistant", "content": text}
                if reasoning:
                    msg["reasoning_content"] = reasoning
                self.messages.append(msg)

                if stream and callback:
                    # Emit text_end event with the complete text
                    callback({
                        "type": "text_end",
                        "content": text or "",
                    })
                elif callback:
                    callback({
                        "type": "text",
                        "content": text or "",
                    })

                return text or ""

        return "Max turns reached without a final response."

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
        text_chunks: List[str] = []
        reasoning_chunks: List[str] = []
        thinking_open = False  # whether a thinking_delta block is currently open
        tool_calls_map: Dict[int, Dict[str, Any]] = {}  # index -> {id, name, arguments_str}
        usage = None

        def _close_thinking() -> None:
            nonlocal thinking_open
            if thinking_open and callback:
                callback({"type": "thinking_end"})
            thinking_open = False

        for chunk in stream:
            # Extract usage from the final chunk if available
            if hasattr(chunk, 'usage') and chunk.usage:
                usage = {
                    'prompt_tokens': chunk.usage.prompt_tokens,
                    'completion_tokens': chunk.usage.completion_tokens,
                    'prompt_tokens_details': {}
                }
                if chunk.usage.prompt_tokens_details:
                    pts = chunk.usage.prompt_tokens_details
                    if hasattr(pts, 'cached_tokens'):
                        usage['prompt_tokens_details']['cached_tokens'] = pts.cached_tokens

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason

            # Handle text content
            if delta.content:
                text_chunks.append(delta.content)
                if callback:
                    callback({
                        "type": "text_delta",
                        "content": delta.content,
                    })

            # Handle reasoning content (different providers use different fields).
            # Accumulate for history, but also emit live thinking_delta events so
            # the CLI can show reasoning progress as it arrives.
            reasoning_delta = self._extract_reasoning_delta(delta)
            if reasoning_delta:
                reasoning_chunks.append(reasoning_delta)
                thinking_open = True
                if callback:
                    callback({
                        "type": "thinking_delta",
                        "content": reasoning_delta,
                    })

            # Handle text content. Once non-reasoning output starts, close the
            # current thinking block so the CLI can finalize its line.
            if delta.content:
                _close_thinking()

            # Handle tool calls
            if delta.tool_calls:
                _close_thinking()
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            "id": tc_delta.id or "",
                            "name": tc_delta.function.name if tc_delta.function and tc_delta.function.name else "",
                            "arguments_str": "",
                        }
                    else:
                        # Update ID if it arrives in a later chunk
                        if tc_delta.id:
                            tool_calls_map[idx]["id"] = tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            tool_calls_map[idx]["name"] = tc_delta.function.name

                    if tc_delta.function and tc_delta.function.arguments:
                        tool_calls_map[idx]["arguments_str"] += tc_delta.function.arguments

        # Parse tool call arguments
        tool_calls = []
        for idx in sorted(tool_calls_map.keys()):
            tc = tool_calls_map[idx]
            try:
                args = json.loads(tc["arguments_str"])
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append({
                "id": tc["id"],
                "name": tc["name"],
                "arguments": args,
            })

        _close_thinking()

        text = "".join(text_chunks) if text_chunks else None
        reasoning = "".join(reasoning_chunks) if reasoning_chunks else None

        return text, reasoning, tool_calls, usage

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
