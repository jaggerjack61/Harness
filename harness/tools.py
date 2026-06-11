"""Tool definitions and executors for the Nasa Level Genius Agent."""

import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Tool definitions (OpenAI function-calling schema) ──────────────────────

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read the contents of a file. Supports reading portions "
            "of large files with offset and limit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read (relative or absolute).",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (1-indexed).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Create or overwrite a file with the given content. "
            "Automatically creates parent directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write (relative or absolute).",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Make precise, targeted edits to a file. Each edit "
            "specifies oldText (exact text to find) and newText (replacement). "
            "Multiple edits can be applied in one call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit (relative or absolute).",
                    },
                    "edits": {
                        "type": "array",
                        "description": "List of edits to apply. Each edit has oldText "
                        "(exact text to replace) and newText (replacement text).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "oldText": {
                                    "type": "string",
                                    "description": "Exact text to find and replace.",
                                },
                                "newText": {
                                    "type": "string",
                                    "description": "Replacement text.",
                                },
                            },
                            "required": ["oldText", "newText"],
                        },
                    },
                },
                "required": ["path", "edits"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command and return its output "
            "(stdout and stderr combined).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                },
                "required": ["command"],
            },
        },
    },
]

# ── Tool implementations ──────────────────────────────────────────────────


def read_file(path: str, offset: Optional[int] = None, limit: Optional[int] = None) -> str:
    """Read the contents of a file.

    Args:
        path: Path to the file.
        offset: 1-indexed line number to start from.
        limit: Maximum number of lines to return.

    Returns:
        File contents as a string, or an error message.
    """
    p = Path(path)
    if not p.exists():
        return f"Error: File not found: {path}"

    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"

    if offset is not None or limit is not None:
        lines = text.splitlines()
        start = (offset - 1) if offset else 0
        end = (start + limit) if limit else None
        text = "\n".join(lines[start:end])

    return text


def write_file(path: str, content: str) -> str:
    """Create or overwrite a file.

    Args:
        path: Path to the file.
        content: Content to write.

    Returns:
        Success or error message.
    """
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Successfully wrote to {path}."
    except Exception as e:
        return f"Error writing file: {e}"


def edit_file(path: str, edits: List[Dict[str, str]]) -> str:
    """Apply one or more precise text replacements to a file.

    Args:
        path: Path to the file.
        edits: List of {"oldText": "...", "newText": "..."} dicts.

    Returns:
        Success or error message.
    """
    p = Path(path)
    if not p.exists():
        return f"Error: File not found: {path}"

    try:
        text = p.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"

    for i, edit in enumerate(edits):
        old = edit["oldText"]
        new = edit["newText"]
        if old not in text:
            return f"Error: Edit {i}: oldText not found in file."
        text = text.replace(old, new, 1)  # Replace first occurrence only

    try:
        p.write_text(text, encoding="utf-8")
        return f"Successfully applied {len(edits)} edit(s) to {path}."
    except Exception as e:
        return f"Error writing file: {e}"


def run_bash(command: str, cwd: Optional[str] = None) -> str:
    """Execute a shell command.

    Args:
        command: The command to run.
        cwd: Working directory for the command.

    Returns:
        Combined stdout and stderr output, or an error message.
    """
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                cwd=cwd,
                timeout=60,
            )
        else:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                cwd=cwd,
                timeout=60,
            )
        output = result.stdout
        if result.stderr:
            output += result.stderr
        if result.returncode != 0:
            output += f"\n[Exit code: {result.returncode}]"
        return output.strip() if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 60 seconds."
    except Exception as e:
        return f"Error executing command: {e}"


# ── Tool registry ─────────────────────────────────────────────────────────


class ToolRegistry:
    """Registry that maps tool names to their implementations."""

    def __init__(self, working_dir: Optional[str] = None):
        self.working_dir = working_dir
        self._handlers = {
            "read": self._handle_read,
            "write": self._handle_write,
            "edit": self._handle_edit,
            "bash": self._handle_bash,
        }

    def get_definitions(self) -> List[Dict[str, Any]]:
        """Return the OpenAI tool definitions."""
        return TOOL_DEFINITIONS

    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool by name with the given arguments.

        Args:
            name: Tool name.
            arguments: Keyword arguments for the tool.

        Returns:
            Result string from the tool.

        Raises:
            ValueError: If the tool name is unknown.
        """
        handler = self._handlers.get(name)
        if handler is None:
            raise ValueError(f"Unknown tool: {name}")
        return handler(arguments)

    def _handle_read(self, args: Dict[str, Any]) -> str:
        return read_file(
            path=args["path"],
            offset=args.get("offset"),
            limit=args.get("limit"),
        )

    def _handle_write(self, args: Dict[str, Any]) -> str:
        return write_file(path=args["path"], content=args["content"])

    def _handle_edit(self, args: Dict[str, Any]) -> str:
        return edit_file(path=args["path"], edits=args["edits"])

    def _handle_bash(self, args: Dict[str, Any]) -> str:
        return run_bash(command=args["command"], cwd=self.working_dir)
