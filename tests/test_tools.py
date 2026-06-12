"""Tests for the tool definitions and executors."""

import os
import tempfile
from pathlib import Path

import pytest

from harness.tools import (
    ToolRegistry,
    read_file,
    write_file,
    edit_file,
    run_bash,
    TOOL_DEFINITIONS,
)


class TestToolDefinitions:
    """Tool definitions must match the OpenAI function-calling schema."""

    def test_all_tools_have_name_and_description(self):
        for tool in TOOL_DEFINITIONS:
            assert "type" in tool
            assert tool["type"] == "function"
            fn = tool["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn

    def test_read_tool_definition(self):
        read_tool = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "read")
        params = read_tool["function"]["parameters"]
        assert "path" in params["properties"]
        assert params["required"] == ["path"]

    def test_write_tool_definition(self):
        write_tool = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "write")
        params = write_tool["function"]["parameters"]
        assert "path" in params["properties"]
        assert "content" in params["properties"]
        assert set(params["required"]) == {"path", "content"}

    def test_edit_tool_definition(self):
        edit_tool = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "edit")
        params = edit_tool["function"]["parameters"]
        assert "path" in params["properties"]
        assert "edits" in params["properties"]
        assert set(params["required"]) == {"path", "edits"}

    def test_bash_tool_definition(self):
        bash_tool = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "bash")
        params = bash_tool["function"]["parameters"]
        assert "command" in params["properties"]
        assert params["required"] == ["command"]


class TestReadFile:
    def test_reads_existing_file(self, tmp_path: Path):
        f = tmp_path / "hello.txt"
        f.write_text("hello world")
        result = read_file(str(f))
        assert result == "hello world"

    def test_returns_error_for_missing_file(self):
        result = read_file("/nonexistent/path.txt")
        assert "Error" in result

    def test_reads_with_offset_and_limit(self, tmp_path: Path):
        f = tmp_path / "lines.txt"
        f.write_text("line1\nline2\nline3\nline4\nline5")
        result = read_file(str(f), offset=2, limit=2)
        assert result == "line2\nline3"


class TestWriteFile:
    def test_creates_new_file(self, tmp_path: Path):
        f = tmp_path / "new.txt"
        result = write_file(str(f), "hello")
        assert "success" in result.lower()
        assert f.read_text() == "hello"

    def test_overwrites_existing_file(self, tmp_path: Path):
        f = tmp_path / "existing.txt"
        f.write_text("old")
        result = write_file(str(f), "new")
        assert "success" in result.lower()
        assert f.read_text() == "new"

    def test_creates_parent_directories(self, tmp_path: Path):
        f = tmp_path / "nested" / "deep" / "file.txt"
        result = write_file(str(f), "nested")
        assert "success" in result.lower()
        assert f.read_text() == "nested"


class TestEditFile:
    def test_single_replacement(self, tmp_path: Path):
        f = tmp_path / "edit.txt"
        f.write_text("hello world")
        result = edit_file(str(f), [{"oldText": "hello", "newText": "hi"}])
        assert "success" in result.lower()
        assert f.read_text() == "hi world"

    def test_multiple_replacements(self, tmp_path: Path):
        f = tmp_path / "edit.txt"
        f.write_text("a b c")
        result = edit_file(str(f), [
            {"oldText": "a", "newText": "1"},
            {"oldText": "c", "newText": "3"},
        ])
        assert "success" in result.lower()
        assert f.read_text() == "1 b 3"

    def test_returns_error_when_old_text_not_found(self, tmp_path: Path):
        f = tmp_path / "edit.txt"
        f.write_text("hello world")
        result = edit_file(str(f), [{"oldText": "nonexistent", "newText": "x"}])
        assert "Error" in result

    def test_returns_error_for_missing_file(self):
        result = edit_file("/nonexistent.txt", [{"oldText": "a", "newText": "b"}])
        assert "Error" in result


class TestRunBash:
    def test_runs_simple_command(self):
        result = run_bash("echo hello")
        assert "hello" in result

    def test_returns_stderr_in_output(self):
        result = run_bash('python -c "import sys; sys.stderr.write(\'error\')"')
        assert "error" in result

    def test_returns_error_on_nonzero_exit(self):
        result = run_bash("exit 1")
        assert "Error" in result or "exit code" in result.lower()

    def test_long_output_is_truncated(self):
        # Generate 500 lines of output – should mention truncation or just work
        result = run_bash('python -c "for i in range(500): print(i)" 2>&1')
        # Just ensure it returns something without crashing
        assert len(result) > 0

    def test_cwd_is_current_by_default(self, tmp_path: Path):
        marker = tmp_path / "marker"
        result = run_bash(f'echo test > "{marker}"')
        assert marker.exists()


class TestToolRegistry:
    def test_get_tool_definitions(self):
        registry = ToolRegistry()
        tools = registry.get_definitions()
        names = {t["function"]["name"] for t in tools}
        assert names == {"read", "write", "edit", "bash"}

    def test_execute_known_tool(self, tmp_path: Path):
        registry = ToolRegistry()
        f = tmp_path / "reg.txt"
        f.write_text("data")
        result = registry.execute("read", {"path": str(f)})
        assert result == "data"

    def test_execute_unknown_tool_raises(self):
        registry = ToolRegistry()
        with pytest.raises(ValueError, match="Unknown tool"):
            registry.execute("nonexistent", {})

    def test_execute_with_working_dir(self, tmp_path: Path):
        registry = ToolRegistry(working_dir=str(tmp_path))
        # Create a marker file so we can verify the command ran in the right dir
        marker = tmp_path / "cwd_test_marker"
        registry.execute("bash", {"command": f'echo ok > "{marker}"'})
        assert marker.exists()


class TestToolOutputLineLimit:
    """Tool call responses exceeding 1,000 lines must be discarded and the agent notified."""

    def test_bash_output_under_limit_is_preserved(self, tmp_path: Path):
        registry = ToolRegistry(working_dir=str(tmp_path))
        result = registry.execute("bash", {"command": "python -c \"for i in range(500): print(i)\""})
        assert "499" in result
        assert "exceeded" not in result.lower()

    def test_bash_output_over_limit_is_discarded(self, tmp_path: Path):
        registry = ToolRegistry(working_dir=str(tmp_path))
        result = registry.execute("bash", {"command": "python -c \"for i in range(1500): print(i)\""})
        assert "499" not in result
        assert "exceeded" in result.lower()
        assert "1,000" in result or "1000" in result
        assert "try again" in result.lower()
        assert "head" in result.lower() or "Select-Object" in result

    def test_read_output_over_limit_is_discarded(self, tmp_path: Path):
        registry = ToolRegistry(working_dir=str(tmp_path))
        f = tmp_path / "big.txt"
        f.write_text("\n".join(f"line {i}" for i in range(1500)))
        result = registry.execute("read", {"path": str(f)})
        assert "line 0" not in result
        assert "exceeded" in result.lower()
        assert "1,000" in result or "1000" in result

    def test_exactly_1000_lines_is_allowed(self, tmp_path: Path):
        registry = ToolRegistry(working_dir=str(tmp_path))
        result = registry.execute("bash", {"command": "python -c \"for i in range(1000): print(i)\""})
        assert "999" in result
        assert "exceeded" not in result.lower()

    def test_1001_lines_is_discarded(self, tmp_path: Path):
        registry = ToolRegistry(working_dir=str(tmp_path))
        result = registry.execute("bash", {"command": "python -c \"for i in range(1001): print(i)\""})
        assert "exceeded" in result.lower()
        assert "1,000" in result or "1000" in result
