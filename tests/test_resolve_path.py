"""Tests for tools.py DRY extractions (path resolution)."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from harness.tools import _resolve_path


class TestResolvePath:
    """_resolve_path should DRY the path resolution logic used by read/write/edit."""

    def test_absolute_path_unchanged(self):
        result = _resolve_path("/etc/hosts", cwd="/some/dir")
        assert result.is_absolute()

    def test_relative_path_resolved_against_cwd(self):
        result = _resolve_path("file.txt", cwd="/some/dir")
        assert result == (Path("/some/dir") / "file.txt").resolve()

    def test_no_cwd_returns_resolved_path(self):
        result = _resolve_path("file.txt", cwd=None)
        assert isinstance(result, Path)

    def test_calls_resolve(self):
        """resolve() should be called to follow symlinks (consistent with previous behavior)."""
        with patch("harness.tools.Path") as MockPath:
            mock_path = MagicMock()
            mock_path.is_absolute.return_value = True
            MockPath.return_value = mock_path
            mock_resolved = MagicMock()
            mock_path.resolve.return_value = mock_resolved
            result = _resolve_path("/etc/hosts", cwd="/some/dir")
            mock_path.resolve.assert_called_once()
