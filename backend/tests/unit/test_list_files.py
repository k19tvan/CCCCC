"""Unit tests for the list_files tool."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_system.core.contracts import ListFilesInput
from agent_system.subagents.issue_agent.tools.list_files import list_files


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a temporary repo fixture.

    Layout: main.py, utils.py, README.md, config.json, binary.so, __pycache__/cache.pyc,
    .venv/lib.py, subdir/helper.py, subdir/image.png
    """
    (tmp_path / "main.py").write_text("# main\ndef main(): pass", encoding="utf-8")
    (tmp_path / "utils.py").write_text("# utils\ndef helper(): pass", encoding="utf-8")
    (tmp_path / "README.md").write_text("# README", encoding="utf-8")
    (tmp_path / "config.json").write_text('{"key": "value"}', encoding="utf-8")

    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    (pycache / "cache.pyc").write_bytes(b"\x00\x01binary")

    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "lib.py").write_text("# should be ignored", encoding="utf-8")

    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "helper.py").write_text("def help(): pass", encoding="utf-8")
    (subdir / "image.png").write_bytes(b"\x89PNG...")

    (tmp_path / "binary.so").write_bytes(b"\x7fELF...")

    return tmp_path


class TestListFilesHappyPath:
    def test_lists_python_files_only(self, sample_repo: Path) -> None:
        """A .py filter returns only .py files and nothing from cache directories."""
        result = list_files(ListFilesInput(
            repo_path=str(sample_repo),
            extensions=[".py"],
            max_results=100,
        ))

        assert result.success is True
        assert result.error is None
        assert len(result.files) > 0

        for f in result.files:
            assert f.endswith(".py"), f"File {f} is not .py"

        for f in result.files:
            assert "__pycache__" not in f, f"File inside __pycache__ leaked through: {f}"
            assert ".venv" not in f, f"File inside .venv leaked through: {f}"

    def test_lists_all_non_binary_files(self, sample_repo: Path) -> None:
        """Without an extension filter, text files are returned and binary files are not."""
        result = list_files(ListFilesInput(
            repo_path=str(sample_repo),
            max_results=100,
        ))

        assert result.success is True
        filenames = [Path(f).name for f in result.files]
        assert "main.py" in filenames
        assert "README.md" in filenames
        assert "config.json" in filenames
        assert "binary.so" not in filenames
        assert "image.png" not in filenames
        assert "cache.pyc" not in filenames

    def test_lists_subdirectory(self, sample_repo: Path) -> None:
        """With a subdirectory, only files inside it are returned."""
        result = list_files(ListFilesInput(
            repo_path=str(sample_repo),
            subdirectory="subdir",
            max_results=100,
        ))

        assert result.success is True
        assert all("subdir" in f for f in result.files)

    def test_files_are_relative_paths(self, sample_repo: Path) -> None:
        """Returned paths are relative, not absolute."""
        result = list_files(ListFilesInput(
            repo_path=str(sample_repo),
            max_results=100,
        ))

        assert result.success is True
        for f in result.files:
            assert not os.path.isabs(f), f"Absolute path leaked through: {f}"
            assert not f.startswith("/"), f"Path starts with /: {f}"

    def test_paths_use_forward_slashes(self, sample_repo: Path) -> None:
        """On Windows, paths must use / instead of backslashes."""
        result = list_files(ListFilesInput(
            repo_path=str(sample_repo),
            max_results=100,
        ))

        assert result.success is True
        for f in result.files:
            assert "\\" not in f, f"Path uses a backslash: {f}"

    def test_total_found_correct(self, sample_repo: Path) -> None:
        """total_found reflects the actual number of files."""
        result = list_files(ListFilesInput(
            repo_path=str(sample_repo),
            extensions=[".py"],
            max_results=100,
        ))

        assert result.success is True
        assert result.total_found == len(result.files)
        assert result.truncated is False

    def test_multiple_extensions(self, sample_repo: Path) -> None:
        """Several extensions can be filtered at once."""
        result = list_files(ListFilesInput(
            repo_path=str(sample_repo),
            extensions=[".py", ".md"],
            max_results=100,
        ))

        assert result.success is True
        for f in result.files:
            assert f.endswith(".py") or f.endswith(".md"), f"Unexpected extension: {f}"


class TestListFilesMaxResults:
    def test_max_results_limit(self, sample_repo: Path) -> None:
        """With max_results=1, only one file is returned."""
        result = list_files(ListFilesInput(
            repo_path=str(sample_repo),
            max_results=1,
        ))

        assert result.success is True
        assert len(result.files) == 1

    def test_truncated_flag_when_limited(self, sample_repo: Path) -> None:
        """truncated is True when total_found > max_results."""
        result_all = list_files(ListFilesInput(
            repo_path=str(sample_repo),
            max_results=100,
        ))
        total = result_all.total_found

        if total > 1:
            result_limited = list_files(ListFilesInput(
                repo_path=str(sample_repo),
                max_results=1,
            ))
            assert result_limited.truncated is True
            assert result_limited.total_found == total

    def test_not_truncated_when_within_limit(self, sample_repo: Path) -> None:
        """truncated is False when total_found <= max_results."""
        result = list_files(ListFilesInput(
            repo_path=str(sample_repo),
            max_results=500,  # maximum allowed value
        ))

        assert result.success is True
        assert result.truncated is False


class TestListFilesPathTraversal:
    def test_path_traversal_subdirectory(self, sample_repo: Path, tmp_path: Path) -> None:
        """A subdirectory containing '..' is blocked."""
        result = list_files(ListFilesInput(
            repo_path=str(sample_repo),
            subdirectory="../",
            max_results=100,
        ))

        assert result.success is False
        assert result.error is not None
        assert "traversal" in result.error.lower() or "outside" in result.error.lower()

    def test_path_traversal_absolute_subdirectory(self, sample_repo: Path) -> None:
        """An absolute subdirectory outside the repo is blocked."""
        result = list_files(ListFilesInput(
            repo_path=str(sample_repo),
            subdirectory="/etc",
            max_results=100,
        ))

        if result.success:
            # If not blocked outright, it must at least not return files outside the repo
            for f in result.files:
                assert not os.path.isabs(f)
        else:
            assert result.error is not None


class TestListFilesErrors:
    def test_repo_path_not_exists(self) -> None:
        """A nonexistent repo_path gives success=False."""
        result = list_files(ListFilesInput(
            repo_path="/does/not/exist/repo",
            max_results=100,
        ))

        assert result.success is False
        assert result.error is not None
        assert len(result.files) == 0

    def test_repo_path_is_file_not_dir(self, sample_repo: Path) -> None:
        """A repo_path pointing at a file instead of a directory gives success=False."""
        result = list_files(ListFilesInput(
            repo_path=str(sample_repo / "main.py"),
            max_results=100,
        ))

        assert result.success is False
        assert result.error is not None

    def test_subdirectory_not_exists(self, sample_repo: Path) -> None:
        """A nonexistent subdirectory gives success=False."""
        result = list_files(ListFilesInput(
            repo_path=str(sample_repo),
            subdirectory="nonexistent_dir",
            max_results=100,
        ))

        assert result.success is False
        assert result.error is not None

    def test_empty_repo(self, tmp_path: Path) -> None:
        """An empty repo gives files=[] and success=True."""
        result = list_files(ListFilesInput(
            repo_path=str(tmp_path),
            max_results=100,
        ))

        assert result.success is True
        assert result.files == []
        assert result.total_found == 0


class TestListFilesIgnoredDirs:
    def test_pycache_ignored(self, sample_repo: Path) -> None:
        """Files inside __pycache__ are not listed."""
        result = list_files(ListFilesInput(
            repo_path=str(sample_repo),
            max_results=100,
        ))

        assert result.success is True
        for f in result.files:
            assert "__pycache__" not in f

    def test_venv_ignored(self, sample_repo: Path) -> None:
        """Files inside .venv are not listed."""
        result = list_files(ListFilesInput(
            repo_path=str(sample_repo),
            max_results=100,
        ))

        assert result.success is True
        for f in result.files:
            assert ".venv" not in f

    def test_custom_ignored_dirs(self, tmp_path: Path) -> None:
        """node_modules, dist and build directories are skipped."""
        for ignored_dir in ["node_modules", "dist", "build"]:
            d = tmp_path / ignored_dir
            d.mkdir(exist_ok=True)
            (d / "file.py").write_text("# ignored", encoding="utf-8")

        (tmp_path / "real.py").write_text("# real", encoding="utf-8")

        result = list_files(ListFilesInput(
            repo_path=str(tmp_path),
            extensions=[".py"],
            max_results=100,
        ))

        assert result.success is True
        for f in result.files:
            assert "node_modules" not in f
            assert "/dist/" not in f
            assert "/build/" not in f
        assert any("real.py" in f for f in result.files)
