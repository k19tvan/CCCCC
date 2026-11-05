"""Unit tests for the read_file tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_system.core.contracts import ReadFileInput
from agent_system.subagents.issue_agent.tools.read_file import read_file


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a repo fixture with sample files."""
    py_content = "\n".join([f"# Line {i}" for i in range(1, 101)])
    (tmp_path / "main.py").write_text(py_content, encoding="utf-8")

    (tmp_path / "small.py").write_text("def foo():\n    return 42\n", encoding="utf-8")

    (tmp_path / "README.md").write_text("# README\nContent", encoding="utf-8")

    # Only pure latin-1 characters
    (tmp_path / "latin.txt").write_bytes(
        "Content with latin-1 chars: cafe, naive, resume".encode("latin-1")
    )

    # Denylisted files
    (tmp_path / ".env").write_text("SECRET=my_secret_value\n", encoding="utf-8")
    (tmp_path / "key.pem").write_text("-----BEGIN RSA PRIVATE KEY-----\n", encoding="utf-8")

    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "module.py").write_text("x = 1\n", encoding="utf-8")

    return tmp_path


class TestReadFileHappyPath:
    def test_read_simple_file(self, sample_repo: Path) -> None:
        result = read_file(ReadFileInput(
            repo_path=str(sample_repo),
            file_path="small.py",
        ))

        assert result.success is True
        assert result.error is None
        assert "def foo" in result.content
        assert result.total_lines == 2
        assert result.file_path == "small.py"

    def test_read_returns_relative_path(self, sample_repo: Path) -> None:
        """file_path in the output is a relative path."""
        result = read_file(ReadFileInput(
            repo_path=str(sample_repo),
            file_path="small.py",
        ))

        assert result.success is True
        assert not result.file_path.startswith("/")
        assert "\\" not in result.file_path

    def test_read_subdirectory_file(self, sample_repo: Path) -> None:
        result = read_file(ReadFileInput(
            repo_path=str(sample_repo),
            file_path="subdir/module.py",
        ))

        assert result.success is True
        assert "x = 1" in result.content

    def test_read_file_line_range(self, sample_repo: Path) -> None:
        """Reads exactly the start_line..end_line range."""
        result = read_file(ReadFileInput(
            repo_path=str(sample_repo),
            file_path="main.py",
            start_line=5,
            end_line=10,
        ))

        assert result.success is True
        assert result.start_line == 5
        assert result.end_line == 10
        lines = result.content.strip().split("\n")
        assert len(lines) == 6  # lines 5 through 10

    def test_read_from_start_line_only(self, sample_repo: Path) -> None:
        """With only start_line set, reads to the end of the file."""
        result = read_file(ReadFileInput(
            repo_path=str(sample_repo),
            file_path="main.py",
            start_line=95,
        ))

        assert result.success is True
        assert result.start_line == 95
        assert result.end_line == 100

    def test_total_lines_accurate(self, sample_repo: Path) -> None:
        result = read_file(ReadFileInput(
            repo_path=str(sample_repo),
            file_path="main.py",
        ))

        assert result.success is True
        assert result.total_lines == 100


class TestReadFileMaxChars:
    def test_content_truncated_at_max_chars(self, sample_repo: Path) -> None:
        """Content longer than max_chars is cut and truncated=True."""
        result = read_file(ReadFileInput(
            repo_path=str(sample_repo),
            file_path="main.py",
            max_chars=50,
        ))

        assert result.success is True
        assert result.truncated is True
        assert len(result.content) <= 50

    def test_not_truncated_small_file(self, sample_repo: Path) -> None:
        """A file smaller than max_chars gives truncated=False."""
        result = read_file(ReadFileInput(
            repo_path=str(sample_repo),
            file_path="small.py",
            max_chars=50_000,
        ))

        assert result.success is True
        assert result.truncated is False


class TestReadFilePathTraversal:
    def test_path_traversal_dotdot(self, sample_repo: Path) -> None:
        """A path containing '..' is blocked."""
        result = read_file(ReadFileInput(
            repo_path=str(sample_repo),
            file_path="../secret.txt",
        ))

        assert result.success is False
        assert result.error is not None
        assert "traversal" in result.error.lower() or "blocked" in result.error.lower()

    def test_path_traversal_multiple_dots(self, sample_repo: Path) -> None:
        """A path with several '..' levels is blocked."""
        result = read_file(ReadFileInput(
            repo_path=str(sample_repo),
            file_path="../../etc/passwd",
        ))

        assert result.success is False
        assert result.error is not None

    def test_absolute_path_outside_repo(self, sample_repo: Path) -> None:
        """An absolute path outside the repo is blocked."""
        result = read_file(ReadFileInput(
            repo_path=str(sample_repo),
            file_path="/etc/hosts",
        ))

        assert result.success is False
        assert result.error is not None


class TestReadFileDenylist:
    def test_env_file_denied(self, sample_repo: Path) -> None:
        result = read_file(ReadFileInput(
            repo_path=str(sample_repo),
            file_path=".env",
        ))

        assert result.success is False
        assert result.error is not None
        assert "denylist" in result.error.lower()

    def test_pem_file_denied(self, sample_repo: Path) -> None:
        result = read_file(ReadFileInput(
            repo_path=str(sample_repo),
            file_path="key.pem",
        ))

        assert result.success is False
        assert result.error is not None

    def test_ssh_path_denied(self, tmp_path: Path) -> None:
        """A path containing '.ssh' is rejected."""
        ssh_dir = tmp_path / ".ssh"
        ssh_dir.mkdir()
        (ssh_dir / "id_rsa").write_text("PRIVATE KEY", encoding="utf-8")

        result = read_file(ReadFileInput(
            repo_path=str(tmp_path),
            file_path=".ssh/id_rsa",
        ))

        assert result.success is False
        assert result.error is not None


class TestReadFileErrors:
    def test_file_not_found(self, sample_repo: Path) -> None:
        result = read_file(ReadFileInput(
            repo_path=str(sample_repo),
            file_path="does_not_exist.py",
        ))

        assert result.success is False
        assert result.error is not None
        assert "exist" in result.error.lower()

    def test_repo_path_not_exists(self) -> None:
        result = read_file(ReadFileInput(
            repo_path="/does/not/exist",
            file_path="main.py",
        ))

        assert result.success is False
        assert result.error is not None

    def test_start_line_beyond_file(self, sample_repo: Path) -> None:
        """start_line beyond the total number of lines gives success=False."""
        result = read_file(ReadFileInput(
            repo_path=str(sample_repo),
            file_path="small.py",  # 2 lines
            start_line=999,
        ))

        assert result.success is False
        assert result.error is not None

    def test_read_directory_instead_of_file(self, sample_repo: Path) -> None:
        result = read_file(ReadFileInput(
            repo_path=str(sample_repo),
            file_path="subdir",
        ))

        assert result.success is False
        assert result.error is not None


class TestReadFileEncoding:
    def test_utf8_file(self, sample_repo: Path) -> None:
        result = read_file(ReadFileInput(
            repo_path=str(sample_repo),
            file_path="README.md",
        ))

        assert result.success is True
        assert "README" in result.content

    def test_latin1_file_fallback(self, sample_repo: Path) -> None:
        """A latin-1 file is read successfully through the encoding fallback."""
        result = read_file(ReadFileInput(
            repo_path=str(sample_repo),
            file_path="latin.txt",
        ))

        assert result.success is True
        assert result.content
