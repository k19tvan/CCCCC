"""Unit tests for the search_code tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_system.core.contracts import SearchCodeInput
from agent_system.subagents.issue_agent.tools.search_code import search_code


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a repo fixture with sample code."""
    (tmp_path / "calculator.py").write_text(
        """\
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def calculate_total(items):
    if not items:
        return 0
    return sum(items)

class Calculator:
    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return a / b
""",
        encoding="utf-8",
    )

    (tmp_path / "utils.py").write_text(
        """\
# utility functions
SECRET_KEY = "not_real_secret"
DEBUG = False

def format_string(text):
    # Process the text
    return text.strip()

class UserManager:
    pass
""",
        encoding="utf-8",
    )

    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "service.py").write_text(
        """\
def get_user(user_id):
    return None

def calculate_total_price(items):
    return sum(item['price'] for item in items)
""",
        encoding="utf-8",
    )

    # Binary file that must be skipped
    (tmp_path / "data.pkl").write_bytes(b"\x80\x03}q\x00.")

    # Directory that must be skipped
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    (pycache / "cached.py").write_text("def add(a,b): pass", encoding="utf-8")

    return tmp_path


class TestSearchCodeLiteral:
    def test_search_simple_function_name(self, sample_repo: Path) -> None:
        result = search_code(SearchCodeInput(
            repo_path=str(sample_repo),
            query="calculate_total",
            max_results=50,
        ))

        assert result.success is True
        assert result.total_matches >= 2  # calculator.py and service.py
        for match in result.matches:
            assert match.file_path
            assert match.line_number >= 1
            assert "calculate_total" in match.line_content

    def test_search_class_name(self, sample_repo: Path) -> None:
        result = search_code(SearchCodeInput(
            repo_path=str(sample_repo),
            query="class Calculator",
            max_results=50,
        ))

        assert result.success is True
        assert result.total_matches >= 1
        assert any("calculator.py" in m.file_path for m in result.matches)

    def test_search_string_content(self, sample_repo: Path) -> None:
        result = search_code(SearchCodeInput(
            repo_path=str(sample_repo),
            query="Cannot divide by zero",
            max_results=50,
        ))

        assert result.success is True
        assert result.total_matches >= 1

    def test_search_returns_line_numbers(self, sample_repo: Path) -> None:
        """Results carry 1-indexed line numbers."""
        result = search_code(SearchCodeInput(
            repo_path=str(sample_repo),
            query="def add",
            max_results=50,
        ))

        assert result.success is True
        assert len(result.matches) > 0
        for match in result.matches:
            assert match.line_number >= 1
            assert "def add" in match.line_content

    def test_search_returns_relative_paths(self, sample_repo: Path) -> None:
        result = search_code(SearchCodeInput(
            repo_path=str(sample_repo),
            query="def",
            max_results=50,
        ))

        assert result.success is True
        for match in result.matches:
            assert not match.file_path.startswith("/")
            assert "\\" not in match.file_path

    def test_search_in_subdirectory(self, sample_repo: Path) -> None:
        result = search_code(SearchCodeInput(
            repo_path=str(sample_repo),
            query="get_user",
            max_results=50,
        ))

        assert result.success is True
        assert result.total_matches >= 1
        assert any("subdir" in m.file_path for m in result.matches)


class TestSearchCodeCaseSensitivity:
    def test_case_insensitive_by_default(self, sample_repo: Path) -> None:
        result_lower = search_code(SearchCodeInput(
            repo_path=str(sample_repo),
            query="calculator",
            case_sensitive=False,
            max_results=50,
        ))
        result_upper = search_code(SearchCodeInput(
            repo_path=str(sample_repo),
            query="CALCULATOR",
            case_sensitive=False,
            max_results=50,
        ))

        assert result_lower.success is True
        assert result_upper.success is True
        assert result_lower.total_matches == result_upper.total_matches

    def test_case_sensitive_search(self, sample_repo: Path) -> None:
        result_exact = search_code(SearchCodeInput(
            repo_path=str(sample_repo),
            query="Calculator",  # PascalCase
            case_sensitive=True,
            max_results=50,
        ))
        result_lower = search_code(SearchCodeInput(
            repo_path=str(sample_repo),
            query="calculator",  # lowercase
            case_sensitive=True,
            max_results=50,
        ))

        assert result_exact.success is True
        assert result_lower.success is True
        assert result_exact.total_matches != result_lower.total_matches


class TestSearchCodeRegex:
    def test_regex_pattern(self, sample_repo: Path) -> None:
        result = search_code(SearchCodeInput(
            repo_path=str(sample_repo),
            query=r"def \w+\(",  # matches all function definitions
            is_regex=True,
            max_results=50,
        ))

        assert result.success is True
        assert result.total_matches >= 5

    def test_invalid_regex(self, sample_repo: Path) -> None:
        """An invalid regex gives success=False."""
        result = search_code(SearchCodeInput(
            repo_path=str(sample_repo),
            query="[invalid regex(",
            is_regex=True,
            max_results=50,
        ))

        assert result.success is False
        assert result.error is not None

    def test_regex_with_groups(self, sample_repo: Path) -> None:
        result = search_code(SearchCodeInput(
            repo_path=str(sample_repo),
            query=r"raise \w+Error",
            is_regex=True,
            max_results=50,
        ))

        assert result.success is True
        assert result.total_matches >= 1


class TestSearchCodeExtensionFilter:
    def test_filter_by_extension(self, sample_repo: Path) -> None:
        """A .py filter does not search other file types."""
        result = search_code(SearchCodeInput(
            repo_path=str(sample_repo),
            query="SECRET",
            file_extensions=[".py"],
            max_results=50,
        ))

        assert result.success is True
        for match in result.matches:
            assert match.file_path.endswith(".py")

    def test_no_extension_filter(self, sample_repo: Path) -> None:
        """Without an extension filter, all text files are searched."""
        result = search_code(SearchCodeInput(
            repo_path=str(sample_repo),
            query="def",
            max_results=50,
        ))

        assert result.success is True
        paths = {m.file_path for m in result.matches}
        assert len(paths) >= 2


class TestSearchCodeContextLines:
    def test_context_lines_before_after(self, sample_repo: Path) -> None:
        result = search_code(SearchCodeInput(
            repo_path=str(sample_repo),
            query="ZeroDivisionError",
            context_lines=2,
            max_results=50,
        ))

        assert result.success is True
        assert len(result.matches) > 0
        for match in result.matches:
            assert len(match.context_before) <= 2
            assert len(match.context_after) <= 2

    def test_no_context_lines(self, sample_repo: Path) -> None:
        """context_lines=0 gives empty context lists."""
        result = search_code(SearchCodeInput(
            repo_path=str(sample_repo),
            query="def add",
            context_lines=0,
            max_results=50,
        ))

        assert result.success is True
        for match in result.matches:
            assert match.context_before == []
            assert match.context_after == []


class TestSearchCodeMaxResults:
    def test_max_results_limit(self, sample_repo: Path) -> None:
        result = search_code(SearchCodeInput(
            repo_path=str(sample_repo),
            query="def",
            max_results=2,
        ))

        assert result.success is True
        assert len(result.matches) <= 2

    def test_truncated_flag(self, sample_repo: Path) -> None:
        """truncated is True when there are more matches than max_results."""
        result_small = search_code(SearchCodeInput(
            repo_path=str(sample_repo),
            query="def",
            max_results=1,
        ))
        result_all = search_code(SearchCodeInput(
            repo_path=str(sample_repo),
            query="def",
            max_results=200,
        ))

        if result_all.total_matches > 1:
            assert result_small.truncated is True
            assert result_small.total_matches == result_all.total_matches

    def test_query_returned_in_output(self, sample_repo: Path) -> None:
        result = search_code(SearchCodeInput(
            repo_path=str(sample_repo),
            query="calculate_total",
            max_results=50,
        ))

        assert result.success is True
        assert result.query == "calculate_total"


class TestSearchCodeIgnoredDirs:
    def test_pycache_not_searched(self, sample_repo: Path) -> None:
        result = search_code(SearchCodeInput(
            repo_path=str(sample_repo),
            query="def add",  # also present in __pycache__/cached.py
            max_results=50,
        ))

        assert result.success is True
        for match in result.matches:
            assert "__pycache__" not in match.file_path

    def test_binary_files_not_searched(self, sample_repo: Path) -> None:
        """Binary files (.pkl, ...) are not searched."""
        result = search_code(SearchCodeInput(
            repo_path=str(sample_repo),
            query=".",  # matches everything (not a regex)
            max_results=200,
        ))

        assert result.success is True
        for match in result.matches:
            assert not match.file_path.endswith(".pkl")


class TestSearchCodeErrors:
    def test_no_match_found(self, sample_repo: Path) -> None:
        """No match gives matches=[] and success=True."""
        result = search_code(SearchCodeInput(
            repo_path=str(sample_repo),
            query="xyz_totally_nonexistent_12345",
            max_results=50,
        ))

        assert result.success is True
        assert result.total_matches == 0
        assert result.matches == []

    def test_repo_path_not_exists(self) -> None:
        result = search_code(SearchCodeInput(
            repo_path="/totally/fake/path",
            query="def",
            max_results=50,
        ))

        assert result.success is False
        assert result.error is not None
