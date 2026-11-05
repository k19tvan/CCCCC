"""Search for strings/symbols in a repository.

- Literal or regex search, optionally filtered by file extension and case sensitivity.
- Returns path, line number, line content and surrounding context lines.
- Limits the number of results.
- Skips cache/virtualenv directories and binary files (same rules as list_files).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from agent_system.core.contracts import SearchCodeInput, SearchCodeOutput, SearchMatch

logger = logging.getLogger(__name__)

_IGNORED_DIRS: frozenset[str] = frozenset({
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".env",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "dist",
    "build",
    ".eggs",
    "htmlcov",
})

_BINARY_EXTENSIONS: frozenset[str] = frozenset({
    ".pyc", ".pyo", ".pyd",
    ".so", ".dll", ".dylib", ".exe", ".bin",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".avi", ".mov", ".wav",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".db", ".sqlite", ".sqlite3",
    ".pkl", ".pickle",
    ".whl",
})

# Maximum file size to search (5 MB)
_MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024


def _is_ignored_dir(dir_name: str) -> bool:
    return dir_name in _IGNORED_DIRS or dir_name.endswith(".egg-info")


def _resolve_and_check(repo_root: Path, target: Path) -> Path | None:
    try:
        resolved = target.resolve()
        resolved_root = repo_root.resolve()
        resolved.relative_to(resolved_root)
        return resolved
    except (ValueError, OSError):
        return None


def _read_lines_safe(file_path: Path) -> list[str] | None:
    """Read a file's lines trying several encodings; return None if it cannot be read."""
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            with file_path.open(encoding=enc, errors="strict") as f:
                return f.readlines()
        except (UnicodeDecodeError, OSError):
            continue
    try:
        with file_path.open(encoding="utf-8", errors="replace") as f:
            return f.readlines()
    except OSError:
        return None


def search_code(tool_input: SearchCodeInput) -> SearchCodeOutput:
    """Search the repo for a literal string or regex and return matches with context lines."""
    repo_root = Path(tool_input.repo_path)

    if not repo_root.exists():
        return SearchCodeOutput(
            success=False,
            error=f"repo_path does not exist: {tool_input.repo_path}",
            query=tool_input.query,
        )

    try:
        if tool_input.is_regex:
            flags = 0 if tool_input.case_sensitive else re.IGNORECASE
            pattern = re.compile(tool_input.query, flags)
        else:
            escaped_query = re.escape(tool_input.query)
            flags = 0 if tool_input.case_sensitive else re.IGNORECASE
            pattern = re.compile(escaped_query, flags)
    except re.error as exc:
        return SearchCodeOutput(
            success=False,
            error=f"Invalid regex pattern: {exc}",
            query=tool_input.query,
        )

    allowed_exts: set[str] | None = None
    if tool_input.file_extensions:
        allowed_exts = {
            ext if ext.startswith(".") else f".{ext}"
            for ext in tool_input.file_extensions
        }

    matches: list[SearchMatch] = []
    total_matches = 0
    resolved_root = repo_root.resolve()

    logger.debug(
        "search_code: query='%s', is_regex=%s, case_sensitive=%s, exts=%s, max=%d",
        tool_input.query,
        tool_input.is_regex,
        tool_input.case_sensitive,
        allowed_exts,
        tool_input.max_results,
    )

    for dirpath, dirnames, filenames in os.walk(resolved_root, followlinks=False):
        # Mutate dirnames in place so os.walk does not descend into ignored directories
        dirnames[:] = [d for d in dirnames if not _is_ignored_dir(d)]

        for filename in sorted(filenames):
            file_path = Path(dirpath) / filename

            if _resolve_and_check(repo_root, file_path) is None:
                continue

            suffix = file_path.suffix.lower()
            if suffix in _BINARY_EXTENSIONS:
                continue

            if allowed_exts is not None and suffix not in allowed_exts:
                continue

            try:
                if file_path.stat().st_size > _MAX_FILE_SIZE_BYTES:
                    logger.debug("search_code: skipping oversized file: %s", file_path)
                    continue
            except OSError:
                continue

            file_lines = _read_lines_safe(file_path)
            if file_lines is None:
                continue

            for line_idx, line_content in enumerate(file_lines):
                line_number = line_idx + 1
                line_stripped = line_content.rstrip("\n\r")

                if not pattern.search(line_stripped):
                    continue

                total_matches += 1

                if len(matches) >= tool_input.max_results:
                    continue  # Keep counting total_matches but do not add more matches

                try:
                    rel_path = str(file_path.relative_to(resolved_root)).replace("\\", "/")
                except ValueError:
                    rel_path = str(file_path)

                ctx = tool_input.context_lines
                context_before = [
                    file_lines[i].rstrip("\n\r")
                    for i in range(max(0, line_idx - ctx), line_idx)
                ]
                context_after = [
                    file_lines[i].rstrip("\n\r")
                    for i in range(line_idx + 1, min(len(file_lines), line_idx + 1 + ctx))
                ]

                matches.append(
                    SearchMatch(
                        file_path=rel_path,
                        line_number=line_number,
                        line_content=line_stripped,
                        context_before=context_before,
                        context_after=context_after,
                    )
                )

    truncated = total_matches > tool_input.max_results
    logger.info(
        "search_code: query='%s' -> %d match(es), returning %d (truncated=%s)",
        tool_input.query,
        total_matches,
        len(matches),
        truncated,
    )

    return SearchCodeOutput(
        success=True,
        matches=matches,
        total_matches=total_matches,
        truncated=truncated,
        query=tool_input.query,
    )
