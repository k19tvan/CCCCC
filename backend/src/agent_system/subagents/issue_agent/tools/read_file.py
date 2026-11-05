"""Read file contents from a repository.

- Only reads files inside repo_path (blocks path traversal and symlink escapes).
- Limits returned characters (max_chars) and supports line ranges (start_line, end_line).
- Tries several encodings in turn.
- Refuses files on the denylist (.env, *.pem, .ssh, ...).
"""

from __future__ import annotations

import logging
from pathlib import Path

from agent_system.core.contracts import ReadFileInput, ReadFileOutput

logger = logging.getLogger(__name__)

_DENIED_FILENAMES: frozenset[str] = frozenset({
    ".env",
    ".env.local",
    ".env.production",
    ".env.staging",
    ".env.development",
    ".gitconfig",
    ".gitcredentials",
})

_DENIED_EXTENSIONS: frozenset[str] = frozenset({
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".cer",
    ".crt",
})

# Matched as substrings of the resolved path
_DENIED_PATH_PATTERNS: list[str] = [
    ".ssh",
    ".git/config",
    ".git/COMMIT_EDITMSG",
]

_ENCODINGS_TO_TRY: list[str] = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]


def _is_denied(resolved_path: Path) -> tuple[bool, str]:
    """Return (True, reason) if the file is on the denylist, otherwise (False, "")."""
    name = resolved_path.name
    suffix = resolved_path.suffix.lower()
    path_str = str(resolved_path).replace("\\", "/")

    if name in _DENIED_FILENAMES:
        return True, f"File '{name}' is on the security denylist."

    if suffix in _DENIED_EXTENSIONS:
        return True, f"Extension '{suffix}' is on the security denylist."

    for pattern in _DENIED_PATH_PATTERNS:
        if pattern in path_str:
            return True, f"Path contains a forbidden pattern: '{pattern}'."

    return False, ""


def _resolve_and_check(repo_root: Path, target: Path) -> Path | None:
    """Return the resolved path if it is inside repo_root, otherwise None."""
    try:
        resolved = target.resolve()
        resolved_root = repo_root.resolve()
        resolved.relative_to(resolved_root)
        return resolved
    except (ValueError, OSError):
        return None


def read_file(tool_input: ReadFileInput) -> ReadFileOutput:
    """Read one file in the repo, optionally a line range, truncated to max_chars."""
    repo_root = Path(tool_input.repo_path)

    if not repo_root.exists():
        return ReadFileOutput(
            success=False,
            error=f"repo_path does not exist: {tool_input.repo_path}",
        )

    file_path = repo_root / tool_input.file_path

    resolved_file = _resolve_and_check(repo_root, file_path)
    if resolved_file is None:
        logger.warning(
            "read_file: path traversal or symlink escape blocked: '%s'",
            tool_input.file_path,
        )
        return ReadFileOutput(
            success=False,
            error=(
                f"Path traversal or symlink escape blocked: '{tool_input.file_path}'. "
                "Only files inside the repo may be read."
            ),
        )

    denied, deny_reason = _is_denied(resolved_file)
    if denied:
        logger.warning("read_file: file rejected by denylist: %s", deny_reason)
        return ReadFileOutput(
            success=False,
            error=deny_reason,
        )

    if not resolved_file.exists():
        return ReadFileOutput(
            success=False,
            error=f"File does not exist: '{tool_input.file_path}'",
        )

    if not resolved_file.is_file():
        return ReadFileOutput(
            success=False,
            error=f"Path is not a file: '{tool_input.file_path}'",
        )

    lines: list[str] = []
    used_encoding = "utf-8"
    for enc in _ENCODINGS_TO_TRY:
        try:
            with resolved_file.open(encoding=enc, errors="strict") as f:
                lines = f.readlines()
            used_encoding = enc
            break
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            return ReadFileOutput(
                success=False,
                error=f"Error reading file: {exc}",
            )
    else:
        try:
            with resolved_file.open(encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            used_encoding = "utf-8 (with replacement)"
        except OSError as exc:
            return ReadFileOutput(
                success=False,
                error=f"Cannot read file: {exc}",
            )

    total_lines = len(lines)

    start = max(1, tool_input.start_line) if tool_input.start_line else 1
    end = (
        min(tool_input.end_line, total_lines)
        if tool_input.end_line
        else total_lines
    )

    if start > total_lines:
        return ReadFileOutput(
            success=False,
            error=(
                f"start_line={start} exceeds the total number of lines ({total_lines}) "
                f"in file '{tool_input.file_path}'."
            ),
        )

    selected_lines = lines[start - 1 : end]

    content = "".join(selected_lines)
    truncated = False

    if len(content) > tool_input.max_chars:
        content = content[: tool_input.max_chars]
        truncated = True
        logger.info(
            "read_file: '%s' truncated at %d characters (max_chars=%d)",
            tool_input.file_path,
            tool_input.max_chars,
            tool_input.max_chars,
        )

    actual_end = end
    if truncated:
        newline_count = content.count("\n")
        actual_end = min(start + newline_count, end)

    try:
        rel_path = str(resolved_file.relative_to(repo_root.resolve())).replace("\\", "/")
    except ValueError:
        rel_path = tool_input.file_path

    logger.info(
        "read_file: read '%s' (lines %d-%d / %d, %d chars, encoding=%s, truncated=%s)",
        rel_path,
        start,
        actual_end,
        total_lines,
        len(content),
        used_encoding,
        truncated,
    )

    return ReadFileOutput(
        success=True,
        content=content,
        file_path=rel_path,
        total_lines=total_lines,
        start_line=start,
        end_line=actual_end,
        truncated=truncated,
        encoding=used_encoding,
    )
