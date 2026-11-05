"""List files in a repository.

- Paths are relative to repo_path.
- Skips cache/virtualenv directories (.git, __pycache__, .venv, node_modules, ...) and binary files.
- Limits the number of results.
- Blocks path traversal and symlink escapes outside the repo.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from agent_system.core.contracts import ListFilesInput, ListFilesOutput

logger = logging.getLogger(__name__)

# Directories that are never descended into
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
    "*.egg-info",
    ".eggs",
    "htmlcov",
    ".coverage",
    ".DS_Store",
})

# Extensions of files that are not returned (binary, or ".lock" which is text but rarely useful)
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
    ".lock",
})


def _is_ignored_dir(dir_name: str) -> bool:
    return dir_name in _IGNORED_DIRS or dir_name.endswith(".egg-info")


def _resolve_and_check(repo_root: Path, target: Path) -> Path | None:
    """Return the resolved path if it is inside repo_root, otherwise None."""
    try:
        resolved = target.resolve()
        resolved_root = repo_root.resolve()
        resolved.relative_to(resolved_root)
        return resolved
    except (ValueError, OSError):
        return None


def list_files(tool_input: ListFilesInput) -> ListFilesOutput:
    """List files in the repo (or a subdirectory) as paths relative to repo_path."""
    repo_root = Path(tool_input.repo_path)

    if not repo_root.exists():
        return ListFilesOutput(
            success=False,
            error=f"repo_path does not exist: {tool_input.repo_path}",
            files=[],
            total_found=0,
            truncated=False,
        )
    if not repo_root.is_dir():
        return ListFilesOutput(
            success=False,
            error=f"repo_path is not a directory: {tool_input.repo_path}",
            files=[],
            total_found=0,
            truncated=False,
        )

    if tool_input.subdirectory:
        subdir_path = repo_root / tool_input.subdirectory
        resolved_subdir = _resolve_and_check(repo_root, subdir_path)
        if resolved_subdir is None:
            return ListFilesOutput(
                success=False,
                error=(
                    f"subdirectory '{tool_input.subdirectory}' is outside the repo or "
                    "contains path traversal."
                ),
                files=[],
                total_found=0,
                truncated=False,
            )
        if not resolved_subdir.exists():
            return ListFilesOutput(
                success=False,
                error=f"subdirectory does not exist: {tool_input.subdirectory}",
                files=[],
                total_found=0,
                truncated=False,
            )
        start_dir = resolved_subdir
    else:
        start_dir = repo_root.resolve()

    allowed_exts: set[str] | None = None
    if tool_input.extensions:
        allowed_exts = {
            ext if ext.startswith(".") else f".{ext}"
            for ext in tool_input.extensions
        }

    collected: list[str] = []
    total_found = 0

    logger.debug(
        "list_files: walking '%s', extensions=%s, max_results=%d",
        start_dir,
        allowed_exts,
        tool_input.max_results,
    )

    for dirpath, dirnames, filenames in os.walk(start_dir, followlinks=False):
        # Mutate dirnames in place so os.walk does not descend into ignored directories
        dirnames[:] = [d for d in dirnames if not _is_ignored_dir(d)]

        current_dir = Path(dirpath)
        if _resolve_and_check(repo_root, current_dir) is None:
            logger.warning(
                "list_files: skipping directory outside the repo: %s", dirpath
            )
            dirnames.clear()
            continue

        for filename in sorted(filenames):
            file_path = current_dir / filename

            # Skip symlinks pointing outside the repo
            if _resolve_and_check(repo_root, file_path) is None:
                logger.debug("list_files: skipping file outside the repo: %s", file_path)
                continue

            suffix = file_path.suffix.lower()
            if suffix in _BINARY_EXTENSIONS:
                continue

            if allowed_exts is not None and suffix not in allowed_exts:
                continue

            total_found += 1
            if len(collected) < tool_input.max_results:
                try:
                    rel_path = file_path.relative_to(repo_root.resolve())
                    collected.append(str(rel_path).replace("\\", "/"))
                except ValueError:
                    pass

    truncated = total_found > tool_input.max_results
    logger.info(
        "list_files: found %d file(s), returning %d (truncated=%s)",
        total_found,
        len(collected),
        truncated,
    )

    return ListFilesOutput(
        success=True,
        files=collected,
        total_found=total_found,
        truncated=truncated,
    )
