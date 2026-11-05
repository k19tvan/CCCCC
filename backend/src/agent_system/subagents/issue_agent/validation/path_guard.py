"""Checks that ticket affected_files are normalized, inside the repo and real files."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath


def _normalize(raw: str) -> tuple[str | None, str | None]:
    """Normalize a relative path. Returns (posix_path, error_reason)."""
    if not isinstance(raw, str) or not raw.strip():
        return None, "path is empty or not a string"
    if "\x00" in raw:
        return None, "contains a null character"

    stripped = raw.strip()
    text = stripped.replace("\\", "/")
    if PurePosixPath(text).is_absolute() or PureWindowsPath(stripped).is_absolute():
        return None, "must not be an absolute path"
    if PureWindowsPath(stripped).drive:
        return None, "must not contain a drive letter"

    parts = [p for p in text.split("/") if p not in ("", ".")]
    if not parts:
        return None, "path does not point to any file"
    if ".." in parts:
        return None, "contains '..' (path traversal)"
    return "/".join(parts), None


def check_affected_files(
    files: list[str], repo_path: str
) -> tuple[list[str], list[str], list[str]]:
    """
    Check each file: well-formed, inside the repo, existing, and a regular file.

    Returns:
        (valid_files, invalid_files, warnings). valid_files are normalized paths
        (posix, relative to the repo), deduplicated, in their original order.
    """
    valid: list[str] = []
    invalid: list[str] = []
    warnings: list[str] = []

    try:
        root = Path(repo_path).resolve(strict=True)
    except (OSError, RuntimeError):
        root = None
    if root is None or not root.is_dir():
        invalid.extend(str(f) for f in files)
        warnings.append(f"repo_path does not exist or is not a directory: {repo_path!r}")
        return valid, invalid, warnings

    seen: set[str] = set()
    for raw in files:
        rel, reason = _normalize(raw)
        if rel is None:
            invalid.append(str(raw))
            warnings.append(f"affected_files: {raw!r} removed — {reason}")
            continue

        try:
            resolved = (root / rel).resolve()
        except (OSError, RuntimeError):
            invalid.append(str(raw))
            warnings.append(f"affected_files: {raw!r} removed — path could not be resolved")
            continue

        # resolve() follows symlinks, so a symlink pointing outside the repo is rejected here.
        if root != resolved and root not in resolved.parents:
            invalid.append(str(raw))
            warnings.append(f"affected_files: {raw!r} removed — outside the repo")
        elif not resolved.exists():
            invalid.append(str(raw))
            warnings.append(f"affected_files: {raw!r} removed — file does not exist")
        elif not resolved.is_file():
            invalid.append(str(raw))
            warnings.append(f"affected_files: {raw!r} removed — not a file")
        elif rel in seen:
            warnings.append(f"affected_files: {raw!r} is a duplicate, keeping only one copy")
        else:
            seen.add(rel)
            valid.append(rel)

    return valid, invalid, warnings
