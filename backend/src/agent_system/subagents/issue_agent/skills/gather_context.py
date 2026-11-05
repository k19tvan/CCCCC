"""Skill that gathers repository context for an issue.

Extracts keywords from the issue, calls list_files, search_code and read_file to collect evidence,
and returns a GatherContextOutput with cited sources for the LLM to build an IssueTicket from.
All context must come from tool output; nothing is invented.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from agent_system.core.contracts import (
    GatherContextInput,
    GatherContextOutput,
    ContextSource,
    ListFilesInput,
    ReadFileInput,
    SearchCodeInput,
)
from agent_system.core.redaction import redact_secrets

logger = logging.getLogger(__name__)

# Per-tool-call result limits
_MAX_FILES = 50
_MAX_SEARCH_RESULTS = 20
_MAX_READ_CHARS = 3_000

# Maximum number of files read in full after searching
_MAX_FILES_TO_READ = 3

# Words never used as keywords; the Vietnamese entries support Vietnamese issue text.
_STOPWORDS: frozenset[str] = frozenset({
    # English
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "it", "its", "this", "that", "these", "those", "i", "we", "you",
    "he", "she", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "our", "their", "what", "which", "who", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "but", "if", "or", "as", "at", "by", "for",
    "in", "of", "on", "to", "up", "out", "off", "over", "under", "again",
    "further", "then", "once", "and", "with", "from", "into", "through",
    "during", "before", "after", "above", "below", "between", "about",
    "against", "along", "around", "get", "make", "work", "use", "also",
    "error", "bug", "issue", "problem", "fix", "code", "function",
    # Vietnamese
    "bị", "lỗi", "không", "chạy", "được", "khi", "hàm", "file",
    "trong", "của", "là", "và", "có", "để", "với", "tôi", "này",
    "đó", "từ", "vào", "ra", "cho", "bằng", "thì", "đã", "sẽ",
})


def _extract_keywords(title: str, description: str, clarified_text: str | None) -> list[str]:
    """
    Extract meaningful keywords from the issue title and description.

    Symbol names (camelCase, snake_case, PascalCase) come first, then file names, then plain
    words of length >= 4 that are not stopwords. At most 10 keywords are returned.
    """
    full_text = f"{title} {description}"
    if clarified_text:
        full_text += f" {clarified_text}"
    # Keys/passwords in the issue must not become search queries (queries end up in the trace)
    full_text = redact_secrets(full_text, placeholder=" ")

    keywords: list[str] = []
    seen: set[str] = set()

    # Symbol names such as calculateTotal, UserService, get_discount_rate
    symbol_pattern = re.compile(
        r"\b([a-z_][a-z0-9_]{2,}(?:_[a-z0-9_]+)+|"  # snake_case (at least 2 parts)
        r"[A-Z][a-zA-Z0-9]{2,}|"  # PascalCase
        r"[a-z][a-zA-Z0-9]{2,}(?=[A-Z]))\b"  # camelCase
    )
    for match in symbol_pattern.finditer(full_text):
        token = match.group(0)
        lower = token.lower()
        if lower not in _STOPWORDS and lower not in seen and len(token) >= 3:
            keywords.append(token)
            seen.add(lower)

    file_pattern = re.compile(r"\b[\w/]+\.(py|json|yaml|yml|toml|ini|cfg|txt|md)\b", re.IGNORECASE)
    for match in file_pattern.finditer(full_text):
        token = match.group(0)
        lower = token.lower()
        if lower not in seen:
            keywords.append(token)
            seen.add(lower)

    # Skip all-uppercase words: instructions like "SYSTEM OVERRIDE" or secret strings must not
    # reach search queries (queries end up in the trace).
    word_pattern = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]{3,}\b")
    for match in word_pattern.finditer(full_text):
        token = match.group(0)
        lower = token.lower()
        if token.isupper() or lower in _STOPWORDS or lower in seen:
            continue
        keywords.append(token)
        seen.add(lower)
        if len(keywords) >= 15:
            break

    return keywords[:10]


def gather_context(
    skill_input: GatherContextInput,
    registry: "ToolRegistry",  # type: ignore[name-defined]  # noqa: F821
) -> GatherContextOutput:
    """
    Gather repository context for an issue using the tools in `registry`.

    Steps: extract keywords, list files, search code for the top keywords, then read the most
    relevant files. Tool failures are logged and skipped so the skill always returns a result.
    """
    logger.info(
        "gather_context: starting context gathering for issue '%s'",
        skill_input.title,
    )

    sources: list[ContextSource] = []
    files_found: set[str] = set()
    tools_called: list[str] = []

    keywords = _extract_keywords(
        skill_input.title,
        skill_input.description,
        skill_input.clarified_text,
    )
    logger.debug("gather_context: extracted keywords: %s", keywords)

    if not keywords:
        # Fallback: first 3 words of the title
        words = redact_secrets(skill_input.title, placeholder=" ").split()[:3]
        keywords = [w for w in words if len(w) >= 3 and not w.isupper()]
        logger.debug("gather_context: fallback keywords: %s", keywords)

    logger.info("gather_context: calling tool list_files")
    tools_called.append("list_files")

    try:
        list_result = registry.call(
            "list_files",
            ListFilesInput(
                repo_path=skill_input.repo_path,
                extensions=[".py", ".md", ".yaml", ".yml", ".toml"],
                max_results=_MAX_FILES,
            ),
        )

        if list_result.success:
            all_files = list_result.files
            logger.info(
                "gather_context: list_files returned %d file(s) (truncated=%s)",
                len(all_files),
                list_result.truncated,
            )

            file_list_preview = all_files[:20]
            sources.append(
                ContextSource(
                    tool_name="list_files",
                    content=(
                        f"Repo structure ({len(all_files)} files found"
                        + (" - truncated" if list_result.truncated else "")
                        + "):\n"
                        + "\n".join(f"  {f}" for f in file_list_preview)
                        + ("\n  ..." if len(all_files) > 20 else "")
                    ),
                    relevance_note="Overall structure of the repo",
                )
            )
            files_found.update(all_files)

            relevant_files = _find_relevant_files(all_files, keywords)
            if relevant_files:
                sources.append(
                    ContextSource(
                        tool_name="list_files",
                        content=(
                            "Files whose names relate to the issue:\n"
                            + "\n".join(f"  {f}" for f in relevant_files)
                        ),
                        relevance_note="Files whose names match the issue keywords",
                    )
                )
        else:
            logger.warning("gather_context: list_files failed: %s", list_result.error)
            sources.append(
                ContextSource(
                    tool_name="list_files",
                    content=f"Could not list files: {list_result.error}",
                    relevance_note="Tool error",
                )
            )
            all_files = []

    except Exception as exc:
        logger.error("gather_context: error calling list_files: %s", exc, exc_info=True)
        all_files = []

    logger.info("gather_context: calling tool search_code with %d keywords", len(keywords))
    tools_called.append("search_code")

    search_hits_by_file: dict[str, list[str]] = {}  # file_path -> list of snippets

    keywords_to_search = keywords[:5]

    for keyword in keywords_to_search:
        try:
            search_result = registry.call(
                "search_code",
                SearchCodeInput(
                    repo_path=skill_input.repo_path,
                    query=keyword,
                    is_regex=False,
                    file_extensions=[".py"],
                    case_sensitive=False,
                    max_results=_MAX_SEARCH_RESULTS,
                    context_lines=2,
                ),
            )

            if search_result.success and search_result.matches:
                logger.debug(
                    "gather_context: search_code '%s' -> %d match(es)",
                    keyword,
                    len(search_result.matches),
                )

                for match in search_result.matches[:5]:  # At most 5 matches per keyword
                    fp = match.file_path
                    files_found.add(fp)
                    if fp not in search_hits_by_file:
                        search_hits_by_file[fp] = []

                    snippet_lines = (
                        match.context_before
                        + [f">>> {match.line_content}"]
                        + match.context_after
                    )
                    search_hits_by_file[fp].append(
                        f"  [line {match.line_number}] "
                        + "\n  ".join(snippet_lines)
                    )

                    sources.append(
                        ContextSource(
                            tool_name="search_code",
                            file_path=fp,
                            line_number=match.line_number,
                            content=(
                                f"Found '{keyword}' in {fp}:{match.line_number}\n"
                                + "\n".join(snippet_lines)
                            ),
                            relevance_note=f"Keyword '{keyword}' appears here",
                        )
                    )
            else:
                logger.debug(
                    "gather_context: search_code '%s' -> no matches", keyword
                )

        except Exception as exc:
            logger.error(
                "gather_context: search_code error for '%s': %s", keyword, exc, exc_info=True
            )

    # Read the files with the most search hits first
    files_to_read = sorted(
        search_hits_by_file.keys(),
        key=lambda f: len(search_hits_by_file[f]),
        reverse=True,
    )[:_MAX_FILES_TO_READ]

    # Top up with files whose names match the keywords
    if len(files_to_read) < _MAX_FILES_TO_READ:
        relevant_from_list = _find_relevant_files(list(files_found), keywords)
        for rf in relevant_from_list:
            if rf not in files_to_read and len(files_to_read) < _MAX_FILES_TO_READ:
                files_to_read.append(rf)

    if files_to_read:
        logger.info(
            "gather_context: calling tool read_file for %d file(s): %s",
            len(files_to_read),
            files_to_read,
        )
        tools_called.append("read_file")

        for file_path in files_to_read:
            try:
                read_result = registry.call(
                    "read_file",
                    ReadFileInput(
                        repo_path=skill_input.repo_path,
                        file_path=file_path,
                        max_chars=_MAX_READ_CHARS,
                    ),
                )

                if read_result.success:
                    sources.append(
                        ContextSource(
                            tool_name="read_file",
                            file_path=file_path,
                            content=(
                                f"Contents of {file_path} "
                                f"(lines {read_result.start_line}-{read_result.end_line}"
                                f"/{read_result.total_lines}"
                                + (" - truncated" if read_result.truncated else "")
                                + "):\n"
                                + read_result.content
                            ),
                            relevance_note="File content related to the issue",
                        )
                    )
                    logger.debug(
                        "gather_context: read_file '%s' -> %d chars",
                        file_path,
                        len(read_result.content),
                    )
                else:
                    logger.warning(
                        "gather_context: read_file '%s' failed: %s",
                        file_path,
                        read_result.error,
                    )

            except Exception as exc:
                logger.error(
                    "gather_context: read_file error for '%s': %s",
                    file_path,
                    exc,
                    exc_info=True,
                )

    summary = _build_summary(
        title=skill_input.title,
        keywords=keywords,
        files_found=list(files_found),
        sources=sources,
        search_hits_by_file=search_hits_by_file,
    )

    logger.info(
        "gather_context: done - %d sources, %d files, %d tools",
        len(sources),
        len(files_found),
        len(set(tools_called)),
    )

    return GatherContextOutput(
        sources=sources,
        files_found=sorted(files_found),
        keywords_used=keywords,
        tools_called=list(dict.fromkeys(tools_called)),  # de-duplicate, keep order
        summary=summary,
    )


def _find_relevant_files(files: list[str], keywords: list[str]) -> list[str]:
    """Return up to 10 files whose name or path matches at least one keyword."""
    relevant: list[str] = []
    lower_keywords = [kw.lower() for kw in keywords]

    for file_path in files:
        path_lower = file_path.lower()
        stem = Path(file_path).stem.lower()

        for kw in lower_keywords:
            kw_parts = re.split(r"[_\-.]", kw)
            if any(part in stem for part in kw_parts if len(part) >= 3):
                relevant.append(file_path)
                break
            if len(kw) >= 4 and kw in path_lower:
                relevant.append(file_path)
                break

    return relevant[:10]


def _build_summary(
    title: str,
    keywords: list[str],
    files_found: list[str],
    sources: list[ContextSource],
    search_hits_by_file: dict[str, list[str]],
) -> str:
    """Build the short summary of collected context that is put into the LLM prompt."""
    lines = [
        f"# Context gathered for issue: \"{title}\"",
        "",
        f"**Keywords searched:** {', '.join(keywords) if keywords else '(none)'}",
        "",
    ]

    if files_found:
        lines.append(f"**Files found in the repo ({len(files_found)}):**")
        for f in sorted(files_found)[:10]:
            hit_count = len(search_hits_by_file.get(f, []))
            marker = f"  - {f}"
            if hit_count > 0:
                marker += f" ({hit_count} match)"
            lines.append(marker)
        if len(files_found) > 10:
            lines.append(f"  ... and {len(files_found) - 10} more files")
        lines.append("")

    search_files = [f for f in search_hits_by_file if search_hits_by_file[f]]
    if search_files:
        lines.append("**Relevant code found in:**")
        for fp in search_files[:5]:
            lines.append(f"  - {fp}: {len(search_hits_by_file[fp])} snippets")
        lines.append("")

    source_count = len(sources)
    lines.append(f"**Total:** {source_count} context sources collected.")

    if not files_found and not search_hits_by_file:
        lines.append("")
        lines.append(
            "**Note:** No relevant files were found in the repo. "
            "The issue may concern functionality that is not implemented yet, "
            "or the keywords may need adjusting."
        )

    return "\n".join(lines)
