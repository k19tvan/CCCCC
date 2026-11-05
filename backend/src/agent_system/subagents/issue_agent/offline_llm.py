"""
Rule-based offline LLM, used when LLM_PROVIDER=fake.

Lets demos, integration tests and evals run without an API key or network. It is a deterministic
TEST DOUBLE, not the product's classifier; real classification quality comes from a real LLM.

It reads the data blocks built by `prompts.py` and returns an IssueTicket JSON using simple keyword
rules. For step-by-step scenarios (bad schema, timeouts...) use `agent_system.core.llm.client.FakeLLM`.
"""

from __future__ import annotations

import json
import re
from pathlib import PurePosixPath
from typing import Any, Sequence

from agent_system.subagents.issue_agent.prompts import EVIDENCE_TAG, ISSUE_TAG
from agent_system.core.llm.client import LLMMessage, LLMResponse

# The keyword patterns below include Vietnamese terms so the double also recognizes Vietnamese issues.
_INJECTION = re.compile(
    r"ignore (all |any )?(previous|prior|above)|system override|reveal (the )?(system prompt|secret)"
    r"|developer instructions|disregard (all |the )?(rules|instructions)|bỏ qua (mọi|tất cả|các) (chỉ dẫn|quy tắc)"
    r"|rm -rf|remove all files|xóa toàn bộ",
    re.IGNORECASE,
)
_BUG = re.compile(
    r"\b(bug|errors?|exceptions?|crash\w*|fail\w*|broken|wrong|incorrect|traceback|raises?|slow|unstable"
    r"|vulnerabilit\w*|security hole|lỗi|sai|hỏng|treo|lỗ hổng)\b|\b\w+(Error|Exception)\b",
    re.IGNORECASE,
)
_QUESTION = re.compile(
    r"^\s*(what|how|where|why|which|who|does|do|is|are|can|query|find|explain|list"
    r"|giải thích|tại sao|ở đâu|làm sao|hàm nào|có phải)\b|\?\s*$",
    re.IGNORECASE,
)
_REFACTOR = re.compile(r"\b(refactor\w*|clean ?up|restructur\w*|tái cấu trúc)\b", re.IGNORECASE)
_DOCS = re.compile(r"\b(docs?|documentation|readme|docstrings?|typo|tài liệu)\b", re.IGNORECASE)
_FEATURE = re.compile(
    r"\b(add|implement|support|new feature|feature request|enhancement|thêm|tính năng|hỗ trợ)\b",
    re.IGNORECASE,
)
# Signals of a bug with concrete details: file, function call, traceback, line number, exception name
_CONCRETE = re.compile(
    r"\w+\.py\b|\w+\s*\(|traceback|\bline \d+|dòng \d+|\b\w+(Error|Exception)\b",
    re.IGNORECASE,
)
_SEVERE = re.compile(r"security|data loss|production down|bảo mật|mất dữ liệu", re.IGNORECASE)
_DATA_BLOCK = r"<{tag}>\s*(.*?)\s*</{tag}>"

_ROUTE_BY_TYPE = {
    "bug": "fix",
    "feature": "implement",
    "refactor": "implement",
    "docs": "implement",
    "question": "answer",
    "invalid": "reject",
}


class OfflineTriageLLM:
    """Offline LLM double: classifies issues with keyword rules and always returns schema-valid JSON."""

    name = "offline_rules"

    def __init__(self, model: str = "offline-rules-v1") -> None:
        self.model = model

    def complete(
        self,
        messages: Sequence[LLMMessage],
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        prompt = next((m.content for m in messages if m.role == "user"), "")
        issue = _read_block(prompt, ISSUE_TAG) or {}
        evidence = _read_block(prompt, EVIDENCE_TAG) or {}
        ticket = _triage(
            title=str(issue.get("title") or ""),
            description=str(issue.get("description") or ""),
            clarified=str(issue.get("clarified_text") or ""),
            evidence=evidence,
        )
        return LLMResponse(text=json.dumps(ticket, ensure_ascii=False), model=self.model)


def _read_block(prompt: str, tag: str) -> Any:
    match = re.search(_DATA_BLOCK.format(tag=tag), prompt, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def _classify(text: str) -> str | None:
    if _BUG.search(text):
        return "bug"
    if _QUESTION.search(text):
        return "question"
    if _REFACTOR.search(text):
        return "refactor"
    if _DOCS.search(text):
        return "docs"
    if _FEATURE.search(text):
        return "feature"
    return None


def _triage(title: str, description: str, clarified: str, evidence: dict[str, Any]) -> dict[str, Any]:
    full_text = " ".join(part for part in (title, description, clarified) if part)

    if _INJECTION.search(full_text):
        return _ticket(
            "invalid",
            "P3",
            "The request contains instructions that try to override the agent's rules",
            summary=(
                "Rejected: the issue content tries to override system instructions or asks for an "
                "unsafe action, so it is not an issue about the repository."
            ),
            acceptance=["The instructions embedded in the issue content are not carried out"],
        )

    issue_type = _classify(title) or _classify(full_text)
    if issue_type is None:
        return _ticket(
            "invalid",
            "P3",
            "The request is unrelated to the repository's source code",
            summary=(
                "Rejected: the request is outside the supported scope (not a bug, feature, "
                "documentation change or question about the repository)."
            ),
            acceptance=["The user receives a clear reason for the refusal"],
        )

    if issue_type == "bug" and not clarified and not _CONCRETE.search(full_text):
        return _ticket(
            "bug",
            "P2",
            "Not enough information to assess the impact",
            summary=f"Issue '{title}' has no specific symptom or location to work with.",
            missing_info=[
                "Specific reproduction steps",
                "Full error message or traceback",
                "Related function/file",
            ],
            questions=[
                "What steps do you take to hit this error?",
                "What is the full error message or traceback?",
                "In which function or file does the error occur?",
            ],
            acceptance=["The error can be reproduced from the information the user provides"],
        )

    files = _pick_files(full_text, evidence)
    missing = _missing_files(full_text, evidence)
    if issue_type == "bug" and missing:
        # The bug mentions a file that is not in the repo: ask the user instead of guessing another file
        return _ticket(
            "bug",
            "P2",
            "Could not identify the related file in the repository",
            summary=f"The issue mentions a file that is not in the repository: {', '.join(missing)}.",
            missing_info=[f"File '{m}' was not found in the repository." for m in missing],
            questions=[
                f"Could you double-check the file name '{m}', or tell me the correct path of the file?"
                for m in missing
            ],
            acceptance=["A file that exists in the repository is identified to locate the bug"],
        )
    if issue_type == "bug":
        priority = "P0" if _SEVERE.search(full_text) else "P1"
        return _ticket(
            "bug",
            priority,
            "Exception or wrong behavior in the reported feature",
            summary=f"Bug: {title}",
            files=files,
            steps=[
                f"Perform the action described in the issue: {title}",
                "Observe the reported error or wrong behavior",
            ],
            acceptance=[
                "The action in the issue no longer causes an error and returns the correct result",
                "A regression test covers the reported case",
            ],
        )
    if issue_type == "question":
        hits = [h for h in evidence.get("hits", []) if any(h.startswith(f + ":") for f in files)]
        found = ", ".join((hits or files)[:5])
        summary = (
            f"Related locations in the repository: {found}."
            if found
            else "No relevant evidence was found in the repository."
        )
        return _ticket(
            "question",
            "P3",
            "Question about the codebase, no impact on operation",
            summary=summary,
            files=files,
            acceptance=["The answer cites files/lines that exist in the repository"],
        )
    return _ticket(
        issue_type,
        "P3" if issue_type == "docs" else "P2",
        "Requested change, does not block existing functionality",
        summary=f"{issue_type.capitalize()}: {title}",
        files=files,
        acceptance=[f"The request '{title}' is fulfilled with an accompanying test/check"],
    )


def _missing_files(text: str, evidence: dict[str, Any]) -> list[str]:
    """.py files mentioned in the issue but absent from the evidence (i.e. not in the repo)."""
    known = {
        PurePosixPath(f).name.lower()
        for f in [*evidence.get("matched", []), *evidence.get("listed", [])]
        if isinstance(f, str)
    }
    mentioned = [m for m in re.findall(r"\b[\w\-./]+\.py\b", text) if not m.startswith("..")]
    return list(dict.fromkeys(m for m in mentioned if PurePosixPath(m).name.lower() not in known))


def _pick_files(text: str, evidence: dict[str, Any]) -> list[str]:
    """Pick files backed by tool evidence that the issue mentions (by name or stem)."""
    matched = [f for f in evidence.get("matched", []) if isinstance(f, str)]
    listed = [f for f in evidence.get("listed", []) if isinstance(f, str)]
    lowered = text.lower()

    by_name = [f for f in matched + listed if PurePosixPath(f).name.lower() in lowered]
    if by_name:
        return list(dict.fromkeys(by_name))[:5]

    # The issue mentions a specific .py file that is not in the repository (the LLM hallucinated a file from the prompt)
    mentioned = [
        m for m in re.findall(r"\b[\w\-./]+\.py\b", text)
        if not m.startswith("..")
    ]
    if mentioned:
        return list(dict.fromkeys(mentioned))[:5]

    by_stem = [
        f
        for f in matched
        if len(PurePosixPath(f).stem) >= 4 and PurePosixPath(f).stem.lower() in lowered
    ]
    return list(dict.fromkeys(by_stem or matched[:2]))[:5]


def _ticket(
    issue_type: str,
    priority: str,
    priority_reason: str,
    *,
    summary: str,
    acceptance: list[str],
    files: list[str] | None = None,
    steps: list[str] | None = None,
    missing_info: list[str] | None = None,
    questions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": issue_type,
        "priority": priority,
        "priority_reason": priority_reason,
        "affected_files": files or [],
        "reproduction_steps": steps or [],
        "acceptance_criteria": acceptance,
        "missing_info": missing_info or [],
        "route": "ask_user" if missing_info else _ROUTE_BY_TYPE[issue_type],
        "clarification_questions": questions or [],
        "summary": summary,
    }
