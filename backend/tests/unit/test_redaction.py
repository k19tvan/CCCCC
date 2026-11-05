"""Tests that secret-shaped strings are masked before they reach traces or search queries."""

from __future__ import annotations

import json

import pytest

from agent_system.core.redaction import REDACTED, redact_secrets
from agent_system.orchestrator import process_issue
from agent_system.subagents.issue_agent.skills.gather_context import _extract_keywords


@pytest.mark.parametrize(
    "text",
    [
        "key sk-AbCd1234EfGh5678IjKl here",
        "password=Hunter2!",
        "DUMMY_CANARY_SECRET=CANARY_SECRET_LEAK_TOKEN_9876",
        "api_key = AbCd1234",
        "blob AbCdEf1234567890GhIjKl end",
        "AKIAIOSFODNN7EXAMPLE",
    ],
)
def test_secret_shaped_text_is_redacted(text: str) -> None:
    assert REDACTED in redact_secrets(text)


@pytest.mark.parametrize(
    "text",
    [
        "verify_token raises ValueError",
        "if token == None: crash",
        "test_calculate_total_with_invalid_items_2 fails",
        "UserService.create_user rejects empty username",
    ],
)
def test_identifiers_and_comparisons_are_not_redacted(text: str) -> None:
    assert redact_secrets(text) == text


def test_mixed_case_secret_is_not_a_search_keyword() -> None:
    keywords = _extract_keywords("Login broken", "token=AbCd1234EfGh and key sk-AbCd1234EfGh5678IjKl", None)

    assert not any("AbCd1234" in k for k in keywords)


def test_secret_in_title_and_description_is_absent_from_trace() -> None:
    secret = "sk-AbCd1234EfGh5678IjKl"
    from agent_system.core.contracts import IssueRequest

    result = process_issue(
        IssueRequest(
            title=f"Login fails, my key is {secret}",
            description=f"Please fix login. password=Hunter2xyz and {secret}",
            repo_path="tests/fixtures/sample_repo",
        )
    )

    trace_text = json.dumps(result.trace, ensure_ascii=False)
    assert secret not in trace_text
    assert "Hunter2xyz" not in trace_text
