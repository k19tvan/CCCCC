"""
Unit tests for the validation module.

Covers path_guard, schema_guard, route_guard, ticket_guard and the issue_schema/routing hooks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest

from agent_system.core.contracts import IssueTicket, ValidationResult
from agent_system.subagents.issue_agent.validation.path_guard import check_affected_files, _normalize
from agent_system.subagents.issue_agent.validation.schema_guard import validate_schema, parse_ticket, check_schema_rules
from agent_system.subagents.issue_agent.validation.route_guard import check_route, route_reason, ALLOWED_ROUTES
from agent_system.subagents.issue_agent.validation.ticket_guard import validate_ticket as guard_validate_ticket, ValidationOutcome
from agent_system.subagents.issue_agent.validation.issue_schema import validate_ticket as hook_validate_ticket
from agent_system.subagents.issue_agent.validation.routing import decide_route


@pytest.fixture
def temp_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "valid_file.py").write_text("# valid\n", encoding="utf-8")
    (repo / "calc.py").write_text("# calc\n", encoding="utf-8")
    sub = repo / "pkg"
    sub.mkdir()
    (sub / "mod.py").write_text("# mod\n", encoding="utf-8")
    return repo


class TestPathGuard:
    def test_normalize_valid_relative_paths(self):
        rel, err = _normalize("foo/bar.py")
        assert rel == "foo/bar.py" and err is None

        rel, err = _normalize("./calc.py")
        assert rel == "calc.py" and err is None

        rel, err = _normalize("pkg\\mod.py")
        assert rel == "pkg/mod.py" and err is None

    def test_normalize_invalid_paths(self):
        assert _normalize("")[1] is not None
        assert _normalize("   ")[1] is not None
        assert _normalize("foo\x00bar.py")[1] is not None
        assert _normalize("../escape.py")[1] is not None
        assert _normalize("foo/../../escape.py")[1] is not None
        assert _normalize("/absolute/path.py")[1] is not None
        assert _normalize("C:\\drive\\path.py")[1] is not None

    def test_check_affected_files_filters_invalid_and_keeps_valid(self, temp_repo: Path):
        files = [
            "valid_file.py",
            "nonexistent.py",
            "pkg",  # directory
            "./calc.py",  # normalized to calc.py
            "valid_file.py",  # duplicate
            "../outside.py",  # traversal
        ]
        valid, invalid, warnings = check_affected_files(files, str(temp_repo))

        assert valid == ["valid_file.py", "calc.py"]
        assert "nonexistent.py" in invalid
        assert "pkg" in invalid
        assert "../outside.py" in invalid
        assert any("does not exist" in w for w in warnings)
        assert any("not a file" in w for w in warnings)
        assert any("duplicate" in w for w in warnings)

    def test_check_affected_files_nonexistent_repo(self, tmp_path: Path):
        valid, invalid, warnings = check_affected_files(["a.py"], str(tmp_path / "no_such_repo"))
        assert valid == []
        assert invalid == ["a.py"]
        assert any("does not exist or is not a directory" in w for w in warnings)

    def test_check_affected_files_symlink_escape(self, temp_repo: Path, tmp_path: Path):
        outside_file = tmp_path / "secret.py"
        outside_file.write_text("secret", encoding="utf-8")
        link = temp_repo / "symlink_outside.py"
        try:
            link.symlink_to(outside_file)
        except (OSError, NotImplementedError):
            pytest.skip("This system does not support symlinks without admin rights")

        valid, invalid, warnings = check_affected_files(["symlink_outside.py"], str(temp_repo))
        assert "symlink_outside.py" in invalid
        assert any("outside the repo" in w for w in warnings)


class TestSchemaGuard:
    def test_valid_ticket_passes_schema(self):
        raw = {
            "type": "bug",
            "priority": "P1",
            "priority_reason": "Crash on divide by zero",
            "affected_files": ["calc.py"],
            "reproduction_steps": ["Call divide(1, 0)"],
            "acceptance_criteria": ["Raises ValueError instead of crash"],
            "missing_info": [],
            "route": "fix",
            "clarification_questions": [],
            "summary": "Fix ZeroDivisionError in divide",
        }
        ticket, errors = validate_schema(raw)
        assert ticket is not None
        assert errors == []

    def test_unknown_fields_reported(self):
        raw = {
            "type": "bug",
            "priority": "P1",
            "priority_reason": "crash",
            "summary": "sum",
            "acceptance_criteria": ["ac"],
            "extra_field": "unknown",
        }
        ticket, errors = validate_schema(raw)
        assert ticket is None
        assert any("fields not in the schema" in e for e in errors)

    def test_more_than_max_affected_files(self):
        raw = {
            "type": "bug",
            "priority": "P1",
            "priority_reason": "reason",
            "affected_files": [f"f{i}.py" for i in range(6)],
            "acceptance_criteria": ["ac"],
            "summary": "summary",
        }
        ticket, errors = validate_schema(raw)
        assert any("at most 5 files" in e for e in errors)

    def test_empty_string_in_list_fields_rejected(self):
        raw = {
            "type": "bug",
            "priority": "P1",
            "priority_reason": "reason",
            "acceptance_criteria": ["valid", "   "],
            "summary": "summary",
        }
        ticket, errors = validate_schema(raw)
        assert any("whitespace-only" in e for e in errors)


class TestRouteGuard:
    def test_valid_routes(self):
        ticket = IssueTicket(
            type="bug",
            priority="P1",
            priority_reason="reason",
            affected_files=[],
            reproduction_steps=[],
            acceptance_criteria=["ac"],
            missing_info=[],
            route="fix",
            clarification_questions=[],
            summary="sum",
        )
        assert check_route(ticket) == []
        assert "routed to the fix branch" in route_reason(ticket)

    def test_inconsistent_route_with_missing_info(self):
        ticket = IssueTicket(
            type="bug",
            priority="P1",
            priority_reason="reason",
            affected_files=[],
            reproduction_steps=[],
            acceptance_criteria=["ac"],
            missing_info=["Need traceback"],
            route="fix",  # Wrong: non-empty missing_info requires ask_user
            clarification_questions=["What is the error?"],
            summary="sum",
        )
        errors = check_route(ticket)
        assert any("missing_info is not empty, so route must be 'ask_user'" in e for e in errors)

    def test_type_route_mismatch(self):
        ticket = IssueTicket(
            type="question",
            priority="P3",
            priority_reason="reason",
            affected_files=[],
            reproduction_steps=[],
            acceptance_criteria=["ac"],
            missing_info=[],
            route="fix",  # Wrong: a question only goes with answer, ask_user, reject
            clarification_questions=[],
            summary="sum",
        )
        errors = check_route(ticket)
        assert any("type='question' does not go with route='fix'" in e for e in errors)


class TestTicketGuardAndHooks:
    def test_guard_validate_ticket_outcome(self, temp_repo: Path):
        raw = {
            "type": "bug",
            "priority": "P1",
            "priority_reason": "Crash in calculator",
            "affected_files": ["calc.py", "nonexistent.py"],
            "reproduction_steps": ["step 1"],
            "acceptance_criteria": ["ac 1"],
            "missing_info": [],
            "route": "fix",
            "clarification_questions": [],
            "summary": "Fix crash",
        }
        outcome = guard_validate_ticket(raw, str(temp_repo))
        assert outcome.result.is_valid is True
        assert outcome.result.valid_files == ["calc.py"]
        assert outcome.result.invalid_files == ["nonexistent.py"]
        assert outcome.ticket is not None
        assert outcome.ticket.affected_files == ["calc.py"]
        assert outcome.status == "invalid_files"
        assert "routed to the fix branch" in outcome.route_reason

    def test_hook_issue_schema_and_routing(self, temp_repo: Path):
        raw = {
            "type": "feature",
            "priority": "P2",
            "priority_reason": "New math function",
            "affected_files": ["valid_file.py"],
            "reproduction_steps": [],
            "acceptance_criteria": ["New function works"],
            "missing_info": [],
            "route": "implement",
            "clarification_questions": [],
            "summary": "Add multiply function",
        }
        res: ValidationResult = hook_validate_ticket(raw, str(temp_repo))
        assert res.is_valid is True
        assert res.valid_files == ["valid_file.py"]

        ticket = IssueTicket.model_validate(raw)
        route, reason = decide_route(ticket)
        assert route == "implement"
        assert "implement branch" in reason
