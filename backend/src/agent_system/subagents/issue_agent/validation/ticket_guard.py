"""Combines schema, route and file checks into one validation outcome for a ticket."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from agent_system.core.contracts import IssueTicket, ValidationResult
from agent_system.subagents.issue_agent.validation.path_guard import check_affected_files
from agent_system.subagents.issue_agent.validation.route_guard import check_route, route_reason
from agent_system.subagents.issue_agent.validation.schema_guard import validate_schema

GuardStatus = Literal["valid", "invalid_schema", "missing_info", "invalid_files"]


@dataclass
class ValidationOutcome:
    result: ValidationResult
    # The parsed ticket. When valid, affected_files keeps only real files (the removed ones are in
    # result.invalid_files/warnings). None when the schema is invalid.
    ticket: IssueTicket | None = None

    @property
    def statuses(self) -> list[GuardStatus]:
        """All applicable statuses in priority order. An invalid schema yields only 'invalid_schema'."""
        if not self.result.is_valid:
            return ["invalid_schema"]
        found: list[GuardStatus] = []
        if self.ticket is not None and self.ticket.missing_info:
            found.append("missing_info")
        if self.result.invalid_files:
            found.append("invalid_files")
        return found or ["valid"]

    @property
    def status(self) -> GuardStatus:
        """Primary status: invalid_schema > missing_info > invalid_files > valid."""
        return self.statuses[0]

    @property
    def route_reason(self) -> str:
        """Route reason for a valid ticket; empty if the ticket is invalid."""
        if not self.result.is_valid or self.ticket is None:
            return ""
        return route_reason(self.ticket)


def validate_ticket(raw: Any, repo_path: str) -> ValidationOutcome:
    """
    Validate an LLM-generated issue ticket.

    - Schema/route errors: is_valid=False, errors to feed back to the LLM for a fix.
    - Nonexistent files: removed from the ticket, recorded in invalid_files + warnings, is_valid stays True.
    - Non-empty missing_info + route='ask_user' + at least one question: valid; the loop hands over to the user.
    """
    ticket, errors = validate_schema(raw)
    if ticket is None:
        return ValidationOutcome(ValidationResult(is_valid=False, errors=errors))

    errors = errors + check_route(ticket)
    valid, invalid, warnings = check_affected_files(ticket.affected_files, repo_path)

    result = ValidationResult(
        is_valid=not errors,
        errors=errors,
        warnings=warnings,
        valid_files=valid,
        invalid_files=invalid,
    )
    if errors:
        return ValidationOutcome(result, ticket)
    return ValidationOutcome(result, ticket.model_copy(update={"affected_files": valid}))
