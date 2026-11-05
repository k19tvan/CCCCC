"""Parsing and schema rules for LLM-generated tickets."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from agent_system.core.contracts import IssueTicket

MAX_AFFECTED_FILES = 5
_LIST_STR_FIELDS = (
    "affected_files",
    "reproduction_steps",
    "acceptance_criteria",
    "missing_info",
    "clarification_questions",
)


def _format_pydantic_errors(exc: ValidationError) -> list[str]:
    errors = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        errors.append(f"field '{loc}': {err['msg']}")
    return errors


def parse_ticket(raw: Any) -> tuple[IssueTicket | None, list[str]]:
    """
    Parse raw data into an IssueTicket. Returns (ticket, errors); ticket is None on error.

    A string input must be plain JSON: ```json fences and LLM preambles are not stripped here; that is the agent's job.
    """
    if isinstance(raw, IssueTicket):
        data: Any = raw.model_dump()
    elif isinstance(raw, (str, bytes)):
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return None, [f"output is not valid JSON: {exc}"]
    else:
        data = raw

    if not isinstance(data, dict):
        return None, [f"output must be a JSON object, got {type(data).__name__}"]

    # Pydantic ignores unknown keys and coerces types by default; check by hand so errors are not masked.
    errors: list[str] = []
    unknown = sorted(set(data) - set(IssueTicket.model_fields))
    if unknown:
        errors.append(f"fields not in the schema: {', '.join(unknown)}")
    for name in _LIST_STR_FIELDS:
        value = data.get(name)
        if value is None:
            continue
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            errors.append(f"field '{name}': must be a list of strings (no other types)")

    try:
        ticket = IssueTicket.model_validate(data)
    except ValidationError as exc:
        return None, errors + _format_pydantic_errors(exc)
    return (None, errors) if errors else (ticket, [])


def check_schema_rules(ticket: IssueTicket) -> list[str]:
    """Cross-field and content rules that Pydantic's type checks do not catch."""
    errors: list[str] = []

    if not ticket.priority_reason.strip():
        errors.append("field 'priority_reason': must not be empty")
    if not ticket.summary.strip():
        errors.append("field 'summary': must not be empty")
    if len(ticket.affected_files) > MAX_AFFECTED_FILES:
        errors.append(
            f"field 'affected_files': at most {MAX_AFFECTED_FILES} files, "
            f"got {len(ticket.affected_files)}"
        )

    for name in _LIST_STR_FIELDS:
        if any(not item.strip() for item in getattr(ticket, name)):
            errors.append(f"field '{name}': contains an empty or whitespace-only item")

    if ticket.route in ("fix", "implement") and not ticket.acceptance_criteria:
        errors.append(
            f"field 'acceptance_criteria': at least 1 criterion is required when route='{ticket.route}'"
        )
    return errors


def validate_schema(raw: Any) -> tuple[IssueTicket | None, list[str]]:
    """Parse + check schema rules. Returns (ticket, errors); empty errors means it passed."""
    ticket, errors = parse_ticket(raw)
    if ticket is None:
        return None, errors
    return ticket, check_schema_rules(ticket)
