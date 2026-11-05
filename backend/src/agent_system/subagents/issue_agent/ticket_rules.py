"""
Ticket validation rules and result messages for triage.

Pure functions (no LLM calls, no trace writes); the graph decides when to call them.
"""

from __future__ import annotations

from typing import Any

from agent_system.core.contracts import IssueTicket


def check_ticket_rules(ticket: IssueTicket) -> list[str]:
    """
    Cross-field constraints that IssueResult status relies on, plus the rules the prompt makes mandatory.

    The validation module may check more; the errors here are fed back to the LLM so it can fix them.
    """
    errors: list[str] = []
    if not [c for c in ticket.acceptance_criteria if c.strip()]:
        errors.append("acceptance_criteria: at least one testable criterion is required")
    if ticket.missing_info and ticket.route != "ask_user":
        errors.append(
            f"route: missing_info is not empty, so route must be 'ask_user' (got '{ticket.route}')"
        )
    if ticket.route == "ask_user":
        if not ticket.missing_info:
            errors.append("missing_info: route='ask_user' requires listing the missing information")
        if not [q for q in ticket.clarification_questions if q.strip()]:
            errors.append(
                "clarification_questions: route='ask_user' requires at least one clarification question"
            )
    return errors


def wrapped_ticket_hint(raw: dict[str, Any]) -> list[str]:
    """
    Detect output that wraps the ticket in one key (e.g. copying the JSON schema layout: {"properties": {...}}).

    It does not unwrap: it only returns a specific error so the LLM can fix it on retry.
    """
    ticket_keys = set(IssueTicket.model_fields)
    for key, value in raw.items():
        if key in ticket_keys or not isinstance(value, dict):
            continue
        if len(ticket_keys & value.keys()) >= 3:
            return [
                f"(root): the ticket fields are wrapped in the key '{key}'. "
                "Put type, priority, route... directly at the top level of the JSON object "
                "and do not return the JSON schema."
            ]
    return []


def format_pydantic_error(error: dict[str, Any]) -> str:
    location = ".".join(str(part) for part in error.get("loc", ())) or "(root)"
    message = f"{location}: {error.get('msg', 'invalid')}"
    value = error.get("input")
    if error.get("type") != "missing" and isinstance(value, (str, int, float, bool)):
        shown = repr(value)
        message += f" (got {shown[:80]})"
    return message


def route_message(ticket: IssueTicket) -> str:
    if ticket.route == "ask_user":
        questions = [q for q in ticket.clarification_questions if q.strip()]
        return "More information is needed before this issue can be handled: " + " ".join(questions)
    if ticket.route == "answer":
        sources = f" Sources: {', '.join(ticket.affected_files)}." if ticket.affected_files else ""
        return f"Answer based on evidence gathered from the repository: {ticket.summary}{sources}"
    if ticket.route == "reject":
        reason = ticket.summary or ticket.priority_reason
        return (
            f"The request was not processed because it is out of scope or invalid. {reason} "
            "Please submit an issue related to this repository."
        ).strip()
    # fix / implement: the agent only triages for now; it has no tools to edit files or run tests.
    return (
        f"Classified the issue as '{ticket.type}' (priority {ticket.priority}) and routed it to "
        f"the '{ticket.route}' branch with a plan in the ticket. The agent has NOT changed code or run "
        "tests: the current version only triages and plans."
    )
