"""Route consistency rules between ticket type, missing_info and route."""

from __future__ import annotations

from agent_system.core.contracts import IssueTicket

# Routes allowed for each issue type.
ALLOWED_ROUTES: dict[str, frozenset[str]] = {
    "bug": frozenset({"fix", "ask_user", "reject"}),
    "feature": frozenset({"implement", "ask_user", "reject"}),
    "refactor": frozenset({"implement", "fix", "ask_user", "reject"}),
    "docs": frozenset({"implement", "answer", "ask_user", "reject"}),
    "question": frozenset({"answer", "ask_user", "reject"}),
    "invalid": frozenset({"reject", "ask_user"}),
}


def check_route(ticket: IssueTicket) -> list[str]:
    """Return route consistency errors; empty if the route is valid."""
    errors: list[str] = []
    route = ticket.route

    if ticket.missing_info and route != "ask_user":
        errors.append(
            f"missing_info is not empty, so route must be 'ask_user', got '{route}'"
        )

    if route == "ask_user":
        if not ticket.missing_info:
            errors.append("route='ask_user' requires missing_info to state the missing information")
        if not ticket.clarification_questions:
            errors.append("route='ask_user' requires at least 1 question in clarification_questions")

    if route not in ALLOWED_ROUTES[ticket.type]:
        allowed = ", ".join(sorted(ALLOWED_ROUTES[ticket.type]))
        errors.append(
            f"type='{ticket.type}' does not go with route='{route}' (allowed: {allowed})"
        )

    if route == "reject" and not ticket.summary.strip():
        errors.append("route='reject' requires a summary stating the reason for the rejection")

    return errors


def route_reason(ticket: IssueTicket) -> str:
    """Short, ticket-specific reason for the chosen route (for traces and messages)."""
    n_files = len(ticket.affected_files)
    files = f"{n_files} affected file(s)" if n_files else "no affected files identified"

    if ticket.route == "ask_user":
        missing = "; ".join(ticket.missing_info) or "not specified"
        return f"{len(ticket.missing_info)} item(s) missing ({missing}): asking the user instead of guessing."
    if ticket.route == "answer":
        return f"Issue of type '{ticket.type}' is a question about the repo ({files}): answering from tool evidence."
    if ticket.route == "fix":
        return f"Issue of type '{ticket.type}', priority {ticket.priority}, enough information ({files}): routed to the fix branch."
    if ticket.route == "implement":
        return f"Issue of type '{ticket.type}', priority {ticket.priority}, enough information ({files}): routed to the implement branch."
    return f"Issue of type '{ticket.type}' was rejected: {ticket.summary.strip() or 'no reason given'}"
