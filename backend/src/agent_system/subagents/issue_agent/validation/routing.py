"""
Route decision hook for tickets.

Loaded dynamically by the issue agent via:
ROUTE_DECIDER_HOOK = ("agent_system.subagents.issue_agent.validation.routing", "decide_route")
"""

from __future__ import annotations

from agent_system.core.contracts import IssueTicket, Route
from agent_system.subagents.issue_agent.validation.route_guard import check_route, route_reason


def decide_route(ticket: IssueTicket) -> tuple[Route, str]:
    """
    Decide the route and its reason from the business rules.

    Pure function: no LLM calls, no retries.
    """
    errors = check_route(ticket)
    if errors and ticket.missing_info and ticket.route != "ask_user":
        return "ask_user", "missing_info is not empty, so the route is changed to 'ask_user'"
    return ticket.route, route_reason(ticket)
