"""Validation and guards for the Issue Agent (UC-01): schema, route, affected_files and the orchestrator hooks."""

from __future__ import annotations

from agent_system.subagents.issue_agent.validation.ticket_guard import ValidationOutcome, validate_ticket

__all__ = ["validate_ticket", "ValidationOutcome"]
