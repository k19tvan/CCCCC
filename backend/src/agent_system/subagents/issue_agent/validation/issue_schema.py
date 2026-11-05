"""
Schema validation hook for tickets.

Loaded dynamically by the issue agent via:
VALIDATOR_HOOK = ("agent_system.subagents.issue_agent.validation.issue_schema", "validate_ticket")
"""

from __future__ import annotations

from typing import Any

from agent_system.core.contracts import ValidationResult
from agent_system.subagents.issue_agent.validation.ticket_guard import validate_ticket as _guard_validate_ticket


def validate_ticket(raw: Any, repo_path: str) -> ValidationResult:
    """
    Validate a ticket and return a ValidationResult.

    - Schema/route errors: is_valid=False, errors for the agent loop to feed back to the LLM.
    - Invalid or nonexistent files go into invalid_files + warnings without failing validation
      when the other fields are valid (is_valid=True).
    - Normalized valid files are in valid_files.
    """
    outcome = _guard_validate_ticket(raw, repo_path)
    return outcome.result
