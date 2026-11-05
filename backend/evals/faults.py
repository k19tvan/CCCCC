"""Fault-injection processors for the mechanism cases in cases.jsonl.

Each processor goes through the public `process_issue` API and only replaces the LLM or a tool
at the boundary via the `process_issue(request, llm=..., registry=...)` hook
(see docs/decisions/02_agent_loop_interfaces.md).

Supported `fault_injection.type` values:
- `schema_retry`: the first `attempts - 1` calls return a ticket with a bad enum, the last returns a valid ticket.
- `schema_exhaustion`: every call returns a ticket with a bad enum (the agent must stop after 3 generations).
- `tool_error`: the tool named by `tool` raises `error` on every call; the LLM uses the current configuration.

When the backend orchestrator is not available on the branch, the processor raises
BackendMissingError so the runner records a precise 'blocked' status instead of crashing.
"""

from __future__ import annotations

import functools
import json
from typing import Any, Callable, Sequence

from agent_system.interfaces.adapter import BackendMissingError
from agent_system.core.contracts import IssueRequest, IssueResult
from agent_system.subagents.issue_agent.tools.registry import default_registry

Processor = Callable[[IssueRequest], IssueResult]

SUPPORTED_FAULTS = ("schema_retry", "schema_exhaustion", "tool_error")
INVALID_PRIORITY = "P9"


class UnsupportedFaultError(ValueError):
    """The case declares an unsupported fault_injection or is missing parameters."""


class FaultyToolRegistry:
    """Wraps the real registry; the tool `tool_name` always raises OSError, other tools run normally."""

    def __init__(self, inner: Any, tool_name: str, error: str) -> None:
        self._inner = inner
        self._tool_name = tool_name
        self._error = error

    def call(self, name: str, tool_input: Any, **kwargs: Any) -> Any:
        if name == self._tool_name:
            raise OSError(self._error)
        return self._inner.call(name, tool_input, **kwargs)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._inner, attr)


def _offline_ticket(messages: Sequence[Any]) -> dict[str, Any]:
    """Deterministic valid ticket built from the prompt the agent sent."""
    try:
        from agent_system.subagents.issue_agent.offline_llm import OfflineTriageLLM
    except ModuleNotFoundError as exc:
        if exc.name == "agent_system.subagents.issue_agent.offline_llm":
            raise BackendMissingError("OfflineTriageLLM is not available on disk") from exc
        raise
    return json.loads(OfflineTriageLLM().complete(messages).text)


def _invalid_ticket(messages: Sequence[Any]) -> dict[str, Any]:
    """Structurally correct ticket with an invalid priority enum: a recoverable validation error."""
    return dict(_offline_ticket(messages), priority=INVALID_PRIORITY)


def build_fault_processor(fault: dict[str, Any]) -> Processor:
    """
    Build a processor for one `fault_injection` declaration.

    Raises:
        UnsupportedFaultError: Unsupported fault type or invalid parameters.
        BackendMissingError: If the orchestrator is not implemented.
    """
    kind = fault.get("type")
    if kind not in SUPPORTED_FAULTS:
        raise UnsupportedFaultError(
            f"fault_injection.type '{kind}' is not supported (supported: {', '.join(SUPPORTED_FAULTS)})"
        )

    try:
        from agent_system.core.llm.client import FakeLLM
        from agent_system.orchestrator import process_issue
    except ModuleNotFoundError as exc:
        if exc.name in ("agent_system.orchestrator", "agent_system.core.llm.client", "agent_system.core.llm", "agent_system"):
            exc_msg = str(exc)

            def _missing_backend_processor(req: IssueRequest) -> IssueResult:
                raise BackendMissingError(
                    f"Cannot run fault injection: backend orchestrator or FakeLLM hook is missing ({exc_msg})"
                )

            return _missing_backend_processor
        raise

    if kind == "schema_retry":
        attempts = fault.get("attempts", 2)
        if not isinstance(attempts, int) or attempts < 2:
            raise UnsupportedFaultError("schema_retry requires 'attempts' to be an integer >= 2")
        llm = FakeLLM([_invalid_ticket] * (attempts - 1) + [_offline_ticket])
        return functools.partial(process_issue, llm=llm)

    if kind == "schema_exhaustion":
        return functools.partial(process_issue, llm=FakeLLM([_invalid_ticket]))

    if kind == "tool_error":
        tool = fault.get("tool")
        if not tool:
            raise UnsupportedFaultError("tool_error requires the 'tool' field")
        registry = FaultyToolRegistry(
            default_registry, tool, str(fault.get("error") or "tool error")
        )
        return functools.partial(process_issue, registry=registry)

    raise UnsupportedFaultError(f"fault_injection.type '{kind}' is not supported")
