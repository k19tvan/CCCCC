"""
Common contract for every subagent the orchestrator calls.

A subagent takes an issue and the run's trace and always returns an IssueResult; how it works
internally (here: a LangGraph graph) is up to the subagent.
"""

from __future__ import annotations

from typing import Protocol

from agent_system.core.contracts import IssueRequest, IssueResult
from agent_system.core.trace import TraceRecorder


class Subagent(Protocol):
    def run(self, request: IssueRequest, trace: TraceRecorder) -> IssueResult: ...
