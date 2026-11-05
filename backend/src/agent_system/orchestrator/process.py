"""
Entry points of the orchestrator.

- `process_issue(request)`: handle one issue directly (CLI, evals, API /api/issues).
- `run_chat_turn(turn)`: handle one chat message; the orchestrator decides between issue and conversation.

Hooks for offline tests and evals:
- `set_llm_client(FakeLLM([...]))` replaces the LLM for every call; `set_llm_client(None)` restores the default.
- Or pass it directly: `process_issue(request, llm=..., registry=...)`.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence

from agent_system.core.config import config
from agent_system.core.contracts import IssueRequest, IssueResult
from agent_system.core.llm.client import FakeLLM, LLMClient, LLMError, create_llm_client
from agent_system.core.trace import TraceRecorder
from agent_system.orchestrator.graph import ChatMessage, build_orchestrator
from agent_system.orchestrator.offline import OfflineAssistantLLM
from agent_system.orchestrator.router import ISSUE

__all__ = [
    "ChatOutcome",
    "ChatTurn",
    "FakeLLM",
    "get_llm_client",
    "process_issue",
    "run_chat_turn",
    "set_llm_client",
]

logger = logging.getLogger(__name__)

_llm_override: LLMClient | None = None


def set_llm_client(client: LLMClient | None) -> None:
    """Use `client` for every call (tests/evals). None restores the env-based configuration."""
    global _llm_override
    _llm_override = client


def get_llm_client() -> LLMClient:
    """
    Return the active LLM client: the one set via the hook, or one created from LLM_PROVIDER.

    LLM_PROVIDER=fake uses OfflineAssistantLLM (deterministic rules, no network).

    Raises:
        LLMConfigError: Unsupported provider or missing configuration.
    """
    if _llm_override is not None:
        return _llm_override
    if config.LLM_PROVIDER.strip().lower() == "fake":
        return OfflineAssistantLLM()
    return create_llm_client()


def process_issue(
    request: IssueRequest,
    *,
    llm: LLMClient | None = None,
    registry: Any | None = None,
) -> IssueResult:
    """
    Process one issue through the orchestrator. Always returns an IssueResult (errors are not raised).

    Args:
        request: Issue from the application.
        llm: LLM client for this call (takes precedence over set_llm_client).
        registry: ToolRegistry for the issue agent (defaults to `default_registry`).
    """
    run_id = uuid.uuid4().hex[:12]
    if not request.request_id:
        request = request.model_copy(update={"request_id": f"req-{uuid.uuid4().hex[:12]}"})
    trace = TraceRecorder(run_id)

    try:
        final_state = build_orchestrator().invoke(
            {
                "llm": llm or get_llm_client(),
                "registry": registry,
                "trace": trace,
                "history": [],
                "user_text": request.description,
                "repo_path": request.repo_path,
                "route": ISSUE,
                "request": request,
            }
        )
        return final_state["result"]
    except LLMError as exc:
        # Usually a configuration error while creating the client (missing API key, wrong provider)
        trace.add("llm_error", error_type=exc.kind, error=str(exc))
        message = f"Could not initialise the LLM ({exc.kind}): {exc}"
    except Exception as exc:
        logger.exception("process_issue failed unexpectedly (run=%s)", run_id)
        trace.add("unexpected_error", error=f"{type(exc).__name__}: {exc}")
        message = f"Internal error while processing the issue: {type(exc).__name__}: {exc}"

    trace.add("finish", status="failed")
    return IssueResult(status="failed", ticket=None, message=message, run_id=run_id, trace=trace.steps)


@dataclass(frozen=True)
class ChatTurn:
    """One chat turn: the new message, previous history, the repo in use and the issue awaiting clarification."""

    user_text: str
    history: Sequence[ChatMessage]
    repo_path: str | None
    pending: IssueRequest | None = None


@dataclass
class ChatOutcome:
    route: str
    reply: str
    run_id: str
    result: IssueResult | None = None
    pending: IssueRequest | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)


def run_chat_turn(
    turn: ChatTurn,
    *,
    llm: LLMClient | None = None,
    registry: Any | None = None,
) -> ChatOutcome:
    """
    Process one chat message. If an issue is awaiting clarification, the message is treated as the answer
    and goes straight to the issue agent; otherwise the orchestrator routes it.
    """
    run_id = uuid.uuid4().hex[:12]
    trace = TraceRecorder(run_id)
    state_in: dict[str, Any] = {
        "llm": llm or get_llm_client(),
        "registry": registry,
        "trace": trace,
        "history": list(turn.history),
        "user_text": turn.user_text,
        "repo_path": turn.repo_path,
    }
    if turn.pending is not None:
        clarified = "\n".join(t for t in (turn.pending.clarified_text, turn.user_text) if t)
        state_in["request"] = turn.pending.model_copy(update={"clarified_text": clarified})
        state_in["route"] = ISSUE

    try:
        final_state = build_orchestrator().invoke(state_in)
    except LLMError as exc:
        trace.add("llm_error", error_type=exc.kind, error=str(exc))
        return _failed_outcome(run_id, state_in, f"Could not initialise the LLM ({exc.kind}): {exc}", trace)
    except Exception as exc:
        logger.exception("run_chat_turn failed unexpectedly (run=%s)", run_id)
        trace.add("unexpected_error", error=f"{type(exc).__name__}: {exc}")
        return _failed_outcome(run_id, state_in, f"Internal error: {type(exc).__name__}: {exc}", trace)

    result = final_state.get("result")
    pending = None
    if result is not None and result.status == "needs_user":
        pending = final_state.get("request")
    return ChatOutcome(
        route=final_state["route"],
        reply=final_state["reply"],
        run_id=run_id,
        result=result,
        pending=pending,
        trace=trace.steps,
    )


def _failed_outcome(
    run_id: str,
    state_in: dict[str, Any],
    message: str,
    trace: TraceRecorder,
) -> ChatOutcome:
    trace.add("finish", status="failed")
    return ChatOutcome(
        route=state_in.get("route", ""),
        reply=message,
        run_id=run_id,
        result=None,
        pending=None,
        trace=trace.steps,
    )
