"""
Adapter between the interfaces (CLI, server) and the backend public API.

Callers can inject a processor (Callable[[IssueRequest], IssueResult]) so unit tests can use test doubles
without global mocking. It never falls back to a fake backend or keyword-based classification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from agent_system.core.contracts import IssueRequest, IssueResult

if TYPE_CHECKING:
    from agent_system.orchestrator.process import ChatOutcome, ChatTurn


class BackendMissingError(RuntimeError):
    """Raised when the backend orchestrator is not implemented or cannot be found."""

    pass


class BackendExecutionError(RuntimeError):
    """Raised when the backend fails while running the pipeline."""

    pass


def resolve_processor() -> Callable[[IssueRequest], IssueResult]:
    """
    Load `process_issue` from `agent_system.orchestrator`.

    Raises:
        BackendMissingError: The orchestrator module or process_issue is missing.
        BackendExecutionError: The module exists but one of its dependencies fails to import.
    """
    try:
        from agent_system.orchestrator import process_issue

        if not callable(process_issue):
            raise BackendMissingError(
                "The 'process_issue' object in agent_system.orchestrator is not callable."
            )
        return process_issue
    except ModuleNotFoundError as e:
        # Distinguish a missing orchestrator module from a missing internal dependency
        if e.name == "agent_system.orchestrator" or (
            e.name and e.name.endswith(".orchestrator")
        ):
            raise BackendMissingError(
                "Backend orchestrator is not implemented: "
                "'agent_system.orchestrator.process_issue' was not found."
            ) from e
        raise BackendExecutionError(
            f"Failed to import a dependency of the backend orchestrator: {e}"
        ) from e
    except AttributeError as e:
        raise BackendMissingError(
            f"Module agent_system.orchestrator does not provide 'process_issue': {e}"
        ) from e
    except Exception as e:
        raise BackendExecutionError(
            f"Unknown error while initialising the backend orchestrator: {e}"
        ) from e


def call_agent(
    request: IssueRequest,
    processor: Callable[[IssueRequest], IssueResult] | None = None,
) -> IssueResult:
    """
    Send an IssueRequest to the agent through the adapter.

    Args:
        request: A valid issue request.
        processor: Optional callable taking an IssueRequest and returning an IssueResult.
                   If None, the real backend is loaded via resolve_processor().

    Raises:
        BackendMissingError: The real backend does not exist (when processor is None).
        BackendExecutionError: The backend raised an error or returned the wrong type.
    """
    if processor is not None:
        target_fn = processor
    else:
        target_fn = resolve_processor()

    try:
        result = target_fn(request)
        if not isinstance(result, IssueResult):
            raise BackendExecutionError(
                f"Backend returned a result of the wrong type: {type(result)}. "
                "Expected an instance of agent_system.core.contracts.IssueResult."
            )
        return result
    except (BackendMissingError, BackendExecutionError):
        raise
    except Exception as e:
        raise BackendExecutionError(
            f"Error while running process_issue: {e}"
        ) from e


def resolve_chat_turn() -> Callable[[ChatTurn], ChatOutcome]:
    """
    Load `run_chat_turn` (chat path: text -> router -> issue agent) from the orchestrator.

    Raises:
        BackendMissingError: The orchestrator is missing or does not export run_chat_turn.
    """
    try:
        from agent_system.orchestrator import run_chat_turn
    except ImportError as e:
        raise BackendMissingError(
            f"The orchestrator chat path is not available: {e}"
        ) from e
    return run_chat_turn


def call_chat_turn(
    user_text: str,
    repo_path: str | None,
    pending: IssueRequest | None = None,
    chat_turn: Callable[[ChatTurn], ChatOutcome] | None = None,
) -> ChatOutcome:
    """
    Send a natural-language message through the chat path, the same flow the UI uses.

    Args:
        user_text: The user's message.
        repo_path: Repository analysed by the chat session.
        pending: Issue awaiting clarification; if set, the message is treated as the answer.
        chat_turn: Optional callable replacing run_chat_turn (for unit tests).
                   If None, the real backend is loaded via resolve_chat_turn().

    Returns:
        ChatOutcome with the route, the reply, and the IssueResult if the message went to the issue agent.

    Raises:
        BackendMissingError: The real backend does not exist (when chat_turn is None).
        BackendExecutionError: The backend raised an error or returned the wrong type.
    """
    from agent_system.orchestrator.process import ChatOutcome, ChatTurn

    target_fn = chat_turn if chat_turn is not None else resolve_chat_turn()
    try:
        outcome = target_fn(
            ChatTurn(user_text=user_text, history=[], repo_path=repo_path, pending=pending)
        )
        if not isinstance(outcome, ChatOutcome):
            raise BackendExecutionError(
                f"Backend returned a result of the wrong type: {type(outcome)}. "
                "Expected an instance of agent_system.orchestrator.process.ChatOutcome."
            )
        return outcome
    except (BackendMissingError, BackendExecutionError):
        raise
    except Exception as e:
        raise BackendExecutionError(
            f"Error while running run_chat_turn: {e}"
        ) from e
