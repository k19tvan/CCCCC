"""
LangGraph graph for the UC-01 issue triage subagent.

Flow:

    receive_request → gather_context → generate ──(LLM error)────────────────────────► END
                                          ▲  └──► validate ──(valid)──► filter_files → route → END
                                          │               ├─(invalid, attempts left)──► retry ─┐
                                          └───────────────┼────────────────────────────────────┘
                                                          └─(invalid, no attempts)──► give_up → END

Rules:
- Only this graph controls retries. The validator and route decider are pure functions: no LLM calls, no retries.
- LLM network/API/timeout failures (`LLMError`) end the run immediately with status="failed"; they are
  not schema errors and are not retried as such.
- Every step is recorded in the trace (same run_id) so a run can be followed end to end.
- Every terminating branch puts `result` into the state; `IssueAgent.run` returns that value.
"""

from __future__ import annotations

import logging
import operator
import time
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from agent_system.core.config import config
from agent_system.core.contracts import (
    GatherContextInput,
    GatherContextOutput,
    IssueRequest,
    IssueResult,
    IssueTicket,
    Route,
    ValidationResult,
)
from agent_system.core.llm.client import LLMClient, LLMError, LLMMessage, parse_json_object
from agent_system.core.redaction import redact_secrets
from agent_system.core.trace import TraceRecorder, ms_since
from agent_system.subagents.issue_agent.prompts import (
    build_initial_messages,
    build_retry_messages,
    ticket_json_schema,
)
from agent_system.subagents.issue_agent.skills.gather_context import gather_context
from agent_system.subagents.issue_agent.ticket_rules import (
    check_ticket_rules,
    format_pydantic_error,
    route_message,
    wrapped_ticket_hint,
)
from agent_system.subagents.issue_agent.tracing import TracingRegistry

logger = logging.getLogger(__name__)

TicketValidator = Callable[[dict[str, Any], str], ValidationResult]
"""(raw_ticket, repo_path) -> ValidationResult. Pure function provided by the validation module."""

RouteDecider = Callable[[IssueTicket], tuple[Route, str]]
"""valid ticket -> (selected route, short reason). Pure function provided by the validation module."""

SkillFn = Callable[[GatherContextInput, Any], GatherContextOutput]

MAX_TRACE_ERRORS = 10

# Each attempt runs 3 nodes (generate, validate, retry); EXTRA_STEPS covers the remaining graph nodes.
STEPS_PER_ATTEMPT = 3
EXTRA_STEPS = 10


class TriageState(TypedDict, total=False):
    request: IssueRequest
    trace: TraceRecorder
    warnings: Annotated[list[str], operator.add]  # nodes only append warnings, never overwrite
    messages: list[LLMMessage]
    attempt: int  # number of LLM calls made so far
    response_text: str  # raw output of the latest LLM call
    errors: list[str]  # errors from the latest validation
    validation: ValidationResult | None
    ticket: IssueTicket | None  # None when the latest validation failed
    result: IssueResult  # only set once the graph has finished


class IssueAgent:
    """
    Triage subagent: coordinates the LLM, skill/tools and validator for one issue.

    Args:
        llm: LLM client (real or FakeLLM).
        registry: ToolRegistry the skill uses to call tools.
        skill: Context-gathering skill, `gather_context` by default.
        validator: Ticket validator (validation module). None = only check the contract schema.
        route_decider: Route rules (validation module). None = keep the route the ticket proposes.
        max_retries: Maximum regenerations after the first attempt (2 => at most 3 generations).
    """

    def __init__(
        self,
        llm: LLMClient,
        registry: Any,
        skill: SkillFn = gather_context,
        validator: TicketValidator | None = None,
        route_decider: RouteDecider | None = None,
        max_retries: int = config.MAX_RETRY_ATTEMPTS,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.llm = llm
        self.registry = registry
        self.skill = skill
        self.validator = validator
        self.route_decider = route_decider
        self.max_retries = max_retries
        self.max_attempts = max_retries + 1
        self.graph = self._build_graph()

    def run(self, request: IssueRequest, trace: TraceRecorder) -> IssueResult:
        final_state = self.graph.invoke(
            {"request": request, "trace": trace, "warnings": [], "attempt": 0},
            {"recursion_limit": STEPS_PER_ATTEMPT * self.max_attempts + EXTRA_STEPS},
        )
        return final_state["result"]

    def _build_graph(self):
        graph = StateGraph(TriageState)
        graph.add_node("receive_request", self._receive_request)
        graph.add_node("gather_context", self._gather_context)
        graph.add_node("generate", self._generate)
        graph.add_node("validate", self._validate_ticket)
        graph.add_node("retry", self._retry)
        graph.add_node("give_up", self._give_up)
        graph.add_node("filter_files", self._filter_files)
        graph.add_node("route", self._route)

        graph.add_edge(START, "receive_request")
        graph.add_edge("receive_request", "gather_context")
        graph.add_edge("gather_context", "generate")
        graph.add_conditional_edges(
            "generate", self._after_generate, {"validate": "validate", "end": END}
        )
        graph.add_conditional_edges(
            "validate",
            self._after_validate,
            {"filter_files": "filter_files", "retry": "retry", "give_up": "give_up"},
        )
        graph.add_edge("retry", "generate")
        graph.add_edge("filter_files", "route")
        graph.add_edge("route", END)
        graph.add_edge("give_up", END)
        return graph.compile()

    def _receive_request(self, state: TriageState) -> dict[str, Any]:
        request = state["request"]
        state["trace"].add(
            "request_received",
            request_id=request.request_id,
            title=redact_secrets(request.title)[:200],
            has_clarified_text=bool(request.clarified_text),
            repo_path=request.repo_path,
        )
        return {}

    def _gather_context(self, state: TriageState) -> dict[str, Any]:
        request, trace = state["request"], state["trace"]
        traced_registry = TracingRegistry(self.registry, trace)
        skill_input = GatherContextInput(
            title=request.title,
            description=request.description,
            repo_path=request.repo_path,
            clarified_text=request.clarified_text,
        )

        trace.add("skill_start", skill="gather_context")
        started = time.perf_counter()
        warnings: list[str] = []
        context: GatherContextOutput | None = None
        context_error: str | None = None
        try:
            context = self.skill(skill_input, traced_registry)
        except Exception as exc:
            context_error = f"{type(exc).__name__}: {exc}"
            logger.exception("gather_context failed (run=%s)", trace.run_id)
            trace.add(
                "skill_error",
                skill="gather_context",
                error=context_error,
                tools=traced_registry.tools_called,
                duration_ms=ms_since(started),
            )
            warnings.append(
                f"Could not gather context from the repository ({context_error}); "
                "the result may lack evidence."
            )
        else:
            trace.add(
                "skill_end",
                skill="gather_context",
                tools=traced_registry.tools_called,
                source_count=len(context.sources),
                files_found=len(context.files_found),
                keywords=context.keywords_used,
                duration_ms=ms_since(started),
            )
            warnings.extend(f"Tool error while gathering context: {e}" for e in traced_registry.errors)

        return {
            "messages": build_initial_messages(request, context, context_error),
            "warnings": warnings,
        }

    def _generate(self, state: TriageState) -> dict[str, Any]:
        trace, messages = state["trace"], state["messages"]
        attempt = state["attempt"] + 1
        trace.add(
            "llm_request",
            attempt=attempt,
            max_attempts=self.max_attempts,
            provider=getattr(self.llm, "name", type(self.llm).__name__),
            model=getattr(self.llm, "model", ""),
            message_count=len(messages),
        )
        started = time.perf_counter()
        try:
            response = self.llm.complete(messages, response_schema=ticket_json_schema())
        except LLMError as exc:
            trace.add(
                "llm_error",
                attempt=attempt,
                error_type=exc.kind,
                error=str(exc),
                duration_ms=ms_since(started),
            )
            message = (
                f"LLM call failed ({exc.kind}): {exc}. This is an LLM connection/service error, "
                "not a schema error; no ticket was created. Please try again later."
            )
            return {"attempt": attempt, "result": _failed(state, message, error_type=exc.kind)}

        trace.add(
            "llm_response",
            attempt=attempt,
            chars=len(response.text),
            usage=response.usage,
            duration_ms=ms_since(started),
        )
        return {"attempt": attempt, "response_text": response.text}

    @staticmethod
    def _after_generate(state: TriageState) -> str:
        return "end" if "result" in state else "validate"

    def _validate_ticket(self, state: TriageState) -> dict[str, Any]:
        ticket, errors, validation = self._check_response(
            state["response_text"], state["request"].repo_path
        )
        state["trace"].add(
            "validation",
            attempt=state["attempt"],
            valid=not errors,
            errors=errors[:MAX_TRACE_ERRORS],
            warnings=validation.warnings if validation else [],
            invalid_files=validation.invalid_files if validation else [],
        )
        return {"ticket": ticket, "errors": errors, "validation": validation}

    def _check_response(
        self, text: str, repo_path: str
    ) -> tuple[IssueTicket | None, list[str], ValidationResult | None]:
        """Return (ticket if valid, list of errors, validator result if any)."""
        try:
            raw = parse_json_object(text)
        except ValueError as exc:
            return None, [f"Output is not a valid JSON object: {exc}"], None

        ticket: IssueTicket | None
        try:
            ticket = IssueTicket.model_validate(raw)
        except ValidationError as exc:
            ticket = None
            errors = wrapped_ticket_hint(raw) + [format_pydantic_error(e) for e in exc.errors()]
        else:
            errors = check_ticket_rules(ticket)

        validation: ValidationResult | None = None
        if self.validator is not None:
            validation = self.validator(raw, repo_path)
            errors.extend(e for e in validation.errors if e not in errors)
            if not validation.is_valid and not validation.errors:
                errors.append("Validator marked the ticket invalid without giving a specific error")

        return (ticket if not errors else None), errors, validation

    def _after_validate(self, state: TriageState) -> str:
        if state["ticket"] is not None:
            return "filter_files"
        return "retry" if state["attempt"] < self.max_attempts else "give_up"

    def _retry(self, state: TriageState) -> dict[str, Any]:
        attempt, errors = state["attempt"], state["errors"]
        state["trace"].add(
            "retry",
            attempt=attempt,
            next_attempt=attempt + 1,
            reason="invalid_schema",
            error_count=len(errors),
        )
        retry_messages = build_retry_messages(
            state["response_text"], errors, attempt, self.max_attempts
        )
        return {"messages": state["messages"] + retry_messages}

    def _give_up(self, state: TriageState) -> dict[str, Any]:
        errors = state["errors"]
        state["trace"].add(
            "retry_exhausted", attempts=self.max_attempts, errors=errors[:MAX_TRACE_ERRORS]
        )
        message = (
            f"Could not produce a valid ticket after {self.max_attempts} generations "
            f"(1 initial + {self.max_retries} retries). Last errors: " + "; ".join(errors[:5])
        )
        return {"result": _failed(state, message, error_type="invalid_schema")}

    def _filter_files(self, state: TriageState) -> dict[str, Any]:
        ticket, validation, trace = state["ticket"], state["validation"], state["trace"]
        if validation is None:
            if ticket.affected_files:
                return {
                    "warnings": [
                        "No validation module: affected_files were not verified to exist in the repo."
                    ]
                }
            return {}

        warnings = list(validation.warnings)
        if not validation.invalid_files and validation.valid_files is None:
            return {"warnings": warnings}

        if validation.valid_files is not None:
            kept = list(dict.fromkeys(validation.valid_files))
        else:
            invalid = set(validation.invalid_files or [])
            kept = [f for f in ticket.affected_files if f not in invalid]

        removed = list(dict.fromkeys(validation.invalid_files or []))
        if removed:
            trace.add("affected_files_filtered", removed=removed, kept=kept)
            warnings.append(
                "Removed affected_files that do not exist or are outside the repo: " + ", ".join(removed)
            )
        elif set(ticket.affected_files) != set(kept):
            trace.add("affected_files_filtered", removed=[], kept=kept)

        return {
            "ticket": ticket.model_copy(update={"affected_files": kept}),
            "warnings": warnings,
        }

    def _route(self, state: TriageState) -> dict[str, Any]:
        ticket = state["ticket"]
        warnings = list(state["warnings"])
        proposed = ticket.route
        if self.route_decider is not None:
            selected, reason = self.route_decider(ticket)
        else:
            selected, reason = proposed, "Route proposed by the agent; passed schema and consistency checks"

        if selected != proposed:
            warnings.append(f"Route adjusted from '{proposed}' to '{selected}': {reason}")
            ticket = ticket.model_copy(update={"route": selected})

        state["trace"].add("route", selected=selected, proposed=proposed, reason=reason)
        status = "needs_user" if selected == "ask_user" else "completed"
        result = _finish(state["trace"], status, ticket, route_message(ticket), warnings)
        return {"ticket": ticket, "result": result}


def _failed(state: TriageState, message: str, error_type: str) -> IssueResult:
    return _finish(state["trace"], "failed", None, message, state["warnings"], error_type=error_type)


def _finish(
    trace: TraceRecorder,
    status: str,
    ticket: IssueTicket | None,
    message: str,
    warnings: list[str],
    **extra: Any,
) -> IssueResult:
    unique_warnings = list(dict.fromkeys(warnings))
    trace.add("finish", status=status, warning_count=len(unique_warnings), **extra)
    return IssueResult(
        status=status,
        ticket=ticket,
        message=message,
        warnings=unique_warnings,
        run_id=trace.run_id,
        trace=trace.steps,
    )
