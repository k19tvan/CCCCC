"""
LangGraph graph of the orchestrator: routes each message to one subagent.

    START ──(decide_route)──► issue_agent ──► END
                         └──► conversation_agent ──► END

`route` may be preset in the state (for example while an issue is being clarified); the routing step is then skipped.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agent_system.core.contracts import GatherContextInput, GatherContextOutput, IssueRequest, IssueResult
from agent_system.core.llm.client import LLMClient
from agent_system.core.trace import TraceRecorder
from agent_system.orchestrator.router import ISSUE, CONVERSATION, route_message
from agent_system.subagents.conversation_agent import ConversationAgent
from agent_system.subagents.issue_agent import create_issue_agent

TITLE_MAX_CHARS = 80
MISSING_REPO_REPLY = (
    "This looks like a task for the issue agent, but no repository is set for this chat. "
    "Enter the repository path below the message box, then send your request again."
)


class ChatMessage(TypedDict):
    role: str
    content: str


class OrchestratorState(TypedDict, total=False):
    llm: LLMClient
    registry: Any
    trace: TraceRecorder
    history: list[ChatMessage]
    user_text: str
    repo_path: str | None
    route: str
    request: IssueRequest
    result: IssueResult
    reply: str


def build_orchestrator():
    graph = StateGraph(OrchestratorState)
    graph.add_node("decide_route", _decide_route)
    graph.add_node(ISSUE, _run_issue)
    graph.add_node(CONVERSATION, _run_conversation)
    graph.add_edge(START, "decide_route")
    graph.add_conditional_edges("decide_route", _selected_route, [ISSUE, CONVERSATION])
    graph.add_edge(ISSUE, END)
    graph.add_edge(CONVERSATION, END)
    return graph.compile()


def _decide_route(state: OrchestratorState) -> dict[str, str]:
    if state.get("route"):
        return {}
    route = route_message(state["llm"], state["history"], state["user_text"], state["trace"])
    return {"route": route}


def _selected_route(state: OrchestratorState) -> str:
    return state["route"]


def _run_issue(state: OrchestratorState) -> dict[str, Any]:
    if not state.get("request") and not state.get("repo_path"):
        return _triage_without_repo(state)
    request = state.get("request") or IssueRequest(
        title=_title_from(state["user_text"]),
        description=state["user_text"],
        repo_path=state["repo_path"],
    )
    result = create_issue_agent(state["llm"], state.get("registry")).run(request, state["trace"])
    return {"request": request, "result": result, "reply": _issue_reply(result)}


def _no_repo_context(skill_input: GatherContextInput, registry: Any) -> GatherContextOutput:
    return GatherContextOutput(
        summary="No repository is selected for this chat session, so there is no evidence from the source code."
    )


def _triage_without_repo(state: OrchestratorState) -> dict[str, Any]:
    """
    No repo yet: still triage the issue (without evidence) so injection attempts and out-of-scope
    requests can be rejected. Only a `reject` ticket is returned; other routes need a repo, so the
    user is asked for the repository path.
    """
    request = IssueRequest(title=_title_from(state["user_text"]), description=state["user_text"])
    result = create_issue_agent(state["llm"], state.get("registry"), skill=_no_repo_context).run(
        request, state["trace"]
    )
    triaged = result.ticket.route if result.ticket else None
    state["trace"].add("missing_repo_path", triaged=triaged)
    if triaged != "reject":
        return {"reply": MISSING_REPO_REPLY}
    # There is no real repo, so no file was verified: return no affected_files
    ticket = result.ticket.model_copy(update={"affected_files": []})
    result = result.model_copy(update={"ticket": ticket})
    return {"request": request, "result": result, "reply": _issue_reply(result)}


def _run_conversation(state: OrchestratorState) -> dict[str, str]:
    reply = ConversationAgent(state["llm"]).run(state["history"], state["user_text"], state["trace"])
    return {"reply": reply}


def _title_from(text: str) -> str:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return first_line[:TITLE_MAX_CHARS] or "Issue from chat"


def _issue_reply(result: IssueResult) -> str:
    lines = [result.message.strip()]
    questions = result.ticket.clarification_questions if result.ticket else []
    if result.status == "needs_user" and questions:
        lines.append("")
        lines.extend(f"- {q}" for q in questions)
    return "\n".join(lines).strip()
