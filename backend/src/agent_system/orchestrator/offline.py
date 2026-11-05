"""
LLM test double for LLM_PROVIDER=fake (chat + issue).

Router and conversation prompts are told apart by the marker in the system prompt; every other prompt
goes to the issue agent's OfflineTriageLLM. It is deterministic and does not reflect real LLM quality.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from agent_system.core.llm.client import LLMMessage, LLMResponse
from agent_system.orchestrator.router import ISSUE, CONVERSATION, ROUTE_WIRE, ROUTER_MARKER
from agent_system.subagents.conversation_agent import CONVERSATION_MARKER
from agent_system.subagents.issue_agent.offline_llm import OfflineTriageLLM

# Greeting phrases include Vietnamese ones to support Vietnamese user input.
_CHAT_HINT = re.compile(
    r"^\s*(xin chào|chào|hello|hi|hey|cảm ơn|thanks|thank you|bạn là ai|bạn làm được gì|who are you"
    r"|what can you do)\b[\s!?.,]*$",
    re.IGNORECASE,
)


class OfflineAssistantLLM:
    name = "offline"
    model = "offline-rules"

    def __init__(self) -> None:
        self._triage = OfflineTriageLLM()

    def complete(
        self,
        messages: Sequence[LLMMessage],
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        system = messages[0].content if messages else ""
        if ROUTER_MARKER in system:
            return LLMResponse(text=json.dumps({"route": ROUTE_WIRE[_offline_route(messages[-1].content)]}))
        if CONVERSATION_MARKER in system:
            return LLMResponse(text=_offline_reply(messages[-1].content))
        return self._triage.complete(messages, response_schema)


def _offline_route(user_block: str) -> str:
    message = user_block.rsplit("<message>", 1)[-1].split("</message>", 1)[0]
    return CONVERSATION if _CHAT_HINT.search(message.strip()) else ISSUE


def _offline_reply(user_text: str) -> str:
    return (
        "[Offline mode] Message received. Configure a real LLM provider to get full answers. "
        f"You said: {user_text.strip()[:200]}"
    )
