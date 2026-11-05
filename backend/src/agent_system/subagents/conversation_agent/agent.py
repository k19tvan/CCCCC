"""Chats naturally with the user; it calls no tools and has no repository access."""

from __future__ import annotations

from typing import Sequence

from agent_system.core.llm.client import LLMClient, LLMMessage
from agent_system.core.trace import TraceRecorder

CONVERSATION_MARKER = "[CONVERSATION]"
MAX_HISTORY_MESSAGES = 20

SYSTEM_PROMPT = f"""{CONVERSATION_MARKER}
You are the AI assistant of the Issue Agent system. Chat naturally and briefly. Reply in the user's language.
You cannot access the repository while chatting. If the user wants to handle a bug, a feature request or a
question about the source code, suggest that they describe it in more detail so the system can take the task."""


class ConversationAgent:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def run(
        self,
        history: Sequence[dict[str, str]],
        user_text: str,
        trace: TraceRecorder,
    ) -> str:
        messages = [
            LLMMessage("system", SYSTEM_PROMPT),
            *(LLMMessage(m["role"], m["content"]) for m in history[-MAX_HISTORY_MESSAGES:]),
            LLMMessage("user", user_text),
        ]
        response = self._llm.complete(messages)
        trace.add("conversation_reply", chars=len(response.text))
        return response.text.strip()
