"""
Issue triage subagent (UC-01), implemented with LangGraph.

Use `create_issue_agent(llm)` to get a subagent wired with the default validator, route decider and
tool registry, or build `IssueAgent(...)` directly to swap individual parts (tests/evals).
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from agent_system.core.config import config
from agent_system.core.llm.client import LLMClient
from agent_system.subagents.issue_agent.graph import IssueAgent, SkillFn
from agent_system.subagents.issue_agent.skills.gather_context import gather_context
from agent_system.subagents.issue_agent.tools.registry import default_registry

__all__ = ["IssueAgent", "create_issue_agent"]

logger = logging.getLogger(__name__)

# Hook functions of the validation module; see docs/decisions/02_agent_loop_interfaces.md
VALIDATOR_HOOK = ("agent_system.subagents.issue_agent.validation.issue_schema", "validate_ticket")
ROUTE_DECIDER_HOOK = ("agent_system.subagents.issue_agent.validation.routing", "decide_route")


def create_issue_agent(
    llm: LLMClient,
    registry: Any | None = None,
    skill: SkillFn | None = None,
) -> IssueAgent:
    """
    Create an IssueAgent with the default configuration.

    Args:
        llm: LLM client for the subagent.
        registry: ToolRegistry (defaults to `default_registry`).
        skill: Context-gathering skill (defaults to `gather_context`).
    """
    return IssueAgent(
        llm=llm,
        registry=registry if registry is not None else default_registry,
        skill=skill if skill is not None else gather_context,
        validator=_load_hook(*VALIDATOR_HOOK),
        route_decider=_load_hook(*ROUTE_DECIDER_HOOK),
        max_retries=config.MAX_RETRY_ATTEMPTS,
    )


def _load_hook(module_name: str, attr: str) -> Any:
    """
    Load a hook function from the validation module if the module exists.

    Only skipped when that module itself is missing; import errors raised inside the module still propagate.
    """
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return None
        raise
    hook = getattr(module, attr, None)
    if hook is None:
        logger.warning("Module %s does not define '%s'; skipping hook.", module_name, attr)
    return hook
