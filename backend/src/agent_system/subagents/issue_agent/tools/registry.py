"""Tool registry for the issue agent.

The registry gives a single interface for calling tools; the agent calls tools through it instead
of importing each one directly.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from agent_system.core.contracts import ToolInput, ToolOutput

logger = logging.getLogger(__name__)

# A tool takes a ToolInput (or subclass) and returns a ToolOutput (or subclass)
ToolFn = Callable[[Any], Any]


class ToolRegistry:
    """
    Registry of all tools in the system.

    Usage:
        registry = ToolRegistry()
        registry.register("list_files", list_files_fn, description="List files in the repo")
        result = registry.call("list_files", ListFilesInput(...))
    """

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        fn: ToolFn,
        description: str = "",
        input_schema: type[ToolInput] | None = None,
        output_schema: type[ToolOutput] | None = None,
    ) -> None:
        """Register a tool under a unique name; re-registering a name overwrites it."""
        if name in self._tools:
            logger.warning("Tool '%s' already exists in the registry, overwriting.", name)
        self._tools[name] = {
            "fn": fn,
            "description": description,
            "input_schema": input_schema,
            "output_schema": output_schema,
        }
        logger.debug("Registered tool: %s", name)

    def call(self, name: str, tool_input: Any, **kwargs: Any) -> Any:
        """
        Call a registered tool by name; extra kwargs are passed to the tool function.

        Raises:
            KeyError: If the tool is not registered.
            Exception: Any error raised by the tool function is logged and re-raised.
        """
        if name not in self._tools:
            available = list(self._tools.keys())
            raise KeyError(
                f"Tool '{name}' does not exist. Available tools: {available}"
            )

        tool_entry = self._tools[name]
        fn = tool_entry["fn"]

        logger.debug("Calling tool '%s' with input: %s", name, tool_input)
        try:
            result = fn(tool_input, **kwargs)
            logger.debug("Tool '%s' returned: %s", name, result)
            return result
        except Exception as exc:
            logger.error("Tool '%s' failed: %s", name, exc, exc_info=True)
            raise

    def list_tools(self) -> list[dict[str, Any]]:
        """Return all registered tools as dicts with keys: name, description, input_schema, output_schema."""
        return [
            {
                "name": name,
                "description": entry["description"],
                "input_schema": entry["input_schema"],
                "output_schema": entry["output_schema"],
            }
            for name, entry in self._tools.items()
        ]

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={list(self._tools.keys())})"


def _build_default_registry() -> ToolRegistry:
    """Build a ToolRegistry with the three default tools registered."""
    from agent_system.subagents.issue_agent.tools.list_files import list_files
    from agent_system.subagents.issue_agent.tools.read_file import read_file
    from agent_system.subagents.issue_agent.tools.search_code import search_code
    from agent_system.core.contracts import (
        ListFilesInput,
        ListFilesOutput,
        ReadFileInput,
        ReadFileOutput,
        SearchCodeInput,
        SearchCodeOutput,
    )

    registry = ToolRegistry()

    registry.register(
        name="list_files",
        fn=list_files,
        description=(
            "List files in a repo/directory. "
            "Skips cache and virtualenv directories and binary files. "
            "Returns paths relative to repo_path."
        ),
        input_schema=ListFilesInput,
        output_schema=ListFilesOutput,
    )

    registry.register(
        name="read_file",
        fn=read_file,
        description=(
            "Read the contents of a file in the repo. "
            "Only reads files inside repo_path and blocks path traversal. "
            "Limits the size and handles encoding automatically."
        ),
        input_schema=ReadFileInput,
        output_schema=ReadFileOutput,
    )

    registry.register(
        name="search_code",
        fn=search_code,
        description=(
            "Search for strings/symbols in the repo. "
            "Returns path, line number and an excerpt for each match. "
            "Supports plain and regex search."
        ),
        input_schema=SearchCodeInput,
        output_schema=SearchCodeOutput,
    )

    logger.info("Default ToolRegistry initialised with %d tool(s).", len(registry.list_tools()))
    return registry


default_registry = _build_default_registry()
