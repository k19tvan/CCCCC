"""Tool-call tracing for the issue agent."""

from __future__ import annotations

import time
from typing import Any

from agent_system.core.trace import TraceRecorder, ms_since


class TracingRegistry:
    """
    Proxy around ToolRegistry: records a trace entry for every tool call without changing tool behavior.

    Tool errors (exception or `success=False`) are collected so the agent can warn the user.
    """

    def __init__(self, inner: Any, trace: TraceRecorder) -> None:
        self._inner = inner
        self._trace = trace
        self.tools_called: list[str] = []
        self.errors: list[str] = []

    def call(self, name: str, tool_input: Any, **kwargs: Any) -> Any:
        if name not in self.tools_called:
            self.tools_called.append(name)
        self._trace.add("tool_call", tool=name, input=_summarize_tool_input(tool_input))
        started = time.perf_counter()
        try:
            result = self._inner.call(name, tool_input, **kwargs)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._trace.add("tool_error", tool=name, error=error, duration_ms=ms_since(started))
            self.errors.append(f"{name}: {error}")
            raise

        success = bool(getattr(result, "success", True))
        error_msg = getattr(result, "error", None)
        self._trace.add(
            "tool_result",
            tool=name,
            success=success,
            error=error_msg,
            output=_summarize_tool_output(result),
            duration_ms=ms_since(started),
        )
        if not success:
            self.errors.append(f"{name}: {error_msg}")
        return result

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


def _summarize_tool_input(tool_input: Any) -> dict[str, Any]:
    if hasattr(tool_input, "model_dump"):
        return tool_input.model_dump(mode="json", exclude={"repo_path"})
    return {"value": str(tool_input)[:200]}


def _summarize_tool_output(result: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if isinstance(getattr(result, "files", None), list):
        summary["file_count"] = len(result.files)
    if isinstance(getattr(result, "matches", None), list):
        summary["match_count"] = len(result.matches)
    if getattr(result, "file_path", None):
        summary["file_path"] = result.file_path
    if isinstance(getattr(result, "truncated", None), bool):
        summary["truncated"] = result.truncated
    return summary
