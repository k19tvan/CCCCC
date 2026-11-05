"""Structured per-run trace shared by the orchestrator and subagents.

Each step is a JSON-serializable dict with `step`, `seq`, `ts` and step-specific data. Steps are also
logged so a run can be followed by run_id.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class TraceRecorder:
    """
    Records a structured trace for one run and also logs each step.

    Only the `route` step carries route keys (`selected`, `proposed`), which evals read to check
    routing decisions.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.steps: list[dict[str, Any]] = []

    def add(self, step: str, **data: Any) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "step": step,
            "seq": len(self.steps) + 1,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            **data,
        }
        self.steps.append(entry)
        logger.info(
            "run=%s step=%s", self.run_id, step, extra={"run_id": self.run_id, "trace": entry}
        )
        return entry


def ms_since(started: float) -> int:
    """Milliseconds elapsed since the `time.perf_counter()` value `started`."""
    return int((time.perf_counter() - started) * 1000)
