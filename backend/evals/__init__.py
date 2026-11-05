"""Evaluation and benchmark suite for the AI Issue Agent (UC-01)."""

import sys
from pathlib import Path

# Make src and agent_system importable when running directly without PYTHONPATH
_repo_root = Path(__file__).resolve().parent.parent
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from evals.evaluator import CaseEvalResult, EvalCase, evaluate_ticket_against_criteria

_RUNNER_EXPORTS = {
    "EXIT_EVAL_ALL_PASSED",
    "EXIT_EVAL_BLOCKED",
    "EXIT_EVAL_CONFIG_ERROR",
    "EXIT_EVAL_FAILED",
    "TARGET_TYPE_ACCURACY",
    "compute_eval_metrics",
    "load_eval_cases",
    "run_evals",
    "run_single_case",
}


def __getattr__(name: str):
    if name in _RUNNER_EXPORTS:
        import evals.runner as _runner

        return getattr(_runner, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "EvalCase",
    "CaseEvalResult",
    "evaluate_ticket_against_criteria",
    "load_eval_cases",
    "run_single_case",
    "compute_eval_metrics",
    "run_evals",
    "TARGET_TYPE_ACCURACY",
    "EXIT_EVAL_ALL_PASSED",
    "EXIT_EVAL_FAILED",
    "EXIT_EVAL_BLOCKED",
    "EXIT_EVAL_CONFIG_ERROR",
]
