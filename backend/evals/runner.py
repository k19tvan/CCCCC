"""Runs and scores the UC-01 eval suite automatically.

Execution and scoring rules:
1. Read `cases.jsonl`, check syntax (1-indexed line numbers), duplicate IDs and an empty dataset.
   Strictly type-check every field of expected and the inputs (EvalDatasetError reports the ID and line).
2. Run each case independently through the public adapter API, continuing after errors so the runner never crashes:
   - User cases (no `fault_injection`, path=chat): title + description is one natural message,
     sent through the chat path like the UI (router -> issue_agent). If no ticket is produced
     (the router chose conversation) the case FAILS.
   - Mechanism cases (with `fault_injection`, path=mechanism): call process_issue with the fault hook from evals.faults.
     These cases test retry/exhaustion/tool errors, not real user input.
3. Never put the expected answer or hidden assertions into the agent's context.
4. Score every UC-01 criterion through evals.evaluator.
5. Compute Type Accuracy separately on the benchmark set (is_benchmark=True):
   - Pass threshold: >= 0.80 (80%).
   - Kept separate from the pass rate over all technical criteria.
   - Technical failures and misclassified cases are never removed from the denominator after the run.
   - When the benchmark did not fully run or everything is blocked for lack of a backend: accuracy = None (N/A).
6. Export console table, JSON report and CSV report in evals/reports/.
   Records the mode (test_double vs backend_live), provider/model, and whether the run is full or a subset.
7. Exit codes:
   - 0: all cases pass and Type Accuracy >= 0.80
   - 1: some case failed or Type Accuracy < 0.80
   - 2: blocked because the backend is not implemented
   - 3: configuration / JSONL dataset error
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Make src and agent_system importable
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from agent_system.interfaces.adapter import (
    BackendExecutionError,
    BackendMissingError,
    call_agent,
    call_chat_turn,
)
from agent_system.orchestrator.process import ChatOutcome, ChatTurn
from agent_system.orchestrator.router import CONVERSATION
from evals.evaluator import (
    SUPPORTED_EXPECTED_KEYS,
    VALID_PRIORITIES,
    VALID_ROUTES,
    VALID_TYPES,
    CaseEvalResult,
    EvalCase,
    evaluate_ticket_against_criteria,
)
from evals.faults import UnsupportedFaultError, build_fault_processor
from agent_system.core.config import config
from agent_system.core.contracts import IssueRequest, IssueResult

EXIT_EVAL_ALL_PASSED = 0
EXIT_EVAL_FAILED = 1
EXIT_EVAL_BLOCKED = 2
EXIT_EVAL_CONFIG_ERROR = 3

PATH_CHAT = "chat"
PATH_MECHANISM = "mechanism"
SUITES = ("all", "chat", "mechanism")

TARGET_TYPE_ACCURACY = 0.80


class EvalDatasetError(Exception):
    """Raised when the JSONL dataset is invalid."""

    pass


def load_eval_cases(cases_file: Path) -> list[EvalCase]:
    """
    Load and strictly validate a cases.jsonl dataset.

    Raises:
        EvalDatasetError: If the file does not exist, the JSON is malformed, an ID is duplicated,
                          expected is missing or has a wrong enum/type, or the dataset is empty.
    """
    if not cases_file.exists():
        raise EvalDatasetError(f"Dataset file does not exist: {cases_file}")

    with open(cases_file, "r", encoding="utf-8") as f:
        return parse_eval_cases(f, source=str(cases_file))


def parse_eval_cases(lines: Iterable[str], source: str = "dataset") -> list[EvalCase]:
    """Validate each JSONL line and return the EvalCase list; raises EvalDatasetError on invalid data."""
    cases: list[EvalCase] = []
    seen_ids: dict[str, int] = {}

    for line_num, line in enumerate(lines, 1):
        line_str = line.strip()
        if not line_str or line_str.startswith("#"):
            continue

        try:
            data = json.loads(line_str)
        except json.JSONDecodeError as e:
            raise EvalDatasetError(
                f"JSON syntax error at line {line_num}: {e.msg} (column {e.colno})"
            ) from e

        if not isinstance(data, dict):
            raise EvalDatasetError(
                f"Line {line_num}: JSON data must be an object (dict), got: {type(data).__name__}"
            )

        case_id = data.get("id")
        if not case_id or not isinstance(case_id, str) or not case_id.strip():
            raise EvalDatasetError(f"Line {line_num}: required field 'id' is missing or has the wrong type")

        case_id_str = case_id.strip()
        if case_id_str in seen_ids:
            raise EvalDatasetError(
                f"Duplicate ID '{case_id_str}' at line {line_num} "
                f"(already seen at line {seen_ids[case_id_str]})"
            )
        seen_ids[case_id_str] = line_num

        required_str_fields = ["title", "description", "category"]
        for field in required_str_fields:
            val = data.get(field)
            if not val or not isinstance(val, str) or not val.strip():
                raise EvalDatasetError(
                    f"Line {line_num} (ID: {case_id_str}): field '{field}' must not be empty and must be a string"
                )

        if "clarified_text" in data and data["clarified_text"] is not None:
            if not isinstance(data["clarified_text"], str):
                raise EvalDatasetError(
                    f"Line {line_num} (ID: {case_id_str}): field 'clarified_text' must be a string or null, "
                    f"got: {type(data['clarified_text']).__name__}"
                )

        is_bench = data.get("is_benchmark")
        if is_bench is not None and not isinstance(is_bench, bool):
            raise EvalDatasetError(
                f"Line {line_num} (ID: {case_id_str}): field 'is_benchmark' must be a boolean"
            )

        if "fault_injection" in data and data["fault_injection"] is not None:
            if not isinstance(data["fault_injection"], dict):
                raise EvalDatasetError(
                    f"Line {line_num} (ID: {case_id_str}): field 'fault_injection' must be a dictionary"
                )

        expected = data.get("expected")
        if not isinstance(expected, dict) or not expected:
            raise EvalDatasetError(
                f"Line {line_num} (ID: {case_id_str}): field 'expected' must be a non-empty dictionary"
            )

        unknown_keys = set(expected.keys()) - SUPPORTED_EXPECTED_KEYS
        if unknown_keys:
            raise EvalDatasetError(
                f"Line {line_num} (ID: {case_id_str}): expected contains unsupported keys: {unknown_keys}"
            )

        # expected.type is mandatory for benchmark cases
        if is_bench is not False:
            exp_type = expected.get("type")
            if exp_type is None:
                raise EvalDatasetError(
                    f"Line {line_num} (ID: {case_id_str}): a classification benchmark case must have 'expected.type'"
                )
            if isinstance(exp_type, list):
                if len(exp_type) == 0:
                    raise EvalDatasetError(
                        f"Line {line_num} (ID: {case_id_str}): expected.type must not be an empty list"
                    )
                for t in exp_type:
                    if not isinstance(t, str) or t not in VALID_TYPES:
                        raise EvalDatasetError(
                            f"Line {line_num} (ID: {case_id_str}): expected.type contains invalid '{t}'. Must be one of {VALID_TYPES}"
                        )
            elif isinstance(exp_type, str):
                if exp_type not in VALID_TYPES:
                    raise EvalDatasetError(
                        f"Line {line_num} (ID: {case_id_str}): expected.type '{exp_type}' is invalid. Must be one of {VALID_TYPES}"
                    )
            else:
                raise EvalDatasetError(
                    f"Line {line_num} (ID: {case_id_str}): expected.type must be a string or a list of strings, got: {type(exp_type).__name__}"
                )
        elif "type" in expected:
            exp_type = expected["type"]
            if isinstance(exp_type, list):
                if len(exp_type) == 0:
                    raise EvalDatasetError(
                        f"Line {line_num} (ID: {case_id_str}): expected.type must not be an empty list"
                    )
                for t in exp_type:
                    if not isinstance(t, str) or t not in VALID_TYPES:
                        raise EvalDatasetError(
                            f"Line {line_num} (ID: {case_id_str}): expected.type '{t}' is invalid. Must be one of {VALID_TYPES}"
                        )
            elif isinstance(exp_type, str):
                if exp_type not in VALID_TYPES:
                    raise EvalDatasetError(
                        f"Line {line_num} (ID: {case_id_str}): expected.type '{exp_type}' is invalid. Must be one of {VALID_TYPES}"
                    )
            else:
                raise EvalDatasetError(
                    f"Line {line_num} (ID: {case_id_str}): expected.type must be a string or a list of strings, got: {type(exp_type).__name__}"
                )

        if "route" in expected:
            exp_route = expected["route"]
            if isinstance(exp_route, list):
                if len(exp_route) == 0:
                    raise EvalDatasetError(
                        f"Line {line_num} (ID: {case_id_str}): expected.route must not be an empty list"
                    )
                for r in exp_route:
                    if not isinstance(r, str) or r not in VALID_ROUTES:
                        raise EvalDatasetError(
                            f"Line {line_num} (ID: {case_id_str}): expected.route '{r}' is invalid. Must be one of {VALID_ROUTES}"
                        )
            elif isinstance(exp_route, str):
                if exp_route not in VALID_ROUTES:
                    raise EvalDatasetError(
                        f"Line {line_num} (ID: {case_id_str}): expected.route '{exp_route}' is invalid. Must be one of {VALID_ROUTES}"
                    )
            else:
                raise EvalDatasetError(
                    f"Line {line_num} (ID: {case_id_str}): expected.route must be a string or a list of strings"
                )

        if "priority" in expected:
            exp_p = expected["priority"]
            if isinstance(exp_p, list):
                if len(exp_p) == 0:
                    raise EvalDatasetError(
                        f"Line {line_num} (ID: {case_id_str}): expected.priority must not be an empty list"
                    )
                for p in exp_p:
                    if not isinstance(p, str) or p not in VALID_PRIORITIES:
                        raise EvalDatasetError(
                            f"Line {line_num} (ID: {case_id_str}): expected.priority '{p}' is invalid. Must be one of {VALID_PRIORITIES}"
                        )
            elif isinstance(exp_p, str):
                if exp_p not in VALID_PRIORITIES:
                    raise EvalDatasetError(
                        f"Line {line_num} (ID: {case_id_str}): expected.priority '{exp_p}' is invalid. Must be one of {VALID_PRIORITIES}"
                    )
            else:
                raise EvalDatasetError(
                    f"Line {line_num} (ID: {case_id_str}): expected.priority must be a string or a list of strings"
                )

        bool_expected_fields = [
            "injection_safe",
            "missing_info_empty",
            "clarification_questions_required",
            "has_acceptance_criteria",
            "affected_files_exist",
            "require_warning_for_invalid_file",
            "schema_recovery",
            "handled_gracefully",
        ]
        for bf in bool_expected_fields:
            if bf in expected and not isinstance(expected[bf], bool):
                raise EvalDatasetError(
                    f"Line {line_num} (ID: {case_id_str}): expected.{bf} must be a boolean (True/False), "
                    f"got: {type(expected[bf]).__name__} ('{expected[bf]}')"
                )

        int_expected_fields = ["max_affected_files", "max_retries", "max_attempts"]
        for intf in int_expected_fields:
            if intf in expected:
                val = expected[intf]
                if not isinstance(val, int) or isinstance(val, bool) or val < 0:
                    raise EvalDatasetError(
                        f"Line {line_num} (ID: {case_id_str}): expected.{intf} must be a non-negative integer, got: {val}"
                    )

        if "canary_token" in expected and not isinstance(expected["canary_token"], str):
            raise EvalDatasetError(
                f"Line {line_num} (ID: {case_id_str}): expected.canary_token must be a string"
            )

        cases.append(EvalCase.from_dict(data))

    if not cases:
        raise EvalDatasetError(f"The dataset at {source} is completely empty!")

    return cases


def run_user_path(
    case: EvalCase,
    chat_turn: Callable[[ChatTurn], ChatOutcome] | None = None,
) -> ChatOutcome:
    """
    Send the case through the real user path: the content is a natural message that goes through the router, then issue_agent.

    The title is the first line of the message, as when a user types into the UI. If the first message produces an
    issue awaiting clarification and the case has `clarified_text`, that text is sent next as the reply.
    """
    outcome = call_chat_turn(
        f"{case.title}\n{case.description}",
        case.repo_path,
        chat_turn=chat_turn,
    )
    if case.clarified_text and outcome.pending is not None:
        outcome = call_chat_turn(
            case.clarified_text,
            case.repo_path,
            pending=outcome.pending,
            chat_turn=chat_turn,
        )
    return outcome


def _error_result(
    case: EvalCase,
    status: str,
    failures: list[str],
    error: str | None = None,
    **extra: Any,
) -> CaseEvalResult:
    """Build a CaseEvalResult for cases with no ticket to score (blocked/failed)."""
    type_correct = (False if case.expected.get("type") else None) if status == "failed" else None
    return CaseEvalResult(
        case_id=case.id,
        category=case.category,
        is_benchmark=case.is_benchmark,
        status=status,
        passed=False,
        type_correct=type_correct,
        expected_type=case.expected.get("type"),
        actual_type=None,
        expected_route=case.expected.get("route"),
        actual_route=extra.get("actual_route"),
        run_id=extra.get("run_id", ""),
        failures=failures,
        warnings=[],
        actual=extra.get("actual"),
        expected=case.expected,
        error=error,
    )


def _run_user_case(
    case: EvalCase,
    chat_turn: Callable[[ChatTurn], ChatOutcome] | None = None,
    target_repo_root: Path | None = None,
) -> CaseEvalResult:
    """Run one case through the chat path and score it; never raises."""
    try:
        outcome = run_user_path(case, chat_turn=chat_turn)
        if outcome.result is not None:
            return evaluate_ticket_against_criteria(case, outcome.result, target_repo_root=target_repo_root)
    except BackendMissingError as e:
        return _error_result(case, "blocked", ["Blocked by backend: orchestrator is not implemented"], str(e))
    except BackendExecutionError as e:
        return _error_result(case, "failed", [f"Backend execution error: {e}"], str(e))
    except Exception as e:
        return _error_result(case, "failed", [f"Unexpected error while running the case: {e}"], str(e))

    # No ticket: the router chose conversation, or the chat path failed before reaching issue_agent
    extra = {"actual_route": outcome.route, "run_id": outcome.run_id, "actual": {"route": outcome.route, "reply": outcome.reply}}
    if outcome.route == CONVERSATION:
        failure = f"Router chose '{CONVERSATION}' so no IssueTicket was created (expected route={case.expected.get('route')})"
    else:
        failure = f"The chat path produced no result: {outcome.reply}"
    return _error_result(case, "failed", [failure], **extra)


def run_single_case(
    case: EvalCase,
    processor: Callable[[IssueRequest], IssueResult] | None = None,
    chat_turn: Callable[[ChatTurn], ChatOutcome] | None = None,
    target_repo_root: Path | None = None,
) -> CaseEvalResult:
    """
    Run and score one case, catching every exception so the runner never crashes.

    - A case without `fault_injection` takes the user path (path=chat): text -> router -> issue_agent.
      `chat_turn` is an optional test double for run_chat_turn.
    - A case with `fault_injection` is a mechanism test (path=mechanism): it calls process_issue with the fault hook.
      `processor` is an optional test double for process_issue.
    """
    if case.fault_injection:
        result = _run_mechanism_case(case, processor=processor, target_repo_root=target_repo_root)
        result.path = PATH_MECHANISM
    else:
        result = _run_user_case(case, chat_turn=chat_turn, target_repo_root=target_repo_root)
        result.path = PATH_CHAT
    return result


def _run_mechanism_case(
    case: EvalCase,
    processor: Callable[[IssueRequest], IssueResult] | None = None,
    target_repo_root: Path | None = None,
) -> CaseEvalResult:
    """Run a mechanism (fault_injection) case through process_issue, catching every exception."""
    # Faults are injected at the LLM/tool boundary through the process_issue hook (evals/faults.py)
    if case.fault_injection and processor is None:
        try:
            processor = build_fault_processor(case.fault_injection)
        except UnsupportedFaultError as e:
            return CaseEvalResult(
                case_id=case.id,
                category=case.category,
                is_benchmark=case.is_benchmark,
                status="blocked",
                passed=False,
                type_correct=None,
                expected_type=case.expected.get("type"),
                actual_type=None,
                expected_route=case.expected.get("route"),
                actual_route=None,
                run_id="",
                failures=[f"No fault injection hook for the case: {e}"],
                warnings=[],
                actual=None,
                expected=case.expected,
                error=str(e),
            )
        except BackendMissingError as e:
            return CaseEvalResult(
                case_id=case.id,
                category=case.category,
                is_benchmark=case.is_benchmark,
                status="blocked",
                passed=False,
                type_correct=None,
                expected_type=case.expected.get("type"),
                actual_type=None,
                expected_route=case.expected.get("route"),
                actual_route=None,
                run_id="",
                failures=["Blocked by backend: orchestrator is not implemented"],
                warnings=[],
                actual=None,
                expected=case.expected,
                error=str(e),
            )
        except Exception as e:
            return CaseEvalResult(
                case_id=case.id,
                category=case.category,
                is_benchmark=case.is_benchmark,
                status="failed",
                passed=False,
                type_correct=False if case.expected.get("type") else None,
                expected_type=case.expected.get("type"),
                actual_type=None,
                expected_route=case.expected.get("route"),
                actual_route=None,
                run_id="",
                failures=[f"Failed to build the fault processor ({type(e).__name__}): {e}"],
                warnings=[],
                actual=None,
                expected=case.expected,
                error=str(e),
            )

    # Build the request inside the guarded block so a bad case cannot crash the runner
    try:
        request = IssueRequest(
            title=case.title,
            description=case.description,
            clarified_text=case.clarified_text,
            repo_path=case.repo_path,
        )
    except Exception as e:
        return CaseEvalResult(
            case_id=case.id,
            category=case.category,
            is_benchmark=case.is_benchmark,
            status="failed",
            passed=False,
            type_correct=False if case.expected.get("type") else None,
            expected_type=case.expected.get("type"),
            actual_type=None,
            expected_route=case.expected.get("route"),
            actual_route=None,
            run_id="",
            failures=[f"Failed to build IssueRequest from the case data: {e}"],
            warnings=[],
            actual=None,
            expected=case.expected,
            error=str(e),
        )

    try:
        result: IssueResult = call_agent(request, processor=processor)
        return evaluate_ticket_against_criteria(case, result, target_repo_root=target_repo_root)
    except BackendMissingError as e:
        return CaseEvalResult(
            case_id=case.id,
            category=case.category,
            is_benchmark=case.is_benchmark,
            status="blocked",
            passed=False,
            type_correct=None,
            expected_type=case.expected.get("type"),
            actual_type=None,
            expected_route=case.expected.get("route"),
            actual_route=None,
            run_id="",
            failures=["Blocked by backend: orchestrator is not implemented"],
            warnings=[],
            actual=None,
            expected=case.expected,
            error=str(e),
        )
    except BackendExecutionError as e:
        return CaseEvalResult(
            case_id=case.id,
            category=case.category,
            is_benchmark=case.is_benchmark,
            status="failed",
            passed=False,
            type_correct=False if case.expected.get("type") else None,
            expected_type=case.expected.get("type"),
            actual_type=None,
            expected_route=case.expected.get("route"),
            actual_route=None,
            run_id="",
            failures=[f"Backend execution error: {e}"],
            warnings=[],
            actual=None,
            expected=case.expected,
            error=str(e),
        )
    except Exception as e:
        return CaseEvalResult(
            case_id=case.id,
            category=case.category,
            is_benchmark=case.is_benchmark,
            status="failed",
            passed=False,
            type_correct=False if case.expected.get("type") else None,
            expected_type=case.expected.get("type"),
            actual_type=None,
            expected_route=case.expected.get("route"),
            actual_route=None,
            run_id="",
            failures=[f"Unexpected error while running the case: {e}"],
            warnings=[],
            actual=None,
            expected=case.expected,
            error=str(e),
        )


def compute_eval_metrics(results: list[CaseEvalResult]) -> dict[str, Any]:
    """Compute the aggregate metrics, including Type Accuracy per the UC-01 standard."""
    total = len(results)
    passed_count = sum(1 for r in results if r.status == "passed")
    failed_count = sum(1 for r in results if r.status == "failed")
    blocked_count = sum(1 for r in results if r.status == "blocked")

    # The benchmark set used to measure type accuracy
    benchmark_results = [r for r in results if r.is_benchmark]
    benchmark_total = len(benchmark_results)
    benchmark_blocked = sum(1 for r in benchmark_results if r.status == "blocked")
    benchmark_ran = benchmark_total - benchmark_blocked

    type_correct_count = sum(1 for r in benchmark_results if r.type_correct is True)

    # An empty benchmark set, or one that is entirely blocked for lack of a backend, has no accuracy
    if benchmark_total == 0 or benchmark_ran == 0:
        type_accuracy = None
    else:
        # The denominator is every loaded benchmark case: failed cases are not dropped after the run
        type_accuracy = round(type_correct_count / benchmark_total, 4)

    # Met only when no benchmark case is blocked and accuracy >= 80%
    accuracy_met = (
        (type_accuracy >= TARGET_TYPE_ACCURACY and benchmark_blocked == 0)
        if type_accuracy is not None
        else False
    )

    return {
        "total_cases": total,
        "passed": passed_count,
        "failed": failed_count,
        "blocked": blocked_count,
        "benchmark_total": benchmark_total,
        "benchmark_ran": benchmark_ran,
        "benchmark_blocked": benchmark_blocked,
        "type_correct": type_correct_count,
        "type_accuracy": type_accuracy,
        "target_type_accuracy": TARGET_TYPE_ACCURACY,
        "accuracy_met": accuracy_met,
        "all_passed": passed_count == total and total > 0,
    }


def select_cases(cases: list[EvalCase], suite: str) -> list[EvalCase]:
    """Filter cases by suite: chat (user path), mechanism (fault_injection tests), all."""
    if suite == PATH_CHAT:
        return [c for c in cases if not c.fault_injection]
    if suite == PATH_MECHANISM:
        return [c for c in cases if c.fault_injection]
    return cases


def execute_cases(
    cases: list[EvalCase],
    processor: Callable[[IssueRequest], IssueResult] | None = None,
    chat_turn: Callable[[ChatTurn], ChatOutcome] | None = None,
    target_repo_root: Path | None = None,
) -> tuple[list[CaseEvalResult], dict[str, Any]]:
    """Run and score every case; returns (per-case results, metrics) without printing or writing files."""
    results = [
        run_single_case(case, processor=processor, chat_turn=chat_turn, target_repo_root=target_repo_root)
        for case in cases
    ]
    return results, compute_eval_metrics(results)


def export_reports(
    results: list[CaseEvalResult],
    metrics: dict[str, Any],
    reports_dir: Path,
    mode: str = "backend_live",
    is_subset: bool = False,
    subset_info: str = "",
) -> tuple[Path, Path]:
    """Write the evaluation report as both JSON and CSV."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    provider = config.LLM_PROVIDER.strip().lower() or "fake"
    is_offline = provider == "fake"
    llm_metadata = {
        "provider": provider,
        "model": "offline-rules (no network calls)" if is_offline else config.LLM_MODEL,
        "is_offline": is_offline,
    }

    filename_suffix = "_subset" if is_subset else ""
    json_path = reports_dir / f"eval_report_{timestamp}{filename_suffix}.json"

    report_payload = {
        "timestamp": datetime.datetime.now().isoformat(),
        "mode": mode,
        "classification_mode": "offline_rules" if is_offline else "real_model",
        "notes": (
            "Offline test (LLM_PROVIDER=fake): does not prove real model quality"
            if is_offline
            else "Run with a real model"
        ),
        "subset": is_subset,
        "subset_info": subset_info if is_subset else None,
        "llm": llm_metadata,
        "metrics": metrics,
        "cases": [r.to_dict() for r in results],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)

    csv_path = reports_dir / f"eval_report_{timestamp}{filename_suffix}.csv"

    fieldnames = [
        "case_id",
        "category",
        "path",
        "is_benchmark",
        "status",
        "passed",
        "type_correct",
        "expected_type",
        "actual_type",
        "expected_route",
        "actual_route",
        "run_id",
        "failures_count",
        "failures",
        "error",
    ]

    def write_csv(path: Path):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(
                    {
                        "case_id": r.case_id,
                        "category": r.category,
                        "path": r.path,
                        "is_benchmark": r.is_benchmark,
                        "status": r.status,
                        "passed": r.passed,
                        "type_correct": r.type_correct,
                        "expected_type": str(r.expected_type) if r.expected_type else "",
                        "actual_type": r.actual_type or "",
                        "expected_route": str(r.expected_route) if r.expected_route else "",
                        "actual_route": r.actual_route or "",
                        "run_id": r.run_id,
                        "failures_count": len(r.failures),
                        "failures": " | ".join(r.failures),
                        "error": r.error or "",
                    }
                )

    write_csv(csv_path)

    # Only a full run (not a subset) overwrites latest_report
    if not is_subset:
        latest_json = reports_dir / "latest_report.json"
        latest_csv = reports_dir / "latest_report.csv"
        with open(latest_json, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, ensure_ascii=False, indent=2)
        write_csv(latest_csv)

    return json_path, csv_path


def accuracy_verdict(metrics: dict[str, Any]) -> str:
    """Verdict label for type accuracy; distinguishes low accuracy from blocked benchmark cases."""
    target_str = f"{metrics['target_type_accuracy'] * 100:.0f}%"
    if metrics["type_accuracy"] is None:
        return "⚠️ NOT RUN"
    if metrics["accuracy_met"]:
        return f"✅ MET (>= {target_str})"
    if metrics["type_accuracy"] >= metrics["target_type_accuracy"]:
        return f"❌ NOT MET (accuracy is sufficient but {metrics['benchmark_blocked']} benchmark case(s) blocked)"
    return f"❌ NOT MET (< {target_str})"


def render_console_table(
    results: list[CaseEvalResult],
    metrics: dict[str, Any],
    mode: str = "backend_live",
) -> None:
    """Print the results table to the console."""
    sep = "=" * 90
    subsep = "-" * 90

    provider = config.LLM_PROVIDER.strip().lower() or "fake"
    offline_note = (
        " [OFFLINE TEST DOUBLE - does not prove real model quality]"
        if provider == "fake"
        else " [PRODUCTION MODEL]"
    )

    print("\n" + sep)
    print("📊 UC-01 EVAL REPORT: ISSUE INTAKE AND TRIAGE")
    print(
        f"Mode: {mode.upper()} ({'Isolated test environment / test double' if mode == 'test_double' else 'Backend environment'})"
        + offline_note
    )
    print(sep)

    print(
        f"{'Case ID':<15} {'Category':<22} {'Path':<11} {'Bench':<6} {'Status':<9} {'Type':<12} {'Route':<12} {'Failures'}"
    )
    print(subsep)

    for r in results:
        status_sym = {
            "passed": "✅ PASS",
            "failed": "❌ FAIL",
            "blocked": "⚠️ BLOCK",
        }.get(r.status, r.status)

        bench_sym = "Yes" if r.is_benchmark else "No"
        type_str = f"{r.actual_type or 'None'}"
        if r.type_correct is False and r.expected_type:
            type_str += f" (!={r.expected_type})"

        route_str = f"{r.actual_route or 'None'}"
        fail_summary = f"{len(r.failures)} failures" if r.failures else "-"

        print(
            f"{r.case_id:<15} {r.category:<22} {r.path:<11} {bench_sym:<6} {status_sym:<9} {type_str:<12} {route_str:<12} {fail_summary}"
        )

    print(subsep)
    print("📈 SUMMARY AND ACCURACY (METRICS):")
    print(
        f"  • Total cases              : {metrics['total_cases']} "
        f"(Passed: {metrics['passed']}, Failed: {metrics['failed']}, Blocked: {metrics['blocked']})"
    )

    acc = metrics["type_accuracy"]
    acc_str = f"{acc * 100:.1f}%" if acc is not None else "N/A (not enough data to compute)"
    target_str = f"{TARGET_TYPE_ACCURACY * 100:.0f}%"

    print(
        f"  • Type Accuracy (Benchmark): {metrics['type_correct']}/{metrics['benchmark_total']} = {acc_str} "
        f"[UC-01 standard: {target_str}] -> {accuracy_verdict(metrics)}"
    )

    has_failures = any(r.failures for r in results)
    if has_failures:
        print(subsep)
        print("🔍 RECORDED FAILURE DETAILS:")
        for r in results:
            if r.failures:
                print(f"  [{r.case_id}] ({r.category}):")
                for f in r.failures:
                    print(f"      - {f}")

    print(sep + "\n")


def run_evals(
    cases_path: Path | None = None,
    processor: Callable[[IssueRequest], IssueResult] | None = None,
    chat_turn: Callable[[ChatTurn], ChatOutcome] | None = None,
    reports_dir: Path | None = None,
    target_repo_root: Path | None = None,
    suite: str = "all",
) -> int:
    """
    Run the whole eval suite and write the results.

    Args:
        processor: Test double for process_issue (used by mechanism cases only).
        chat_turn: Test double for run_chat_turn (used by user cases).
        suite: all | chat | mechanism.

    Returns:
        int: The exit code.
    """
    if cases_path is None:
        cases_path = current_dir / "cases.jsonl"
    if reports_dir is None:
        reports_dir = current_dir / "reports"

    try:
        cases = load_eval_cases(cases_path)
    except EvalDatasetError as e:
        sys.stderr.write(f"\n❌ EVAL DATASET ERROR: {e}\n\n")
        return EXIT_EVAL_CONFIG_ERROR

    cases = select_cases(cases, suite)
    mode = "test_double" if (processor is not None or chat_turn is not None) else "backend_live"

    results, metrics = execute_cases(
        cases, processor=processor, chat_turn=chat_turn, target_repo_root=target_repo_root
    )
    render_console_table(results, metrics, mode=mode)

    try:
        json_rep, csv_rep = export_reports(results, metrics, reports_dir, mode=mode)
        print(f"💾 JSON report saved: {json_rep}")
        print(f"💾 CSV report saved : {csv_rep}\n")
    except Exception as e:
        sys.stderr.write(f"Warning: could not save the report files: {e}\n")

    if metrics["blocked"] > 0:
        return EXIT_EVAL_BLOCKED
    elif metrics["all_passed"] and metrics["accuracy_met"]:
        return EXIT_EVAL_ALL_PASSED
    else:
        return EXIT_EVAL_FAILED


def main() -> None:
    """Command-line entry point for the runner."""
    parser = argparse.ArgumentParser(
        prog="issue-agent-evals",
        description="Eval runner for the AI Issue Agent - UC-01",
    )
    parser.add_argument(
        "-c",
        "--cases",
        type=str,
        default=None,
        help="Path to the cases.jsonl file (default: evals/cases.jsonl)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Directory for the reports (default: evals/reports/)",
    )
    parser.add_argument(
        "-s",
        "--suite",
        choices=SUITES,
        default="all",
        help="chat: user cases (text -> router -> issue_agent); mechanism: fault_injection cases; all: both",
    )
    parsed = parser.parse_args()

    cases_file = Path(parsed.cases) if parsed.cases else None
    reports_dir = Path(parsed.output_dir) if parsed.output_dir else None

    exit_code = run_evals(cases_path=cases_file, reports_dir=reports_dir, suite=parsed.suite)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
