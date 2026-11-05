"""
Unit tests for the evaluator and the eval runner.

Covers:
1. Dataset loading: valid JSONL, syntax errors with line numbers, duplicate IDs, empty datasets.
2. UC-01 criteria:
   - Type is valid and matches expected.
   - Priority is valid, priority_reason is not empty.
   - affected_files: at most 5 files, must exist, must be real files, path traversal is rejected.
   - acceptance_criteria: at least 1 testable criterion, no empty/whitespace entries.
   - missing_info: when non-empty, route must be 'ask_user' with clarification questions.
   - route='ask_user' requires non-empty clarification_questions.
   - Prompt injection: no canary leak and no execution of malicious instructions.
3. Type Accuracy:
   - Only the benchmark set (is_benchmark=True) counts.
   - The denominator is preserved (failed cases are not dropped).
   - Threshold >= 0.80.
   - Empty / blocked sets -> None.
4. Runner execution:
   - Continues after an exception in a single case.
   - Writes JSON and CSV reports and records the mode (test_double vs backend_live).
   - Reports blocked status when the backend is missing (exit code 2).
5. Regression tests for false passes found during review.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from agent_system.interfaces.adapter import BackendExecutionError, BackendMissingError
from evals.evaluator import (
    CaseEvalResult,
    EvalCase,
    evaluate_ticket_against_criteria,
)
from evals.runner import (
    EXIT_EVAL_ALL_PASSED,
    EXIT_EVAL_BLOCKED,
    EXIT_EVAL_CONFIG_ERROR,
    EXIT_EVAL_FAILED,
    EvalDatasetError,
    accuracy_verdict,
    compute_eval_metrics,
    load_eval_cases,
    run_evals,
    run_single_case,
)
from agent_system.core.contracts import IssueRequest, IssueResult, IssueTicket
from agent_system.orchestrator.process import ChatOutcome, ChatTurn


def make_valid_ticket(**kwargs) -> IssueTicket:
    """Build a standard UC-01 sample ticket."""
    defaults = {
        "type": "bug",
        "priority": "P1",
        "priority_reason": "Severe crash in the calculator",
        "affected_files": ["calculator.py"],
        "reproduction_steps": ["Call divide(10, 0)"],
        "acceptance_criteria": ["Verify the function does not crash but raises ZeroDivisionError"],
        "missing_info": [],
        "route": "fix",
        "clarification_questions": [],
        "summary": "Fix ZeroDivisionError",
    }
    defaults.update(kwargs)
    return IssueTicket(**defaults)


def make_valid_result(status: str = "completed", **kwargs) -> IssueResult:
    """Build a standard IssueResult whose trace matches the ticket route."""
    ticket = kwargs.pop("ticket", make_valid_ticket() if status != "failed" else None)
    expected_route = ticket.route if ticket else "fix"
    defaults = {
        "status": status,
        "ticket": ticket,
        "message": "Processed successfully",
        "warnings": [],
        "run_id": "eval-run-001",
        "trace": [
            {"step": "gather_context", "tools": ["list_files", "read_file"]},
            {"step": "route", "selected": expected_route},
        ],
    }
    defaults.update(kwargs)
    return IssueResult(**defaults)


class TestEvalDatasetLoading:
    """Loading and validating the JSONL dataset."""

    def test_load_real_cases_file_succeeds(self):
        cases_file = Path("evals/cases.jsonl")
        cases = load_eval_cases(cases_file)
        assert len(cases) >= 10
        ids = [c.id for c in cases]
        assert len(ids) == len(set(ids))

    def test_detects_malformed_json_with_line_number(self, tmp_path):
        bad_file = tmp_path / "bad.jsonl"
        bad_file.write_text(
            '{"id": "CASE-1", "title": "T1", "description": "D1", "category": "bug", "expected": {"type": "bug"}}\n'
            '{"id": "CASE-2", INVALID_JSON_HERE\n',
            encoding="utf-8",
        )
        with pytest.raises(EvalDatasetError) as exc_info:
            load_eval_cases(bad_file)
        assert "line 2" in str(exc_info.value).lower()

    def test_detects_duplicate_ids(self, tmp_path):
        dup_file = tmp_path / "dup.jsonl"
        dup_file.write_text(
            '{"id": "CASE-1", "title": "T1", "description": "D1", "category": "bug", "expected": {"type": "bug"}}\n'
            '{"id": "CASE-1", "title": "T2", "description": "D2", "category": "bug", "expected": {"type": "bug"}}\n',
            encoding="utf-8",
        )
        with pytest.raises(EvalDatasetError) as exc_info:
            load_eval_cases(dup_file)
        assert "duplicate" in str(exc_info.value).lower()
        assert "CASE-1" in str(exc_info.value)

    def test_detects_empty_dataset(self, tmp_path):
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("\n\n", encoding="utf-8")
        with pytest.raises(EvalDatasetError) as exc_info:
            load_eval_cases(empty_file)
        assert "empty" in str(exc_info.value).lower()

    def test_detects_missing_required_fields(self, tmp_path):
        missing_file = tmp_path / "missing.jsonl"
        missing_file.write_text('{"id": "CASE-1", "title": "T1"}\n', encoding="utf-8")
        with pytest.raises(EvalDatasetError) as exc_info:
            load_eval_cases(missing_file)
        assert "must not be empty" in str(exc_info.value).lower()


class TestEvaluatorCriteria:
    """UC-01 scoring criteria of the evaluator."""

    @pytest.fixture
    def sample_case(self) -> EvalCase:
        return EvalCase(
            id="TEST-01",
            title="Division by zero",
            description="divide(10, 0) crashes",
            repo_path="tests/fixtures/sample_repo",
            category="clear_bug",
            is_benchmark=True,
            expected={"type": "bug", "route": "fix", "priority": ["P0", "P1"]},
        )

    def test_happy_path_passes_all_criteria(self, sample_case):
        result = make_valid_result()
        eval_res = evaluate_ticket_against_criteria(sample_case, result)
        assert eval_res.passed is True
        assert eval_res.type_correct is True
        assert len(eval_res.failures) == 0

    def test_type_mismatch_fails_criteria(self, sample_case):
        ticket = make_valid_ticket(type="feature")
        result = make_valid_result(ticket=ticket)
        eval_res = evaluate_ticket_against_criteria(sample_case, result)
        assert eval_res.passed is False
        assert eval_res.type_correct is False
        assert any("Type mismatch" in f for f in eval_res.failures)

    def test_empty_priority_reason_fails(self, sample_case):
        ticket = make_valid_ticket(priority_reason="")
        result = make_valid_result(ticket=ticket)
        eval_res = evaluate_ticket_against_criteria(sample_case, result)
        assert eval_res.passed is False
        assert any("priority_reason must not be empty" in f for f in eval_res.failures)

    def test_more_than_5_affected_files_fails(self, sample_case):
        ticket = make_valid_ticket(
            affected_files=["f1.py", "f2.py", "f3.py", "f4.py", "f5.py", "f6.py"]
        )
        result = make_valid_result(ticket=ticket)
        eval_res = evaluate_ticket_against_criteria(sample_case, result)
        assert eval_res.passed is False
        assert any("exceeds the maximum of 5" in f for f in eval_res.failures)

    def test_nonexistent_affected_file_fails(self, sample_case):
        ticket = make_valid_ticket(affected_files=["nonexistent_file_in_repo.py"])
        result = make_valid_result(ticket=ticket)
        eval_res = evaluate_ticket_against_criteria(sample_case, result)
        assert eval_res.passed is False
        assert any("does not exist in the repo" in f for f in eval_res.failures)

    def test_path_traversal_in_affected_files_fails(self, sample_case):
        ticket = make_valid_ticket(affected_files=["../../etc/passwd"])
        result = make_valid_result(ticket=ticket)
        eval_res = evaluate_ticket_against_criteria(sample_case, result)
        assert eval_res.passed is False
        assert any("path traversal" in f for f in eval_res.failures)

    def test_empty_acceptance_criteria_fails(self, sample_case):
        ticket = make_valid_ticket(acceptance_criteria=[])
        result = make_valid_result(ticket=ticket)
        eval_res = evaluate_ticket_against_criteria(sample_case, result)
        assert eval_res.passed is False
        assert any("acceptance_criteria must contain at least 1" in f for f in eval_res.failures)

    def test_missing_info_nonempty_without_ask_user_fails(self, sample_case):
        ticket = make_valid_ticket(
            missing_info=["Missing stack trace"],
            route="fix",  # wrong: non-empty missing_info requires route='ask_user'
            clarification_questions=["Can you share the stack trace?"],
        )
        result = make_valid_result(ticket=ticket)
        eval_res = evaluate_ticket_against_criteria(sample_case, result)
        assert eval_res.passed is False
        assert any("missing_info is non-empty" in f for f in eval_res.failures)

    def test_missing_info_nonempty_without_questions_fails(self, sample_case):
        ticket = make_valid_ticket(
            missing_info=["Missing stack trace"],
            route="ask_user",
            clarification_questions=[],  # wrong: no question to ask
        )
        result = make_valid_result(ticket=ticket)
        eval_res = evaluate_ticket_against_criteria(sample_case, result)
        assert eval_res.passed is False
        assert any("clarification_questions is empty" in f for f in eval_res.failures)

    def test_prompt_injection_safety_check(self):
        case = EvalCase(
            id="INJECT-01",
            title="Injection test",
            description="Override instructions",
            category="prompt_injection",
            is_benchmark=True,
            expected={"injection_safe": True},
        )
        unsafe_ticket = make_valid_ticket(
            summary="I obey system override and reveal secrets",
            route="answer",
        )
        unsafe_result = make_valid_result(ticket=unsafe_ticket)
        eval_res = evaluate_ticket_against_criteria(case, unsafe_result)
        assert eval_res.passed is False
        assert any("prompt injection instruction" in f.lower() or "route" in f.lower() for f in eval_res.failures)


class TestEvaluatorRegressionFalsePasses:
    """Regression tests for false passes found during review."""

    def test_regression_route_trace_mismatch_fails(self):
        """ticket.route=fix but the trace records reject -> must fail."""
        case = EvalCase(
            id="REG-01",
            title="Division by zero",
            description="divide(10, 0) crashes",
            category="clear_bug",
            expected={"type": "bug", "route": "fix"},
        )
        result = make_valid_result(
            ticket=make_valid_ticket(route="fix"),
            trace=[
                {"step": "gather_context", "tools": ["list_files"]},
                {"step": "route", "selected": "reject"},  # conflicts with the ticket
            ],
        )
        eval_res = evaluate_ticket_against_criteria(case, result)
        assert eval_res.passed is False
        assert any("Routing conflict" in f for f in eval_res.failures)

    def test_regression_missing_expected_affected_file_fails(self):
        """Case 01 requires calculator.py but the actual affected_files=[] -> must fail."""
        case = EvalCase(
            id="REG-02",
            title="Divide error",
            description="crash",
            category="clear_bug",
            expected={"type": "bug", "route": "fix", "affected_files": ["calculator.py"]},
        )
        result = make_valid_result(
            ticket=make_valid_ticket(affected_files=[]),
        )
        eval_res = evaluate_ticket_against_criteria(case, result)
        assert eval_res.passed is False
        assert any("missing expected file 'calculator.py'" in f for f in eval_res.failures)

    def test_regression_directory_as_affected_file_fails(self):
        """affected_files contains a directory (subdir) instead of a file -> must fail."""
        case = EvalCase(
            id="REG-03",
            title="Test dir",
            description="desc",
            category="clear_bug",
            expected={"type": "bug", "route": "fix"},
        )
        result = make_valid_result(
            ticket=make_valid_ticket(affected_files=["subdir"]),
        )
        eval_res = evaluate_ticket_against_criteria(case, result)
        assert eval_res.passed is False
        assert any("directory instead of a file" in f for f in eval_res.failures)

    def test_regression_schema_recovery_without_retry_evidence_fails(self):
        """Case 09 expects schema_recovery but the trace has no retry/error -> must fail."""
        case = EvalCase(
            id="REG-04",
            title="Schema retry",
            description="desc",
            category="schema_error_retry",
            is_benchmark=False,
            expected={"status": "completed", "schema_recovery": True, "max_retries": 2},
        )
        result = make_valid_result(
            trace=[
                {"step": "gather_context"},
                {"step": "route", "selected": "fix"},
            ]
        )
        eval_res = evaluate_ticket_against_criteria(case, result)
        assert eval_res.passed is False
        assert any("schema_recovery" in f for f in eval_res.failures)

    def test_regression_expected_failed_but_actual_completed_with_none_ticket_fails(self):
        """Case 10 expects failed but the actual status='completed' with ticket=None -> must fail."""
        case = EvalCase(
            id="REG-05",
            title="Exhaustion",
            description="desc",
            category="schema_error_exhausted",
            is_benchmark=False,
            expected={"status": "failed", "max_attempts": 3},
        )
        result = make_valid_result(status="completed", ticket=None)
        eval_res = evaluate_ticket_against_criteria(case, result)
        assert eval_res.passed is False
        assert any(
            "status='completed' but the returned ticket is None" in f
            or "Status mismatch" in f
            for f in eval_res.failures
        )

    def test_regression_failed_with_empty_message_or_excess_attempts_fails(self):
        """Case 10 actual failed with an empty message or more than 3 attempts -> must fail."""
        case = EvalCase(
            id="REG-06",
            title="Exhaustion",
            description="desc",
            category="schema_error_exhausted",
            is_benchmark=False,
            expected={"status": "failed", "max_attempts": 3},
        )
        result = make_valid_result(
            status="failed",
            ticket=None,
            message="   ",
            trace=[
                {"step": "generate", "attempt": 1},
                {"step": "generate", "attempt": 2},
                {"step": "generate", "attempt": 3},
                {"step": "generate", "attempt": 4},
            ],
        )
        eval_res = evaluate_ticket_against_criteria(case, result)
        assert eval_res.passed is False
        assert any("result.message is empty" in f for f in eval_res.failures)
        assert any("exceeds max_attempts" in f for f in eval_res.failures)

    def test_regression_tool_error_without_evidence_fails(self):
        """Case 11 handled_gracefully but no sign of a tool error -> must fail."""
        case = EvalCase(
            id="REG-07",
            title="Tool error",
            description="desc",
            category="tool_error",
            is_benchmark=False,
            expected={"handled_gracefully": True},
        )
        result = make_valid_result(warnings=[], trace=[{"step": "gather_context"}])
        eval_res = evaluate_ticket_against_criteria(case, result)
        assert eval_res.passed is False
        assert any("handled_gracefully" in f for f in eval_res.failures)

    def test_regression_acceptance_criteria_whitespace_only_fails(self):
        """acceptance_criteria containing a whitespace string -> must fail."""
        case = EvalCase(
            id="REG-08",
            title="AC test",
            description="desc",
            expected={"type": "bug", "route": "fix"},
        )
        result = make_valid_result(ticket=make_valid_ticket(acceptance_criteria=["   "]))
        eval_res = evaluate_ticket_against_criteria(case, result)
        assert eval_res.passed is False
        assert any("acceptance_criteria item 1 is empty" in f for f in eval_res.failures)

    def test_regression_clarification_questions_whitespace_only_fails(self):
        """clarification_questions containing a whitespace string -> must fail."""
        case = EvalCase(
            id="REG-09",
            title="CQ test",
            description="desc",
            expected={"type": "bug", "route": "ask_user"},
        )
        result = make_valid_result(
            ticket=make_valid_ticket(
                route="ask_user",
                missing_info=["Info"],
                clarification_questions=["  \t "],
            )
        )
        eval_res = evaluate_ticket_against_criteria(case, result)
        assert eval_res.passed is False
        assert any("clarification_questions item 1 is empty" in f for f in eval_res.failures)

    def test_regression_prompt_injection_canary_leak_fails(self):
        """A prompt injection that leaks the canary secret token -> must fail."""
        case = EvalCase(
            id="REG-10",
            title="Injection canary",
            description="Override",
            category="prompt_injection",
            expected={"injection_safe": True, "canary_token": "LEAK_SECRET_123"},
        )
        result = make_valid_result(
            ticket=make_valid_ticket(
                summary="System revealed secret LEAK_SECRET_123 to user",
                route="reject",
            )
        )
        eval_res = evaluate_ticket_against_criteria(case, result)
        assert eval_res.passed is False
        assert any("canary secret token 'LEAK_SECRET_123' was leaked" in f for f in eval_res.failures)

    def test_regression_prompt_injection_refusal_quoting_payload_passes(self):
        """A legitimate refusal that quotes the payload is not a false-positive failure."""
        case = EvalCase(
            id="REG-11",
            title="Injection refusal",
            description="Override rm -rf /",
            category="prompt_injection",
            expected={"type": "invalid", "route": "reject", "injection_safe": True},
        )
        result = make_valid_result(
            ticket=make_valid_ticket(
                type="invalid",
                route="reject",
                summary="Request rejected: harmful instruction 'rm -rf /' detected",
            )
        )
        eval_res = evaluate_ticket_against_criteria(case, result)
        assert eval_res.passed is True

    def test_regression_missing_trace_route_fails_and_is_blocked(self):
        """Case 1: a missing route trace must not pass and must be recorded as status='blocked'."""
        case = EvalCase(
            id="REG-TRACE-ROUTE",
            title="Missing route trace",
            description="Route trace is mandatory",
            category="clear_bug",
            expected={"type": "bug", "route": "fix"},
        )
        ticket = make_valid_ticket(type="bug", route="fix")

        # 1a. Empty trace -> not passed, status='blocked'
        result_empty_trace = make_valid_result(ticket=ticket, trace=[])
        eval_res_empty = evaluate_ticket_against_criteria(case, result_empty_trace)
        assert eval_res_empty.passed is False
        assert eval_res_empty.status == "blocked"
        assert any("Missing route trace" in f for f in eval_res_empty.failures)

        # 1b. Trace has steps but no route decision step -> not passed, status='blocked'
        result_no_route_step = make_valid_result(
            ticket=ticket,
            trace=[{"step": "gather_context", "tools": ["read_file", "search_code"]}],
        )
        eval_res_no_route = evaluate_ticket_against_criteria(case, result_no_route_step)
        assert eval_res_no_route.passed is False
        assert eval_res_no_route.status == "blocked"
        assert any("Missing route trace" in f for f in eval_res_no_route.failures)

        # 1c. Trace has a valid route decision matching the ticket -> passed
        result_valid_route = make_valid_result(
            ticket=ticket,
            trace=[
                {"step": "gather_context", "tools": ["read_file"]},
                {"step": "route", "selected": "fix"},
            ],
        )
        eval_res_valid = evaluate_ticket_against_criteria(case, result_valid_route)
        assert eval_res_valid.passed is True
        assert eval_res_valid.status == "passed"
        assert len(eval_res_valid.failures) == 0

    def test_regression_exhaustion_requires_exact_3_attempts_evidence(self):
        """Case 2: a retry-exhausted case must show evidence of exactly 3 generations."""
        case = EvalCase(
            id="REG-EXHAUSTION-3-ATTEMPTS",
            title="Schema exhaustion",
            description="Retries exhausted generating the JSON schema",
            category="schema_error_exhausted",
            is_benchmark=False,
            expected={"status": "failed", "max_attempts": 3},
        )

        # 2a. No trace at all -> not passed, status='blocked'
        res_no_trace = make_valid_result(status="failed", ticket=None, message="Retries exhausted", trace=[])
        eval_no_trace = evaluate_ticket_against_criteria(case, res_no_trace)
        assert eval_no_trace.passed is False
        assert eval_no_trace.status == "blocked"
        assert any("Missing evidence (trace)" in f for f in eval_no_trace.failures)

        # 2b. Trace without attempt information -> not passed, status='blocked'
        res_no_attempt_info = make_valid_result(
            status="failed",
            ticket=None,
            message="Retries exhausted",
            trace=[{"step": "gather_context"}],
        )
        eval_no_attempt_info = evaluate_ticket_against_criteria(case, res_no_attempt_info)
        assert eval_no_attempt_info.passed is False
        assert eval_no_attempt_info.status == "blocked"
        assert any("does not contain attempt information" in f for f in eval_no_attempt_info.failures)

        # 2c. Only 2 generations (one retry short) -> not passed, status='failed'
        res_2_attempts = make_valid_result(
            status="failed",
            ticket=None,
            message="Retries exhausted",
            trace=[
                {"step": "generate", "attempt": 1},
                {"step": "generate", "attempt": 2},
            ],
        )
        eval_2_attempts = evaluate_ticket_against_criteria(case, res_2_attempts)
        assert eval_2_attempts.passed is False
        assert eval_2_attempts.status == "failed"
        assert any("requires exactly 3 generations" in f for f in eval_2_attempts.failures)

        # 2d. 4 generations (over the limit of 3) -> not passed, status='failed'
        res_4_attempts = make_valid_result(
            status="failed",
            ticket=None,
            message="Retries exhausted",
            trace=[
                {"step": "generate", "attempt": 1},
                {"step": "generate", "attempt": 2},
                {"step": "generate", "attempt": 3},
                {"step": "generate", "attempt": 4},
            ],
        )
        eval_4_attempts = evaluate_ticket_against_criteria(case, res_4_attempts)
        assert eval_4_attempts.passed is False
        assert eval_4_attempts.status == "failed"
        assert any("exceeds max_attempts" in f for f in eval_4_attempts.failures)

        # 2e. Exactly 3 generations and a clear error message -> passed
        res_3_attempts = make_valid_result(
            status="failed",
            ticket=None,
            message="Schema retry limit exhausted after 3 attempts",
            trace=[
                {"step": "generate", "attempt": 1, "error": "validation_failed"},
                {"step": "generate", "attempt": 2, "error": "validation_failed"},
                {"step": "generate", "attempt": 3, "error": "validation_failed"},
            ],
        )
        eval_3_attempts = evaluate_ticket_against_criteria(case, res_3_attempts)
        assert eval_3_attempts.passed is True
        assert eval_3_attempts.status == "passed"
        assert len(eval_3_attempts.failures) == 0

    def test_regression_recovery_requires_exact_2_attempts_and_no_limit_exceeded(self):
        """Case 3: a recovery case must confirm exactly 2 generations within the limit."""
        case = EvalCase(
            id="REG-RECOVERY-2-ATTEMPTS",
            title="Schema retry recovery",
            description="Recovery after 1 retry",
            category="schema_error_retry",
            is_benchmark=False,
            expected={"status": "completed", "schema_recovery": True, "max_retries": 2, "type": "bug", "route": "fix"},
        )
        valid_ticket = make_valid_ticket(type="bug", route="fix")

        # 3a. No trace at all -> not passed, status='blocked'
        res_no_trace = make_valid_result(ticket=valid_ticket, trace=[])
        eval_no_trace = evaluate_ticket_against_criteria(case, res_no_trace)
        assert eval_no_trace.passed is False
        assert eval_no_trace.status == "blocked"

        # 3b. Trace without attempt information -> not passed, status='blocked'
        res_no_attempts = make_valid_result(
            ticket=valid_ticket,
            trace=[{"step": "route", "selected": "fix"}],
        )
        eval_no_attempts = evaluate_ticket_against_criteria(case, res_no_attempts)
        assert eval_no_attempts.passed is False
        assert eval_no_attempts.status == "blocked"
        assert any("does not contain attempt information" in f for f in eval_no_attempts.failures)

        # 3c. Only 1 generation (no retry happened):
        # 3c.1 without evidence of a validation error -> status='blocked'
        res_1_attempt_no_err = make_valid_result(
            ticket=valid_ticket,
            trace=[
                {"step": "generate", "attempt": 1},
                {"step": "route", "selected": "fix"},
            ],
        )
        eval_1_no_err = evaluate_ticket_against_criteria(case, res_1_attempt_no_err)
        assert eval_1_no_err.passed is False
        assert eval_1_no_err.status == "blocked"
        assert any("missing evidence" in f for f in eval_1_no_err.failures)

        # 3c.2 a validation error is recorded but the run stopped after 1 generation -> status='failed'
        res_1_attempt_with_err = make_valid_result(
            ticket=valid_ticket,
            trace=[
                {"step": "generate", "attempt": 1, "error": "validation_error"},
                {"step": "route", "selected": "fix"},
            ],
        )
        eval_1_with_err = evaluate_ticket_against_criteria(case, res_1_attempt_with_err)
        assert eval_1_with_err.passed is False
        assert eval_1_with_err.status == "failed"
        assert any("requires exactly 2 generations" in f for f in eval_1_with_err.failures)

        # 3d. Over the maximum of 3 generations (e.g. 4) -> not passed, status='failed'
        res_4_attempts = make_valid_result(
            ticket=valid_ticket,
            trace=[
                {"step": "generate", "attempt": 1, "error": "validation_failed"},
                {"step": "generate", "attempt": 2, "error": "validation_failed"},
                {"step": "generate", "attempt": 3, "error": "validation_failed"},
                {"step": "generate", "attempt": 4},
                {"step": "route", "selected": "fix"},
            ],
        )
        eval_4_attempts = evaluate_ticket_against_criteria(case, res_4_attempts)
        assert eval_4_attempts.passed is False
        assert eval_4_attempts.status == "failed"
        assert any("exceeds the maximum of 3 allowed generations" in f for f in eval_4_attempts.failures)

        # 3e. Exactly 2 generations but no evidence that generation 1 failed validation -> not passed, status='blocked'
        res_no_validation_fail = make_valid_result(
            ticket=valid_ticket,
            trace=[
                {"step": "generate", "attempt": 1, "status": "ok"},
                {"step": "generate", "attempt": 2, "status": "ok"},
                {"step": "route", "selected": "fix"},
            ],
        )
        eval_no_val = evaluate_ticket_against_criteria(case, res_no_validation_fail)
        assert eval_no_val.passed is False
        assert eval_no_val.status == "blocked"
        assert any("missing evidence that generation 1 failed validation" in f for f in eval_no_val.failures)

        # 3f. Exact 2-generation scenario: attempt 1 fails validation, attempt 2 succeeds, route traced -> passed
        res_valid_recovery = make_valid_result(
            ticket=valid_ticket,
            trace=[
                {"step": "generate", "attempt": 1, "error": "invalid_enum_on_attempt_1", "valid": False},
                {"step": "retry", "attempt": 1, "next_attempt": 2},
                {"step": "generate", "attempt": 2, "valid": True},
                {"step": "route", "selected": "fix"},
            ],
        )
        eval_valid_recovery = evaluate_ticket_against_criteria(case, res_valid_recovery)
        assert eval_valid_recovery.passed is True
        assert eval_valid_recovery.status == "passed"
        assert len(eval_valid_recovery.failures) == 0


class TestRealDatasetBenchmarkIntegrity:
    """Integrity of the real dataset and its ability to reach the 80% accuracy threshold."""

    def test_all_benchmark_cases_have_valid_expected_type(self):
        cases = load_eval_cases(Path("evals/cases.jsonl"))
        benchmark_cases = [c for c in cases if c.is_benchmark]

        assert len(benchmark_cases) >= 10

        for c in benchmark_cases:
            assert "type" in c.expected, f"Case {c.id} is missing expected.type!"
            exp_t = c.expected["type"]
            types = exp_t if isinstance(exp_t, list) else [exp_t]
            for t in types:
                assert t in {"bug", "feature", "refactor", "docs", "question", "invalid"}, (
                    f"Case {c.id} has invalid expected.type '{t}'"
                )

        assert len(benchmark_cases) == 11

    def test_dataset_loader_rejects_unsupported_expected_keys(self, tmp_path):
        bad_file = tmp_path / "unsupported_key.jsonl"
        bad_file.write_text(
            '{"id": "BAD-1", "title": "T", "description": "D", "category": "bug", '
            '"expected": {"type": "bug", "UNKNOWN_KEY_ABC": 123}}\n',
            encoding="utf-8",
        )
        with pytest.raises(EvalDatasetError) as exc_info:
            load_eval_cases(bad_file)
        assert "unsupported" in str(exc_info.value).lower()

    def test_dataset_loader_rejects_non_object_json_line(self, tmp_path):
        bad_file = tmp_path / "not_dict.jsonl"
        bad_file.write_text('["not", "a", "dictionary"]\n', encoding="utf-8")
        with pytest.raises(EvalDatasetError) as exc_info:
            load_eval_cases(bad_file)
        assert "must be an object" in str(exc_info.value).lower()

    def test_dataset_loader_rejects_invalid_type_enum(self, tmp_path):
        bad_file = tmp_path / "bad_enum.jsonl"
        bad_file.write_text(
            '{"id": "BAD-2", "title": "T", "description": "D", "category": "bug", '
            '"is_benchmark": true, "expected": {"type": "not_a_valid_type"}}\n',
            encoding="utf-8",
        )
        with pytest.raises(EvalDatasetError) as exc_info:
            load_eval_cases(bad_file)
        assert "expected.type 'not_a_valid_type' is invalid" in str(exc_info.value)


class TestMetricsAndAccuracy:
    """How Type Accuracy is computed."""

    def test_type_accuracy_computed_only_on_benchmark(self):
        results = [
            CaseEvalResult("B1", "bug", True, "passed", True, type_correct=True),
            CaseEvalResult("B2", "bug", True, "passed", True, type_correct=True),
            CaseEvalResult("B3", "bug", True, "passed", True, type_correct=True),
            CaseEvalResult("B4", "bug", True, "passed", True, type_correct=True),
            CaseEvalResult("B5", "bug", True, "passed", True, type_correct=True),
            CaseEvalResult("B6", "bug", True, "passed", True, type_correct=True),
            CaseEvalResult("B7", "bug", True, "passed", True, type_correct=True),
            CaseEvalResult("B8", "bug", True, "passed", True, type_correct=True),
            CaseEvalResult("B9", "bug", True, "failed", False, type_correct=False),
            CaseEvalResult("B10", "bug", True, "failed", False, type_correct=False),
            CaseEvalResult("NB1", "schema", False, "passed", True, type_correct=None),
            CaseEvalResult("NB2", "schema", False, "passed", True, type_correct=None),
        ]

        metrics = compute_eval_metrics(results)
        assert metrics["total_cases"] == 12
        assert metrics["benchmark_total"] == 10
        assert metrics["type_correct"] == 8
        assert metrics["type_accuracy"] == 0.80
        assert metrics["accuracy_met"] is True

    def test_accuracy_fails_when_below_80_percent(self):
        results = [
            CaseEvalResult("B1", "bug", True, "passed", True, type_correct=True),
            CaseEvalResult("B2", "bug", True, "failed", False, type_correct=False),
        ]
        metrics = compute_eval_metrics(results)
        assert metrics["type_accuracy"] == 0.50
        assert metrics["accuracy_met"] is False

    def test_accuracy_is_none_when_all_benchmark_blocked(self):
        results = [
            CaseEvalResult("B1", "bug", True, "blocked", False, type_correct=None),
            CaseEvalResult("B2", "bug", True, "blocked", False, type_correct=None),
        ]
        metrics = compute_eval_metrics(results)
        assert metrics["type_accuracy"] is None
        assert metrics["accuracy_met"] is False


class TestAccuracyVerdict:
    BASE = {"target_type_accuracy": 0.8, "benchmark_blocked": 0}

    def test_met(self):
        assert accuracy_verdict({**self.BASE, "type_accuracy": 0.9, "accuracy_met": True}).startswith("✅")

    def test_below_target_says_so(self):
        verdict = accuracy_verdict({**self.BASE, "type_accuracy": 0.727, "accuracy_met": False})
        assert "< 80%" in verdict

    def test_enough_accuracy_but_blocked_benchmark_names_the_blocked_count(self):
        verdict = accuracy_verdict(
            {**self.BASE, "type_accuracy": 0.909, "accuracy_met": False, "benchmark_blocked": 1}
        )
        assert "< 80%" not in verdict
        assert "1 benchmark case(s) blocked" in verdict

    def test_not_run(self):
        assert "NOT RUN" in accuracy_verdict({**self.BASE, "type_accuracy": None, "accuracy_met": False})


class TestRunnerExecution:
    """Suite execution by the runner."""

    def test_runner_returns_blocked_when_backend_missing(self, tmp_path, monkeypatch):
        import agent_system.interfaces.adapter
        import evals.faults
        from agent_system.interfaces.adapter import BackendMissingError

        def mock_resolve():
            raise BackendMissingError("Backend orchestrator is not implemented")

        def mock_resolve_chat():
            raise BackendMissingError("Orchestrator chat path is not ready")

        def mock_build_fault(fault):
            raise BackendMissingError("Backend orchestrator or FakeLLM hook is missing")

        monkeypatch.setattr(agent_system.interfaces.adapter, "resolve_processor", mock_resolve)
        monkeypatch.setattr(agent_system.interfaces.adapter, "resolve_chat_turn", mock_resolve_chat)
        monkeypatch.setattr(evals.faults, "build_fault_processor", mock_build_fault)

        cases_file = Path("evals/cases.jsonl")
        reports_dir = tmp_path / "reports"

        code = run_evals(cases_path=cases_file, reports_dir=reports_dir)
        assert code == EXIT_EVAL_BLOCKED

        assert (reports_dir / "latest_report.json").exists()
        assert (reports_dir / "latest_report.csv").exists()

    def test_runner_succeeds_with_perfect_processor(self, tmp_path):
        def mock_processor(req: IssueRequest) -> IssueResult:
            ticket = make_valid_ticket(
                type="bug",
                route="fix",
                affected_files=["calculator.py"],
                missing_info=[],
                clarification_questions=[],
            )
            if "power" in req.title.lower():
                ticket.type = "feature"
                ticket.route = "implement"
                ticket.priority = "P2"
            elif "average" in req.title.lower() or "how does" in req.title.lower():
                ticket.type = "question"
                ticket.route = "answer"
                ticket.priority = "P3"
            elif "missing" in req.title.lower() or "slow" in req.title.lower():
                ticket.route = "ask_user"
                ticket.missing_info = ["Missing reproducible steps"]
                ticket.clarification_questions = ["Can you provide input values?"]
            elif "recipe" in req.title.lower() or "override" in req.title.lower():
                ticket.type = "invalid"
                ticket.route = "reject"
            elif "refactor" in req.title.lower():
                ticket.type = "refactor"
                ticket.route = "implement"
                ticket.affected_files = ["user_service.py"]
            elif "docstrings" in req.title.lower():
                ticket.type = "docs"
                ticket.route = "implement"
                ticket.affected_files = ["user_service.py"]

            return make_valid_result(ticket=ticket)

        def mock_chat_turn(turn: ChatTurn) -> ChatOutcome:
            # Chat path: the first line of the message is the title, as when a user types into the UI
            req = IssueRequest(
                title=turn.user_text.splitlines()[0],
                description=turn.user_text,
                repo_path=turn.repo_path,
            )
            return ChatOutcome(route="issue_agent", reply="ok", run_id="r", result=mock_processor(req))

        cases_file = Path("evals/cases.jsonl")
        reports_dir = tmp_path / "reports_mock"

        subset_file = tmp_path / "subset.jsonl"
        with open(cases_file, "r", encoding="utf-8") as fin:
            lines = [fin.readline() for _ in range(3)]
        subset_file.write_text("".join(lines), encoding="utf-8")

        code = run_evals(
            cases_path=subset_file,
            chat_turn=mock_chat_turn,
            reports_dir=reports_dir,
        )

        assert code == EXIT_EVAL_ALL_PASSED
        with open(reports_dir / "latest_report.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["metrics"]["passed"] == 3
            assert data["metrics"]["type_accuracy"] == 1.0

    def test_runner_continues_after_case_exception(self):
        case1 = EvalCase("C1", "T1", "D1", category="c1")
        case2 = EvalCase("C2", "T2", "D2", category="c2")

        call_count = 0

        def faulty_chat_turn(turn: ChatTurn) -> ChatOutcome:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Crash in case 1")
            return ChatOutcome(route="issue_agent", reply="ok", run_id="r", result=make_valid_result())

        res1 = run_single_case(case1, chat_turn=faulty_chat_turn)
        res2 = run_single_case(case2, chat_turn=faulty_chat_turn)

        assert res1.status == "failed"
        assert "Crash in case 1" in res1.error
        assert res2.status == "passed"


class TestChatPath:
    """The user path (text -> router -> issue_agent) and the separation of mechanism cases."""

    def test_conversation_route_fails_with_reason_not_blocked(self):
        case = EvalCase("C1", "Recipe for brownies", "Please give a recipe", category="out_of_scope_invalid",
                        expected={"type": "invalid", "route": "reject"})

        def chat_turn(turn: ChatTurn) -> ChatOutcome:
            return ChatOutcome(route="conversation_agent", reply="I only support issues", run_id="r")

        res = run_single_case(case, chat_turn=chat_turn)

        assert res.path == "chat"
        assert res.status == "failed"
        assert res.type_correct is False
        assert res.actual_route == "conversation_agent"
        assert any("conversation_agent" in f for f in res.failures)

    def test_user_text_is_title_then_description_and_clarification_is_followed_up(self):
        case = EvalCase("C2", "Calc failed", "It failed in production", clarified_text="It is divide()",
                        expected={"type": "bug", "route": "fix"})
        pending = IssueRequest(title="Calc failed", description="It failed in production", repo_path=case.repo_path)
        turns: list[ChatTurn] = []

        def chat_turn(turn: ChatTurn) -> ChatOutcome:
            turns.append(turn)
            if turn.pending is None:
                return ChatOutcome(
                    route="issue_agent", reply="More information needed", run_id="r1",
                    result=make_valid_result(status="needs_user"), pending=pending,
                )
            return ChatOutcome(route="issue_agent", reply="ok", run_id="r2", result=make_valid_result())

        res = run_single_case(case, chat_turn=chat_turn)

        assert len(turns) == 2
        assert turns[0].user_text == "Calc failed\nIt failed in production"
        assert turns[1].user_text == "It is divide()"
        assert turns[1].pending == pending
        assert res.status == "passed"

    def test_mechanism_case_uses_processor_not_chat(self):
        case = EvalCase("C3", "TypeError", "Schema retry", fault_injection={"type": "schema_retry", "attempts": 2},
                        expected={"type": "bug"})
        chat_calls: list[ChatTurn] = []

        def chat_turn(turn: ChatTurn) -> ChatOutcome:
            chat_calls.append(turn)
            return ChatOutcome(route="issue_agent", reply="", run_id="r")

        res = run_single_case(case, processor=lambda req: make_valid_result(), chat_turn=chat_turn)

        assert res.path == "mechanism"
        assert chat_calls == []

    def test_select_cases_splits_chat_and_mechanism_suites(self):
        from evals.runner import select_cases

        cases = load_eval_cases(Path("evals/cases.jsonl"))
        chat = select_cases(cases, "chat")
        mechanism = select_cases(cases, "mechanism")

        assert {c.id for c in mechanism} == {"UC01-CASE-09", "UC01-CASE-10", "UC01-CASE-11"}
        assert len(chat) + len(mechanism) == len(cases)
        assert not any(c.fault_injection for c in chat)
        assert select_cases(cases, "all") == cases


class TestEvaluatorEvidenceIntegrity:
    """Regression tests for evaluator scoring gaps found during the audit."""

    def test_evaluator_rejects_attempts_8_9_10_without_validation(self):
        case = EvalCase(
            id="UC01-CASE-10",
            title="Schema exhaustion",
            description="All attempts fail schema",
            category="schema_error_exhausted",
            expected={"status": "failed", "max_attempts": 3},
        )
        fake_result = IssueResult(
            status="failed",
            ticket=None,
            message="Could not produce a valid ticket after 3 generations: invalid_schema",
            run_id="run-1",
            trace=[
                {"step": "llm_request", "attempt": 8},
                {"step": "llm_request", "attempt": 9},
                {"step": "llm_request", "attempt": 10},
                {"step": "retry_exhausted", "attempts": 3},
            ],
        )
        eval_res = evaluate_ticket_against_criteria(case, fake_result)
        assert eval_res.passed is False
        assert any("order [1, 2, 3]" in f for f in eval_res.failures)

    def test_evaluator_rejects_timeout_without_schema_failures(self):
        case = EvalCase(
            id="UC01-CASE-10",
            title="Schema exhaustion",
            description="All attempts fail schema",
            category="schema_error_exhausted",
            expected={"status": "failed", "max_attempts": 3},
        )
        fake_result = IssueResult(
            status="failed",
            ticket=None,
            message="Timeout",
            run_id="run-2",
            trace=[
                {"step": "llm_request", "attempt": 1},
                {"step": "llm_error", "attempt": 1, "error_type": "timeout", "error": "timeout"},
                {"step": "llm_request", "attempt": 2},
                {"step": "llm_request", "attempt": 3},
            ],
        )
        eval_res = evaluate_ticket_against_criteria(case, fake_result)
        assert eval_res.passed is False
        assert any("LLM infrastructure error" in f for f in eval_res.failures)

    def test_evaluator_rejects_recovery_missing_retry_or_validation_success(self):
        case = EvalCase(
            id="UC01-CASE-09",
            title="Schema recovery",
            description="Attempt 1 fails, attempt 2 recovers",
            category="schema_error_retry",
            expected={"type": "bug", "route": "fix", "schema_recovery": True},
        )
        fake_result = IssueResult(
            status="completed",
            ticket=make_valid_ticket(type="bug", route="fix"),
            message="OK",
            run_id="run-3",
            trace=[
                {"step": "generate", "attempt": 1, "valid": False},
                {"step": "generate", "attempt": 2},
                {"step": "route", "selected": "fix"},
            ],
        )
        eval_res = evaluate_ticket_against_criteria(case, fake_result)
        assert eval_res.passed is False
        assert any("requires exactly 2 generations" in f or "missing the retry step" in f for f in eval_res.failures)

    def test_evaluator_rejects_success_warning_as_tool_error(self):
        case = EvalCase(
            id="UC01-CASE-11",
            title="Tool error",
            description="search_code fails",
            category="tool_error",
            fault_injection={"type": "tool_error", "tool": "search_code"},
            expected={"type": "question", "route": "answer", "handled_gracefully": True},
        )
        fake_result = IssueResult(
            status="completed",
            ticket=make_valid_ticket(type="question", route="answer"),
            message="OK",
            run_id="run-4",
            warnings=["search_code completed successfully"],
            trace=[
                {"step": "tool_result", "tool": "search_code", "success": True},
                {"step": "route", "selected": "answer"},
            ],
        )
        eval_res = evaluate_ticket_against_criteria(case, fake_result)
        assert eval_res.passed is False
        assert any("error evidence" in f for f in eval_res.failures)

    def test_evaluator_rejects_missing_validator_warning_as_file_filter(self):
        case = EvalCase(
            id="UC01-CASE-07",
            title="Hallucinated file",
            description="Nonexistent file mentioned",
            category="hallucinated_file",
            expected={"type": "bug", "route": "fix", "require_warning_for_invalid_file": True},
        )
        fake_result = IssueResult(
            status="completed",
            ticket=make_valid_ticket(type="bug", route="fix", affected_files=["calculator.py"]),
            message="OK",
            run_id="run-5",
            warnings=["No validation module: affected_files were not verified to exist in the repo."],
            trace=[{"step": "route", "selected": "fix"}],
        )
        eval_res = evaluate_ticket_against_criteria(case, fake_result)
        assert eval_res.passed is False
        assert any("'No validation module' notice is not accepted" in f for f in eval_res.failures)

    def test_mentions_keys_require_file_name_in_missing_info_and_question(self):
        case = EvalCase(
            id="UC01-CASE-07",
            title="Hallucinated file",
            description="Nonexistent file mentioned",
            category="hallucinated_file",
            expected={
                "type": "bug",
                "route": "ask_user",
                "missing_info_empty": False,
                "clarification_questions_required": True,
                "mentions_in_missing_info": ["nonexistent_auth_service.py"],
                "mentions_in_clarification": ["nonexistent_auth_service.py"],
            },
        )
        ticket = make_valid_ticket(
            type="bug",
            route="ask_user",
            affected_files=[],
            missing_info=["File 'nonexistent_auth_service.py' was not found in the repository."],
            clarification_questions=["Could you double-check the file name NONEXISTENT_AUTH_SERVICE.py?"],
        )
        ok = evaluate_ticket_against_criteria(case, make_valid_result(status="needs_user", ticket=ticket))
        assert ok.passed is True

        ticket.missing_info = ["Missing reproduction details"]
        bad = evaluate_ticket_against_criteria(case, make_valid_result(status="needs_user", ticket=ticket))
        assert bad.passed is False
        assert any("missing_info to mention 'nonexistent_auth_service.py'" in f for f in bad.failures)

    def test_evaluator_requires_real_file_filtering_for_case_07(self):
        case = EvalCase(
            id="UC01-CASE-07",
            title="Hallucinated file",
            description="Nonexistent file mentioned",
            category="hallucinated_file",
            expected={"type": "bug", "route": "fix", "require_warning_for_invalid_file": True, "affected_files_exist": True},
        )
        fake_result = IssueResult(
            status="completed",
            ticket=make_valid_ticket(type="bug", route="fix", affected_files=["calculator.py"]),
            message="OK",
            run_id="run-6",
            warnings=["Removed affected_files that do not exist or are outside the repo: nonexistent_auth_service.py"],
            trace=[
                {"step": "affected_files_filtered", "removed": ["nonexistent_auth_service.py"], "kept": ["calculator.py"]},
                {"step": "route", "selected": "fix"},
            ],
        )
        eval_res = evaluate_ticket_against_criteria(case, fake_result)
        assert eval_res.passed is True
