"""Unit tests for the demo CLI.

Covers argument parsing, input validation (exit code 1), a single processor call per request,
human-readable and JSON output, the trace flag, exit codes (0, 1, 2, 3, 4, 130) and sample loading.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from agent_system.interfaces.adapter import BackendExecutionError, BackendMissingError, call_agent
from agent_system.interfaces.cli import (
    EXIT_BACKEND_MISSING,
    EXIT_FAILED,
    EXIT_INPUT_ERROR,
    EXIT_INTERRUPT,
    EXIT_NEEDS_USER,
    EXIT_SUCCESS,
    run_cli,
)
from agent_system.core.contracts import IssueRequest, IssueResult, IssueTicket


def make_sample_ticket(**kwargs) -> IssueTicket:
    defaults = {
        "type": "bug",
        "priority": "P1",
        "priority_reason": "divide crashes the system",
        "affected_files": ["calculator.py"],
        "reproduction_steps": ["Call divide(10, 0)"],
        "acceptance_criteria": ["No crash, raise ValueError instead"],
        "missing_info": [],
        "route": "fix",
        "clarification_questions": [],
        "summary": "Division by zero in calculator",
    }
    defaults.update(kwargs)
    return IssueTicket(**defaults)


def make_sample_result(status: str = "completed", **kwargs) -> IssueResult:
    ticket = kwargs.pop("ticket", make_sample_ticket() if status != "failed" else None)
    defaults = {
        "status": status,
        "ticket": ticket,
        "message": "Processed successfully",
        "warnings": [],
        "run_id": "test-run-12345",
        "trace": [
            {"step": "gather_context", "tools": ["list_files", "read_file"]},
            {"step": "validate_schema", "valid": True},
            {"step": "route", "selected": "fix"},
        ],
    }
    defaults.update(kwargs)
    return IssueResult(**defaults)


class TestAppCLIValidation:
    def test_missing_title_returns_input_error(self, capsys):
        code = run_cli(["-d", "Bug description"])
        assert code == EXIT_INPUT_ERROR
        stderr = capsys.readouterr().err
        assert "issue title" in stderr

    def test_empty_whitespace_title_returns_input_error(self, capsys):
        code = run_cli(["-t", "   ", "-d", "Bug description"])
        assert code == EXIT_INPUT_ERROR
        stderr = capsys.readouterr().err
        assert "issue title" in stderr

    def test_missing_description_returns_input_error(self, capsys):
        code = run_cli(["-t", "Bug title"])
        assert code == EXIT_INPUT_ERROR
        stderr = capsys.readouterr().err
        assert "issue description" in stderr

    def test_nonexistent_repo_path_returns_input_error(self, capsys):
        code = run_cli(["-t", "Title", "-d", "Description", "-r", "path/does/not/exist"])
        assert code == EXIT_INPUT_ERROR
        stderr = capsys.readouterr().err
        assert "target repository path does not exist" in stderr

    def test_invalid_sample_id_returns_input_error(self, capsys):
        code = run_cli(["--sample", "NO-SUCH-ID"])
        assert code == EXIT_INPUT_ERROR
        stderr = capsys.readouterr().err
        assert "sample issue with ID 'NO-SUCH-ID' not found" in stderr


class TestAppCLIExecution:
    def test_calls_processor_once_with_proper_request(self):
        mock_proc = MagicMock(return_value=make_sample_result(status="completed"))

        code = run_cli(
            [
                "-t",
                "Test title",
                "-d",
                "Test description",
                "-r",
                "tests/fixtures/sample_repo",
                "-c",
                "More details",
            ],
            processor=mock_proc,
        )

        assert code == EXIT_SUCCESS
        assert mock_proc.call_count == 1
        req: IssueRequest = mock_proc.call_args[0][0]
        assert req.title == "Test title"
        assert req.description == "Test description"
        assert req.clarified_text == "More details"
        assert req.repo_path == "tests/fixtures/sample_repo"

    def test_exit_code_completed_is_0(self):
        mock_proc = MagicMock(return_value=make_sample_result(status="completed"))
        code = run_cli(
            ["-t", "Bug", "-d", "Crash", "-r", "tests/fixtures/sample_repo"],
            processor=mock_proc,
        )
        assert code == EXIT_SUCCESS

    def test_exit_code_needs_user_is_2(self):
        ticket = make_sample_ticket(
            route="ask_user",
            missing_info=["Missing log file"],
            clarification_questions=["Can you attach the log file?"],
        )
        result = make_sample_result(status="needs_user", ticket=ticket)
        mock_proc = MagicMock(return_value=result)

        code = run_cli(
            ["-t", "Bug", "-d", "Missing info", "-r", "tests/fixtures/sample_repo"],
            processor=mock_proc,
        )
        assert code == EXIT_NEEDS_USER

    def test_exit_code_failed_is_3(self):
        result = make_sample_result(status="failed", ticket=None, message="Schema retry exceeded")
        mock_proc = MagicMock(return_value=result)

        code = run_cli(
            ["-t", "Bug", "-d", "Fail", "-r", "tests/fixtures/sample_repo"],
            processor=mock_proc,
        )
        assert code == EXIT_FAILED


class TestAppCLIOutputs:
    def test_json_mode_outputs_valid_json_to_stdout(self, capsys):
        expected_result = make_sample_result(status="completed")
        mock_proc = MagicMock(return_value=expected_result)

        code = run_cli(
            [
                "-t",
                "Bug",
                "-d",
                "Crash",
                "-r",
                "tests/fixtures/sample_repo",
                "--json",
            ],
            processor=mock_proc,
        )

        assert code == EXIT_SUCCESS
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["status"] == "completed"
        assert parsed["ticket"]["type"] == "bug"
        assert parsed["ticket"]["priority"] == "P1"
        assert parsed["run_id"] == "test-run-12345"

    def test_trace_flag_displays_trace_steps(self, capsys):
        result = make_sample_result(status="completed")
        mock_proc = MagicMock(return_value=result)

        code = run_cli(
            [
                "-t",
                "Bug",
                "-d",
                "Crash",
                "-r",
                "tests/fixtures/sample_repo",
                "--trace",
            ],
            processor=mock_proc,
        )

        assert code == EXIT_SUCCESS
        out = capsys.readouterr().out
        assert "EXECUTION STEP TRACE LOG" in out
        assert "gather_context" in out

    def test_route_fix_displays_uc01_triage_notice(self, capsys):
        ticket = make_sample_ticket(route="fix")
        result = make_sample_result(status="completed", ticket=ticket)
        mock_proc = MagicMock(return_value=result)

        run_cli(
            ["-t", "Bug", "-d", "Crash", "-r", "tests/fixtures/sample_repo"],
            processor=mock_proc,
        )

        out = capsys.readouterr().out
        assert "code is not modified yet" in out.lower()


class TestAppBackendHandling:
    def test_backend_missing_returns_exit_code_4(self, capsys):
        def failing_proc(_):
            raise BackendMissingError("Backend orchestrator is not implemented.")

        code = run_cli(
            ["-t", "Bug", "-d", "Crash", "-r", "tests/fixtures/sample_repo"],
            processor=failing_proc,
        )

        assert code == EXIT_BACKEND_MISSING
        stderr = capsys.readouterr().err
        assert "Backend orchestrator is not implemented" in stderr

    def test_backend_execution_error_returns_exit_code_3(self, capsys):
        def failing_proc(_):
            raise BackendExecutionError("LLM connection error.")

        code = run_cli(
            ["-t", "Bug", "-d", "Crash", "-r", "tests/fixtures/sample_repo"],
            processor=failing_proc,
        )

        assert code == EXIT_FAILED
        stderr = capsys.readouterr().err
        assert "LLM connection error" in stderr

    def test_keyboard_interrupt_returns_130(self, capsys):
        def interrupt_proc(_):
            raise KeyboardInterrupt()

        code = run_cli(
            ["-t", "Bug", "-d", "Crash", "-r", "tests/fixtures/sample_repo"],
            processor=interrupt_proc,
        )

        assert code == EXIT_INTERRUPT
        stderr = capsys.readouterr().err
        assert "Cancelled by the user" in stderr

    def test_resolve_processor_raises_backend_missing_when_unimportable(self, monkeypatch):
        from agent_system.interfaces.adapter import resolve_processor
        import sys
        # Keep the test independent of the real orchestrator being importable
        monkeypatch.setitem(sys.modules, "agent_system.orchestrator", None)
        with pytest.raises(BackendMissingError):
            resolve_processor()


class TestAppInteractiveMode:
    def test_interactive_followup_success_returns_0(self, monkeypatch):
        calls = []

        def two_stage_proc(req: IssueRequest) -> IssueResult:
            calls.append(req)
            if len(calls) == 1:
                return make_sample_result(
                    status="needs_user",
                    ticket=make_sample_ticket(
                        route="ask_user",
                        missing_info=["Missing data"],
                        clarification_questions=["Enter a list of numbers?"],
                    ),
                )
            else:
                return make_sample_result(status="completed")

        monkeypatch.setattr("builtins.input", lambda _: "Clarification: [1, 2, 3]")

        code = run_cli(
            ["-t", "Bug", "-d", "Crash", "-r", "tests/fixtures/sample_repo", "-i"],
            processor=two_stage_proc,
        )

        assert code == EXIT_SUCCESS
        assert len(calls) == 2
        assert calls[1].clarified_text == "Clarification: [1, 2, 3]"

    def test_interactive_followup_backend_execution_error_returns_exit_code_3(
        self, monkeypatch, capsys
    ):
        call_count = 0

        def failing_second_call(_: IssueRequest) -> IssueResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_sample_result(status="needs_user")
            raise BackendExecutionError("Network failure on the second call!")

        monkeypatch.setattr("builtins.input", lambda _: "Clarifying answer")

        code = run_cli(
            ["-t", "Bug", "-d", "Crash", "-r", "tests/fixtures/sample_repo", "-i"],
            processor=failing_second_call,
        )

        assert code == EXIT_FAILED
        stderr = capsys.readouterr().err
        assert "Network failure on the second call" in stderr

    def test_interactive_followup_eof_returns_130(self, monkeypatch, capsys):
        def first_call_proc(_: IssueRequest) -> IssueResult:
            return make_sample_result(status="needs_user")

        def raise_eof(_):
            raise EOFError()

        monkeypatch.setattr("builtins.input", raise_eof)

        code = run_cli(
            ["-t", "Bug", "-d", "Crash", "-r", "tests/fixtures/sample_repo", "-i"],
            processor=first_call_proc,
        )

        assert code == EXIT_INTERRUPT
        stderr = capsys.readouterr().err
        assert "Interactive mode stopped" in stderr

    def test_interactive_followup_keyboard_interrupt_returns_130(self, monkeypatch, capsys):
        def first_call_proc(_: IssueRequest) -> IssueResult:
            return make_sample_result(status="needs_user")

        def raise_sigint(_):
            raise KeyboardInterrupt()

        monkeypatch.setattr("builtins.input", raise_sigint)

        code = run_cli(
            ["-t", "Bug", "-d", "Crash", "-r", "tests/fixtures/sample_repo", "-i"],
            processor=first_call_proc,
        )

        assert code == EXIT_INTERRUPT
