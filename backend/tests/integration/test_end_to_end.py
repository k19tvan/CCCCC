"""
End-to-end integration tests for the whole UC-01 pipeline.

- Skipped only when the `agent_system.orchestrator` module does not exist, or when it exposes no
  FakeLLM/offline hook to run without a network.
- If the module exists but has a syntax or dependency import error, the test fails instead of
  hiding the error behind a "skip".
- Everything runs fully OFFLINE through the public `process_issue` API; a guard blocks any socket connection.
"""

import importlib.util
import json
from pathlib import Path
import pytest

from agent_system.interfaces.cli import EXIT_SUCCESS, run_cli
from agent_system.interfaces.server import handle_issue
from evals.faults import FaultyToolRegistry
from agent_system.core.contracts import IssueRequest, IssueResult
from agent_system.core.llm.client import FakeLLM
from agent_system.subagents.issue_agent.tools.registry import default_registry

SAMPLE_REPO_PATH = str(Path(__file__).resolve().parent.parent / "fixtures" / "sample_repo")

orchestrator_spec = importlib.util.find_spec("agent_system.orchestrator")
MODULE_EXISTS = orchestrator_spec is not None

HAS_BACKEND = False
HAS_OFFLINE_HOOK = False
SKIP_REASON = ""

if not MODULE_EXISTS:
    SKIP_REASON = "Module 'agent_system.orchestrator' does not exist yet."
else:
    # Deliberately no try/except: an internal import error in the backend must surface.
    import agent_system.orchestrator as orchestrator

    if not hasattr(orchestrator, "process_issue") or not callable(orchestrator.process_issue):
        SKIP_REASON = "Module agent_system.orchestrator exists but does not define a callable 'process_issue'."
    else:
        HAS_BACKEND = True
        process_issue = orchestrator.process_issue

        # Make sure the tests cannot hit a real, paid network/API.
        if hasattr(orchestrator, "set_llm_client") or hasattr(orchestrator, "FakeLLM"):
            HAS_OFFLINE_HOOK = True
        else:
            SKIP_REASON = (
                "Module orchestrator exists but provides no FakeLLM/offline hook "
                "to run integration tests safely without the real network."
            )


@pytest.fixture(autouse=True)
def guard_against_network_calls(monkeypatch):
    """Block every outgoing socket connection so integration tests stay fully offline."""
    import socket

    def forbidden_connect(*args, **kwargs):
        raise RuntimeError(
            "Offline test violation: integration tests must not make network/LLM API connections."
        )

    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)


@pytest.mark.skipif(
    not (HAS_BACKEND and HAS_OFFLINE_HOOK),
    reason=SKIP_REASON or "Backend or FakeLLM hook not ready.",
)
class TestEndToEndPipeline:
    """End-to-end suite, run once the backend orchestrator and the FakeLLM hook are available."""

    def test_e2e_clear_bug_success_flow(self):
        """Success flow: a specific bug triggers gather_context (>=2 tools), classification and the fix route."""
        request = IssueRequest(
            title="ZeroDivisionError in Calculator.divide when denominator is zero",
            description=(
                "Running divide(10, 0) in calculator.py causes ZeroDivisionError. "
                "Stacktrace shows line 25 in calculator.py: raise ZeroDivisionError('Cannot divide by zero')."
            ),
            repo_path=SAMPLE_REPO_PATH,
        )

        result: IssueResult = process_issue(request)

        assert result.status == "completed"
        assert result.run_id != ""
        assert result.ticket is not None

        ticket = result.ticket
        assert ticket.type == "bug"
        assert ticket.route == "fix"
        assert ticket.priority in {"P0", "P1"}
        assert ticket.priority_reason != ""

        assert len(ticket.affected_files) <= 5
        for f in ticket.affected_files:
            full_file = Path(SAMPLE_REPO_PATH) / f
            assert full_file.exists() and full_file.is_file(), (
                f"File {f} does not exist or is a directory in sample_repo"
            )

        assert len(ticket.acceptance_criteria) >= 1
        assert all(ac.strip() for ac in ticket.acceptance_criteria)

        tools_called = set()
        for step in result.trace:
            if "tools" in step and isinstance(step["tools"], list):
                tools_called.update(step["tools"])
            if step.get("tool"):
                tools_called.add(step["tool"])

        valid_tools = {"list_files", "read_file", "search_code"}
        called_known_tools = tools_called.intersection(valid_tools)
        assert len(called_known_tools) >= 2, (
            f"Trace must show at least 2 tools called, actual: {tools_called}"
        )

    def test_e2e_missing_info_needs_user_flow(self):
        """Missing-information flow: a vague issue is routed to ask_user with clarification questions."""
        request = IssueRequest(
            title="Calculator crashed unexpectedly",
            description="A math function crashed without details.",
            repo_path=SAMPLE_REPO_PATH,
        )

        result: IssueResult = process_issue(request)

        assert result.status == "needs_user"
        assert result.ticket is not None
        assert result.ticket.route == "ask_user"
        assert len(result.ticket.missing_info) >= 1
        assert len(result.ticket.clarification_questions) >= 1
        assert all(cq.strip() for cq in result.ticket.clarification_questions)

    def test_e2e_followup_with_clarified_text(self):
        """Follow-up flow: resend the request with clarified_text answering the agent's questions."""
        followup_request = IssueRequest(
            title="Calculator crashed unexpectedly",
            description="A math function crashed without details.",
            clarified_text="The function is calculate_average([]) with an empty list raising ValueError.",
            repo_path=SAMPLE_REPO_PATH,
        )

        result: IssueResult = process_issue(followup_request)

        assert result.status == "completed"
        assert result.ticket is not None
        assert result.ticket.type == "bug"
        assert result.ticket.route == "fix"
        assert len(result.ticket.missing_info) == 0

    def test_e2e_web_api_http_flow(self):
        """Web API boundary: handle_issue calls the adapter -> the real public process_issue."""
        payload = {
            "title": "ZeroDivisionError in Calculator.divide when denominator is zero",
            "description": "Running divide(10, 0) in calculator.py causes ZeroDivisionError.",
            "repo_path": SAMPLE_REPO_PATH,
        }
        res = handle_issue(payload)
        assert res["status"] == "completed"
        assert res["ticket"]["type"] == "bug"
        assert res["ticket"]["route"] == "fix"
        assert len(res["run_id"]) > 0
        assert len(res["trace"]) >= 2

    def test_e2e_cli_entry_flow(self, capsys):
        """CLI boundary: run_cli with sample SAMPLE-01 and the --json flag."""
        exit_code = run_cli(["--sample", "SAMPLE-01", "--json"])
        assert exit_code == EXIT_SUCCESS
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "completed"
        assert data["ticket"]["type"] == "bug"
        assert data["ticket"]["route"] == "fix"

    def test_e2e_schema_recovery_retry_flow(self):
        """Schema recovery: wrong priority on attempt 1, fixed on attempt 2 -> completed."""
        valid_ticket = {
            "summary": "Fix ZeroDivisionError in Calculator.divide",
            "type": "bug",
            "priority": "P1",
            "priority_reason": "Crash on zero division",
            "affected_files": ["calculator.py"],
            "reproduction_steps": ["divide(10, 0)"],
            "acceptance_criteria": ["Raise ZeroDivisionError instead of crash"],
            "missing_info": [],
            "route": "fix",
            "clarification_questions": [],
        }
        invalid_ticket = dict(valid_ticket, priority="P9")
        llm = FakeLLM([invalid_ticket, valid_ticket])

        request = IssueRequest(
            title="Fix ZeroDivisionError in Calculator",
            description="divide(10, 0) crashes in calculator.py",
            repo_path=SAMPLE_REPO_PATH,
        )
        result = process_issue(request, llm=llm)

        assert result.status == "completed"
        assert result.ticket is not None
        assert result.ticket.priority == "P1"

        # Exactly 2 requests: attempt 1 validation false, a retry step, attempt 2 validation true
        gen_steps = [s for s in result.trace if s.get("step") == "llm_request"]
        assert len(gen_steps) == 2
        assert [s.get("attempt") for s in gen_steps] == [1, 2]

        val_steps = [s for s in result.trace if s.get("step") == "validation"]
        assert len(val_steps) == 2
        assert val_steps[0]["valid"] is False
        assert val_steps[1]["valid"] is True

        retry_steps = [s for s in result.trace if s.get("step") == "retry"]
        assert len(retry_steps) == 1
        assert retry_steps[0]["attempt"] == 1
        assert retry_steps[0]["next_attempt"] == 2

    def test_e2e_schema_exhaustion_flow(self):
        """Schema retries exhausted: all 3 attempts invalid -> failed with retry_exhausted."""
        invalid_ticket = {
            "summary": "Invalid schema ticket",
            "type": "bug",
            "priority": "P9",  # invalid enum
            "priority_reason": "Testing exhaustion",
            "affected_files": ["calculator.py"],
            "reproduction_steps": ["crash"],
            "acceptance_criteria": ["done"],
            "missing_info": [],
            "route": "fix",
            "clarification_questions": [],
        }
        llm = FakeLLM([invalid_ticket, invalid_ticket, invalid_ticket])

        request = IssueRequest(
            title="Crash in Calculator",
            description="Testing schema exhaustion limit",
            repo_path=SAMPLE_REPO_PATH,
        )
        result = process_issue(request, llm=llm)

        assert result.status == "failed"
        assert result.ticket is None
        assert "3 generations" in result.message

        gen_steps = [s for s in result.trace if s.get("step") == "llm_request"]
        assert len(gen_steps) == 3

        exhausted_steps = [s for s in result.trace if s.get("step") == "retry_exhausted"]
        assert len(exhausted_steps) == 1
        assert exhausted_steps[0]["attempts"] == 3

    def test_e2e_hallucinated_file_filtered_flow(self):
        """Hallucinated file filtering: the LLM proposes a nonexistent file; the pipeline removes it and warns."""
        ticket_with_hallucination = {
            "summary": "Fix ZeroDivisionError with hallucinated file",
            "type": "bug",
            "priority": "P1",
            "priority_reason": "Crash on zero division",
            "affected_files": ["calculator.py", "ghost_nonexistent_service.py"],
            "reproduction_steps": ["divide(10, 0)"],
            "acceptance_criteria": ["Raise ZeroDivisionError"],
            "missing_info": [],
            "route": "fix",
            "clarification_questions": [],
        }
        llm = FakeLLM([ticket_with_hallucination])

        request = IssueRequest(
            title="ZeroDivisionError in Calculator",
            description="Testing filtering of nonexistent files",
            repo_path=SAMPLE_REPO_PATH,
        )
        result = process_issue(request, llm=llm)

        assert result.status == "completed"
        assert result.ticket is not None
        assert "ghost_nonexistent_service.py" not in result.ticket.affected_files
        assert "calculator.py" in result.ticket.affected_files

        # The trace records affected_files_filtered
        filter_steps = [s for s in result.trace if s.get("step") == "affected_files_filtered"]
        assert len(filter_steps) == 1
        assert "ghost_nonexistent_service.py" in filter_steps[0]["removed"]
        assert "calculator.py" in filter_steps[0]["kept"]

        # The result warnings mention the removed file
        assert any("ghost_nonexistent_service.py" in w for w in result.warnings)

    def test_e2e_tool_error_resilience_flow(self):
        """Tool failure resilience: a tool raising OSError is handled gracefully."""
        faulty_reg = FaultyToolRegistry(default_registry, "search_code", "Simulated disk permission error")

        request = IssueRequest(
            title="How does calculator divide handle zero?",
            description="Codebase question about divide method in calculator.py",
            repo_path=SAMPLE_REPO_PATH,
        )
        result = process_issue(request, registry=faulty_reg)

        assert result.status in ("completed", "needs_user")
        assert result.ticket is not None

        # The trace records a failed tool_result or a tool_error
        tool_results = [
            s for s in result.trace
            if (s.get("step") == "tool_result" and s.get("tool") == "search_code")
            or (s.get("step") == "tool_error" and s.get("tool") == "search_code")
        ]
        assert len(tool_results) >= 1
        assert any(t.get("success") is False or t.get("step") == "tool_error" for t in tool_results)

        # The warnings carry the tool error
        assert any("search_code" in w for w in result.warnings)
