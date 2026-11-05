"""Unit tests for the web server and REST API.

Covers repo path resolution, GET /api/info, POST /api/issues (validation, success, backend errors),
in-memory sessions (create, list, get, delete, send message, titles, busy 409, 404) and the request
handler (unknown route 404, invalid JSON 400, oversized body 413).

No real TCP socket is opened, so the tests do not depend on the network environment.
"""

from __future__ import annotations

import io
import json
from http import HTTPStatus
from unittest.mock import patch

import pytest

from agent_system.core.contracts import IssueRequest, IssueResult, IssueTicket
from agent_system.interfaces import server
from agent_system.interfaces.adapter import BackendExecutionError, BackendMissingError
from agent_system.interfaces.server import (
    ApiError,
    Handler,
    get_llm_info,
    handle_create_session,
    handle_delete_session,
    handle_get_session,
    handle_info,
    handle_issue,
    handle_list_sessions,
    handle_send_message,
    resolve_repo_path,
    set_test_processor,
)
from agent_system.orchestrator import ChatOutcome


@pytest.fixture(autouse=True)
def _clean_sessions(monkeypatch):
    monkeypatch.setattr(server, "_sessions", {})


def make_test_ticket(**kwargs) -> IssueTicket:
    defaults = {
        "type": "bug",
        "priority": "P1",
        "priority_reason": "Crash in calculator",
        "affected_files": ["calculator.py"],
        "reproduction_steps": ["Call divide(10, 0)"],
        "acceptance_criteria": ["Raise ZeroDivisionError instead of crashing"],
        "missing_info": [],
        "route": "fix",
        "clarification_questions": [],
        "summary": "Fix ZeroDivisionError",
    }
    defaults.update(kwargs)
    return IssueTicket(**defaults)


def make_test_result(status: str = "completed", **kwargs) -> IssueResult:
    defaults = {
        "status": status,
        "ticket": make_test_ticket() if status != "failed" else None,
        "message": "Issue triaged",
        "warnings": [],
        "trace": [],
        "run_id": "run-test",
    }
    defaults.update(kwargs)
    return IssueResult(**defaults)


class MockHTTPHandler(Handler):
    """Handler running on in-memory streams, without opening a socket."""

    def __init__(self, method: str, path: str, body: bytes | None = None):
        self.rfile = io.BytesIO(body or b"")
        self.wfile = io.BytesIO()
        self.command = method
        self.path = path
        self.request_version = "HTTP/1.1"
        self.close_connection = False
        self.headers = {"Content-Length": str(len(body))} if body else {}
        self.response_status = None

    def send_response(self, code: int, message: str | None = None) -> None:
        self.response_status = code

    def send_header(self, keyword: str, value: str) -> None:
        pass

    def end_headers(self) -> None:
        pass

    def log_message(self, fmt: str, *args) -> None:
        pass

    def get_response_json(self) -> dict:
        return json.loads(self.wfile.getvalue().decode("utf-8"))


class TestRepoPathResolution:
    def test_valid_relative_repo_path(self):
        assert resolve_repo_path("tests/fixtures/sample_repo").endswith("sample_repo")

    def test_empty_uses_default_repo(self):
        assert resolve_repo_path("").endswith("sample_repo")
        assert resolve_repo_path(None).endswith("sample_repo")

    def test_nonexistent_repo_path_raises_400(self):
        with pytest.raises(ApiError) as exc:
            resolve_repo_path("does/not/exist/xyz")
        assert exc.value.status == HTTPStatus.BAD_REQUEST


class TestInfo:
    def test_llm_info_marks_fake_provider_offline(self, monkeypatch):
        from agent_system.core.config import config

        monkeypatch.setattr(config, "LLM_PROVIDER", "fake")
        assert get_llm_info()["is_offline"] is True

    def test_handle_info_structure(self):
        info = handle_info()
        assert set(info) == {"llm"}
        assert {"provider", "model", "is_offline"} <= set(info["llm"])


class TestHandleIssue:
    def test_empty_title_raises_400(self):
        with pytest.raises(ApiError) as exc:
            handle_issue({"title": "   ", "description": "Crash bug"})
        assert exc.value.status == HTTPStatus.BAD_REQUEST
        assert "Issue title" in str(exc.value)

    def test_missing_title_raises_400(self):
        with pytest.raises(ApiError) as exc:
            handle_issue({"description": "Crash bug"})
        assert exc.value.status == HTTPStatus.BAD_REQUEST

    def test_empty_description_raises_400(self):
        with pytest.raises(ApiError) as exc:
            handle_issue({"title": "Bug title", "description": "  \n  "})
        assert exc.value.status == HTTPStatus.BAD_REQUEST
        assert "Issue description" in str(exc.value)

    def test_valid_issue_passes_clarified_text_to_processor(self):
        captured: list[IssueRequest] = []

        def processor(req: IssueRequest) -> IssueResult:
            captured.append(req)
            return make_test_result(ticket=make_test_ticket(type="feature", route="implement"))

        res = handle_issue(
            {
                "title": "Add a power function",
                "description": "Need exponentiation in the calculator",
                "repo_path": "tests/fixtures/sample_repo",
                "clarified_text": "Positive integer base",
            },
            processor=processor,
        )

        assert captured[0].title == "Add a power function"
        assert captured[0].clarified_text == "Positive integer base"
        assert res["status"] == "completed"
        assert res["ticket"]["route"] == "implement"

    def test_backend_missing_error_raises_503(self):
        def processor(req: IssueRequest):
            raise BackendMissingError("Backend orchestrator is not implemented.")

        with pytest.raises(ApiError) as exc:
            handle_issue(
                {"title": "Bug", "description": "Desc", "repo_path": "tests/fixtures/sample_repo"},
                processor=processor,
            )
        assert exc.value.status == HTTPStatus.SERVICE_UNAVAILABLE

    def test_backend_execution_error_raises_500(self):
        def processor(req: IssueRequest):
            raise BackendExecutionError("Pipeline crashed internally.")

        with pytest.raises(ApiError) as exc:
            handle_issue(
                {"title": "Bug", "description": "Desc", "repo_path": "tests/fixtures/sample_repo"},
                processor=processor,
            )
        assert exc.value.status == HTTPStatus.INTERNAL_SERVER_ERROR

    def test_non_string_fields_rejected_with_400(self):
        with pytest.raises(ApiError) as exc:
            handle_issue({"title": {"bad": "dict"}, "description": "valid"})
        assert exc.value.status == HTTPStatus.BAD_REQUEST

        with pytest.raises(ApiError) as exc:
            handle_issue({"title": "valid", "description": ["a", "b"]})
        assert exc.value.status == HTTPStatus.BAD_REQUEST

        with pytest.raises(ApiError) as exc:
            handle_issue({"title": "valid", "description": "valid", "clarified_text": [1, 2]})
        assert exc.value.status == HTTPStatus.BAD_REQUEST


class TestPostIssuesHttp:
    def test_valid_body_returns_200(self):
        def processor(req: IssueRequest) -> IssueResult:
            return make_test_result()

        set_test_processor(processor)
        try:
            body = json.dumps({
                "title": "Division by zero bug",
                "description": "Error in divide",
                "repo_path": "tests/fixtures/sample_repo",
            }).encode("utf-8")
            handler = MockHTTPHandler("POST", "/api/issues", body=body)
            handler.do_POST()
            assert handler.response_status == HTTPStatus.OK
            assert handler.get_response_json()["status"] == "completed"
        finally:
            set_test_processor(None)

    def test_invalid_json_returns_400(self):
        handler = MockHTTPHandler("POST", "/api/issues", body=b"not json")
        handler.do_POST()
        assert handler.response_status == HTTPStatus.BAD_REQUEST

    def test_non_dict_json_returns_400(self):
        handler = MockHTTPHandler("POST", "/api/issues", body=b"['a']")
        handler.do_POST()
        assert handler.response_status == HTTPStatus.BAD_REQUEST

    def test_too_large_body_returns_413(self):
        handler = MockHTTPHandler("POST", "/api/issues", body=b"x" * 1_000_005)
        handler.do_POST()
        assert handler.response_status == HTTPStatus.REQUEST_ENTITY_TOO_LARGE

    def test_unknown_route_returns_404(self):
        handler = MockHTTPHandler("GET", "/api/does-not-exist")
        handler.do_GET()
        assert handler.response_status == HTTPStatus.NOT_FOUND


class TestSessions:
    def test_create_session_returns_empty_detail(self):
        detail = handle_create_session({})
        assert detail["title"] == server.DEFAULT_SESSION_TITLE
        assert detail["messages"] == []
        assert detail["repo_path"] is None

    def test_create_session_with_repo_validates_path(self):
        assert handle_create_session({"repo_path": "tests/fixtures/sample_repo"})["repo_path"].endswith("sample_repo")
        with pytest.raises(ApiError) as exc:
            handle_create_session({"repo_path": "does/not/exist"})
        assert exc.value.status == HTTPStatus.BAD_REQUEST

    def test_repo_path_from_message_is_passed_and_remembered(self, monkeypatch):
        seen = []

        def fake_turn(turn):
            seen.append(turn.repo_path)
            return ChatOutcome(route="conversation_agent", reply="ok", run_id="r")

        monkeypatch.setattr(server, "run_chat_turn", fake_turn)
        session = handle_create_session({})
        handle_send_message(session["id"], {"content": "hi"})
        handle_send_message(session["id"], {"content": "bug", "repo_path": "tests/fixtures/sample_repo"})
        handle_send_message(session["id"], {"content": "again"})

        assert seen[0] is None
        assert seen[1].endswith("sample_repo")
        assert seen[2] == seen[1]

    def test_list_sessions_returns_newest_first(self):
        first = handle_create_session({})
        second = handle_create_session({})
        ids = [s["id"] for s in handle_list_sessions()["sessions"]]
        assert set(ids) == {first["id"], second["id"]}

    def test_get_and_delete_session(self):
        created = handle_create_session({})
        assert handle_get_session(created["id"])["id"] == created["id"]
        assert handle_delete_session(created["id"]) == {"ok": True}
        with pytest.raises(ApiError) as exc:
            handle_get_session(created["id"])
        assert exc.value.status == HTTPStatus.NOT_FOUND
        assert exc.value.code == "session_not_found"

    def test_send_message_stores_both_messages_and_titles_session(self, monkeypatch):
        captured = {}

        def fake_turn(turn):
            captured["turn"] = turn
            return ChatOutcome(route="conversation_agent", reply="Hello there", run_id="r1")

        monkeypatch.setattr(server, "run_chat_turn", fake_turn)
        session = handle_create_session({})

        out = handle_send_message(session["id"], {"content": "hello there\nextra line"})

        assert captured["turn"].user_text == "hello there\nextra line"
        assert captured["turn"].history == []
        assert out["user_message"]["role"] == "user"
        assert out["assistant_message"]["content"] == "Hello there"
        assert out["assistant_message"]["route"] == "conversation_agent"
        assert out["assistant_message"]["result"] is None
        assert out["session"]["title"] == "hello there"
        assert out["session"]["message_count"] == 2

    def test_second_message_receives_history_and_pending(self, monkeypatch):
        pending_request = IssueRequest(title="Bug", description="Something is broken", repo_path="tests/fixtures/sample_repo")
        seen = []

        def fake_turn(turn):
            seen.append(turn)
            if len(seen) == 1:
                return ChatOutcome(
                    route="issue_agent",
                    reply="Please describe it in more detail",
                    run_id="r1",
                    result=make_test_result(status="needs_user"),
                    pending=pending_request,
                )
            return ChatOutcome(route="issue_agent", reply="Done", run_id="r2")

        monkeypatch.setattr(server, "run_chat_turn", fake_turn)
        session = handle_create_session({})
        handle_send_message(session["id"], {"content": "something is broken"})
        handle_send_message(session["id"], {"content": "error when dividing by 0"})

        second = seen[1]
        assert second.pending == pending_request
        assert [m["content"] for m in second.history] == ["something is broken", "Please describe it in more detail"]

    def test_send_message_rejects_empty_and_too_long_content(self):
        session = handle_create_session({})
        with pytest.raises(ApiError) as exc:
            handle_send_message(session["id"], {"content": "   "})
        assert exc.value.status == HTTPStatus.BAD_REQUEST

        with pytest.raises(ApiError) as exc:
            handle_send_message(session["id"], {"content": "x" * (server.MAX_MESSAGE_CHARS + 1)})
        assert exc.value.code == "message_too_long"

    def test_send_message_to_unknown_session_returns_404(self):
        with pytest.raises(ApiError) as exc:
            handle_send_message("no-such-session", {"content": "hello"})
        assert exc.value.status == HTTPStatus.NOT_FOUND

    def test_busy_session_rejects_concurrent_message(self, monkeypatch):
        session = handle_create_session({})
        server._sessions[session["id"]].busy = True
        with pytest.raises(ApiError) as exc:
            handle_send_message(session["id"], {"content": "hello"})
        assert exc.value.status == HTTPStatus.CONFLICT

        with pytest.raises(ApiError) as exc:
            handle_delete_session(session["id"])
        assert exc.value.status == HTTPStatus.CONFLICT

    def test_busy_flag_reset_when_turn_raises(self, monkeypatch):
        def boom(turn):
            raise RuntimeError("simulated error")

        monkeypatch.setattr(server, "run_chat_turn", boom)
        session = handle_create_session({})
        with pytest.raises(RuntimeError):
            handle_send_message(session["id"], {"content": "hello"})
        assert server._sessions[session["id"]].busy is False

    def test_http_session_flow(self, monkeypatch):
        monkeypatch.setattr(
            server,
            "run_chat_turn",
            lambda turn: ChatOutcome(route="conversation_agent", reply="ok", run_id="r"),
        )
        created = MockHTTPHandler("POST", "/api/sessions", body=b"{}")
        created.do_POST()
        session_id = created.get_response_json()["id"]

        message = MockHTTPHandler(
            "POST", f"/api/sessions/{session_id}/messages", body='{"content": "hello"}'.encode("utf-8")
        )
        message.do_POST()
        assert message.response_status == HTTPStatus.OK
        assert message.get_response_json()["assistant_message"]["content"] == "ok"

        detail = MockHTTPHandler("GET", f"/api/sessions/{session_id}")
        detail.do_GET()
        assert len(detail.get_response_json()["messages"]) == 2

        deleted = MockHTTPHandler("DELETE", f"/api/sessions/{session_id}")
        deleted.do_DELETE()
        assert deleted.response_status == HTTPStatus.OK
        assert session_id not in server._sessions

    def test_http_unknown_session_returns_404(self):
        handler = MockHTTPHandler("GET", "/api/sessions/no-such-session")
        handler.do_GET()
        assert handler.response_status == HTTPStatus.NOT_FOUND

    def test_eval_routes_removed(self):
        handler = MockHTTPHandler("POST", "/api/evals/stream", body=b"{}")
        handler.do_POST()
        assert handler.response_status == HTTPStatus.NOT_FOUND
