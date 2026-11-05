"""
REST API (and static host for the built web UI) for the AI Issue Agent.

Uses only the Python standard library (http.server). Every message goes through
`agent_system.orchestrator.run_chat_turn`, which decides between conversation (conversation agent) and
issue handling (issue agent). Chat sessions live in memory only and are lost on restart.

Run (from backend/):
    python -m agent_system.interfaces.server                 # Listens on http://127.0.0.1:8000
    python -m agent_system.interfaces.server --port 8080      # Listens on port 8080

The web UI lives in frontend/ (React). If frontend/dist has been built, this server serves it at `/`;
during development run `npm run dev` in frontend/ (it proxies /api to this server).

Endpoints:
    GET    /                                   UI (if the frontend is built)
    GET    /assets/<file>                      JS/CSS/font files of the built UI
    GET    /api/info                           LLM information
    GET    /api/sessions                       List chat sessions -> {sessions}
    POST   /api/sessions                       {repo_path?} -> new chat session
    GET    /api/sessions/<id>                  Session detail with messages
    DELETE /api/sessions/<id>                  Delete a session -> {ok}
    POST   /api/sessions/<id>/messages         {content, repo_path?} -> {user_message, assistant_message, session}
    POST   /api/issues                         {title, description, repo_path, clarified_text} -> IssueResult
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import threading
import urllib.parse
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

# Backend root directory (contains examples/, evals/, tests/); must be on sys.path to import internal modules
BACKEND_ROOT = Path(__file__).resolve().parents[3]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from pydantic import ValidationError

from agent_system.core.contracts import IssueRequest, IssueResult
from agent_system.interfaces.adapter import (
    BackendExecutionError,
    BackendMissingError,
    call_agent,
)
from agent_system.orchestrator import ChatTurn, run_chat_turn

logger = logging.getLogger(__name__)

# The React UI is built into frontend/dist (override with the FRONTEND_DIST environment variable)
FRONTEND_DIST = Path(os.getenv("FRONTEND_DIST") or BACKEND_ROOT.parent / "frontend" / "dist")
ASSET_TYPES = {
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".woff2": "font/woff2",
}
DEFAULT_REPO_PATH = "tests/fixtures/sample_repo"
DEFAULT_SESSION_TITLE = "New chat"
MAX_BODY_BYTES = 1_000_000
MAX_MESSAGE_CHARS = 8_000
MAX_TITLE_CHARS = 40
SESSION_PREFIX = "/api/sessions/"

# Optional hook for unit tests and test doubles for /api/issues
_TEST_PROCESSOR: Callable[[IssueRequest], IssueResult] | None = None

_sessions: dict[str, "ChatSession"] = {}
_sessions_lock = threading.Lock()


def set_test_processor(processor: Callable[[IssueRequest], IssueResult] | None) -> None:
    """Set an optional processor for /api/issues (useful for unit tests)."""
    global _TEST_PROCESSOR
    _TEST_PROCESSOR = processor


class ApiError(Exception):
    """API error carrying a specific HTTP status and a JSON message."""

    def __init__(self, status: HTTPStatus, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code or status.name.lower()


def resolve_repo_path(raw: Any) -> str:
    """
    Normalise repo_path and check that it exists.
    Relative paths are resolved from the backend/ directory, independent of the server's cwd.
    """
    if raw is not None and not isinstance(raw, str):
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "The repository path must be a valid string",
            code="invalid_repo_path",
        )
    cleaned = (raw or "").strip() or DEFAULT_REPO_PATH
    path = Path(cleaned).expanduser()
    if not path.is_absolute():
        path = BACKEND_ROOT / path
    path = path.resolve()
    if not path.is_dir():
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            f"The repository path does not exist or is not a directory: {raw or DEFAULT_REPO_PATH}",
            code="invalid_repo_path",
        )
    return str(path)


def get_llm_info() -> dict[str, Any]:
    """Return the model and whether it runs offline or online."""
    try:
        from agent_system.core.config import config

        provider = (config.LLM_PROVIDER or "fake").strip().lower()
        model = config.LLM_MODEL or "offline-rules"
    except Exception:
        provider = "fake"
        model = "offline-rules"

    is_offline = provider in ("fake", "offline", "none", "mock")
    return {
        "provider": provider,
        "model": model,
        "is_offline": is_offline,
    }


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _required_text(body: dict[str, Any], key: str, max_chars: int) -> str:
    raw = body.get(key)
    if not isinstance(raw, str):
        raise ApiError(HTTPStatus.BAD_REQUEST, f"Field '{key}' must be a string", f"invalid_{key}")
    text = raw.strip()
    if not text:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"Field '{key}' must not be empty", f"empty_{key}")
    if len(text) > max_chars:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"Message exceeds {max_chars} characters", "message_too_long")
    return text


def _optional_repo_path(body: dict[str, Any]) -> str | None:
    raw = body.get("repo_path")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    try:
        return resolve_repo_path(raw)
    except ApiError as exc:
        raise ApiError(exc.status, f"Repository path not found or not a directory: {raw}", exc.code) from exc


def _new_message(role: str, content: str, **extra: Any) -> dict[str, Any]:
    return {"id": uuid.uuid4().hex[:12], "role": role, "content": content, "created_at": _now(), **extra}


@dataclass
class ChatSession:
    id: str
    repo_path: str | None
    created_at: str
    updated_at: str
    title: str = DEFAULT_SESSION_TITLE
    messages: list[dict[str, Any]] = field(default_factory=list)
    pending: IssueRequest | None = None
    busy: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "updated_at": self.updated_at,
            "message_count": len(self.messages),
        }

    def detail(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "repo_path": self.repo_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": list(self.messages),
        }


def _get_session_locked(session_id: str) -> ChatSession:
    session = _sessions.get(session_id)
    if session is None:
        raise ApiError(HTTPStatus.NOT_FOUND, f"Chat not found: {session_id}", "session_not_found")
    return session


def handle_info() -> dict[str, Any]:
    """Endpoint GET /api/info."""
    return {"llm": get_llm_info()}


def handle_list_sessions() -> dict[str, Any]:
    """Endpoint GET /api/sessions: list chat sessions, newest first."""
    with _sessions_lock:
        summaries = [s.summary() for s in _sessions.values()]
    summaries.sort(key=lambda s: s["updated_at"], reverse=True)
    return {"sessions": summaries}


def handle_create_session(body: dict[str, Any]) -> dict[str, Any]:
    """Endpoint POST /api/sessions: create a new chat session."""
    repo_path = _optional_repo_path(body)
    now = _now()
    session = ChatSession(id=uuid.uuid4().hex[:12], repo_path=repo_path, created_at=now, updated_at=now)
    with _sessions_lock:
        _sessions[session.id] = session
        return session.detail()


def handle_get_session(session_id: str) -> dict[str, Any]:
    """Endpoint GET /api/sessions/<id>."""
    with _sessions_lock:
        return _get_session_locked(session_id).detail()


def handle_delete_session(session_id: str) -> dict[str, Any]:
    """Endpoint DELETE /api/sessions/<id>."""
    with _sessions_lock:
        session = _get_session_locked(session_id)
        if session.busy:
            raise ApiError(HTTPStatus.CONFLICT, "Chat is still processing a message, try again later", "session_busy")
        del _sessions[session_id]
    return {"ok": True}


def handle_send_message(session_id: str, body: dict[str, Any]) -> dict[str, Any]:
    """Endpoint POST /api/sessions/<id>/messages: send a message and return the orchestrator's reply."""
    content = _required_text(body, "content", MAX_MESSAGE_CHARS)
    requested_repo = _optional_repo_path(body)

    with _sessions_lock:
        session = _get_session_locked(session_id)
        if session.busy:
            raise ApiError(HTTPStatus.CONFLICT, "Chat is still processing the previous message", "session_busy")
        session.busy = True
        history = [{"role": m["role"], "content": m["content"]} for m in session.messages]
        pending = session.pending
        repo_path = requested_repo or session.repo_path

    try:
        outcome = run_chat_turn(
            ChatTurn(user_text=content, history=history, repo_path=repo_path, pending=pending)
        )
    finally:
        with _sessions_lock:
            session.busy = False

    user_message = _new_message("user", content)
    assistant_message = _new_message(
        "assistant",
        outcome.reply,
        route=outcome.route,
        result=outcome.result.model_dump(mode="json") if outcome.result else None,
    )
    with _sessions_lock:
        session.repo_path = repo_path
        session.pending = outcome.pending
        session.messages.extend([user_message, assistant_message])
        if session.title == DEFAULT_SESSION_TITLE:
            session.title = content.splitlines()[0][:MAX_TITLE_CHARS]
        session.updated_at = _now()
        summary = session.summary()

    return {"user_message": user_message, "assistant_message": assistant_message, "session": summary}


def handle_issue(
    body: dict[str, Any],
    processor: Callable[[IssueRequest], IssueResult] | None = None,
) -> dict[str, Any]:
    """Endpoint POST /api/issues: accept an issue, call the agent and return the structured result."""
    if not isinstance(body, dict):
        raise ApiError(HTTPStatus.BAD_REQUEST, "Request body must be a JSON object", "invalid_body")

    raw_title = body.get("title")
    if raw_title is None or not isinstance(raw_title, str):
        raise ApiError(HTTPStatus.BAD_REQUEST, "Issue title is required and must be a string", "missing_title")
    title = raw_title.strip()
    if not title:
        raise ApiError(HTTPStatus.BAD_REQUEST, "Issue title must not be empty or whitespace only", "missing_title")

    raw_description = body.get("description")
    if raw_description is None or not isinstance(raw_description, str):
        raise ApiError(HTTPStatus.BAD_REQUEST, "Issue description is required and must be a string", "missing_description")
    description = raw_description.strip()
    if not description:
        raise ApiError(HTTPStatus.BAD_REQUEST, "Issue description must not be empty or whitespace only", "missing_description")

    raw_repo_path = body.get("repo_path")
    repo_path = resolve_repo_path(raw_repo_path)

    raw_clarified = body.get("clarified_text")
    if raw_clarified is not None and not isinstance(raw_clarified, str):
        raise ApiError(HTTPStatus.BAD_REQUEST, "Field clarified_text must be a string if provided", "invalid_clarified_text")
    clarified_text = raw_clarified.strip() if raw_clarified and raw_clarified.strip() else None

    try:
        request = IssueRequest(
            title=title,
            description=description,
            repo_path=repo_path,
            clarified_text=clarified_text,
        )
    except ValidationError as exc:
        fields = ", ".join(str(e.get("loc", ["?"])[0]) for e in exc.errors())
        raise ApiError(HTTPStatus.BAD_REQUEST, f"Invalid issue data: {fields}", "validation_error") from exc

    effective_processor = processor if processor is not None else _TEST_PROCESSOR
    try:
        result = call_agent(request, processor=effective_processor)
    except BackendMissingError as exc:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, str(exc), "backend_missing") from exc
    except BackendExecutionError as exc:
        raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc), "backend_error") from exc

    return result.model_dump(mode="json")


class Handler(BaseHTTPRequestHandler):
    """HTTP request handler for the Issue Agent web app."""

    server_version = "IssueAgentWeb/1.0"

    def do_GET(self) -> None:  # noqa: N802
        path = urllib.parse.urlparse(self.path).path

        if path in ("/", "/index.html"):
            index_path = FRONTEND_DIST / "index.html"
            if not index_path.exists():
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": "Frontend is not built. Run `npm run build` in frontend/, or use `npm run dev`."},
                )
                return
            self._send_file(index_path, "text/html; charset=utf-8")
        elif path.startswith("/assets/") and Path(path).suffix in ASSET_TYPES:
            asset_path = FRONTEND_DIST / "assets" / Path(path).name
            if not asset_path.is_file():
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Endpoint not found: {path}"})
                return
            self._send_file(asset_path, ASSET_TYPES[asset_path.suffix])
        elif path == "/api/info":
            self._run(handle_info)
        elif path == "/api/sessions":
            self._run(handle_list_sessions)
        elif path.startswith(SESSION_PREFIX):
            session_id = urllib.parse.unquote(path[len(SESSION_PREFIX):])
            self._run(lambda: handle_get_session(session_id))
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Endpoint not found: {path}"})

    def do_POST(self) -> None:  # noqa: N802
        path = urllib.parse.urlparse(self.path).path

        is_message_route = path.startswith(SESSION_PREFIX) and path.endswith("/messages")
        if path not in ("/api/sessions", "/api/issues") and not is_message_route:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Endpoint not found: {path}"})
            return

        try:
            body = self._read_json()
        except ApiError as exc:
            self._send_json(exc.status, {"error": str(exc), "code": exc.code})
            return

        if path == "/api/sessions":
            self._run(lambda: handle_create_session(body))
        elif path == "/api/issues":
            self._run(lambda: handle_issue(body))
        else:
            session_id = urllib.parse.unquote(path[len(SESSION_PREFIX):-len("/messages")])
            self._run(lambda: handle_send_message(session_id, body))

    def do_DELETE(self) -> None:  # noqa: N802
        path = urllib.parse.urlparse(self.path).path
        if not path.startswith(SESSION_PREFIX):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Endpoint not found: {path}"})
            return
        session_id = urllib.parse.unquote(path[len(SESSION_PREFIX):])
        self._run(lambda: handle_delete_session(session_id))

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s %s", self.address_string(), fmt % args)

    def _run(self, fn: Callable[[], Any]) -> None:
        try:
            result = fn()
            self._send_json(HTTPStatus.OK, result)
        except ApiError as exc:
            self._send_json(exc.status, {"error": str(exc), "code": exc.code})
        except Exception as exc:
            logger.exception("Error while handling request %s", self.path)
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"{type(exc).__name__}: {exc}", "code": "internal_error"},
            )

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (ValueError, TypeError):
            length = 0

        if length > MAX_BODY_BYTES:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Request size exceeds the 1MB limit")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            data = json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"Request body is not valid JSON: {exc}", "invalid_json") from exc
        if not isinstance(data, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "Request body must be a JSON object", "invalid_json_body")
        return data

    def _send_json(self, status: HTTPStatus, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self._send(status, data, "application/json; charset=utf-8")

    def _send_file(self, path: Path, content_type: str) -> None:
        self._send(HTTPStatus.OK, path.read_bytes(), content_type)

    def _send(self, status: HTTPStatus, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        self.wfile.write(data)


def create_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    """Create a ThreadingHTTPServer instance."""
    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    """Start the web server from the command line."""
    parser = argparse.ArgumentParser(
        prog="issue-agent-web",
        description="Chat interface for the AI Issue Agent",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="IP address to listen on (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    server = create_server(args.host, args.port)
    info = get_llm_info()
    print("=" * 65)
    print(f"  AI Issue Agent API: http://{args.host}:{server.server_port}")
    print(f"  LLM Provider: {info['provider']} | Model: {info['model']}")
    print(f"  Mode: {'Offline Test Double' if info['is_offline'] else 'Live Model'}")
    ui = "served at /" if (FRONTEND_DIST / "index.html").exists() else "not built (use `npm run dev` in frontend/)"
    print(f"  UI: {ui}")
    print("=" * 65)
    print("Press Ctrl+C to stop the server.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the web server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
