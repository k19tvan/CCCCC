# Backend

Python agent system: a LangGraph orchestrator that dispatches to subagents, plus a CLI and a small REST
API. The API uses only the standard library (`http.server`).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+. Run all commands from `backend/`.

```bash
uv sync          # creates .venv from pyproject.toml / uv.lock (includes the dev group)
```

Add or change dependencies with `uv add <pkg>` / `uv add --dev <pkg>`; commit `uv.lock` with the change.

## Run

```bash
uv run issue-agent-server            # API → http://127.0.0.1:8000  (--port 8080 to change)
uv run issue-agent --help            # CLI

LLM_PROVIDER=fake uv run pytest
LLM_PROVIDER=fake uv run python -m evals.runner --output-dir evals/reports
```

`LLM_PROVIDER=fake` uses an offline test double. For a real model, copy `.env.example` to `.env`, set
`LLM_API_KEY`, and run with `uv run --env-file .env ...`. The code does not read `.env` by itself.

The server also serves the web UI at `/` when `../frontend/dist` exists
(override the location with `FRONTEND_DIST`). See [frontend/README.md](../frontend/README.md).

## Architecture

```text
process_issue(request)
  └─ orchestrator graph ── route_to_subagent ──► issue_agent (LangGraph graph) ──► IssueResult
```

- **Orchestrator** (`orchestrator/`): creates `run_id` and the trace, builds the registered subagents, and
  runs the orchestrator graph. `process_issue` never raises; failures come back as `status="failed"`.
- **Subagent** (`subagents/`): anything that implements `Subagent.run(request, trace) -> IssueResult`.
  The orchestrator only knows this contract.
- **issue_agent**: gathers context from the repo, asks the LLM for a ticket, validates it (retrying on
  schema errors), filters `affected_files`, and picks a route. Retry lives only in this graph.

To add a subagent: implement `run`, add it to the `subagents` mapping in `orchestrator/process.py`, and
extend `route_to_subagent` in `orchestrator/graph.py`.

## Layout

```text
backend/
├── src/agent_system/
│   ├── orchestrator/
│   │   ├── process.py          process_issue — public entry point, LLM selection
│   │   └── graph.py            orchestrator graph and subagent routing
│   ├── subagents/
│   │   ├── base.py             Subagent contract
│   │   └── issue_agent/
│   │       ├── graph.py        triage graph (IssueAgent)
│   │       ├── ticket_rules.py cross-field ticket rules and result messages
│   │       ├── tracing.py      traces every tool call
│   │       ├── prompts.py, offline_llm.py
│   │       ├── skills/         gather_context
│   │       ├── tools/          list_files, read_file, search_code, registry
│   │       └── validation/     schema, path/route guards, routing
│   ├── core/                   contracts, config, logging, trace, llm/ (OpenAI-compatible client)
│   └── interfaces/             adapter (single gateway), cli, server
├── evals/                      eval cases, scorer, runner  → evals/README.md
├── examples/                   sample repo, sample issues, demo scripts
└── tests/                      unit + integration tests
```

## REST API

| Method | Endpoint | Description |
|---|---|---|
| `GET`  | `/api/info` | Backend status and LLM metadata |
| `GET`  | `/api/sessions` | List chat sessions |
| `POST` | `/api/sessions` | Create a chat session `{repo_path?}` |
| `GET`  | `/api/sessions/<id>` | Session detail with messages |
| `DELETE` | `/api/sessions/<id>` | Delete a session |
| `POST` | `/api/sessions/<id>/messages` | Send `{content, repo_path?}`, get the orchestrator's reply |
| `POST` | `/api/issues` | Send `{title, description, repo_path, clarified_text}`, get `IssueResult` |
| `GET`  | `/`, `/assets/<file>` | Built frontend (`.js`, `.css`, `.woff2` only), if present |

## CLI

```bash
# Classify an issue and show the trace
uv run issue-agent -t "ZeroDivisionError in Calculator.divide" \
  -d "divide(10, 0) crashes" -r tests/fixtures/sample_repo --trace

# Plain JSON output (CI/CD)
uv run issue-agent --sample SAMPLE-01 --json

# Interactive mode when info is missing
uv run issue-agent -t "Calculation bug" -d "Something breaks" \
  -r tests/fixtures/sample_repo --interactive
```

### Exit codes

| Code | Meaning |
|:---:|---|
| `0`   | Classified OK (`status == completed`) |
| `1`   | Bad input (empty title/description, missing repo_path) |
| `2`   | Needs more info (`status == needs_user`) |
| `3`   | Pipeline failed (`status == failed`) |
| `4`   | Backend `process_issue` not ready |
| `130` | Cancelled with Ctrl+C |

## When the backend isn't ready

- CLI returns exit code `4` with a hint.
- The API returns `503 Service Unavailable`; the server stays up.
- Neither one fakes a successful result.
