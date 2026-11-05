# Frontend

React + TypeScript + Vite web console. It talks to the backend only through `/api/*`.

## Develop

Requires Node 20+. Run the backend first, then the dev server:

```bash
# Terminal 1 — backend API (from backend/)
LLM_PROVIDER=fake uv run issue-agent-server  # http://127.0.0.1:8000

# Terminal 2 — Vite dev server with hot reload (proxies /api to :8000)
cd frontend
npm install
npm run dev                                    # http://localhost:5173
```

If the terminal shows `http proxy error ... ECONNREFUSED 127.0.0.1:8000`, the backend (terminal 1)
isn't running.

## Build

```bash
npm run build      # type-checks, then writes to dist/ (not committed)
```

When `frontend/dist` exists, the backend serves it at `/`, so a single process runs the whole app.

## Structure

```text
src/
├── main.tsx              entry point
├── App.tsx               app state: chat sessions, backend status, repo path
├── api.ts                typed fetch helpers for /api/*
├── types.ts              API response types
├── styles.css            styles + bundled Exo 2 font (SIL OFL)
├── fonts/                Exo 2 woff2 files
└── components/
    ├── ChatView.tsx      message thread, input box, repo path field
    ├── Sidebar.tsx       session list, new/delete chat, backend status
    ├── TicketCard.tsx    issue ticket rendering with trace
    └── ui.tsx            Badge
```
