import type { AppInfo, SessionSummary } from "../types";

interface Props {
  sessions: SessionSummary[];
  activeId: string | null;
  info: AppInfo | null;
  connectError: boolean;
  locked: boolean;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
}

const formatTime = (iso: string) => {
  const date = new Date(iso);
  const sameDay = date.toDateString() === new Date().toDateString();
  return sameDay
    ? date.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" })
    : date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
};

export function Sidebar({ sessions, activeId, info, connectError, locked, onSelect, onNew, onDelete }: Props) {
  const confirmDelete = (id: string, title: string) => {
    if (window.confirm(`Delete "${title}"?`)) onDelete(id);
  };

  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="logo" aria-hidden="true" />
        <strong>Issue Agent</strong>
      </div>

      <button className="new-chat" onClick={onNew} disabled={locked}>
        <span aria-hidden="true">+</span> New chat
      </button>

      <div className="section-label">Recent</div>
      <nav className="session-list" aria-label="Chats">
        {sessions.length === 0 && <p className="empty-list">No chats yet</p>}
        {sessions.map((s) => (
          <div key={s.id} className={`session-item ${s.id === activeId ? "active" : ""}`}>
            <button className="session-open" onClick={() => onSelect(s.id)} disabled={locked} title={s.title}>
              <span className="session-title">{s.title}</span>
              <span className="session-meta">{formatTime(s.updated_at)}</span>
            </button>
            <button className="icon-btn" aria-label={`Delete ${s.title}`} onClick={() => confirmDelete(s.id, s.title)}>
              <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
                <path
                  d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          </div>
        ))}
      </nav>

      <div className="sidebar-footer">
        <span className={`dot ${connectError ? "bad" : info ? (info.llm.is_offline ? "warn" : "ok") : ""}`} />
        <span className="footer-text">
          {connectError
            ? "Backend unreachable"
            : info
              ? info.llm.is_offline
                ? "Offline mode"
                : `${info.llm.provider} · ${info.llm.model}`
              : "Connecting…"}
        </span>
      </div>
    </aside>
  );
}
