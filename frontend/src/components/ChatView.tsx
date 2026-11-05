import { useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent } from "react";
import type { ChatMessage, SessionDetail } from "../types";
import { TicketCard } from "./TicketCard";

const ROUTE_LABEL: Record<string, string> = {
  conversation_agent: "Chat",
  issue_agent: "Issue analysis",
};

interface Props {
  session: SessionDetail | null;
  repoPath: string;
  pendingText: string | null;
  error: string | null;
  onRepoChange: (value: string) => void;
  onSend: (content: string) => Promise<boolean>;
}

export function ChatView({ session, repoPath, pendingText, error, onRepoChange, onSend }: Props) {
  const [draft, setDraft] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const busy = pendingText !== null;
  const messages = session?.messages ?? [];
  const isEmpty = messages.length === 0 && !busy;

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, busy, error]);

  useEffect(() => {
    textareaRef.current?.focus();
  }, [session?.id]);

  useLayoutEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [draft]);

  const submit = async () => {
    const content = draft.trim();
    if (!content || busy) return;
    setDraft("");
    const sent = await onSend(content);
    if (!sent) setDraft((current) => current || content);
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void submit();
    }
  };

  const composer = (
    <form
      className="composer"
      onSubmit={(e) => {
        e.preventDefault();
        void submit();
      }}
    >
      <textarea
        ref={textareaRef}
        value={draft}
        rows={1}
        placeholder="Message Issue Agent…"
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={handleKeyDown}
      />
      <div className="composer-bar">
        <label className="repo-input" title="Repository the issue agent analyzes">
          <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
            <path
              d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinejoin="round"
            />
          </svg>
          <input
            value={repoPath}
            placeholder="Repository path"
            aria-label="Repository path"
            onChange={(e) => onRepoChange(e.target.value)}
            spellCheck={false}
          />
        </label>
        <button type="submit" className="send" disabled={busy || !draft.trim()} aria-label="Send">
          <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
            <path d="M12 19V5M5 12l7-7 7 7" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </div>
    </form>
  );

  if (isEmpty) {
    return (
      <main className="main main-empty">
        <div className="hero">
          <span className="hero-mark" aria-hidden="true" />
          <h1>
            What are we <span className="grad-text">working on</span>?
          </h1>
          {composer}
          <p className="hint">Enter to send · Shift + Enter for a new line</p>
          {error && (
            <div className="error-banner" role="alert">
              {error}
            </div>
          )}
        </div>
      </main>
    );
  }

  return (
    <main className="main">
      <header className="chat-header">
        <h1 title={session?.title}>{session?.title ?? "New chat"}</h1>
      </header>

      <div className="messages">
        <div className="thread">
          {messages.map((message) => (
            <MessageItem key={message.id} message={message} />
          ))}
          {busy && (
            <>
              <div className="msg user">
                <div className="bubble">{pendingText}</div>
              </div>
              <div className="msg assistant">
                <div className="avatar" aria-hidden="true" />
                <div className="typing" aria-label="Thinking">
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            </>
          )}
          {error && (
            <div className="error-banner" role="alert">
              {error}
            </div>
          )}
          <div ref={endRef} />
        </div>
      </div>

      <div className="composer-dock">{composer}</div>
    </main>
  );
}

function MessageItem({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return (
      <div className="msg user">
        <div className="bubble">{message.content}</div>
      </div>
    );
  }
  return (
    <div className="msg assistant">
      <div className="avatar" aria-hidden="true" />
      <div className="assistant-body">
        {message.route && <span className="route-tag">{ROUTE_LABEL[message.route] ?? message.route}</span>}
        <p className="reply">{message.content}</p>
        {message.result && <TicketCard result={message.result} />}
      </div>
    </div>
  );
}
