import { useCallback, useEffect, useState } from "react";
import * as api from "./api";
import { ChatView } from "./components/ChatView";
import { Sidebar } from "./components/Sidebar";
import type { AppInfo, SessionDetail, SessionSummary } from "./types";

const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));

export default function App() {
  const [info, setInfo] = useState<AppInfo | null>(null);
  const [connectError, setConnectError] = useState(false);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [active, setActive] = useState<SessionDetail | null>(null);
  const [repoPath, setRepoPath] = useState("");
  const [pendingText, setPendingText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refreshSessions = useCallback(async () => {
    const res = await api.listSessions();
    setSessions(res.sessions);
  }, []);

  useEffect(() => {
    api
      .getInfo()
      .then(setInfo)
      .catch(() => setConnectError(true));
    refreshSessions().catch(() => setConnectError(true));
  }, [refreshSessions]);

  const openSession = async (id: string) => {
    if (pendingText !== null) return;
    setError(null);
    try {
      const detail = await api.getSession(id);
      setActive(detail);
      setRepoPath(detail.repo_path ?? "");
    } catch (e) {
      setError(errorText(e));
    }
  };

  const newChat = () => {
    if (pendingText !== null) return;
    setActive(null);
    setRepoPath("");
    setError(null);
  };

  const removeSession = async (id: string) => {
    setError(null);
    try {
      await api.deleteSession(id);
      if (active?.id === id) newChat();
      await refreshSessions();
    } catch (e) {
      setError(errorText(e));
    }
  };

  const send = async (content: string): Promise<boolean> => {
    setPendingText(content);
    setError(null);
    try {
      const session = active ?? (await api.createSession());
      if (!active) setActive(session);
      const res = await api.sendMessage(session.id, content, repoPath);
      setActive((latest) =>
        latest?.id === session.id
          ? {
              ...latest,
              title: res.session.title,
              repo_path: repoPath.trim() || latest.repo_path,
              messages: [...latest.messages, res.user_message, res.assistant_message],
            }
          : latest,
      );
      await refreshSessions();
      return true;
    } catch (e) {
      setError(errorText(e));
      await refreshSessions().catch(() => undefined);
      return false;
    } finally {
      setPendingText(null);
    }
  };

  return (
    <div className="app">
      <Sidebar
        sessions={sessions}
        activeId={active?.id ?? null}
        info={info}
        connectError={connectError}
        locked={pendingText !== null}
        onSelect={openSession}
        onNew={newChat}
        onDelete={removeSession}
      />
      <ChatView
        session={active}
        repoPath={repoPath}
        pendingText={pendingText}
        error={error}
        onRepoChange={setRepoPath}
        onSend={send}
      />
    </div>
  );
}
