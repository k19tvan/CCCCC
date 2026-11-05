import type { AppInfo, SendMessageResponse, SessionDetail, SessionSummary } from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public code?: string,
  ) {
    super(message);
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  let data: unknown = null;
  try {
    data = await res.json();
  } catch {
    // Non-JSON body; the status check below reports the failure.
  }
  if (!res.ok) {
    const body = data as { error?: string; code?: string } | null;
    throw new ApiError(res.status, body?.error ?? `HTTP ${res.status}`, body?.code);
  }
  return data as T;
}

const jsonRequest = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

const sessionUrl = (id: string) => `/api/sessions/${encodeURIComponent(id)}`;

export const getInfo = () => request<AppInfo>("/api/info");

export const listSessions = () => request<{ sessions: SessionSummary[] }>("/api/sessions");

export const createSession = () => request<SessionDetail>("/api/sessions", jsonRequest("POST", {}));

export const getSession = (id: string) => request<SessionDetail>(sessionUrl(id));

export const deleteSession = (id: string) => request<{ ok: boolean }>(sessionUrl(id), { method: "DELETE" });

export const sendMessage = (id: string, content: string, repoPath: string) =>
  request<SendMessageResponse>(
    `${sessionUrl(id)}/messages`,
    jsonRequest("POST", repoPath.trim() ? { content, repo_path: repoPath.trim() } : { content }),
  );
