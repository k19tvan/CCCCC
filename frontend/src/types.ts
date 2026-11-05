export interface Ticket {
  summary?: string;
  type?: string;
  priority?: string;
  priority_reason?: string;
  route?: string;
  affected_files?: string[];
  acceptance_criteria?: string[];
  clarification_questions?: string[];
}

export interface TraceStep {
  step?: string;
  action?: string;
  selected?: string;
  tools?: string[] | string;
  attempt?: number;
  tool?: string;
  status?: string;
  error?: string;
}

export interface IssueResult {
  status: "completed" | "needs_user" | "failed";
  message?: string;
  ticket?: Ticket | null;
  warnings?: string[];
  trace?: TraceStep[];
  run_id?: string;
}

export interface AppInfo {
  llm: { provider: string; model: string; is_offline: boolean };
}

export interface SessionSummary {
  id: string;
  title: string;
  updated_at: string;
  message_count: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  route?: string;
  result?: IssueResult | null;
}

export interface SessionDetail {
  id: string;
  title: string;
  repo_path: string | null;
  created_at: string;
  updated_at: string;
  messages: ChatMessage[];
}

export interface SendMessageResponse {
  user_message: ChatMessage;
  assistant_message: ChatMessage;
  session: SessionSummary;
}
