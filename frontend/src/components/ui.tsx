const TONES: Record<string, string> = {
  completed: "ok",
  needs_user: "warn",
  failed: "bad",
};

const LABELS: Record<string, string> = {
  completed: "Completed",
  needs_user: "Needs info",
  failed: "Failed",
};

export function Badge({ status }: { status: string }) {
  return <span className={`badge ${TONES[status] ?? ""}`}>{LABELS[status] ?? status}</span>;
}
