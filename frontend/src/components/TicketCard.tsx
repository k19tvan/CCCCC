import { useState } from "react";
import type { IssueResult, TraceStep } from "../types";
import { Badge } from "./ui";

function Section({ title, items }: { title: string; items?: string[] }) {
  if (!items?.length) return null;
  return (
    <section className="ticket-section">
      <h4>{title}</h4>
      <ul>
        {items.map((item, i) => (
          <li key={i}>{item}</li>
        ))}
      </ul>
    </section>
  );
}

const traceLabel = (step: TraceStep) =>
  [step.step ?? step.action ?? "step", step.selected ?? step.tool, step.status].filter(Boolean).join(" · ");

export function TicketCard({ result }: { result: IssueResult }) {
  const [showTrace, setShowTrace] = useState(false);
  const ticket = result.ticket;
  const trace = result.trace ?? [];

  return (
    <div className="ticket">
      <div className="ticket-head">
        <Badge status={result.status} />
        {ticket?.type && <span className="chip">{ticket.type}</span>}
        {ticket?.priority && <span className="chip">{ticket.priority}</span>}
        {ticket?.route && <span className="chip">{ticket.route}</span>}
      </div>
      {ticket?.summary && <p className="ticket-summary">{ticket.summary}</p>}
      {ticket?.priority_reason && <p className="ticket-note">{ticket.priority_reason}</p>}
      <Section title="Affected files" items={ticket?.affected_files} />
      <Section title="Acceptance criteria" items={ticket?.acceptance_criteria} />
      <Section title="Warnings" items={result.warnings} />
      {trace.length > 0 && (
        <>
          <button type="button" className="link" onClick={() => setShowTrace((v) => !v)}>
            {showTrace ? "Hide" : "Show"} trace ({trace.length} steps)
          </button>
          {showTrace && (
            <ol className="trace">
              {trace.map((step, i) => (
                <li key={i}>
                  <code>{traceLabel(step)}</code>
                </li>
              ))}
            </ol>
          )}
        </>
      )}
    </div>
  );
}
