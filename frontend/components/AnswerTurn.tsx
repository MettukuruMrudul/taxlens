"use client";

import { useState } from "react";

import { type AnswerSentence, type Citation, sendFeedback, type Status } from "@/lib/api";

export type Turn = {
  id: string;
  question: string;
  status: Status | "streaming" | "error";
  taxYear: number | null;
  yearSource: string | null;
  compareYears: number[] | null;
  sentences: AnswerSentence[];
  citations: Record<number, Citation>;
  message: string | null;
  queryId: number | null;
  disclaimer: string | null;
  cacheHit: boolean;
};

const YEAR_SOURCE: Record<string, string> = {
  explicit: "chosen above",
  tax_year: "read from your question",
  filing_year: "filing year in your question, minus one",
  relative: "worked out from today's date",
};

type Props = { turn: Turn; activeRef: string | null; onCite: (turnId: string, c: Citation) => void };

export default function AnswerTurn({ turn, activeRef, onCite }: Props) {
  const yearNote = turn.compareYears
    ? `Comparing tax years ${turn.compareYears.join(" and ")}`
    : turn.taxYear
      ? `Tax year ${turn.taxYear}${turn.yearSource && YEAR_SOURCE[turn.yearSource] ? `, ${YEAR_SOURCE[turn.yearSource]}` : ""}`
      : null;

  return (
    <article className="turn" aria-busy={turn.status === "streaming"}>
      <p className="question">{turn.question}</p>
      {yearNote && <div className="turn-meta num">{yearNote}</div>}

      {turn.sentences.length > 0 && turn.status !== "abstained" && (
        <div className="answer">
          {turn.sentences.map((s, i) => (
            <div className="line" key={i}>
              <div className="line-refs">
                {s.citations.map((n) => {
                  const c = turn.citations[n];
                  const key = `${turn.id}:${n}`;
                  return (
                    <button key={n} type="button" className="ref" aria-pressed={activeRef === key}
                            aria-label={`Show source ${n}${c?.title ? `: ${c.title} ${c.tax_year ?? ""}` : ""}`}
                            onClick={() => c && onCite(turn.id, c)}>
                      {n}
                    </button>
                  );
                })}
              </div>
              <div className="line-text">{s.text}</div>
            </div>
          ))}
        </div>
      )}

      {turn.status === "streaming" && turn.sentences.length === 0 && (
        <p className="pending">Finding the {turn.taxYear ?? ""} text…</p>
      )}
      {turn.message && turn.status !== "answered" && turn.status !== "streaming" && (
        <p className={`notice${turn.status === "error" ? " warn" : ""}`}>{turn.message}</p>
      )}
      {turn.disclaimer && turn.status !== "streaming" && <p className="disclaimer">{turn.disclaimer}</p>}
      {turn.queryId !== null && turn.status !== "streaming" && <Feedback queryId={turn.queryId} />}
    </article>
  );
}

function Feedback({ queryId }: { queryId: number }) {
  const [rating, setRating] = useState<1 | -1 | null>(null);
  const [comment, setComment] = useState("");
  const [state, setState] = useState<"idle" | "sent" | "error">("idle");

  async function submit(r: 1 | -1, note?: string) {
    setRating(r);
    try {
      await sendFeedback(queryId, r, note);
      setState("sent");
    } catch {
      setState("error");
    }
  }

  if (state === "sent" && (rating === 1 || comment)) return <div className="feedback muted">Feedback saved.</div>;
  return (
    <div className="feedback">
      <span className="muted">Was this right?</span>
      <button type="button" className="btn-quiet" aria-pressed={rating === 1} onClick={() => submit(1)}>Yes</button>
      <button type="button" className="btn-quiet" aria-pressed={rating === -1} onClick={() => submit(-1)}>No</button>
      {rating === -1 && (
        <form onSubmit={(e) => { e.preventDefault(); submit(-1, comment); }} style={{ display: "contents" }}>
          <input value={comment} onChange={(e) => setComment(e.target.value)} maxLength={2000}
                 placeholder="What was wrong? (optional)" aria-label="What was wrong" />
          <button type="submit" className="btn-quiet">Send</button>
        </form>
      )}
      {state === "error" && <span className="error">Couldn't save feedback.</span>}
    </div>
  );
}
