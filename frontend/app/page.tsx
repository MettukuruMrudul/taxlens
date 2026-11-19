"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import AnswerTurn, { type Turn } from "@/components/AnswerTurn";
import HistoryDrawer from "@/components/HistoryDrawer";
import SignIn from "@/components/SignIn";
import SourceViewer from "@/components/SourceViewer";
import YearPicker from "@/components/YearPicker";
import { ApiError, type Citation, fetchMeta, getToken, getUser, signOut, streamAnswer } from "@/lib/api";

const EXAMPLES = [
  "What is the standard deduction for a single filer?",
  "I'm filing in 2024. What does line 30 of Form 1040 cover?",
  "How did line 30 change from 2021 to 2022?",
  "Where do I report my W-2 wages?",
];
const YEAR_KEY = "selectedYear";

export default function Home() {
  const [ready, setReady] = useState(false);
  const [signedIn, setSignedIn] = useState(false);
  const [years, setYears] = useState<number[]>([]);
  const [year, setYear] = useState<number | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [active, setActive] = useState<{ key: string; citation: Citation } | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const threadEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setSignedIn(Boolean(getToken()));
    const saved = sessionStorage.getItem(YEAR_KEY);
    if (saved) setYear(Number(saved));
    fetchMeta()
      .then((m) => {
        setYears(m.years);
        if (!m.index_loaded) setNotice(m.error ?? "The index isn't loaded yet.");
      })
      .catch((e: Error) => setNotice(e.message))
      .finally(() => setReady(true));
  }, []);

  useEffect(() => threadEnd.current?.scrollIntoView({ behavior: "smooth", block: "end" }), [turns]);

  const chooseYear = (y: number | null) => {
    setYear(y);
    if (y === null) sessionStorage.removeItem(YEAR_KEY);
    else sessionStorage.setItem(YEAR_KEY, String(y));
  };

  const update = (id: string, patch: (t: Turn) => Turn) =>
    setTurns((all) => all.map((t) => (t.id === id ? patch(t) : t)));

  const ask = useCallback(async (question: string, askYear: number | null) => {
    const q = question.trim();
    if (q.length < 3 || busy) return;
    const id = `${Date.now()}`;
    setTurns((all) => [...all, {
      id, question: q, status: "streaming", taxYear: askYear, yearSource: askYear ? "explicit" : null,
      compareYears: null, sentences: [], citations: {}, message: null, queryId: null, disclaimer: null,
      cacheHit: false,
    }]);
    setDraft("");
    setBusy(true);
    try {
      await streamAnswer(q, askYear, (e) => {
        if (e.type === "meta") {
          update(id, (t) => ({ ...t, taxYear: e.tax_year, yearSource: e.year_source,
            compareYears: e.compare_years, message: e.message,
            status: e.status === "answered" ? "streaming" : e.status }));
        } else if (e.type === "sentence") {
          update(id, (t) => ({ ...t, sentences: [...t.sentences, e.sentence],
            citations: { ...t.citations, ...Object.fromEntries(e.citations.map((c) => [c.n, c])) } }));
        } else if (e.type === "abstain") {
          update(id, (t) => ({ ...t, status: "abstained", message: e.message, sentences: [] }));
        } else if (e.type === "error") {
          update(id, (t) => ({ ...t, status: "error", message: e.error }));
        } else if (e.type === "done") {
          const r = e.result;
          update(id, (t) => ({ ...t, status: r.status, message: r.message, queryId: r.query_id,
            disclaimer: r.disclaimer, cacheHit: r.cache_hit, taxYear: r.tax_year,
            sentences: r.status === "answered" ? r.sentences : [],
            citations: Object.fromEntries(r.citations.map((c) => [c.n, c])) }));
          if (r.status === "answered" && r.citations[0]) {
            setActive({ key: `${id}:${r.citations[0].n}`, citation: r.citations[0] });
          }
        }
      });
    } catch (err) {
      const e = err as ApiError;
      if (e.status === 401) {
        signOut();
        setSignedIn(false);
      }
      update(id, (t) => ({ ...t, status: "error", message: e.message }));
    } finally {
      setBusy(false);
    }
  }, [busy]);

  if (!ready) return null;
  if (!signedIn) return <SignIn onSignedIn={() => setSignedIn(true)} />;

  return (
    <div className="shell">
      <div className="main">
        <header className="topbar">
          <div className="wordmark">TaxLens <span>IRS answers by tax year</span></div>
          <div className="topbar-actions">
            <button type="button" className="btn-quiet" onClick={() => setHistoryOpen(true)}>History</button>
            <button type="button" className="btn-quiet" onClick={() => { signOut(); setSignedIn(false); }}>
              Sign out {getUser()}
            </button>
          </div>
        </header>

        <YearPicker years={years} value={year} onChange={chooseYear} />

        <section className="thread" aria-live="polite">
          {notice && <p className="notice warn">{notice}</p>}
          {turns.length === 0 && (
            <div className="empty">
              <h1>Ask about Form 1040, its schedules, or Publication 17.</h1>
              <p className="muted">
                Every answer uses only that year&apos;s IRS text and points to the exact line. If the text
                doesn&apos;t support an answer, TaxLens says so instead of guessing.
              </p>
              <div className="examples">
                {EXAMPLES.map((ex) => (
                  <button key={ex} type="button" onClick={() => ask(ex, year)}>{ex}</button>
                ))}
              </div>
            </div>
          )}
          {turns.map((t) => (
            <AnswerTurn key={t.id} turn={t} activeRef={active?.key ?? null}
                        onCite={(turnId, c) => setActive({ key: `${turnId}:${c.n}`, citation: c })} />
          ))}
          <div ref={threadEnd} />
        </section>

        <form className="composer" onSubmit={(e) => { e.preventDefault(); ask(draft, year); }}>
          <label htmlFor="q" className="muted" style={{ position: "absolute", left: "-9999px" }}>Question</label>
          <textarea id="q" value={draft} onChange={(e) => setDraft(e.target.value)} maxLength={1000}
                    placeholder={year ? `Ask about tax year ${year}` : "Ask a question, and mention the tax year"}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(draft, year); }
                    }} />
          <button className="btn" type="submit" disabled={busy || draft.trim().length < 3}>
            {busy ? "Answering…" : "Ask"}
          </button>
        </form>
      </div>

      <SourceViewer citation={active?.citation ?? null} />

      {historyOpen && (
        <HistoryDrawer onClose={() => setHistoryOpen(false)}
                       onPick={(q, y) => { setHistoryOpen(false); ask(q, y); }} />
      )}
    </div>
  );
}
