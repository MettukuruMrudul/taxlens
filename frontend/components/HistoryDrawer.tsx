"use client";

import { useEffect, useState } from "react";

import { fetchHistory, type HistoryItem } from "@/lib/api";

type Props = { onClose: () => void; onPick: (question: string, year: number | null) => void };

export default function HistoryDrawer({ onClose, onPick }: Props) {
  const [items, setItems] = useState<HistoryItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchHistory().then((d) => setItems(d.items)).catch((e: Error) => setError(e.message));
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <nav className="drawer" aria-label="Your past questions">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
        <strong>Past questions</strong>
        <button type="button" className="btn-quiet" onClick={onClose}>Close</button>
      </div>
      {error && <p className="error">{error}</p>}
      {items === null && !error && <p className="muted">Loading…</p>}
      {items?.length === 0 && <p className="muted">Questions you ask will appear here.</p>}
      <ul>
        {items?.map((it) => (
          <li key={it.query_id}>
            <button type="button" onClick={() => onPick(it.question, it.tax_year)}>
              <div>{it.question}</div>
              <div className="muted num" style={{ fontSize: "0.8rem" }}>
                {it.tax_year ? `Tax year ${it.tax_year}` : "No year"}
                {it.abstained ? ", not answered" : ""}, {new Date(it.created_at).toLocaleDateString()}
              </div>
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
}
