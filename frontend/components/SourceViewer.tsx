"use client";

import { useEffect, useState } from "react";

import { type Citation, fetchPageImage } from "@/lib/api";

export default function SourceViewer({ citation }: { citation: Citation | null }) {
  const [src, setSrc] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let url: string | null = null;
    let cancelled = false;
    setSrc(null);
    setError(null);
    if (!citation?.doc_id || !citation.page) return;
    fetchPageImage(citation)
      .then((u) => {
        url = u;
        if (!cancelled) setSrc(u);
      })
      .catch((e: Error) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [citation]);

  if (!citation) {
    return (
      <aside className="source" aria-label="Source document">
        <p className="source-empty">
          Every answer line has a numbered box beside it. Select one to see the exact IRS page it came from,
          with the cited text highlighted.
        </p>
      </aside>
    );
  }

  const where = citation.line_label ? `Line ${citation.line_label}` : citation.heading_path ?? "";
  return (
    <aside className="source" aria-label="Source document">
      <h2>
        {citation.title} <span className="num">({citation.tax_year})</span>
      </h2>
      <div className="source-meta num">
        {[where, citation.page ? `page ${citation.page}` : null].filter(Boolean).join(", ")}
      </div>
      {citation.snippet && <blockquote className="source-quote">{citation.snippet}</blockquote>}
      {error && <p className="error">{error}</p>}
      {!error && citation.page && !src && <p className="muted">Loading page {citation.page}…</p>}
      {src && (
        // eslint-disable-next-line @next/next/no-img-element
        <img className="page-img" src={src}
             alt={`${citation.title} ${citation.tax_year}, page ${citation.page}, cited text highlighted`} />
      )}
    </aside>
  );
}
