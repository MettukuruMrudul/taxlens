"""Retrievers: a naive baseline and the year-aware TaxLens retriever.

NaiveRetriever is what a standard RAG system does: index every year's copy
of every passage and return the best keyword matches, whatever their year.

YearAwareRetriever searches the validity-interval units and only returns
units that apply to the requested tax year.  When bi-encoder embeddings and
a query vector are provided it combines BM25 and dense scores via Reciprocal
Rank Fusion (RRF), and an optional cross-encoder re-ranks the top results.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from taxlens.index.intervals import Unit
from taxlens.retrieve.bm25 import BM25


def index_text(heading_path: str | None, line_label: str | None, text: str) -> str:
    parts = [heading_path or ""]
    if line_label:
        parts.append(f"line {line_label}")
    parts.append(text)
    return " ".join(parts)


def rrf_merge(ranked_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion over multiple ranked lists of document indices."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


@dataclass
class Hit:
    text: str
    years: tuple[int, int]        # (first, last) tax year the passage applies to
    score: float
    ref: object                   # what to cite
    line_label: str | None = None
    heading_path: str | None = None
    form_code: str | None = None
    bm25: float = 0.0             # raw BM25 score (used as an abstention feature)
    pinned: bool = False          # placed first by the structured line lookup


class NaiveRetriever:
    """Baseline: one entry per record per year, no year filter."""

    def __init__(self, records: list[dict]):
        self.records = records
        self.bm25 = BM25([index_text(r.get("heading_path"), r.get("line_label"), r["text"]) for r in records])

    def search(self, query: str, year: int | None = None, k: int = 5) -> list[Hit]:
        scores = self.bm25.scores(query)
        best = sorted(scores, key=scores.get, reverse=True)[:k]
        return [Hit(self.records[i]["text"], (self.records[i]["tax_year"],) * 2, scores[i],
                    self.records[i]["ref"], self.records[i].get("line_label"),
                    self.records[i].get("heading_path"), self.records[i].get("form_code"),
                    scores[i]) for i in best]


LINE_REF = re.compile(r"\bline\s+(\d{1,2}[a-z]?)\b", re.I)
FORM_HINTS = {
    "f1040s1": re.compile(r"\bschedule 1\b", re.I), "f1040s2": re.compile(r"\bschedule 2\b", re.I),
    "f1040s3": re.compile(r"\bschedule 3\b", re.I), "f1040sa": re.compile(r"\bschedule a\b", re.I),
    "f1040sb": re.compile(r"\bschedule b\b", re.I),
}
RERANK_DEPTH = 20


class YearAwareRetriever:
    """TaxLens: validity-interval units, filtered to the requested year.

    Ranking stages (each optional stage switches on when its model is given):

    1. Year filter. Only units whose interval contains the year are scored,
       so text from another tax year can never be returned.
    2. BM25 keyword ranking, plus dense cosine ranking if unit embeddings and
       a query vector are available, fused with Reciprocal Rank Fusion.
    3. Structured line lookup. A question naming "line 30" gets that form
       line for the requested year ranked first (Form 1040 unless the
       question names a schedule).
    4. Cross-encoder re-ranking of the top 20 (``reranker(query, texts)``
       returning one score per text), keeping line-lookup hits on top.
    """

    def __init__(self, units: list[Unit], line_lookup: bool = True, embeddings=None, reranker=None):
        self.units = units
        self.line_lookup = line_lookup
        self.embeddings = embeddings  # (n_units, dim) float32, L2-normalised, aligned with units
        self.reranker = reranker
        self.bm25 = BM25([index_text(u.heading_path, u.line_label, u.text) for u in units])

    def _line_hits(self, query: str, candidates: set[int]) -> list[int]:
        wanted = {m.lower() for m in LINE_REF.findall(query)}
        if not wanted:
            return []
        forms = {fc for fc, pat in FORM_HINTS.items() if pat.search(query)} or {"f1040"}
        found = [i for i in candidates
                 if self.units[i].record_type == "form_line" and self.units[i].line_label in wanted]
        preferred = [i for i in found if self.units[i].form_code in forms]
        return sorted(preferred or found, key=lambda i: self.units[i].form_code != "f1040")

    def search(self, query: str, year: int | None = None, k: int = 5, query_vector=None) -> list[Hit]:
        allowed = {i for i in range(len(self.units)) if year is None or self.units[i].applies_to(year)}
        bm25 = {i: v for i, v in self.bm25.scores(query).items() if i in allowed}
        bm25_rank = sorted(bm25, key=bm25.get, reverse=True)

        if self.embeddings is not None and query_vector is not None and allowed:
            import numpy as np
            idx = np.fromiter(allowed, dtype=np.int64)
            sims = self.embeddings[idx] @ np.asarray(query_vector, dtype=np.float32)
            top = idx[np.argsort(-sims)[: max(100, 4 * k)]]
            fused = rrf_merge([bm25_rank[:100], [int(i) for i in top]])
            ranked = sorted(fused, key=fused.get, reverse=True)
            scores = fused
        else:
            ranked, scores = bm25_rank, bm25

        pinned = self._line_hits(query, allowed) if self.line_lookup else []
        pinned_set = set(pinned)
        rest = [i for i in ranked if i not in pinned_set]

        if self.reranker is not None and rest:
            head = rest[:RERANK_DEPTH]
            rr = self.reranker(query, [index_text(self.units[i].heading_path, self.units[i].line_label,
                                                  self.units[i].text) for i in head])
            order = sorted(range(len(head)), key=lambda j: -float(rr[j]))
            scores = dict(scores)
            for j in order:
                scores[head[j]] = float(rr[j])
            rest = [head[j] for j in order] + rest[RERANK_DEPTH:]

        top_score = max((scores.get(i, 0.0) for i in rest[:1]), default=0.0)
        best = pinned + rest
        hits = []
        for i in best[:k]:
            u = self.units[i]
            ref = u.sources.get(year) if year is not None else next(iter(u.sources.values()))
            score = top_score + 1.0 if i in pinned_set else scores.get(i, 0.0)
            hits.append(Hit(u.text, (u.valid_from, u.valid_to), score, ref, u.line_label,
                            u.heading_path, u.form_code, bm25.get(i, 0.0), i in pinned_set))
        return hits

    def count_for_year(self, year: int) -> int:
        return sum(1 for u in self.units if u.applies_to(year))
