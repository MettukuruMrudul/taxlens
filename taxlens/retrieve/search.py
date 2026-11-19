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

# Everyday words people use that IRS text doesn't (added to the BM25 query).
THESAURUS = {
    r"\bcrypto(currenc(y|ies))?\b|\bbitcoin\b|\bethereum\b": "virtual currency digital asset",
    r"\bnfts?\b": "digital asset",
    r"\bstimulus (check|payment)s?\b": "recovery rebate credit economic impact payment",
    r"\bsocial security\b": "social security benefits",
    r"\bw2\b": "w-2",
    r"\bbirth ?dates?\b|\bdates? of birth\b|\bbirthdays?\b": "born",
}
# Asking where something goes on a form, or what a named form says.
NAVIGATION = re.compile(
    r"\b(where (do|should|can|would) (i|we) (enter|report|put|write|elect|claim|show|list)|"
    r"where on|which line|what line|on (the|my) (\d{4} )?form|what does (the )?(\d{4} )?form)\b", re.I)
NAMED_FORM = re.compile(r"\b(form 1040|1040 form|the 1040)\b", re.I)
FORM_BOOST = 1.6          # BM25 multiplier for units of the form the question is about
PROMOTE_RATIO = 0.3       # promote the best form unit if it scores >= 30% of the top hit


# Words that say which document or year, not what the question is about.
_FORM_WORDS = re.compile(r"\b(forms?|schedule [123ab]|1040(-sr)?|lines? \d{1,2}[a-z]?|20[0-4]\d|tax year)\b", re.I)


def content_query(query: str) -> str:
    """The question minus form names, line numbers and years: what it is about."""
    return _FORM_WORDS.sub(" ", query)


def expand_query(query: str) -> str:
    extra = [words for pat, words in THESAURUS.items() if re.search(pat, query, re.I)]
    return " ".join([query, *extra])


def preferred_forms(query: str) -> set[str]:
    """Form codes whose own text should outrank instructions and publications."""
    if re.search(r"\binstructions?\b|\bpub(lication)?\.? ?17\b", query, re.I):
        return set()          # the user asked for the instructions or Pub 17
    forms = {fc for fc, pat in FORM_HINTS.items() if pat.search(query)}
    if NAMED_FORM.search(query):
        forms.add("f1040")
    if not forms and NAVIGATION.search(query):
        forms = {"f1040", *FORM_HINTS}   # "where do I enter X": some form line
    return forms


class YearAwareRetriever:
    """TaxLens: validity-interval units, filtered to the requested year.

    Ranking stages (each optional stage switches on when its model is given):

    1. Year filter. Only units whose interval contains the year are scored,
       so text from another tax year can never be returned.
    2. BM25 keyword ranking, plus dense cosine ranking if unit embeddings and
       a query vector are available, fused with Reciprocal Rank Fusion.
       The query is expanded with a small tax thesaurus ("crypto" also searches
       "virtual currency digital asset"), and when the question names a form or
       asks where something goes on one, that form's units get a BM25 boost so
       the form line outranks long instruction passages that mention it; the
       form unit that best matches the question's topic is moved to the top if
       it scores at least 30% of the best hit on that topic.
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
        raw = {i: v for i, v in self.bm25.scores(expand_query(query)).items() if i in allowed}
        forms = preferred_forms(query)
        bm25 = {i: v * (FORM_BOOST if self.units[i].form_code in forms else 1.0) for i, v in raw.items()}
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

        if forms and rest:
            # "Answer from the form, explain from the instructions": when the question is
            # about a form, the form unit that best matches the question's *topic* goes
            # first. Scoring on the topic (form names, years and line numbers removed)
            # stops "Form 1040 (2021) Page 2"-style text winning on the form name alone.
            topic = self.bm25.scores(expand_query(content_query(query)))
            cands = [i for i in rest if self.units[i].form_code in forms and topic.get(i, 0.0) > 0]
            if cands:
                cand = max(cands, key=lambda i: topic[i])
                best_topic = max(topic.get(i, 0.0) for i in rest[:RERANK_DEPTH])
                if cand != rest[0] and topic[cand] >= PROMOTE_RATIO * best_topic:
                    rest.remove(cand)
                    rest.insert(0, cand)

        top_score = max((scores.get(i, 0.0) for i in rest[:1]), default=0.0)
        best = pinned + rest
        hits = []
        for i in best[:k]:
            u = self.units[i]
            ref = u.sources.get(year) if year is not None else next(iter(u.sources.values()))
            score = top_score + 1.0 if i in pinned_set else scores.get(i, 0.0)
            hits.append(Hit(u.text, (u.valid_from, u.valid_to), score, ref, u.line_label,
                            u.heading_path, u.form_code, raw.get(i, 0.0), i in pinned_set))
        return hits

    def count_for_year(self, year: int) -> int:
        return sum(1 for u in self.units if u.applies_to(year))
