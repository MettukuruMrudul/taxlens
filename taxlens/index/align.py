"""Cross-year line alignment (component C11).

The IRS renumbers form lines between years (2021's line 30 "Recovery rebate
credit" became 2022's "Reserved for future use"; "Wages" became line 1a and
gained sub-lines). To answer "how did X change?" TaxLens needs to know which
line in year Y+1 is the *same logical line* as a line in year Y.

For each form and each pair of adjacent years, every pair of form lines is
scored:

    sim = 0.4 * semantic   (TF-IDF cosine; bi-encoder cosine if vectors given)
        + 0.3 * (1 - normalised edit distance)
        + 0.2 * (1 - |relative position difference|)
        + 0.1 * same line label

and the Hungarian algorithm (scipy.optimize.linear_sum_assignment) picks
the globally best one-to-one matching. Pairs scoring below TAU_ALIGN are
recorded as 'unmatched' (a line that was added, removed, or rewritten).
Matched lines share a canonical_line_id across all years.

    python -m taxlens.index.align
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from difflib import SequenceMatcher

from taxlens.index.intervals import normalize

TAU_ALIGN = 0.55
WEIGHTS = (0.4, 0.3, 0.2, 0.1)


@dataclass
class Line:
    record_id: object
    tax_year: int
    line_label: str | None
    text: str
    rel_pos: float          # position in the form, 0.0 (first line) .. 1.0 (last line)


@dataclass
class Pair:
    a: Line
    b: Line
    score: float
    matched: bool


def _semantic(texts_a: list[str], texts_b: list[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(texts_a + texts_b)
    return cosine_similarity(vec.transform(texts_a), vec.transform(texts_b))


def similarity_matrix(a: list[Line], b: list[Line], vectors_a=None, vectors_b=None):
    import numpy as np
    ta = [normalize(x.text).lower() for x in a]
    tb = [normalize(x.text).lower() for x in b]
    if vectors_a is not None and vectors_b is not None:
        sem = np.clip(np.asarray(vectors_a) @ np.asarray(vectors_b).T, 0.0, 1.0)
    else:
        sem = _semantic(ta, tb)
    w_sem, w_edit, w_pos, w_label = WEIGHTS
    sim = np.zeros((len(a), len(b)))
    for i, la in enumerate(a):
        for j, lb in enumerate(b):
            edit = SequenceMatcher(None, ta[i], tb[j]).ratio()
            pos = 1.0 - abs(la.rel_pos - lb.rel_pos)
            label = 1.0 if la.line_label and la.line_label == lb.line_label else 0.0
            sim[i, j] = w_sem * sem[i, j] + w_edit * edit + w_pos * pos + w_label * label
    return np.clip(sim, 0.0, 1.0)


def align(a: list[Line], b: list[Line], tau: float = TAU_ALIGN, vectors_a=None,
          vectors_b=None) -> list[Pair]:
    """Globally optimal one-to-one alignment of two years' lines."""
    if not a or not b:
        return []
    from scipy.optimize import linear_sum_assignment
    sim = similarity_matrix(a, b, vectors_a, vectors_b)
    rows, cols = linear_sum_assignment(-sim)
    pairs = [Pair(a[i], b[j], float(sim[i, j]), float(sim[i, j]) >= tau) for i, j in zip(rows, cols)]
    # lines of `a` left without any partner (more lines in a than b): record their best guess
    assigned = set(rows)
    for i in range(len(a)):
        if i not in assigned:
            j = int(sim[i].argmax())
            pairs.append(Pair(a[i], b[j], float(sim[i, j]), False))
    return pairs


def canonical_ids(form_code: str, years: list[int], pairs_by_year: dict[int, list[Pair]],
                  lines_by_year: dict[int, list[Line]]) -> dict[object, str]:
    """Give every line an id that stays the same along chains of matches."""
    ids: dict[object, str] = {}
    for y in years:
        for line in lines_by_year[y]:
            if line.record_id not in ids:
                ids[line.record_id] = f"{form_code}:{line.line_label or 'x'}@{y}"
        for p in pairs_by_year.get(y, []):
            if p.matched:
                ids[p.b.record_id] = ids[p.a.record_id]
    return ids


def main() -> int:
    from taxlens.db import connect
    with connect() as conn:
        rows = conn.execute(
            "SELECT r.record_id, d.form_code, d.tax_year, r.line_label, r.text, r.seq "
            "FROM line_records r JOIN documents d USING (doc_id) "
            "WHERE r.record_type = 'form_line' ORDER BY d.form_code, d.tax_year, r.seq").fetchall()
        if not rows:
            print("No form lines. Run the parser first.")
            return 1
        grouped: dict[str, dict[int, list[tuple]]] = {}
        for rid, fc, y, label, text, _seq in rows:
            grouped.setdefault(fc, {}).setdefault(y, []).append((rid, label, text))

        out_rows, summary = [], []
        for fc, by_year in sorted(grouped.items()):
            years = sorted(by_year)
            lines = {y: [Line(rid, y, lab, txt, i / max(1, len(rs) - 1))
                         for i, (rid, lab, txt) in enumerate(rs)] for y, rs in by_year.items()}
            pairs_by_year = {}
            for y in years[:-1]:
                if y + 1 not in lines:
                    continue
                pairs_by_year[y] = align(lines[y], lines[y + 1])
            ids = canonical_ids(fc, years, pairs_by_year, lines)
            for y, pairs in pairs_by_year.items():
                m = sum(p.matched for p in pairs)
                summary.append((fc, y, y + 1, len(pairs), m))
                out_rows += [(ids[p.a.record_id], fc, p.a.record_id, p.b.record_id, p.a.tax_year,
                              p.b.tax_year, p.score, "matched" if p.matched else "unmatched")
                             for p in pairs]

        with conn.transaction():
            conn.execute("TRUNCATE line_alignment RESTART IDENTITY")
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO line_alignment (canonical_line_id, form_code, record_id_a, record_id_b,"
                    " year_a, year_b, similarity_score, match_type) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
                    " ON CONFLICT (record_id_a, record_id_b) DO NOTHING", out_rows)

    print(f"{'form':<10} {'years':<11} {'lines':>6} {'matched':>8}")
    for fc, ya, yb, n, m in summary:
        print(f"{fc:<10} {ya}->{yb:<5} {n:>6} {m:>8} ({m / max(1, n):.0%})")
    print(f"\nWrote {len(out_rows)} alignment rows. Spot-check renumbered lines before trusting them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
