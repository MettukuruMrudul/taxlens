"""Retrieval evaluation: naive RAG baseline vs year-aware TaxLens.

Metrics (all with 95% bootstrap confidence intervals):

* recall@5     - a passage containing the gold answer, from the asked tax
                 year, is among the top 5 results
* hit@1        - the top result contains the gold answer for the asked year
* wrong_year@1 - the top result's text does not exist in the asked tax
                 year's documents (it came from another year's version).
                 Identical text shared across years is NOT counted as wrong.
* year detection accuracy (year-aware system only)

This measures retrieval, not final answers. Answer-level metrics need the
generator (Week 7) and human grading.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from taxlens.index.intervals import normalize
from taxlens.retrieve.yeardetect import detect_tax_year


def load_benchmark(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _contains_gold(text: str, gold_any: list[str]) -> bool:
    t = normalize(text).lower()
    return any(normalize(g).lower() in t for g in gold_any)


@dataclass
class QResult:
    qid: str
    detected_year: int | None
    hit_at_1: bool
    recall_at_5: bool
    wrong_year_at_1: bool


def evaluate(questions: list[dict], retriever, year_texts: dict[int, set[str]],
             year_aware: bool, k: int = 5) -> list[QResult]:
    out = []
    for q in questions:
        target = q["tax_year"]
        year = detect_tax_year(q["question"]).year if year_aware else None
        hits = retriever.search(q["question"], year=year, k=k)

        def correct(h) -> bool:
            in_target_year = normalize(h.text) in year_texts[target]
            return in_target_year and _contains_gold(h.text, q["gold_any"])

        top = hits[0] if hits else None
        out.append(QResult(
            qid=q["id"],
            detected_year=year,
            hit_at_1=bool(top) and correct(top),
            recall_at_5=any(correct(h) for h in hits),
            wrong_year_at_1=bool(top) and normalize(top.text) not in year_texts[target],
        ))
    return out


def bootstrap_ci(values: list[bool], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    xs = [1.0 if v else 0.0 for v in values]
    if not xs:
        return (0.0, 0.0, 0.0)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return (sum(xs) / len(xs), means[int(0.025 * n)], means[int(0.975 * n) - 1])


def summarize(results: list[QResult], questions: list[dict] | None = None) -> dict:
    s = {}
    for metric in ("hit_at_1", "recall_at_5", "wrong_year_at_1"):
        s[metric] = bootstrap_ci([getattr(r, metric) for r in results])
    if questions is not None and any(r.detected_year is not None for r in results):
        gold = {q["id"]: q["tax_year"] for q in questions}
        s["year_detection"] = bootstrap_ci([r.detected_year == gold[r.qid] for r in results])
    return s


def mcnemar_exact(a: list[bool], b: list[bool]) -> float:
    """Two-sided exact McNemar p-value for paired binary outcomes."""
    from math import comb
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def format_table(rows: dict[str, dict]) -> str:
    lines = ["| System | hit@1 | recall@5 | wrong-year@1 | year detection |", "|---|---|---|---|---|"]
    fmt = lambda t: f"{t[0]:.1%} [{t[1]:.0%}–{t[2]:.0%}]"
    for name, s in rows.items():
        yd = fmt(s["year_detection"]) if "year_detection" in s else "—"
        lines.append(f"| {name} | {fmt(s['hit_at_1'])} | {fmt(s['recall_at_5'])} | {fmt(s['wrong_year_at_1'])} | {yd} |")
    return "\n".join(lines)
