"""Training triples with cross-year hard negatives (task 21.3).

A generic embedding model thinks 2021's "line 30: Recovery rebate credit"
and 2022's "line 30: Reserved for future use" are near-duplicates of a
question about line 30. To fine-tune a bi-encoder or cross-encoder that
respects tax years, each training example pairs a query with:

* positive          - the unit that answers it, for the query's year
* cross_year_neg    - same form + line label, adjacent year (±1), different text
* in_year_neg       - a high-BM25 unit from the *same* year that is not the answer
* random_neg        - any other unit

Queries come from templates over form lines (no LLM needed). Add
LLM-written paraphrases later to reach the ~40,000 triples in the plan.

    python -m taxlens.index.hard_negatives          # writes data/training/triples.jsonl
"""
from __future__ import annotations

import json
import random
import sys

from taxlens.config import PROJECT_ROOT
from taxlens.index.intervals import Unit
from taxlens.retrieve.search import YearAwareRetriever

FORM_NAMES = {"f1040": "Form 1040", "f1040s1": "Schedule 1", "f1040s2": "Schedule 2",
              "f1040s3": "Schedule 3", "f1040sa": "Schedule A", "f1040sb": "Schedule B"}
TEMPLATES = [
    "What is line {label} of {form} used for in {year}?",
    "What goes on line {label} of the {year} {form}?",
    "Which line of the {year} {form} is for {topic}?",
    "Where do I report {topic} on my {year} {form}?",
]
OUT_PATH = PROJECT_ROOT / "data" / "training" / "triples.jsonl"


def _topic(text: str) -> str:
    return text.split(".")[0].strip().lower()[:80]


def make_triples(units: list[Unit], seed: int = 0, per_unit: int = 2) -> list[dict]:
    rng = random.Random(seed)
    retriever = YearAwareRetriever(units, line_lookup=False)
    by_key: dict[tuple, list[Unit]] = {}
    for u in units:
        if u.record_type == "form_line" and u.line_label:
            by_key.setdefault((u.form_code, u.line_label), []).append(u)
    triples = []
    for u in units:
        if u.record_type != "form_line" or not u.line_label or u.form_code not in FORM_NAMES:
            continue
        for year in range(u.valid_from, u.valid_to + 1):
            cross = [o for o in by_key[(u.form_code, u.line_label)]
                     if o is not u and (o.applies_to(year - 1) or o.applies_to(year + 1))
                     and not o.applies_to(year)]
            if not cross:
                continue  # the line never changed: not a useful hard example
            for template in rng.sample(TEMPLATES, k=min(per_unit, len(TEMPLATES))):
                query = template.format(label=u.line_label, form=FORM_NAMES[u.form_code], year=year,
                                        topic=_topic(u.text))
                in_year = next((h.text for h in retriever.search(query, year=year, k=10)
                                if h.text != u.text), None)
                random_neg = rng.choice(units).text
                triples.append({"query": query, "year": year, "positive": u.text,
                                "cross_year_neg": rng.choice(cross).text,
                                "in_year_neg": in_year, "random_neg": random_neg})
    return triples


def main() -> int:
    from taxlens.db import connect
    from taxlens.index.build import load_units
    with connect() as conn:
        units = load_units(conn)
    if not units:
        print("index_units is empty. Run: python -m taxlens.index.build")
        return 1
    triples = make_triples(units)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for t in triples:
            f.write(json.dumps(t) + "\n")
    print(f"Wrote {len(triples)} triples to {OUT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
