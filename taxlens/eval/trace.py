"""Show exactly what TaxLens retrieves for a question, and where each hit comes from.

    python -m taxlens.eval.trace "What does the 2022 Form 1040 ask about crypto?"
    python -m taxlens.eval.trace "line 30?" --year 2023 --k 8

For each hit: the unit's validity interval, the record cited for the asked
year, and the document that record actually belongs to. A citation whose
document year differs from the asked year is flagged, since that should
never happen.
"""
from __future__ import annotations

import argparse
import sys

from taxlens.db import connect
from taxlens.index.build import load_units
from taxlens.retrieve.search import YearAwareRetriever
from taxlens.retrieve.yeardetect import detect_tax_year


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace retrieval for one question.")
    parser.add_argument("question")
    parser.add_argument("--year", type=int)
    parser.add_argument("--k", type=int, default=6)
    args = parser.parse_args()

    guess = detect_tax_year(args.question, explicit=args.year)
    print(f"tax year: {guess.year} ({guess.source}, confidence {guess.confidence})\n")
    if guess.year is None:
        return 1
    with connect() as conn:
        units = load_units(conn)
        hits = YearAwareRetriever(units).search(args.question, year=guess.year, k=args.k)
        refs = [h.ref for h in hits if h.ref is not None]
        docs = {}
        if refs:
            for rid, doc_id, ty in conn.execute(
                    "SELECT r.record_id, d.doc_id, d.tax_year FROM line_records r JOIN documents d"
                    " USING (doc_id) WHERE r.record_id = ANY(%s)", (refs,)):
                docs[rid] = (doc_id, ty)

    problems = 0
    for i, h in enumerate(hits, start=1):
        unit = next(u for u in units if u.text == h.text and (u.valid_from, u.valid_to) == h.years)
        doc_id, doc_year = docs.get(h.ref, (None, None))
        flag = ""
        if h.ref is None:
            flag, problems = "  <-- NO RECORD FOR THIS YEAR", problems + 1
        elif doc_year != guess.year:
            flag, problems = f"  <-- CITES A {doc_year} RECORD", problems + 1
        print(f"[{i}] {h.form_code} {h.line_label or ''} valid {h.years[0]}-{h.years[1]}"
              f"  bm25={h.bm25:.2f}{'  pinned' if h.pinned else ''}{flag}")
        print(f"    cites record {h.ref} from {doc_id}; unit sources: {unit.sources}")
        print(f"    {h.text[:160]}\n")
    print("No year problems." if not problems else f"{problems} hit(s) with a year problem.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
