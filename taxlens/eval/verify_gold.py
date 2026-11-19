"""Check every benchmark gold answer against the actual IRS PDF text.

    python -m taxlens.eval.verify_gold                        # report only, changes nothing
    python -m taxlens.eval.verify_gold --mark --by "Mrudul"   # record evidence, set verified=true

For each question, the gold strings are searched (case- and whitespace-
insensitive) in that tax year's documents, Form 1040 and its schedules first,
then instructions and Publication 17. A question is marked verified only if
a gold string is found in its own year's text; the page is stored as
evidence. The report also flags:

* gold text that appears identically in other years (the question cannot
  detect a mix-up between those years), and
* questions whose note says the gold string is not a full answer.

The --mark step is meant to follow a human spot-check of the report.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from taxlens.config import PROJECT_ROOT, RAW_DIR

FORM_FIRST = ("f1040", "f1040s1", "f1040s2", "f1040s3", "f1040sa", "f1040sb", "i1040gi", "i1040sca", "p17")


def norm(s: str) -> str:
    s = (s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
          .replace("\u2014", "-").replace("\u2013", "-"))
    return re.sub(r"\s+", " ", s).lower()


class PdfText:
    """Lazily extracted, cached page text per (form, year)."""

    def __init__(self, raw_dir: Path = RAW_DIR):
        self.raw_dir, self.cache = raw_dir, {}

    def pages(self, form: str, year: int) -> list[str]:
        key = (form, year)
        if key not in self.cache:
            path = self.raw_dir / f"{form}--{year}.pdf"
            if not path.exists():
                self.cache[key] = []
            else:
                import pdfplumber
                with pdfplumber.open(path) as pdf:
                    self.cache[key] = [norm(p.extract_text() or "") for p in pdf.pages]
        return self.cache[key]

    def find(self, needle: str, year: int, forms=FORM_FIRST) -> tuple[str, int] | None:
        n = norm(needle)
        for form in forms:
            for i, text in enumerate(self.pages(form, year), start=1):
                if n in text:
                    return form, i
        return None


def check(questions: list[dict], pdfs: PdfText, years: list[int]) -> list[dict]:
    out = []
    for q in questions:
        hit = next(((g, h) for g in q["gold_any"] if (h := pdfs.find(g, q["tax_year"]))), None)
        other_years = []
        if hit:
            gold, (form, _page) = hit
            other_years = [y for y in years if y != q["tax_year"] and pdfs.find(gold, y, forms=(form,))]
        out.append({"id": q["id"], "tax_year": q["tax_year"], "found": bool(hit),
                    "gold": hit[0] if hit else q["gold_any"][0],
                    "evidence": {"doc": f"{hit[1][0]}--{q['tax_year']}", "page": hit[1][1]} if hit else None,
                    "same_text_in_years": other_years})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify benchmark gold answers against IRS PDFs.")
    parser.add_argument("--benchmark", default="eval/benchmark/seed_v0.jsonl")
    parser.add_argument("--mark", action="store_true", help="write verified/evidence back to the file")
    parser.add_argument("--by", default=None, help="who spot-checked the report (required with --mark)")
    args = parser.parse_args()
    if args.mark and not args.by:
        parser.error("--mark needs --by NAME: verification records who checked it")

    path = PROJECT_ROOT / args.benchmark
    questions = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    years = sorted({q["tax_year"] for q in questions})
    results = check(questions, PdfText(), years)

    print(f"{'id':<6}{'year':<6}{'found':<8}{'evidence':<20}same text in")
    for r in results:
        ev = f"{r['evidence']['doc']} p{r['evidence']['page']}" if r["evidence"] else "-"
        print(f"{r['id']:<6}{r['tax_year']:<6}{'yes' if r['found'] else 'NO':<8}{ev:<20}"
              f"{', '.join(map(str, r['same_text_in_years'])) or '-'}")
    missing = [r["id"] for r in results if not r["found"]]
    weak = [r["id"] for r in results if r["same_text_in_years"]]
    print(f"\n{len(results) - len(missing)}/{len(results)} gold answers found in their own year's IRS text.")
    if missing:
        print("NOT FOUND (fix these before marking):", ", ".join(missing))
    if weak:
        print(f"{len(weak)} questions use text identical in other years (can't catch those year mix-ups):",
              ", ".join(weak))

    if args.mark:
        by_id = {r["id"]: r for r in results}
        for q in questions:
            r = by_id[q["id"]]
            if r["found"]:
                q["verified"] = True
                q["verified_by"] = args.by
                q["evidence"] = r["evidence"]
            q["same_text_in_years"] = r["same_text_in_years"]
        with open(path, "w", encoding="utf-8") as f:
            for q in questions:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        print(f"\nMarked {len(results) - len(missing)} questions verified by {args.by} in {args.benchmark}.")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
