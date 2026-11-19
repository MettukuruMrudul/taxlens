"""Parser coverage audit (Week 3).

Measures how much of the IRS text the parser captures correctly, using a
sample of "gold" pages that a person checks by eye.

    python -m taxlens.eval.parse_audit sample   # pick pages, draw boxes, write audit.csv
    python -m taxlens.eval.parse_audit score    # read your verdicts, report coverage
    python -m taxlens.eval.parse_audit gaps     # automatic check for missing form lines

How the human check works
-------------------------
`sample` renders each gold page to a PNG with every parsed record drawn on
top: numbered form lines in green (tagged with their line number), other form
text in grey, and prose blocks in blue (tagged S1, S2, ... in reading order).
It also writes audit.csv with one row per scored unit:

* form pages: one row per numbered form line
* prose pages: one row per text block, in reading order

Open each PNG next to audit.csv and fill the `verdict` column:

    ok       captured correctly (right label / complete text / right order)
    wrong    captured, but with a wrong label, cut-off or mixed-in text, or
             out of reading order
    missed   add a NEW row for each numbered line or text block on the page
             that has no box at all

Coverage = ok / (ok + wrong + missed).
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pdfplumber
from PIL import ImageDraw

from taxlens.config import PROJECT_ROOT

AUDIT_DIR = PROJECT_ROOT / "eval" / "parse_audit"
AUDIT_CSV = AUDIT_DIR / "audit.csv"
RESULTS_MD = AUDIT_DIR / "results.md"
PAGES_DIR = AUDIT_DIR / "pages"
FIELDS = ["doc_id", "doc_type", "page", "tag", "record_type", "text", "verdict", "note"]
VERDICTS = {"ok", "wrong", "missed"}

# How many gold pages to draw from each document type.
SAMPLE_PLAN = {"form": 12, "instructions": 8, "publication": 10}
RESOLUTION = 110

COLORS = {"form_line": (0, 150, 60), "form_text": (150, 150, 150), "section": (30, 90, 220)}


# --------------------------------------------------------------------------- selection

def select_gold_pages(pages_by_doc: dict[str, tuple[str, list[int]]], seed: int = 7) -> list[tuple[str, int]]:
    """Stratified random sample of (doc_id, page), spread across document
    types and tax years. `pages_by_doc` maps doc_id -> (doc_type, pages)."""
    rng = random.Random(seed)
    chosen: list[tuple[str, int]] = []
    for doc_type, quota in SAMPLE_PLAN.items():
        docs = sorted(d for d, (t, _) in pages_by_doc.items() if t == doc_type)
        candidates = [(d, p) for d in docs for p in pages_by_doc[d][1]]
        rng.shuffle(candidates)
        # round-robin over documents so no single document dominates
        by_doc: dict[str, list[int]] = defaultdict(list)
        for d, p in candidates:
            by_doc[d].append(p)
        order = sorted(by_doc, key=lambda d: rng.random())
        picked: list[tuple[str, int]] = []
        while len(picked) < quota and any(by_doc.values()):
            for d in order:
                if by_doc[d] and len(picked) < quota:
                    picked.append((d, by_doc[d].pop()))
        chosen.extend(sorted(picked))
    return chosen


# --------------------------------------------------------------------------- audit rows

def page_units(records: list[dict], page: int) -> list[dict]:
    """The units to score on one page, each with a tag drawn on the PNG."""
    units, n = [], 0
    for r in records:
        for box in r["boxes"]:
            if box["page"] != page:
                continue
            if r["record_type"] == "form_line":
                units.append({"tag": r["line_label"], "record": r, "box": box})
            elif r["record_type"] == "section":
                n += 1
                units.append({"tag": f"S{n}", "record": r, "box": box})
    return units


def render_page(pdf_path: Path, page: int, records: list[dict], out: Path) -> None:
    """Draw every parsed record on the page image."""
    with pdfplumber.open(pdf_path) as pdf:
        img = pdf.pages[page - 1].to_image(resolution=RESOLUTION).original.convert("RGB")
    scale = RESOLUTION / 72
    draw = ImageDraw.Draw(img, "RGBA")
    tags = {id(u["box"]): u["tag"] for u in page_units(records, page)}
    for r in records:
        color = COLORS[r["record_type"]]
        for box in r["boxes"]:
            if box["page"] != page:
                continue
            x0, y0 = box["x0"] * scale, box["top"] * scale
            x1, y1 = box["x1"] * scale, box["bottom"] * scale
            draw.rectangle([x0 - 1, y0 - 1, x1 + 1, y1 + 1], outline=color + (255,), width=2,
                           fill=color + (25,))
            tag = tags.get(id(box))
            if tag:
                tw = 7 * len(tag) + 6
                draw.rectangle([x0 - 1, y0 - 13, x0 - 1 + tw, y0 - 1], fill=color + (230,))
                draw.text((x0 + 2, y0 - 12), tag, fill=(255, 255, 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)


def audit_rows(doc_id: str, doc_type: str, page: int, records: list[dict]) -> list[dict]:
    rows = []
    for u in page_units(records, page):
        rows.append({
            "doc_id": doc_id, "doc_type": doc_type, "page": page, "tag": u["tag"],
            "record_type": u["record"]["record_type"],
            "text": u["record"]["text"][:400], "verdict": "", "note": "",
        })
    return rows


# --------------------------------------------------------------------------- scoring

def score(rows: list[dict]) -> dict:
    """Coverage overall and per document type. Rows without a verdict are
    reported as unchecked and left out of the score."""
    counts: dict[str, Counter] = defaultdict(Counter)
    unchecked, invalid = 0, []
    for i, row in enumerate(rows, start=2):           # row 1 is the header
        v = (row.get("verdict") or "").strip().lower()
        if not v:
            unchecked += 1
            continue
        if v not in VERDICTS:
            invalid.append(f"line {i}: '{row.get('verdict')}'")
            continue
        for key in ("all", row["doc_type"]):
            counts[key][v] += 1
    result = {}
    for key, c in counts.items():
        total = c["ok"] + c["wrong"] + c["missed"]
        result[key] = {**c, "total": total, "coverage": c["ok"] / total if total else None}
    return {"by_type": result, "unchecked": unchecked, "invalid": invalid}


def format_report(s: dict, pages: int) -> str:
    lines = ["# Parser coverage audit", "",
             f"Gold pages: {pages}. Unchecked rows: {s['unchecked']}.", "",
             "| Scope | ok | wrong | missed | coverage |", "|---|---|---|---|---|"]
    for key in ["all", "form", "instructions", "publication"]:
        if key in s["by_type"]:
            r = s["by_type"][key]
            cov = f"{r['coverage']:.1%}" if r["coverage"] is not None else "—"
            lines.append(f"| {key} | {r['ok']} | {r['wrong']} | {r['missed']} | {cov} |")
    if s["invalid"]:
        lines += ["", "Invalid verdicts (use ok / wrong / missed): " + ", ".join(s["invalid"])]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- numbering gaps

def numbering_gaps(labels: list[str]) -> list[int]:
    """Line numbers missing between the lowest and highest numbered line."""
    nums = sorted({int(m.group(1)) for l in labels if (m := re.match(r"^(\d+)", l or ""))})
    if not nums:
        return []
    return [n for n in range(nums[0], nums[-1] + 1) if n not in nums]


# --------------------------------------------------------------------------- CLI

def _load(conn):
    docs = {d: (t, p) for d, t, p in conn.execute(
        "SELECT doc_id, doc_type, local_path FROM documents")}
    recs: dict[str, list[dict]] = defaultdict(list)
    for doc_id, rtype, label, text, boxes in conn.execute(
            "SELECT doc_id, record_type, line_label, text, boxes FROM line_records ORDER BY doc_id, seq"):
        recs[doc_id].append({"record_type": rtype, "line_label": label, "text": text, "boxes": boxes})
    return docs, recs


def cmd_sample(args) -> int:
    from taxlens.db import connect
    if AUDIT_CSV.exists() and not args.overwrite:
        print(f"{AUDIT_CSV} already exists. It may contain your verdicts, so it was not "
              f"replaced. Use --overwrite to start over.")
        return 1
    with connect() as conn:
        docs, recs = _load(conn)
    if not recs:
        print("No line_records yet. Run: python -m taxlens.ingest.run_parser")
        return 1
    pages_by_doc = {
        d: (docs[d][0], sorted({b["page"] for r in rs for b in r["boxes"]}))
        for d, rs in recs.items()
    }
    gold = select_gold_pages(pages_by_doc, seed=args.seed)
    rows = []
    for doc_id, page in gold:
        doc_type, local_path = docs[doc_id]
        render_page(PROJECT_ROOT / local_path, page, recs[doc_id], PAGES_DIR / f"{doc_id}_p{page}.png")
        rows.extend(audit_rows(doc_id, doc_type, page, recs[doc_id]))
        print(f"rendered {doc_id} page {page}")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with AUDIT_CSV.open("w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig opens cleanly in Excel
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(gold)} gold pages, {len(rows)} rows to check.")
    print(f"Images: {PAGES_DIR}\nChecklist: {AUDIT_CSV}")
    return 0


def cmd_score(args) -> int:
    if not AUDIT_CSV.exists():
        print("No audit.csv yet. Run the 'sample' command first.")
        return 1
    with AUDIT_CSV.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    pages = len({(r["doc_id"], r["page"]) for r in rows})
    report = format_report(score(rows), pages)
    RESULTS_MD.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved to {RESULTS_MD}")
    return 0


def cmd_gaps(args) -> int:
    from taxlens.db import connect
    with connect() as conn:
        docs, recs = _load(conn)
    found = False
    for doc_id in sorted(recs):
        if docs[doc_id][0] != "form":
            continue
        labels = [r["line_label"] for r in recs[doc_id] if r["record_type"] == "form_line"]
        missing = numbering_gaps(labels)
        status = "complete" if not missing else "MISSING lines " + ", ".join(map(str, missing))
        found |= bool(missing)
        print(f"{doc_id:<16} {len(labels):>3} numbered lines  {status}")
    return 1 if found else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Parser coverage audit.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="pick gold pages and write the checklist")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--overwrite", action="store_true")
    sub.add_parser("score", help="compute coverage from your verdicts")
    sub.add_parser("gaps", help="list missing numbered lines in each form")
    args = parser.parse_args()
    return {"sample": cmd_sample, "score": cmd_score, "gaps": cmd_gaps}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
