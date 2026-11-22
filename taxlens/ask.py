"""Ask TaxLens a question from the command line.

    python -m taxlens.ask "What is the standard deduction for a single filer for 2023?"
    python -m taxlens.ask "What is line 30 used for?" --year 2021
    python -m taxlens.ask "How did line 30 change from 2021 to 2022?" --debug

Runs the same pipeline as the API: year detection, intent, year-aware
retrieval, cited answer generation, verification, and abstention. With
LLM_PROVIDER=none (the default) answers quote the IRS text directly.
"""
import argparse
import sys

from taxlens.db import connect
from taxlens.pipeline import build_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask TaxLens a question.")
    parser.add_argument("question")
    parser.add_argument("--year", type=int, help="tax year (overrides detection)")
    parser.add_argument("--debug", action="store_true", help="show intent, timings and abstention details")
    args = parser.parse_args()

    pipeline = build_pipeline(connect)
    if not pipeline.corpus.units:
        print("index_units is empty. Run: python -m taxlens.index.build")
        return 1
    res = pipeline.run(args.question, args.year)

    if res.tax_year:
        print(f"Tax year {res.tax_year} ({(res.year_source or '').replace('_', ' ')})\n")
    if res.status != "answered":
        print(res.message)
    else:
        for s in res.sentences:
            marks = "".join(f"[{n}]" for n in s["citations"])
            print(f"{s['text']} {marks}")
        print()
        for c in res.citations:
            where = f"line {c['line_label']}" if c["line_label"] else (c["heading_path"] or "")
            print(f"[{c['n']}] {c['title']} ({c['tax_year']}), page {c['page']} - {where}")
    if args.debug:
        print(f"\nintent={res.intent} mode={res.mode} p_correct={res.p_correct} "
              f"unsupported_claim_rate={res.unsupported_claim_rate:.2f} reason={res.abstain_reason}")
        print("timings (ms):", res.timings)
        for note in res.notes:
            print("note:", note)
        if res.mode == "extractive_fallback" and not res.notes:
            print("note: the LLM answered, but with no usable cited sentence (it may have said it can't answer)")
    print(f"\n{res.disclaimer}")
    return 0 if res.status == "answered" else 2


if __name__ == "__main__":
    sys.exit(main())
