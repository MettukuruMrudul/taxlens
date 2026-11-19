"""Run the retrieval benchmark against the database and save the results.

Run from the project folder (after taxlens.index.build):
    python -m taxlens.eval.run_retrieval
    python -m taxlens.eval.run_retrieval --benchmark eval/benchmark/seed_v0.jsonl

Compares three systems on the same questions:
  1. Naive RAG: BM25 over every year's copy of every passage
  2. Year-aware: BM25 over validity-interval units, filtered to the detected year
  3. TaxLens v1: year-aware + exact form-line lookup
Prints a results table, saves it under eval/results/, and stores every
question's outcome in eval_runs / eval_items.
"""
import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from psycopg.types.json import Jsonb

from taxlens.config import PROJECT_ROOT
from taxlens.db import connect
from taxlens.eval.retrieval_eval import (evaluate, format_table, load_benchmark, mcnemar_exact,
                                         summarize)
from taxlens.index.build import load_records, load_units
from taxlens.index.intervals import normalize
from taxlens.retrieve.search import NaiveRetriever, YearAwareRetriever

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval benchmark.")
    parser.add_argument("--benchmark", default="eval/benchmark/seed_v0.jsonl")
    args = parser.parse_args()
    bench_path = PROJECT_ROOT / args.benchmark
    questions = load_benchmark(bench_path)

    with connect() as conn:
        records = load_records(conn)
        units = load_units(conn)
        conn.commit()
        if not units:
            print("index_units is empty. Run: python -m taxlens.index.build")
            return 1

        year_texts: dict[int, set[str]] = defaultdict(set)
        for r in records:
            year_texts[r["tax_year"]].add(normalize(r["text"]))
        missing = sorted({q["tax_year"] for q in questions} - set(year_texts))
        if missing:
            print(f"The benchmark asks about years not in the corpus: {missing}")
            return 1

        systems = {
            "Naive RAG (BM25, every year's copy)": (NaiveRetriever(records), False),
            "Year-aware (intervals, no line lookup)": (YearAwareRetriever(units, line_lookup=False), True),
            "TaxLens v1 (year-aware + line lookup)": (YearAwareRetriever(units), True),
        }
        results, summaries = {}, {}
        for name, (retriever, aware) in systems.items():
            results[name] = evaluate(questions, retriever, year_texts, year_aware=aware)
            summaries[name] = summarize(results[name], questions if aware else None)

        with conn.transaction():
            for name, res in results.items():
                run_id = conn.execute(
                    "INSERT INTO eval_runs (system, benchmark, n_questions, metrics)"
                    " VALUES (%s, %s, %s, %s) RETURNING run_id",
                    (name, args.benchmark, len(questions), Jsonb(summaries[name]))).fetchone()[0]
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO eval_items (run_id, question_id, detected_year, hit_at_1,"
                        " recall_at_5, wrong_year_at_1) VALUES (%s, %s, %s, %s, %s, %s)",
                        [(run_id, r.qid, r.detected_year, r.hit_at_1, r.recall_at_5, r.wrong_year_at_1)
                         for r in res])

    names = list(results)
    p_hit = mcnemar_exact([r.hit_at_1 for r in results[names[0]]], [r.hit_at_1 for r in results[names[2]]])
    p_wy = mcnemar_exact([r.wrong_year_at_1 for r in results[names[0]]],
                         [r.wrong_year_at_1 for r in results[names[2]]])
    table = format_table(summaries)
    years = sorted(year_texts)
    report = "\n".join([
        f"# Retrieval results: {Path(args.benchmark).name}",
        "",
        f"Run: {datetime.now():%Y-%m-%d %H:%M}. Questions: {len(questions)}. "
        f"Corpus: {len(records)} records / {len(units)} index units, tax years {years[0]}-{years[-1]}.",
        "",
        table,
        "",
        "Values are means with 95% bootstrap confidence intervals.",
        f"McNemar exact test, naive vs TaxLens v1: hit@1 p = {p_hit:.4f}; wrong-year@1 p = {p_wy:.4f}.",
        "",
        "These are retrieval metrics, not answer metrics. wrong-year@1 counts a top result whose",
        "text does not exist in the asked year's documents; identical text shared across years is",
        "not counted as wrong.",
    ]) + "\n"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"retrieval_{Path(args.benchmark).stem}.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
