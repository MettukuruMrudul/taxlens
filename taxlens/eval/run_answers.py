"""Answer-level evaluation (component C10, tasks 14.1-14.3).

    python -m taxlens.eval.run_answers
    python -m taxlens.eval.run_answers --benchmark eval/benchmark/seed_v0.jsonl --gate

Runs every benchmark question through the full pipeline (no year hint, so
year detection is tested too) and measures, with 95% bootstrap CIs:

  answer_correct          answered, and the answer contains a gold string
  abstained               TaxLens declined to answer
  wrong_year              the answer contains another year's gold value for
                          the same topic and not this year's (e.g. 2022's
                          standard deduction given for 2023)
  citation_year_precision share of citations that come from the asked tax year
  year_detection          detected tax year == the benchmark's tax year
  unsupported_claim_rate  verifier's unsupported claims / claims (answered only)
  selective_accuracy      correctness among answered questions

Writes eval/results/answers_<benchmark>.md (with failure clusters by
intent, tax year and topic), stores the run in eval_runs/eval_items, and
writes eval/results/answer_features.jsonl for training the abstainer:
    python -m taxlens.verify.abstain train eval/results/answer_features.jsonl

Gold strings in seed_v0 were drafted by Claude and are not yet verified by
a human; treat results as development numbers until they are.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from taxlens.config import PROJECT_ROOT
from taxlens.eval.retrieval_eval import bootstrap_ci, load_benchmark
from taxlens.index.intervals import normalize

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


def contains(text: str, needles: list[str]) -> bool:
    t = normalize(text).lower()
    return any(normalize(n).lower() in t for n in needles)


def other_year_gold(questions: list[dict]) -> dict[str, list[str]]:
    """question id -> gold strings that belong to the same topic in *other* years."""
    by_topic: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for q in questions:
        by_topic[q["topic"]][q["tax_year"]].update(q["gold_any"])
    out = {}
    for q in questions:
        mine = set(q["gold_any"])
        others = {g for y, gs in by_topic[q["topic"]].items() if y != q["tax_year"] for g in gs}
        out[q["id"]] = sorted(others - mine)
    return out


def grade(q: dict, res: dict, wrong_pool: list[str]) -> dict:
    text = " ".join(s["text"] for s in res.get("sentences", []))
    answered = res["status"] == "answered"
    correct = answered and contains(text, q["gold_any"])
    wrong_year = answered and not correct and bool(wrong_pool) and contains(text, wrong_pool)
    cites = res.get("citations", [])
    cite_ok = [c.get("tax_year") == q["tax_year"] for c in cites]
    gold_in_sources = any(contains(c.get("snippet") or "", q["gold_any"]) for c in cites)
    return {"id": q["id"], "topic": q["topic"], "tax_year": q["tax_year"], "intent": res.get("intent"),
            "status": res["status"], "answered": answered, "correct": correct,
            "wrong_year": wrong_year, "year_ok": res.get("tax_year") == q["tax_year"],
            "citation_year_precision": (sum(cite_ok) / len(cite_ok)) if cite_ok else None,
            "gold_in_sources": gold_in_sources,
            "unsupported_claim_rate": res.get("unsupported_claim_rate", 0.0),
            "answer": text, "mode": res.get("mode")}


def summarize(rows: list[dict]) -> dict:
    s = {
        "answer_correct": bootstrap_ci([r["correct"] for r in rows]),
        "abstained": bootstrap_ci([not r["answered"] for r in rows]),
        "wrong_year": bootstrap_ci([r["wrong_year"] for r in rows]),
        "year_detection": bootstrap_ci([r["year_ok"] for r in rows]),
    }
    answered = [r for r in rows if r["answered"]]
    if answered:
        s["selective_accuracy"] = bootstrap_ci([r["correct"] for r in answered])
        s["unsupported_claim_rate"] = sum(r["unsupported_claim_rate"] for r in answered) / len(answered)
    prec = [r["citation_year_precision"] for r in rows if r["citation_year_precision"] is not None]
    if prec:
        s["citation_year_precision"] = sum(prec) / len(prec)
    return s


def failure_clusters(rows: list[dict]) -> dict[str, Counter]:
    bad = [r for r in rows if not r["correct"]]
    return {"intent": Counter(r["intent"] for r in bad),
            "tax_year": Counter(r["tax_year"] for r in bad),
            "topic": Counter(r["topic"] for r in bad),
            "status": Counter(r["status"] for r in bad)}


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def fmt(t) -> str:
    if isinstance(t, (list, tuple)):
        return f"{t[0]:.1%} [{t[1]:.0%}–{t[2]:.0%}]"
    return f"{t:.1%}"


def report(bench: str, system: str, rows: list[dict], summary: dict, n_years: list[int],
           n_verified: int = 0) -> str:
    lines = [f"# Answer evaluation: {Path(bench).name}", "",
             f"Run {datetime.now():%Y-%m-%d %H:%M}. System: `{system}`. Questions: {len(rows)}. "
             f"Corpus years: {n_years[0]}-{n_years[-1]}.", "",
             "| Metric | Value (95% CI) |", "|---|---|"]
    for k, v in summary.items():
        lines.append(f"| {k} | {fmt(v)} |")
    lines += ["", "## Failure clusters (incorrect or abstained)", ""]
    for name, counts in failure_clusters(rows).items():
        if counts:
            lines.append(f"- **{name}**: " + ", ".join(f"{k}: {v}" for k, v in counts.most_common()))
    lines += ["", "## Incorrect answers", "", "| id | year | status | answer |", "|---|---|---|---|"]
    for r in rows:
        if not r["correct"]:
            ans = (r["answer"] or "—").replace("|", "/")[:160]
            lines.append(f"| {r['id']} | {r['tax_year']} | {r['status']} | {ans} |")
    if n_verified == len(rows):
        lines += ["", "All gold answers verified against the IRS source PDFs (python -m taxlens.eval.verify_gold)."]
    else:
        lines += ["", f"{n_verified} of {len(rows)} gold answers verified; treat results as provisional."]
    return "\n".join(lines) + "\n"


def main() -> int:
    from psycopg.types.json import Jsonb

    from taxlens import config
    from taxlens.db import connect
    from taxlens.pipeline import build_pipeline

    parser = argparse.ArgumentParser(description="Answer-level benchmark.")
    parser.add_argument("--benchmark", default="eval/benchmark/seed_v0.jsonl")
    parser.add_argument("--gate", action="store_true", help="exit 1 if the quality gate fails")
    parser.add_argument("--no-models", action="store_true", help="skip optional neural models")
    args = parser.parse_args()

    questions = load_benchmark(PROJECT_ROOT / args.benchmark)
    pipeline = build_pipeline(connect, with_models=not args.no_models)
    if not pipeline.corpus.units:
        print("index_units is empty. Run: python -m taxlens.index.build")
        return 1
    system = f"taxlens-answers:{config.LLM_PROVIDER}"
    pools = other_year_gold(questions)

    rows, features = [], []
    for i, q in enumerate(questions, start=1):
        res = pipeline.run(q["question"])
        d = res.to_dict()
        row = grade(q, d, pools[q["id"]])
        rows.append(row)
        if res.features is not None:
            # label = whether the attempted answer (shown or not) was right; that is
            # exactly what the abstainer has to predict
            features.append({"id": q["id"], "features": res.features,
                             "correct": bool(res.attempt) and contains(res.attempt, q["gold_any"])})
        print(f"[{i}/{len(questions)}] {q['id']} {row['status']:<12} {'OK ' if row['correct'] else 'XX '} "
              f"{q['question'][:60]}")

    summary = summarize(rows)
    with connect() as conn, conn.transaction():
        run_id = conn.execute(
            "INSERT INTO eval_runs (system, benchmark, n_questions, metrics, git_sha)"
            " VALUES (%s, %s, %s, %s, %s) RETURNING run_id",
            (system, args.benchmark, len(rows), Jsonb(summary), git_sha())).fetchone()[0]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO eval_items (run_id, question_id, detected_year, hit_at_1, recall_at_5,"
                " wrong_year_at_1, answer, grade) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                [(run_id, r["id"], r["tax_year"] if r["year_ok"] else None, r["correct"],
                  r["correct"] or r["gold_in_sources"], r["wrong_year"], r["answer"],
                  "correct" if r["correct"] else ("abstained" if not r["answered"] else "incorrect"))
                 for r in rows])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"answers_{Path(args.benchmark).stem}.md"
    text = report(args.benchmark, system, rows, summary, pipeline.corpus.years,
                  sum(bool(q.get("verified")) for q in questions))
    out.write_text(text, encoding="utf-8")
    with open(RESULTS_DIR / "answer_features.jsonl", "w", encoding="utf-8") as f:
        for row in features:
            f.write(json.dumps(row) + "\n")
    print("\n" + text)
    print(f"Saved {out} (eval run {run_id}).")

    if args.gate:
        from taxlens.eval.gate import check_gate, latest_runs
        with connect() as conn:
            current, previous = latest_runs(conn, system)
        failures = check_gate(current, previous)
        if failures:
            print("QUALITY GATE FAILED:\n  - " + "\n  - ".join(failures))
            return 1
        print("Quality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
