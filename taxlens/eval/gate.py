"""CI quality gate for answer-level evaluation (task 14.3).

    python -m taxlens.eval.gate            # checks the latest run against the previous one

Fails (exit code 1) when:
  * answer correctness drops more than 2 percentage points versus the
    previous run of the same system on the same benchmark,
  * the wrong-year answer rate is above 3%, or
  * the unsupported-claim rate is above 5%.
"""
from __future__ import annotations

import sys

GATE_RULES = {
    "max_correctness_drop": 0.02,
    "max_wrong_year_rate": 0.03,
    "max_unsupported_claim_rate": 0.05,
}


def _mean(metric) -> float:
    """Metrics are stored as [mean, ci_low, ci_high] or as a plain number."""
    if isinstance(metric, (list, tuple)):
        return float(metric[0])
    return float(metric)


def check_gate(current: dict, previous: dict | None, rules: dict = GATE_RULES) -> list[str]:
    """Return human-readable failures (an empty list means the gate passes)."""
    failures = []
    if previous is not None and "answer_correct" in previous and "answer_correct" in current:
        drop = _mean(previous["answer_correct"]) - _mean(current["answer_correct"])
        if drop > rules["max_correctness_drop"]:
            failures.append(f"answer correctness fell {drop:.1%} "
                            f"({_mean(previous['answer_correct']):.1%} -> {_mean(current['answer_correct']):.1%})")
    if "wrong_year" in current and _mean(current["wrong_year"]) > rules["max_wrong_year_rate"]:
        failures.append(f"wrong-year rate {_mean(current['wrong_year']):.1%} "
                        f"> {rules['max_wrong_year_rate']:.0%}")
    if ("unsupported_claim_rate" in current
            and _mean(current["unsupported_claim_rate"]) > rules["max_unsupported_claim_rate"]):
        failures.append(f"unsupported-claim rate {_mean(current['unsupported_claim_rate']):.1%} "
                        f"> {rules['max_unsupported_claim_rate']:.0%}")
    return failures


def latest_runs(conn, system_prefix: str = "taxlens-answers") -> tuple[dict | None, dict | None]:
    rows = conn.execute(
        "SELECT system, benchmark, metrics FROM eval_runs WHERE system LIKE %s"
        " ORDER BY run_id DESC LIMIT 50", (system_prefix + "%",)).fetchall()
    if not rows:
        return None, None
    system, bench, current = rows[0]
    previous = next((m for s, b, m in rows[1:] if s == system and b == bench), None)
    return current, previous


def main() -> int:
    from taxlens.db import connect
    with connect() as conn:
        current, previous = latest_runs(conn)
    if current is None:
        print("No answer evaluation runs yet. Run: python -m taxlens.eval.run_answers")
        return 1
    failures = check_gate(current, previous)
    if failures:
        print("QUALITY GATE FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("Quality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
