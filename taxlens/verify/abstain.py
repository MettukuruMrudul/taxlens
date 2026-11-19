"""Abstention: decide when TaxLens should say "I can't answer that reliably".

A wrong tax answer is worse than no answer, so TaxLens answers only when it
is confident. The decision uses features of the whole pipeline run:

    max_retrieval_score     best retrieval score, squashed to [0, 1]
    mean_retrieval_score    mean of the top-5 squashed scores
    year_confidence         from the year detector (explicit = 1.0 ... none = 0)
    intent_confidence       intent classifier probability
    supported_fraction      share of answer sentences that passed verification
    unsupported_claim_rate  from the verifier
    n_passages_for_year     index units valid for the asked year (log-scaled)
    has_figure              1 if the answer came from the figures table
    intent one-hot          seven columns

Two decision makers share this interface:

* ``RuleAbstainer`` (default until you have graded data): abstain if nothing
  was verified, or retrieval found nothing.
* ``LearnedAbstainer``: logistic regression trained on graded evaluation
  runs (``python -m taxlens.verify.abstain train <features.jsonl>``), with
  the threshold chosen to hit a target selective accuracy. Training reports
  AUROC and the selective-accuracy curve (accuracy vs. share answered).
"""
from __future__ import annotations

import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

from taxlens.config import MODELS_DIR
from taxlens.intent.classify import INTENTS

MODEL_PATH = MODELS_DIR / "abstain.pkl"
ABSTAIN_MESSAGE = ("I couldn't find IRS text for that tax year that supports a reliable answer, "
                   "so I'm not going to guess. Try rephrasing, naming the form or line, or "
                   "checking a different tax year.")


@dataclass
class AbstainFeatures:
    max_retrieval_score: float
    mean_retrieval_score: float
    year_confidence: float
    intent_confidence: float
    supported_fraction: float
    unsupported_claim_rate: float
    n_passages_for_year: float
    has_figure: float
    intent: str

    def vector(self) -> list[float]:
        base = [self.max_retrieval_score, self.mean_retrieval_score, self.year_confidence,
                self.intent_confidence, self.supported_fraction, self.unsupported_claim_rate,
                math.log1p(self.n_passages_for_year), self.has_figure]
        return base + [1.0 if self.intent == i else 0.0 for i in INTENTS]


def squash(score: float, scale: float = 10.0) -> float:
    """Map an unbounded retrieval score (BM25, RRF*k, logits) into [0, 1)."""
    return 1.0 - math.exp(-max(score, 0.0) / scale)


@dataclass
class Decision:
    abstain: bool
    p_correct: float
    reason: str


class RuleAbstainer:
    def decide(self, f: AbstainFeatures) -> Decision:
        if f.has_figure and f.supported_fraction > 0:
            return Decision(False, 0.95, "answered from the figures table")
        if f.n_passages_for_year == 0 or f.max_retrieval_score == 0:
            return Decision(True, 0.0, "no passages for this tax year")
        if f.supported_fraction == 0:
            return Decision(True, 0.05, "no answer sentence was supported by its citation")
        p = 0.5 * f.supported_fraction + 0.3 * f.max_retrieval_score + 0.2 * f.year_confidence
        return Decision(p < 0.45, p, "rule score")


class LearnedAbstainer:
    def __init__(self, model, threshold: float):
        self.model, self.threshold = model, threshold

    def decide(self, f: AbstainFeatures) -> Decision:
        p = float(self.model.predict_proba([f.vector()])[0][1])
        return Decision(p < self.threshold, p, f"learned p(correct)={p:.2f}")


def load_abstainer():
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as fh:
            saved = pickle.load(fh)
        return LearnedAbstainer(saved["model"], saved["threshold"])
    return RuleAbstainer()


def selective_accuracy_curve(p_correct: list[float], correct: list[bool]) -> list[tuple[float, float]]:
    """(coverage, accuracy) points from answering the most-confident first.
    Coverage = share of questions answered. Accuracy is over answered ones."""
    order = sorted(range(len(p_correct)), key=lambda i: -p_correct[i])
    points, right = [], 0
    for n, i in enumerate(order, start=1):
        right += bool(correct[i])
        points.append((n / len(order), right / n))
    return points


def choose_threshold(p_correct: list[float], correct: list[bool], target_accuracy: float) -> float:
    """Lowest threshold whose answered set still reaches target accuracy
    (i.e. answer as many questions as possible at that accuracy)."""
    best = 1.01
    for t in sorted(set(p_correct)):
        answered = [c for p, c in zip(p_correct, correct) if p >= t]
        if answered and sum(answered) / len(answered) >= target_accuracy:
            best = t
            break
    return best


def train(features_path: Path, target_accuracy: float = 0.95) -> dict:
    """features_path: JSONL rows {"features": {...AbstainFeatures...}, "correct": bool}."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    rows = [json.loads(line) for line in open(features_path, encoding="utf-8") if line.strip()]
    X = [AbstainFeatures(**r["features"]).vector() for r in rows]
    y = [bool(r["correct"]) for r in rows]
    if len(set(y)) < 2:
        raise ValueError("Need both correct and incorrect examples to train the abstainer")
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    folds = min(5, sum(y), len(y) - sum(y))
    p_cv = cross_val_predict(model, X, y, cv=max(2, folds), method="predict_proba")[:, 1].tolist()
    auroc = roc_auc_score(y, p_cv)
    threshold = choose_threshold(p_cv, y, target_accuracy)
    model.fit(X, y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump({"model": model, "threshold": threshold}, fh)
    curve = selective_accuracy_curve(p_cv, y)
    answered = [c for p, c in zip(p_cv, y) if p >= threshold]
    return {"n": len(y), "auroc_cv": auroc, "threshold": threshold,
            "coverage": len(answered) / len(y),
            "selective_accuracy": (sum(answered) / len(answered)) if answered else None,
            "curve": curve}


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] != "train":
        print("usage: python -m taxlens.verify.abstain train <features.jsonl> [target_accuracy]")
        return 2
    target = float(argv[2]) if len(argv) > 2 else 0.95
    r = train(Path(argv[1]), target)
    print(f"examples: {r['n']}  AUROC (cross-validated): {r['auroc_cv']:.3f}")
    print(f"threshold: {r['threshold']:.3f}  coverage: {r['coverage']:.1%}  "
          f"selective accuracy: {r['selective_accuracy']}")
    step = max(1, len(r["curve"]) // 10)
    print("coverage -> accuracy:", ", ".join(f"{c:.0%}:{a:.0%}" for c, a in r["curve"][::step]))
    print(f"saved {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
