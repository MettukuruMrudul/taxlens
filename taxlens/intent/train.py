"""Compare intent models and save the best one.

    python -m taxlens.intent.train
    python -m taxlens.intent.train --data eval/benchmark/intent_seed_v0.jsonl

Candidates (each scored by 5-fold stratified cross-validated macro-F1):
  A. TF-IDF + logistic regression          (always available)
  B. TF-IDF + linear SVM (calibrated)      (always available)
  C. TF-IDF + XGBoost                      (if xgboost is installed)
A fine-tuned DistilBERT is the planned candidate D; it needs
requirements-ml.txt and a larger labeled set than the 140-question seed.

The winner is refit on all data and saved to models/intent.pkl, and the
report goes to eval/results/intent_model_selection.md.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

from taxlens.config import MODELS_DIR, PROJECT_ROOT
from taxlens.intent.classify import MODEL_PATH, build_tfidf_lr, load_seed


def candidates() -> dict:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    models = {
        "A. TF-IDF + logistic regression": build_tfidf_lr(),
        "B. TF-IDF + linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", CalibratedClassifierCV(LinearSVC(C=1.0), cv=3))]),
    }
    try:
        from xgboost import XGBClassifier

        class _XGBText:
            """XGBoost needs integer labels; wrap it so it looks like the others."""
            def __init__(self):
                from sklearn.preprocessing import LabelEncoder
                self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
                self.enc = LabelEncoder()
                self.clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1)

            def fit(self, X, y):
                self.clf.fit(self.vec.fit_transform(X), self.enc.fit_transform(y))
                self.classes_ = self.enc.classes_
                return self

            def predict(self, X):
                return self.enc.inverse_transform(self.clf.predict(self.vec.transform(X)))

            def predict_proba(self, X):
                return self.clf.predict_proba(self.vec.transform(X))

        models["C. TF-IDF + XGBoost"] = _XGBText()
    except ImportError:
        pass
    return models


def main() -> int:
    from sklearn.metrics import classification_report, f1_score
    from sklearn.model_selection import StratifiedKFold

    parser = argparse.ArgumentParser(description="Train and select the intent classifier.")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()
    texts, labels = load_seed(Path(args.data)) if args.data else load_seed()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores, reports = {}, {}
    for name in candidates():
        preds = [None] * len(texts)
        for train_idx, test_idx in skf.split(texts, labels):
            m = candidates()[name]  # a fresh, unfitted model for every fold
            m.fit([texts[i] for i in train_idx], [labels[i] for i in train_idx])
            for i, p in zip(test_idx, m.predict([texts[i] for i in test_idx])):
                preds[i] = p
        scores[name] = f1_score(labels, preds, average="macro")
        reports[name] = classification_report(labels, preds, digits=3, zero_division=0)
        print(f"{name:<36} macro-F1 = {scores[name]:.3f}")

    best = max(scores, key=scores.get)
    final = candidates()[best].fit(texts, labels)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final, f)

    out = PROJECT_ROOT / "eval" / "results" / "intent_model_selection.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Intent model selection ({datetime.now():%Y-%m-%d %H:%M})", "",
             f"Data: {len(texts)} labeled questions, 5-fold stratified cross-validation.", "",
             "| Model | macro-F1 |", "|---|---|"]
    lines += [f"| {n} | {s:.3f} |" for n, s in scores.items()]
    lines += ["", f"Deployed: **{best}** (saved to models/intent.pkl).", "",
              "Seed labels were drafted by Claude and are not yet human-verified;",
              "treat these numbers as development estimates.", "",
              "## Per-class report for the deployed model", "", "```", reports[best], "```"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nDeployed {best}. Report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
