"""Intent classifier (component C9).

Every question is routed to one of seven intent types. The intent decides
what happens next:

* ``out_of_scope``      -> explain what TaxLens covers; no retrieval, no LLM
* ``figure_lookup``     -> answer from the year_figures table first
* ``cross_year_change`` -> retrieve from two tax years and compare
* everything else       -> normal year-aware retrieval + cited answer

Model: TF-IDF (word 1-2 grams + character 3-5 grams) + logistic regression,
trained on ``eval/benchmark/intent_seed_v0.jsonl``. ``python -m
taxlens.intent.train`` compares it against other models with cross-validated
macro-F1 and saves the winner to ``models/intent.pkl``. If no saved model
exists, the classifier trains itself from the seed file on first use (it
takes well under a second).

A small domain guard sits in front of the model: a question with no tax
vocabulary at all is out of scope whatever the model says, so "best pizza
in Boston" can never be answered with IRS text.
"""
from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from taxlens.config import MODELS_DIR, PROJECT_ROOT

INTENTS = ("figure_lookup", "rule_or_eligibility", "form_navigation",
           "procedure_or_computation", "definition", "cross_year_change", "out_of_scope")

SEED_PATH = PROJECT_ROOT / "eval" / "benchmark" / "intent_seed_v0.jsonl"
MODEL_PATH = MODELS_DIR / "intent.pkl"

TAX_VOCAB = re.compile(
    r"\b(tax|taxes|taxable|irs|1040|1099|w-?2|schedule|deduct\w*|credit|refund|return|filing|"
    r"file|filer|dependent|income|wages|line|form|itemiz\w*|exemption|withholding|agi|"
    r"social security|ira|irs|standard deduction|head of household|jointly|separately|"
    r"estimated|overpayment|digital asset|crypto\w*|virtual currency|surviving spouse|"
    r"earned income|instructions|publication|pub 17|mortgage interest|charit\w*|alimony|"
    r"unemployment|scholarship|tips|gambling|medical|student loan|educator|amend\w*)\b",
    re.I)
# Clearly non-federal or advice topics that share tax words.
OUT_OF_SCOPE_HINTS = re.compile(
    r"\b(state tax(es)?|california|new york state|texas|canada|uk|germany|vat|sales tax|"
    r"property tax|stock|invest\w*|tax software|accountant|cpa near|llc|congress|"
    r"crypto price|bitcoin price|credit score|credit card)\b", re.I)


@dataclass
class IntentResult:
    intent: str          # one of INTENTS
    confidence: float    # probability of the chosen intent, in [0, 1]


def load_seed(path: Path = SEED_PATH) -> tuple[list[str], list[str]]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r["question"] for r in rows], [r["intent_type"] for r in rows]


def build_tfidf_lr():
    """The default model: word + character TF-IDF features, logistic regression."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
    ])
    return Pipeline([("features", features),
                     ("clf", LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced"))])


class IntentClassifier:
    def __init__(self, model=None):
        self.model = model

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "IntentClassifier":
        if path.exists():
            with open(path, "rb") as f:
                return cls(pickle.load(f))
        texts, labels = load_seed()
        return cls(build_tfidf_lr().fit(texts, labels))

    def classify(self, question: str) -> IntentResult:
        q = question.strip()
        if not q:
            return IntentResult("out_of_scope", 1.0)
        if not TAX_VOCAB.search(q):
            return IntentResult("out_of_scope", 0.95)
        probs = self.model.predict_proba([q])[0]
        classes = list(self.model.classes_)
        best = int(probs.argmax())
        intent, conf = str(classes[best]), float(probs[best])
        hinted = bool(OUT_OF_SCOPE_HINTS.search(q))
        if hinted and intent != "out_of_scope":
            oos = probs[classes.index("out_of_scope")] if "out_of_scope" in classes else 0.0
            if oos >= 0.15:
                intent, conf = "out_of_scope", float(max(oos, 0.6))
        elif not hinted and intent == "out_of_scope":
            # The question uses tax vocabulary and names nothing outside the corpus:
            # trust the domain signal and take the best in-scope intent instead.
            order = probs.argsort()[::-1]
            alt = next(int(i) for i in order if classes[int(i)] != "out_of_scope")
            intent, conf = str(classes[alt]), float(probs[alt])
        return IntentResult(intent, conf)


_default: IntentClassifier | None = None


def classify(question: str) -> IntentResult:
    """Classify with a lazily loaded, process-wide classifier."""
    global _default
    if _default is None:
        _default = IntentClassifier.load()
    return _default.classify(question)
