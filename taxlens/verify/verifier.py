"""Claim-level verification of generated answers (component C8).

An answer sentence is only shown if the IRS passages it cites support it.
Each sentence is split into atomic claims, and every claim must pass:

1. **Number grounding** (always on, deterministic). Every dollar amount,
   percentage, date, and other number in the claim must appear in a cited
   passage or in the year's figures table. In tax answers the number *is*
   the answer, and an LLM that "rounds" $13,850 to $13,800 is wrong. This
   check cannot be fooled by fluent wording.
2. **Textual support**, by one of:
   * an NLI cross-encoder (if NLI_MODEL is set): some cited passage must
     entail the claim with probability >= TAU_NLI (0.7), or
   * lexical support (default): at least TAU_LEXICAL of the claim's content
     words appear in one cited passage.

``unsupported_claim_rate`` = unsupported claims / all claims, the headline
hallucination metric in the evaluation harness.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from taxlens.index.intervals import normalize
from taxlens.retrieve.bm25 import STOPWORDS

TAU_NLI = 0.7
TAU_LEXICAL = 0.5

# $12,550   12,550   7.65%   2023   (line labels and form numbers are removed first)
_NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
_MONTH_DATE = re.compile(r"\b(January|February|March|April|May|June|July|August|September|"
                         r"October|November|December)\s+\d{1,2},\s+\d{4}\b")
_WORD = re.compile(r"[a-z0-9]+")
# Words that carry no evidence either way.
_FILLER = STOPWORDS | {"also", "irs", "taxlens", "according", "says", "states", "per", "use",
                       "used", "see", "enter", "may", "must", "should", "would", "s"}
# "Form 1040", "Schedule 8812", "line 12", "Pub. 17", "boxes 1-3": identifiers, not facts.
_IDENTIFIER = re.compile(r"\b(?:forms?|schedules?|publications?|pub\.?|lines?|box(?:es)?|"
                         r"worksheets?|w|part)\s*[-#]?\s*\d[\w-]*(?:\s*(?:through|to|and|-)\s*\d[\w]*)?",
                         re.I)


@dataclass
class ClaimCheck:
    claim: str
    supported: bool
    numbers_ok: bool
    support_score: float          # best lexical overlap or NLI probability
    missing_numbers: list[str] = field(default_factory=list)


@dataclass
class SentenceCheck:
    text: str
    record_ids: list
    claims: list[ClaimCheck]

    @property
    def supported(self) -> bool:
        return bool(self.claims) and all(c.supported for c in self.claims)


@dataclass
class VerificationReport:
    sentences: list[SentenceCheck]

    @property
    def n_claims(self) -> int:
        return sum(len(s.claims) for s in self.sentences)

    @property
    def n_unsupported(self) -> int:
        return sum(1 for s in self.sentences for c in s.claims if not c.supported)

    @property
    def unsupported_claim_rate(self) -> float:
        return self.n_unsupported / self.n_claims if self.n_claims else 0.0

    @property
    def supported_fraction(self) -> float:
        return (sum(s.supported for s in self.sentences) / len(self.sentences)
                if self.sentences else 0.0)


def numbers_in(text: str) -> set[str]:
    """Canonical numbers mentioned in text: '$12,550' -> '12550', '7.65%' -> '7.65%'.
    Single digits are ignored (they are usually list markers or line numbers)."""
    out = set()
    text = _IDENTIFIER.sub(" ", text)
    for m in _NUMBER.findall(text):
        n = m.lstrip("$").replace(",", "").rstrip(".")
        if n.rstrip("%").replace(".", "").isdigit() and len(n.rstrip("%").replace(".", "")) >= 2:
            out.add(n)
    return out


def split_claims(sentence: str) -> list[str]:
    """Rule-based atomic claim splitter.

    Splits on semicolons and on ', and' / ', but' between clauses. Conservative
    on purpose: an unsplit compound claim is checked as a whole, which is
    stricter, not looser.
    """
    parts = [p.strip() for p in re.split(r";\s+|,\s+(?:and|but|while)\s+", sentence) if p.strip()]
    claims = [p for p in parts if len(_content_words(p)) >= 2]
    return claims or [sentence.strip()]


def _content_words(text: str) -> list[str]:
    return [w for w in _WORD.findall(normalize(text).lower()) if w not in _FILLER]


def lexical_support(claim: str, passage: str) -> float:
    words = set(_content_words(claim))
    if not words:
        return 1.0
    pw = set(_content_words(passage))
    return len(words & pw) / len(words)


class Verifier:
    """Checks answer sentences against the passages they cite."""

    def __init__(self, nli_model=None, tau_nli: float = TAU_NLI, tau_lexical: float = TAU_LEXICAL):
        self.nli = nli_model          # sentence_transformers.CrossEncoder (NLI head) or None
        self.tau_nli = tau_nli
        self.tau_lexical = tau_lexical

    @classmethod
    def from_config(cls) -> "Verifier":
        from taxlens import config
        if config.NLI_MODEL:
            try:
                from sentence_transformers import CrossEncoder
                return cls(CrossEncoder(config.NLI_MODEL))
            except Exception as e:  # missing package or model download failure
                print(f"NLI model unavailable ({e}); using lexical verification")
        return cls()

    def _entailment(self, claim: str, passages: list[str]) -> float:
        # NLI cross-encoders output logits for (contradiction, entailment, neutral)
        # in that order for the common cross-encoder/nli-* checkpoints.
        import numpy as np
        logits = np.asarray(self.nli.predict([(p, claim) for p in passages]))
        if logits.ndim == 1:
            return float(logits.max())
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        return float(probs[:, 1].max())

    def check_claim(self, claim: str, passages: list[str], extra_numbers: set[str]) -> ClaimCheck:
        available = set(extra_numbers)
        for p in passages:
            available |= numbers_in(p)
        # a date like "January 2, 1958" must appear as a whole
        missing_dates = []
        for m in _MONTH_DATE.finditer(claim):
            if not any(normalize(m.group(0)) in normalize(p) for p in passages):
                missing_dates.append(m.group(0))
        date_digits = {n for m in _MONTH_DATE.finditer(claim) for n in numbers_in(m.group(0))}
        missing = sorted(n for n in numbers_in(claim) - date_digits if n not in available)
        missing += missing_dates
        numbers_ok = not missing
        if not passages:
            return ClaimCheck(claim, False, numbers_ok, 0.0, missing)
        if self.nli is not None:
            score = self._entailment(claim, passages)
            ok = score >= self.tau_nli
        else:
            score = max(lexical_support(claim, p) for p in passages)
            ok = score >= self.tau_lexical
        return ClaimCheck(claim, numbers_ok and ok, numbers_ok, score, missing)

    def check_sentence(self, text: str, record_ids: list, passages_by_id: dict,
                       figure_numbers: set[str] | None = None) -> SentenceCheck:
        """figure_numbers: numbers that are allowed without a passage, i.e. the
        year's figures-table values and the tax year(s) being asked about."""
        cited = [passages_by_id[r] for r in record_ids if r in passages_by_id]
        extra = figure_numbers or set()
        claims = [self.check_claim(c, cited, extra) for c in split_claims(text)]
        return SentenceCheck(text, list(record_ids), claims)

    def verify(self, sentences: list[dict], passages_by_id: dict,
               figure_numbers: set[str] | None = None) -> VerificationReport:
        """sentences: [{"text": ..., "record_ids": [...]}, ...]"""
        return VerificationReport([
            self.check_sentence(s["text"], s.get("record_ids", []), passages_by_id, figure_numbers)
            for s in sentences])
