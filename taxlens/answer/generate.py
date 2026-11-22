"""Answer generator (component C7).

Turns retrieved IRS passages into a short answer in which **every sentence
cites the passages it came from**, then lets only verified sentences
through. Three modes, chosen per question:

``figure``      Dollar amounts for a parameter in the figures table (e.g. the
                standard deduction) are written from the table, never by an
                LLM. The sentence cites the exact Form 1040 text the number
                was extracted from.
``llm``         The configured LLM writes JSON Lines, one sentence per line:
                ``{"text": "...", "cites": ["P1", "P3"]}``. Lines are parsed
                as they stream in, verified against the cited passages, and
                yielded immediately (low time-to-first-sentence). A sentence
                that fails verification is sent back for repair up to
                MAX_ATTRIBUTION_RETRIES times, then dropped.
``extractive``  No LLM configured (or the LLM failed): quote the best
                passages directly. Always faithful, less fluent.

The caller decides whether to abstain from the verification results.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from taxlens.llm import LLM, LLMError, LLMTimeout
from taxlens.verify.verifier import Verifier, numbers_in

DISCLAIMER = ("TaxLens provides information from IRS publications for educational purposes "
              "only and is not tax advice.")
MAX_ATTRIBUTION_RETRIES = 3
MAX_PASSAGES = 8

STATUS_PATTERNS = {
    "single_or_mfs": r"\bsingle\b|\bmarried filing separately\b|\bmfs\b|\bfiling separately\b",
    "mfj_or_qss": (r"\bmarried filing jointly\b|\bfiling jointly\b|\bjointly\b|\bmfj\b|"
                   r"\bmarried couple\b|\bqualifying (?:widow|surviving spouse)|\bwidow"),
    "head_of_household": r"\bhead of household\b|\bhoh\b",
}
STATUS_NAMES = {"single_or_mfs": "single or married filing separately",
                "mfj_or_qss": "married filing jointly or qualifying surviving spouse",
                "head_of_household": "head of household"}
FIGURE_TRIGGERS = {"standard_deduction": re.compile(r"\bstandard deduction\b", re.I)}
# "...while taking the standard deduction?" asks about the choice, not the amount.
FIGURE_CHOICE = re.compile(r"\b(if|while|when|instead of|or)\b[^?.]*\b(take|taking|claim|claiming|use|using|"
                           r"choose|choosing|itemiz\w*)\b[^?.]*\bstandard deduction\b", re.I)


@dataclass
class Passage:
    """A retrieved passage as the generator sees it."""
    ref: object                      # line_records.record_id to cite
    text: str
    years: tuple[int, int]
    line_label: str | None = None
    heading_path: str | None = None
    title: str | None = None         # e.g. "Form 1040 (2023)"
    score: float = 0.0               # retrieval score
    pinned: bool = False             # exact line-lookup match


@dataclass
class Sentence:
    text: str
    record_ids: list
    supported: bool = True
    attempts: int = 1


@dataclass
class Answer:
    sentences: list[Sentence]
    mode: str                        # figure | llm | extractive | extractive_fallback
    disclaimer: str = DISCLAIMER
    dropped: int = 0                 # sentences removed after failed attribution
    n_claims: int = 0
    n_unsupported: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def unsupported_claim_rate(self) -> float:
        return self.n_unsupported / self.n_claims if self.n_claims else 0.0

    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)

    def to_dict(self) -> dict:
        return {"sentences": [{"text": s.text, "record_ids": s.record_ids} for s in self.sentences],
                "disclaimer": self.disclaimer, "mode": self.mode}


# --------------------------------------------------------------------------- figures

def detect_filing_status(question: str) -> str | None:
    q = question.lower()
    hits = [s for s, pat in STATUS_PATTERNS.items() if re.search(pat, q)]
    return hits[0] if len(hits) == 1 else None


def figure_answer(question: str, tax_year: int, figures: list[dict]) -> list[Sentence] | None:
    """Answer straight from the figures table when the question asks for one.
    figures: rows of year_figures for the tax year."""
    if FIGURE_CHOICE.search(question):
        return None
    for parameter, trigger in FIGURE_TRIGGERS.items():
        if not trigger.search(question):
            continue
        rows = [f for f in figures if f["parameter"] == parameter and f["tax_year"] == tax_year]
        if not rows:
            return None
        status = detect_filing_status(question)
        chosen = [r for r in rows if r["filing_status"] == status] if status else rows
        if not chosen:
            chosen = rows
        label = parameter.replace("_", " ")
        if len(chosen) == 1:
            r = chosen[0]
            return [Sentence(f"For tax year {tax_year}, the {label} for "
                             f"{STATUS_NAMES.get(r['filing_status'], r['filing_status'])} is "
                             f"${r['value']:,}.", [r["source_record_id"]])]
        order = list(STATUS_NAMES)
        chosen.sort(key=lambda r: order.index(r["filing_status"]) if r["filing_status"] in order else 9)
        parts = [f"${r['value']:,} for {STATUS_NAMES.get(r['filing_status'], r['filing_status'])}"
                 for r in chosen]
        return [Sentence(f"For tax year {tax_year}, the {label} is " + ", ".join(parts[:-1])
                         + f", and {parts[-1]}.", sorted({r['source_record_id'] for r in chosen}))]
    return None


# --------------------------------------------------------------------------- extractive

def _where(p: Passage) -> str:
    title = p.title or "IRS document"
    if p.line_label:
        return f"{title}, line {p.line_label}"
    if p.heading_path:
        return f"{title} ({p.heading_path.split(' > ')[-1]})"
    return title


def extractive_answer(passages: list[Passage], k: int = 2, max_chars: int = 400,
                      close_ratio: float = 0.75) -> list[Sentence]:
    """Quote the best passage, plus runners-up only if they scored nearly as well
    (an exact line-lookup match stands alone unless another line was also named)."""
    if not passages:
        return []
    first = passages[0]
    chosen = [first] + [p for p in passages[1:k]
                        if (p.pinned and first.pinned)
                        or (not first.pinned and first.score > 0 and p.score >= close_ratio * first.score)]
    out = []
    for p in chosen:
        text = p.text if len(p.text) <= max_chars else p.text[:max_chars].rsplit(" ", 1)[0] + "…"
        out.append(Sentence(f"{_where(p)} says: “{text}”", [p.ref]))
    return out


# --------------------------------------------------------------------------- LLM prompts

SYSTEM_TEMPLATE = """You are TaxLens, an assistant that answers questions about U.S. federal \
tax forms using ONLY the IRS passages provided.

The question is about tax year {year}. Passages are labeled P1, P2, ... and show the tax \
year(s) each applies to. Never use a passage from a different tax year to answer, never \
use outside knowledge, and copy every dollar amount, percentage, and date exactly as it \
appears in a passage or in the figures below.

Figures for tax year {year} (authoritative):
{figures}

Write 1 to 4 short sentences. Output JSON Lines and nothing else: one JSON object per \
line, {{"text": "<sentence>", "cites": ["P1"]}}. Every sentence must cite the passage(s) \
that support it. If the passages do not answer the question, output exactly one line: \
{{"text": "", "cites": []}}"""

USER_TEMPLATE = """Passages:
{passages}

Question: {question}"""

REPAIR_TEMPLATE = """This sentence is not fully supported by the passages it cites:
{sentence}
Problem: {problem}

Rewrite it so that every word of it is supported by the passages, keeping numbers exactly \
as written there, or drop it. Reply with one JSON line: {{"text": "<sentence or empty>", \
"cites": ["P.."]}}"""


def format_passages(passages: list[Passage]) -> tuple[str, dict[str, object]]:
    lines, labels = [], {}
    for i, p in enumerate(passages[:MAX_PASSAGES], start=1):
        label = f"P{i}"
        labels[label] = p.ref
        yrs = f"{p.years[0]}" if p.years[0] == p.years[1] else f"{p.years[0]}-{p.years[1]}"
        lines.append(f"[{label}] ({_where(p)}; tax year {yrs}) {p.text}")
    return "\n".join(lines), labels


def format_figures(figures: list[dict]) -> str:
    if not figures:
        return "(none)"
    return "\n".join(f"- {f['parameter']} / {f['filing_status']}: ${f['value']:,}" for f in figures)


def parse_line(line: str, labels: dict[str, object]) -> Sentence | None:
    line = line.strip().strip(",")
    if not line.startswith("{"):
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    text = str(obj.get("text", "")).strip()
    if not text:
        return None
    cites = obj.get("cites") or obj.get("record_ids") or []
    refs = [labels[c] for c in cites if isinstance(c, str) and c in labels]
    return Sentence(text, refs)


def iter_lines(chunks: Iterable[str]) -> Iterator[str]:
    """Re-chunk a token stream into complete lines."""
    buf = ""
    for chunk in chunks:
        buf += chunk
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            yield line
    if buf.strip():
        yield buf


# --------------------------------------------------------------------------- generator

class AnswerGenerator:
    def __init__(self, llm: LLM | None = None, verifier: Verifier | None = None):
        self.llm = llm
        self.verifier = verifier or Verifier()

    def _allowed_numbers(self, question: str, tax_year: int, figures: list[dict]) -> set[str]:
        allowed = {str(tax_year)} | numbers_in(question)
        allowed |= {str(f["value"]) for f in figures}
        return allowed

    def stream(self, question: str, tax_year: int, passages: list[Passage],
               figures: list[dict] | None = None, stats: Answer | None = None,
               compare_years: list[int] | None = None) -> Iterator[Sentence]:
        """Yield verified sentences one at a time. Pass an Answer as ``stats``
        to have mode, dropped count, and claim counts filled in. With
        ``compare_years`` the answer contrasts the passages of those years."""
        figures = figures or []
        stats = stats or Answer([], mode="")
        # A citation covers the passage and its source label (form name and year).
        by_ref = {p.ref: f"{p.title or ''} {p.text}".strip() for p in passages}
        for f in figures:
            if f.get("source_record_id") is not None:
                by_ref.setdefault(f["source_record_id"], f.get("source_text", ""))
        allowed = self._allowed_numbers(question, tax_year, figures)
        if compare_years:
            allowed |= {str(y) for y in compare_years}

        fig = None if compare_years else figure_answer(question, tax_year, figures)
        if fig:
            stats.mode = "figure"
            for s in fig:
                self._count(stats, s, by_ref, allowed)
                stats.sentences.append(s)
                yield s
            return

        if self.llm is None:
            stats.mode = "extractive"
            yield from self._emit_extractive(passages, stats, by_ref, allowed, compare_years)
            return

        stats.mode = "llm"
        block, labels = format_passages(passages)
        system = SYSTEM_TEMPLATE.format(year=tax_year, figures=format_figures(figures))
        if compare_years:
            system += (f"\n\nThis question compares tax years {compare_years[0]} and "
                       f"{compare_years[-1]}. Describe each year using only that year's passages, "
                       "then state the difference. If the text is identical, say so.")
        user = USER_TEMPLATE.format(passages=block, question=question)
        produced = 0
        try:
            for line in iter_lines(self.llm.stream(system, user)):
                s = parse_line(line, labels)
                if s is None:
                    continue
                produced += 1
                fixed = self._verify_or_repair(s, labels, by_ref, allowed, system, block, stats)
                if fixed is not None:
                    stats.sentences.append(fixed)
                    yield fixed
        except LLMTimeout as e:
            # A slow model must not crash a request or a whole benchmark run:
            # record why and fall back to quoting the passages below.
            stats.notes.append(f"llm timeout: {e}")
        except LLMError as e:
            stats.notes.append(f"llm error: {e}")
        if produced == 0 and not stats.sentences:
            # The model said it can't answer, returned no usable JSON, or failed.
            # Fall back to quoting the passages; the abstainer decides if that's enough.
            stats.mode = "extractive_fallback"
            yield from self._emit_extractive(passages, stats, by_ref, allowed, compare_years)

    def generate(self, question: str, tax_year: int, passages: list[Passage],
                 figures: list[dict] | None = None, compare_years: list[int] | None = None) -> Answer:
        answer = Answer([], mode="")
        for _ in self.stream(question, tax_year, passages, figures, stats=answer,
                             compare_years=compare_years):
            pass
        return answer

    # ----------------------------------------------------------------- helpers

    def _emit_extractive(self, passages, stats, by_ref, allowed, compare_years=None) -> Iterator[Sentence]:
        if compare_years:
            # the best passage for each compared year, oldest first
            best = {}
            for p in passages:
                best.setdefault(p.years[0], p)
            chosen = [extractive_answer([best[y]], k=1)[0] for y in sorted(best)]
        else:
            chosen = extractive_answer(passages)
        for s in chosen:
            self._count(stats, s, by_ref, allowed)
            stats.sentences.append(s)
            yield s

    def _count(self, stats: Answer, s: Sentence, by_ref: dict, allowed: set[str]) -> bool:
        check = self.verifier.check_sentence(s.text, s.record_ids, by_ref, allowed)
        stats.n_claims += len(check.claims)
        stats.n_unsupported += sum(not c.supported for c in check.claims)
        s.supported = check.supported
        return check.supported

    def _verify_or_repair(self, s: Sentence, labels, by_ref, allowed, system, block,
                          stats: Answer) -> Sentence | None:
        reverse = {v: k for k, v in labels.items()}
        for attempt in range(1, MAX_ATTRIBUTION_RETRIES + 1):
            check = self.verifier.check_sentence(s.text, s.record_ids, by_ref, allowed)
            if check.supported:
                stats.n_claims += len(check.claims)
                s.attempts = attempt
                return s
            if attempt == MAX_ATTRIBUTION_RETRIES:
                break
            problems = []
            for c in check.claims:
                if c.missing_numbers:
                    problems.append(f"numbers not in the cited passages: {', '.join(c.missing_numbers)}")
                elif not c.supported:
                    problems.append(f"claim not supported: {c.claim}")
            if not s.record_ids:
                problems.append("no passage cited")
            cites = [reverse.get(r) for r in s.record_ids if r in reverse]
            prompt = REPAIR_TEMPLATE.format(
                sentence=json.dumps({"text": s.text, "cites": cites}),
                problem="; ".join(problems) or "unsupported")
            try:
                reply = self.llm.complete(system, f"Passages:\n{block}\n\n{prompt}", max_tokens=200)
            except LLMError as e:
                stats.notes.append(f"repair failed: {e}")
                break
            repaired = next((x for x in (parse_line(ln, labels) for ln in reply.splitlines()) if x), None)
            if repaired is None:
                break                      # the model chose to drop the sentence
            s = repaired
        # count the final failing version once
        check = self.verifier.check_sentence(s.text, s.record_ids, by_ref, allowed)
        stats.n_claims += len(check.claims)
        stats.n_unsupported += sum(not c.supported for c in check.claims)
        stats.dropped += 1
        return None
