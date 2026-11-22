"""The TaxLens question-answering pipeline.

    question
      -> year detection       which tax year? ("filing in 2024" means tax year 2023)
      -> intent               out of scope? a figure? a cross-year comparison?
      -> retrieval            year-filtered units, BM25 (+ dense, + rerank), line lookup
      -> generation           figures table / LLM with citations / extractive quotes
      -> verification         every claim checked against the passages it cites
      -> abstention           answer only when the evidence supports it

The pipeline is database-agnostic: it gets a ``Corpus`` (index units,
figures, a source lookup) so tests can run it on a handful of in-memory
records, and the CLI and API share exactly the same code path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Iterator

from taxlens.answer.generate import DISCLAIMER, Answer, AnswerGenerator, Passage, Sentence
from taxlens.index.intervals import Unit
from taxlens.intent.classify import IntentClassifier, IntentResult
from taxlens.retrieve.search import Hit, YearAwareRetriever
from taxlens.retrieve.yeardetect import YearGuess, detect_tax_year
from taxlens.tracing import span
from taxlens.verify.abstain import ABSTAIN_MESSAGE, AbstainFeatures, load_abstainer, squash

OUT_OF_SCOPE_MESSAGE = ("TaxLens answers questions about U.S. federal individual income tax "
                        "forms and IRS publications (Form 1040, its schedules and instructions, "
                        "and Publication 17). That question is outside what it covers.")
FORM_TITLES = {"f1040": "Form 1040", "f1040s1": "Schedule 1 (Form 1040)",
               "f1040s2": "Schedule 2 (Form 1040)", "f1040s3": "Schedule 3 (Form 1040)",
               "f1040sa": "Schedule A (Form 1040)", "f1040sb": "Schedule B (Form 1040)",
               "i1040gi": "Instructions for Form 1040", "i1040sca": "Instructions for Schedule A",
               "p17": "Publication 17"}
K_PASSAGES = 6
_YEAR = re.compile(r"\b(20[0-4]\d)\b")


@dataclass
class Source:
    record_id: object
    doc_id: str
    title: str
    tax_year: int
    page: int | None
    boxes: list = field(default_factory=list)
    form_code: str | None = None


SourceLookup = Callable[[list], dict]       # record_ids -> {record_id: Source}


@dataclass
class Corpus:
    units: list[Unit]
    figures: dict[int, list[dict]]           # tax_year -> year_figures rows
    lookup_sources: SourceLookup
    embeddings: object = None                # (n_units, dim) array or None
    encoder: object = None                   # taxlens.retrieve.dense.Encoder or None
    reranker: object = None
    retriever: YearAwareRetriever = field(init=False)
    years: list[int] = field(init=False)

    def __post_init__(self):
        self.retriever = YearAwareRetriever(self.units, embeddings=self.embeddings,
                                            reranker=self.reranker)
        self.years = sorted({y for u in self.units for y in range(u.valid_from, u.valid_to + 1)})


@dataclass
class Result:
    status: str                  # answered | abstained | needs_year | out_of_scope | year_unavailable
    question: str
    tax_year: int | None = None
    year_source: str | None = None
    intent: str | None = None
    sentences: list[dict] = field(default_factory=list)     # {"text", "citations": [n, ...]}
    citations: list[dict] = field(default_factory=list)     # numbered sources
    message: str | None = None
    mode: str | None = None
    compare_years: list[int] | None = None
    unsupported_claim_rate: float = 0.0
    p_correct: float | None = None
    abstain_reason: str | None = None
    features: dict | None = None
    attempt: str = ""           # text generated before the abstention decision (for training)
    notes: list = field(default_factory=list)  # e.g. LLM errors behind an extractive fallback
    timings: dict = field(default_factory=dict)
    disclaimer: str = DISCLAIMER

    @property
    def abstained(self) -> bool:
        return self.status != "answered"

    def answer_text(self) -> str:
        return " ".join(s["text"] for s in self.sentences)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "status", "question", "tax_year", "year_source", "intent", "sentences", "citations",
            "message", "mode", "compare_years", "unsupported_claim_rate", "p_correct",
            "abstain_reason", "timings", "disclaimer")}


class Pipeline:
    def __init__(self, corpus: Corpus, generator: AnswerGenerator | None = None,
                 intent: IntentClassifier | None = None, abstainer=None):
        self.corpus = corpus
        self.generator = generator or AnswerGenerator()
        self.intent = intent or IntentClassifier.load()
        self.abstainer = abstainer or load_abstainer()

    # ------------------------------------------------------------------ public

    def run(self, question: str, tax_year: int | None = None, today: date | None = None) -> Result:
        result = None
        for event in self.stream(question, tax_year, today):
            if event["type"] == "done":
                result = event["result"]
        return result

    def stream(self, question: str, tax_year: int | None = None,
               today: date | None = None) -> Iterator[dict]:
        """Events: meta, sentence (one per verified sentence), abstain, done."""
        timings: dict = {}
        res = Result("answered", question)
        with span("taxlens.request", timings, explicit_year=tax_year) as root:
            with span("taxlens.year_detect", timings) as sp:
                guess = detect_tax_year(question, explicit=tax_year, today=today)
                sp.set("tax_year", guess.year)
                sp.set("source", guess.source)
            with span("taxlens.intent_classify", timings) as sp:
                intent = self.intent.classify(question)
                sp.set("intent_type", intent.intent)
            if guess.year is None and intent.intent == "cross_year_change":
                mentioned = sorted({int(y) for y in _YEAR.findall(question)})
                if len(mentioned) >= 2:      # "from 2021 to 2022": answer for the later year
                    guess = YearGuess(mentioned[-1], "tax_year", 0.9)
            res.tax_year, res.year_source, res.intent = guess.year, guess.source, intent.intent
            root.set("tax_year", guess.year)
            root.set("intent_type", intent.intent)

            if self._early_exit(res, guess.year, intent):
                res.timings = timings
                root.set("abstained", True)
                yield {"type": "meta", **self._meta(res)}
                yield {"type": "done", "result": res}
                return

            year = guess.year
            compare = self._compare_years(question, year, intent)
            res.compare_years = compare
            yield {"type": "meta", **self._meta(res)}

            with span("taxlens.retrieval", timings, retrieval_k=K_PASSAGES) as sp:
                hits = self._retrieve(question, compare or [year])
                sp.set("n_hits", len(hits))
            sources = self.corpus.lookup_sources([h.ref for h in hits]
                                                 + [f["source_record_id"] for f in self._figures(year)])
            passages = [self._passage(h, sources) for h in hits]

            with span("taxlens.answer_generate", timings) as sp:
                answer = Answer([], mode="")
                numbering: dict = {}
                streamed = []
                for s in self.generator.stream(question, year, passages, self._figures(year),
                                               stats=answer, compare_years=compare):
                    item = self._sentence_dict(s, numbering)
                    streamed.append(item)
                    yield {"type": "sentence", "sentence": item,
                           "citations": [self._citation(n, r, sources, hits) for r, n in numbering.items()
                                         if n in item["citations"]]}
                sp.set("mode", answer.mode)
                sp.set("answer_sentence_count", len(answer.sentences))

            with span("taxlens.verify", timings) as sp:
                features = self._features(guess, intent, hits, answer, year)
                decision = self.abstainer.decide(features)
                if not answer.sentences:
                    decision.abstain, decision.reason = True, "no sentence survived verification"
                sp.set("abstained", decision.abstain)
                sp.set("unsupported_claim_rate", round(answer.unsupported_claim_rate, 4))

            res.mode = answer.mode
            res.notes = list(answer.notes)
            res.unsupported_claim_rate = answer.unsupported_claim_rate
            res.p_correct = round(decision.p_correct, 3)
            res.features = dict(features.__dict__)
            res.abstain_reason = decision.reason
            res.attempt = " ".join(item["text"] for item in streamed)
            if decision.abstain:
                res.status, res.message = "abstained", ABSTAIN_MESSAGE
                yield {"type": "abstain", "message": ABSTAIN_MESSAGE, "reason": decision.reason}
            else:
                res.sentences = streamed
                res.citations = [self._citation(n, r, sources, hits) for r, n in numbering.items()]
            root.set("abstained", decision.abstain)
            root.set("cache_hit", False)
        res.timings = timings
        yield {"type": "done", "result": res}

    # ------------------------------------------------------------------ steps

    def _early_exit(self, res: Result, year: int | None, intent: IntentResult) -> bool:
        if intent.intent == "out_of_scope":
            res.status, res.message = "out_of_scope", OUT_OF_SCOPE_MESSAGE
            return True
        if year is None:
            res.status = "needs_year"
            latest = self.corpus.years[-1] if self.corpus.years else 2024
            res.message = (f"Which tax year is this about? TaxLens covers tax years "
                           f"{self._year_range()}. Pick one with the year selector or add it to "
                           f"the question (for example, \"for tax year {latest}\").")
            return True
        if year not in self.corpus.years:
            res.status = "year_unavailable"
            hint = ""
            if res.year_source == "relative":
                hint = " Questions like \"this year\" refer to the current calendar year."
            res.message = (f"Tax year {year} isn't in TaxLens yet; it covers {self._year_range()}."
                           + hint)
            return True
        return False

    def _year_range(self) -> str:
        ys = self.corpus.years
        return f"{ys[0]}–{ys[-1]}" if ys else "no years yet"

    def _compare_years(self, question: str, year: int, intent: IntentResult) -> list[int] | None:
        if intent.intent != "cross_year_change":
            return None
        mentioned = sorted({int(y) for y in _YEAR.findall(question)} & set(self.corpus.years))
        if len(mentioned) >= 2:
            return [mentioned[0], mentioned[-1]]
        other = year - 1 if (year - 1) in self.corpus.years else year + 1
        return sorted({year, other} & set(self.corpus.years)) or None

    def _retrieve(self, question: str, years: list[int]) -> list[Hit]:
        qvec = self.corpus.encoder.encode_query(question) if self.corpus.encoder else None
        k = max(2, K_PASSAGES // len(years)) if len(years) > 1 else K_PASSAGES
        hits: list[Hit] = []
        for y in years:
            for h in self.corpus.retriever.search(question, year=y, k=k, query_vector=qvec):
                if len(years) > 1:
                    h.years = (y, y)       # in a comparison, each passage speaks for one year
                hits.append(h)
        return hits

    def _figures(self, year: int) -> list[dict]:
        return self.corpus.figures.get(year, [])

    def _passage(self, h: Hit, sources: dict) -> Passage:
        src = sources.get(h.ref)
        if src:
            title = f"{src.title} ({src.tax_year})"
        else:
            title = f"{FORM_TITLES.get(h.form_code or '', h.form_code or 'IRS document')} ({h.years[0]})"
        return Passage(h.ref, h.text, h.years, h.line_label, h.heading_path, title, h.bm25, h.pinned)

    @staticmethod
    def _sentence_dict(s: Sentence, numbering: dict) -> dict:
        nums = []
        for r in s.record_ids:
            if r not in numbering:
                numbering[r] = len(numbering) + 1
            nums.append(numbering[r])
        return {"text": s.text, "citations": nums}

    def _citation(self, n: int, ref, sources: dict, hits: list[Hit]) -> dict:
        src: Source | None = sources.get(ref)
        hit = next((h for h in hits if h.ref == ref), None)
        fallback_title = FORM_TITLES.get(hit.form_code or "", "IRS document") if hit else None
        return {"n": n, "record_id": ref,
                "doc_id": src.doc_id if src else None,
                "title": src.title if src else fallback_title,
                "tax_year": src.tax_year if src else (hit.years[0] if hit else None),
                "page": src.page if src else None,
                "boxes": src.boxes if src else [],
                "line_label": hit.line_label if hit else None,
                "heading_path": hit.heading_path if hit else None,
                "snippet": (hit.text[:280] if hit else None)}

    def _features(self, guess: YearGuess, intent: IntentResult, hits: list[Hit], answer: Answer,
                  year: int) -> AbstainFeatures:
        bm = sorted((h.bm25 for h in hits), reverse=True)[:5]
        produced = len(answer.sentences) + answer.dropped
        return AbstainFeatures(
            max_retrieval_score=squash(bm[0]) if bm else 0.0,
            mean_retrieval_score=squash(sum(bm) / len(bm)) if bm else 0.0,
            year_confidence=guess.confidence,
            intent_confidence=intent.confidence,
            supported_fraction=(sum(s.supported for s in answer.sentences) / produced) if produced else 0.0,
            unsupported_claim_rate=answer.unsupported_claim_rate,
            n_passages_for_year=float(self.corpus.retriever.count_for_year(year)),
            has_figure=1.0 if answer.mode == "figure" else 0.0,
            intent=intent.intent)

    def _meta(self, res: Result) -> dict:
        return {"status": res.status, "tax_year": res.tax_year, "year_source": res.year_source,
                "intent": res.intent, "message": res.message, "compare_years": res.compare_years,
                "available_years": self.corpus.years}


# ---------------------------------------------------------------------- database wiring

def db_source_lookup(conn_factory) -> SourceLookup:
    """Build a SourceLookup that reads line_records + documents."""
    def lookup(record_ids: list) -> dict:
        ids = sorted({int(r) for r in record_ids if r is not None})
        if not ids:
            return {}
        with conn_factory() as conn:
            rows = conn.execute(
                "SELECT r.record_id, d.doc_id, d.title, d.tax_year, r.page, r.boxes, d.form_code "
                "FROM line_records r JOIN documents d USING (doc_id) WHERE r.record_id = ANY(%s)",
                (ids,)).fetchall()
        return {rid: Source(rid, doc, title, ty, page, boxes or [], fc)
                for rid, doc, title, ty, page, boxes, fc in rows}
    return lookup


def load_corpus(conn_factory, with_models: bool = True) -> Corpus:
    """Load everything the pipeline needs from Postgres (once, at start-up)."""
    from taxlens.index.build import load_units
    with conn_factory() as conn:
        units = load_units(conn)
        figures: dict[int, list[dict]] = {}
        for p, fs, y, v, src, txt in conn.execute(
                "SELECT parameter, filing_status, tax_year, value, source_record_id, source_text "
                "FROM year_figures"):
            figures.setdefault(y, []).append({"parameter": p, "filing_status": fs, "tax_year": y,
                                              "value": v, "source_record_id": src, "source_text": txt})
        embeddings = encoder = reranker = None
        if with_models:
            from taxlens.retrieve.dense import get_encoder, load_embeddings, load_reranker
            encoder = get_encoder()
            if encoder is not None:
                embeddings = load_embeddings(conn)
                if embeddings is None:
                    print("EMBED_MODEL is set but index_units has no embeddings; "
                          "run python -m taxlens.index.embed. Dense retrieval off.")
                    encoder = None
            reranker = load_reranker()
    return Corpus(units, figures, db_source_lookup(conn_factory), embeddings, encoder, reranker)


def build_pipeline(conn_factory, with_models: bool = True) -> Pipeline:
    from taxlens.llm import get_llm
    from taxlens.verify.verifier import Verifier
    corpus = load_corpus(conn_factory, with_models)
    verifier = Verifier.from_config() if with_models else Verifier()
    return Pipeline(corpus, AnswerGenerator(get_llm(), verifier))
