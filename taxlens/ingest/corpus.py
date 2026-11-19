"""The MVP corpus: the Form 1040 family plus Publication 17, tax years 2021-2024.

Every file comes from the IRS prior-year archive, where each PDF's name
includes its tax year (for example f1040--2021.pdf).

There is deliberately NO fallback to the IRS "current forms" folder. That
folder always holds the newest year, so falling back to it could silently
save the wrong year's PDF under an older year's name. That is exactly the
kind of wrong-year error TaxLens exists to prevent.
"""
from dataclasses import dataclass

IRS_PRIOR_BASE = "https://www.irs.gov/pub/irs-prior"

MVP_YEARS = (2021, 2022, 2023, 2024)

# (form_code, doc_type, title)
MVP_DOCS = (
    ("f1040", "form", "Form 1040"),
    ("i1040gi", "instructions", "Instructions for Form 1040"),
    ("f1040s1", "form", "Schedule 1 (Form 1040)"),
    ("f1040s2", "form", "Schedule 2 (Form 1040)"),
    ("f1040s3", "form", "Schedule 3 (Form 1040)"),
    ("f1040sa", "form", "Schedule A (Form 1040)"),
    ("i1040sca", "instructions", "Instructions for Schedule A (Form 1040)"),
    ("f1040sb", "form", "Schedule B (Form 1040)"),
    ("p17", "publication", "Publication 17, Your Federal Income Tax"),
)


@dataclass(frozen=True)
class CorpusDoc:
    form_code: str
    doc_type: str
    title: str
    tax_year: int

    @property
    def doc_id(self) -> str:
        return f"{self.form_code}-{self.tax_year}"

    @property
    def filename(self) -> str:
        return f"{self.form_code}--{self.tax_year}.pdf"

    @property
    def url(self) -> str:
        return f"{IRS_PRIOR_BASE}/{self.filename}"


MVP_DOC_COUNT = 36  # len(MVP_DOCS) * len(MVP_YEARS) == 9 * 4


def mvp_corpus() -> list[CorpusDoc]:
    """Every (document, tax year) pair in the MVP corpus."""
    return [
        CorpusDoc(code, doc_type, title, year)
        for year in MVP_YEARS
        for code, doc_type, title in MVP_DOCS
    ]


def count() -> int:
    """Return the expected number of documents in the MVP corpus (36)."""
    return MVP_DOC_COUNT


def validate_url(url: str) -> bool:
    """Return True iff *url* comes from the IRS prior-year archive.

    Raises ``ValueError`` if the URL does not start with ``IRS_PRIOR_BASE``
    or if it contains the string ``"irs-current"`` (which would mean we
    accidentally fell back to the current-forms folder).
    """
    if not url.startswith(IRS_PRIOR_BASE):
        raise ValueError(
            f"URL does not start with IRS_PRIOR_BASE ({IRS_PRIOR_BASE!r}): {url!r}"
        )
    if "irs-current" in url:
        raise ValueError(
            f"URL must not reference the current-forms folder; got: {url!r}"
        )
    return True


def full_corpus_doc_ids() -> list[str]:
    """Return all doc_id strings for the full corpus.

    .. note::
        Not yet implemented — reserved for Phase 2 when the corpus expands
        beyond the MVP set.
    """
    raise NotImplementedError("full_corpus_doc_ids is a Phase 2 placeholder")
