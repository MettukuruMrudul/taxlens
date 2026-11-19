import pytest

from taxlens.ingest.corpus import (
    IRS_PRIOR_BASE,
    MVP_DOC_COUNT,
    MVP_DOCS,
    MVP_YEARS,
    count,
    full_corpus_doc_ids,
    mvp_corpus,
    validate_url,
)


def test_corpus_covers_every_doc_and_year():
    assert len(mvp_corpus()) == len(MVP_DOCS) * len(MVP_YEARS)


def test_doc_ids_are_unique():
    ids = [doc.doc_id for doc in mvp_corpus()]
    assert len(ids) == len(set(ids))


def test_urls_point_at_the_prior_year_archive():
    doc = next(d for d in mvp_corpus() if d.doc_id == "f1040-2021")
    assert doc.url == "https://www.irs.gov/pub/irs-prior/f1040--2021.pdf"


def test_every_url_contains_its_own_tax_year():
    for doc in mvp_corpus():
        assert f"--{doc.tax_year}.pdf" in doc.url


# --- new additions ---

def test_mvp_doc_count_constant_is_36():
    assert MVP_DOC_COUNT == 36


def test_count_returns_36():
    assert count() == 36


def test_count_matches_actual_corpus_length():
    assert count() == len(mvp_corpus())


def test_validate_url_accepts_prior_base_url():
    url = f"{IRS_PRIOR_BASE}/f1040--2023.pdf"
    assert validate_url(url) is True


def test_validate_url_rejects_url_not_starting_with_prior_base():
    with pytest.raises(ValueError, match="IRS_PRIOR_BASE"):
        validate_url("https://www.irs.gov/pub/irs-pdf/f1040.pdf")


def test_validate_url_rejects_url_containing_irs_current():
    with pytest.raises(ValueError, match="irs-current"):
        validate_url(f"https://www.irs.gov/pub/irs-current/f1040.pdf")


def test_every_corpus_url_passes_validate_url():
    for doc in mvp_corpus():
        assert validate_url(doc.url) is True


def test_full_corpus_doc_ids_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        full_corpus_doc_ids()
