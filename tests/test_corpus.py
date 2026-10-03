from taxlens.ingest.corpus import MVP_DOCS, MVP_YEARS, mvp_corpus


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
