from datetime import date

from taxlens.pipeline import Pipeline

TODAY = date(2026, 11, 14)


def run(corpus, question, year=None):
    return Pipeline(corpus).run(question, year, today=TODAY)


def test_standard_deduction_comes_from_the_figures_table(corpus):
    r = run(corpus, "What is the standard deduction for a single filer for 2022?")
    assert r.status == "answered" and r.mode == "figure"
    assert "$12,950" in r.answer_text()
    assert r.citations[0]["tax_year"] == 2022 and r.citations[0]["boxes"]


def test_filing_year_is_converted_to_tax_year(corpus):
    r = run(corpus, "I'm filing my return in 2024. What standard deduction do I get as a single filer?")
    assert r.tax_year == 2023 and "$13,850" in r.answer_text()


def test_line_question_never_cites_another_year(corpus):
    r = run(corpus, "What is line 30 of Form 1040 used for in 2022?")
    assert r.status == "answered"
    assert "Reserved for future use" in r.answer_text()
    assert all(c["tax_year"] == 2022 for c in r.citations)


def test_out_of_scope_skips_retrieval(corpus):
    r = run(corpus, "What's the best pizza place in Boston?")
    assert r.status == "out_of_scope" and r.sentences == [] and "timings" in r.to_dict()
    assert "retrieval" not in r.timings


def test_missing_year_asks_for_one(corpus):
    r = run(corpus, "Where do I report my W-2 wages?")
    assert r.status == "needs_year" and "2021" in r.message


def test_explicit_year_overrides_question(corpus):
    r = run(corpus, "Where do I report my W-2 wages?", year=2023)
    assert r.status == "answered" and r.tax_year == 2023 and r.year_source == "explicit"


def test_year_outside_corpus(corpus):
    r = run(corpus, "What is line 30 for this year?")
    assert r.status == "year_unavailable" and "2026" in r.message


def test_cross_year_question_shows_both_years(corpus):
    r = run(corpus, "How did line 30 change from 2021 to 2022?")
    assert r.status == "answered" and r.compare_years == [2021, 2022]
    years = {c["tax_year"] for c in r.citations}
    assert years == {2021, 2022}


def test_stream_emits_meta_sentences_then_done(corpus):
    events = list(Pipeline(corpus).stream("What is line 30 of Form 1040 used for in 2023?", today=TODAY))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "meta" and kinds[-1] == "done" and "sentence" in kinds
    assert events[0]["available_years"] == [2021, 2022, 2023]


def test_every_answer_carries_the_disclaimer(corpus):
    for q in ["What is the standard deduction for 2021?", "best pizza", "line 30 in 2023?"]:
        assert "not tax advice" in run(corpus, q).to_dict()["disclaimer"]
