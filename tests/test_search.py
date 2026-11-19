from taxlens.index.intervals import build_units
from taxlens.retrieve.bm25 import tokenize
from taxlens.retrieve.search import NaiveRetriever, YearAwareRetriever


def recs():
    rows = [
        (2021, "form_text", None, "Standard Deduction for Single or Married filing separately, $12,550"),
        (2022, "form_text", None, "Standard Deduction for Single or Married filing separately, $12,950"),
        (2021, "form_line", "30", "Recovery rebate credit. See instructions"),
        (2022, "form_line", "30", "Reserved for future use"),
        (2022, "form_line", "12", "Standard deduction or itemized deductions (from Schedule A)"),
    ]
    return [{"form_code": "f1040", "tax_year": y, "record_type": t, "line_label": l,
             "heading_path": None, "text": x, "ref": (y, i)} for i, (y, t, l, x) in enumerate(rows)]


def test_tokenize_keeps_dollar_amounts_whole():
    assert "12550" in tokenize("Single, $12,550")


def test_year_aware_never_returns_another_years_text():
    r = YearAwareRetriever(build_units(recs()))
    for hit in r.search("standard deduction single", year=2022, k=5):
        assert hit.years[0] <= 2022 <= hit.years[1]


def test_line_lookup_puts_the_named_line_first():
    r = YearAwareRetriever(build_units(recs()))
    assert r.search("what is line 30 used for", year=2022)[0].text == "Reserved for future use"
    assert r.search("what is line 30 used for", year=2021)[0].text.startswith("Recovery rebate")


def test_naive_retriever_can_mix_years():
    r = NaiveRetriever(recs())
    years = {h.years[0] for h in r.search("standard deduction single", k=5)}
    assert years == {2021, 2022}
