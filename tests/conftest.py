"""Shared fixtures: a tiny in-memory corpus so pipeline tests need no database."""
import pytest

from taxlens.index.intervals import build_units
from taxlens.pipeline import Corpus, Source

ROWS = [
    # (tax_year, form_code, record_type, line_label, text)
    (2021, "f1040", "form_text", None, "Standard Deduction for— Single or Married filing separately, $12,550 "
                                       "Married filing jointly or Qualifying widow(er), $25,100 Head of household, $18,800"),
    (2022, "f1040", "form_text", None, "Standard Deduction for— Single or Married filing separately, $12,950 "
                                       "Married filing jointly or Qualifying surviving spouse, $25,900 Head of household, $19,400"),
    (2021, "f1040", "form_line", "30", "Recovery rebate credit. See instructions"),
    (2022, "f1040", "form_line", "30", "Reserved for future use"),
    (2023, "f1040", "form_line", "30", "Reserved for future use"),
    (2021, "f1040", "form_line", "1", "Wages, salaries, tips, etc. Attach Form(s) W-2"),
    (2022, "f1040", "form_line", "1a", "Total amount from Form(s) W-2, box 1 (see instructions)"),
    (2023, "f1040", "form_line", "1a", "Total amount from Form(s) W-2, box 1 (see instructions)"),
    (2023, "f1040", "form_text", None, "Standard Deduction for— Single or Married filing separately, $13,850 "
                                       "Married filing jointly or Qualifying surviving spouse, $27,700 Head of household, $20,800"),
]
FIGURES = {
    2021: [("single_or_mfs", 12550, 1), ("mfj_or_qss", 25100, 1), ("head_of_household", 18800, 1)],
    2022: [("single_or_mfs", 12950, 2), ("mfj_or_qss", 25900, 2), ("head_of_household", 19400, 2)],
    2023: [("single_or_mfs", 13850, 9), ("mfj_or_qss", 27700, 9), ("head_of_household", 20800, 9)],
}


def make_corpus() -> Corpus:
    records = [{"form_code": fc, "tax_year": y, "record_type": rt, "line_label": ll,
                "heading_path": None, "text": tx, "ref": i + 1}
               for i, (y, fc, rt, ll, tx) in enumerate(ROWS)]
    sources = {i + 1: Source(i + 1, f"{fc}-{y}", "Form 1040", y, 1,
                             [{"page": 1, "x0": 50.0, "top": 100.0, "x1": 300.0, "bottom": 112.0}], fc)
               for i, (y, fc, *_rest) in enumerate(ROWS)}
    figures = {y: [{"parameter": "standard_deduction", "filing_status": s, "tax_year": y, "value": v,
                    "source_record_id": ref, "source_text": ROWS[ref - 1][4]} for s, v, ref in rows]
               for y, rows in FIGURES.items()}
    return Corpus(build_units(records), figures,
                  lambda ids: {r: sources[r] for r in ids if r in sources})


@pytest.fixture
def corpus() -> Corpus:
    return make_corpus()
