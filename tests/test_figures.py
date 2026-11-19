from taxlens.index.figures import extract_standard_deduction

MARGIN_2024 = ("Standard Deduction for\u2014 \u2022 Single or Married filing separately, $14,600 "
               "\u2022 Married filing jointly or Qualifying surviving spouse, $29,200 "
               "\u2022 Head of household, $21,900 \u2022 If you checked any box under Standard Deduction, see instructions.")
MARGIN_2021 = ("Standard Deduction for\u2014 \u2022 Single or Married filing separately, $12,550 "
               "\u2022 Married filing jointly or Qualifying widow(er), $25,100 \u2022 Head of household, $18,800")


def test_extracts_all_three_amounts_with_sources():
    figs = extract_standard_deduction([
        {"tax_year": 2024, "record_type": "form_text", "text": MARGIN_2024, "ref": 7},
        {"tax_year": 2021, "record_type": "form_text", "text": MARGIN_2021, "ref": 3},
    ])
    got = {(f.tax_year, f.filing_status): f.value for f in figs}
    assert got[(2024, "single_or_mfs")] == 14600
    assert got[(2024, "mfj_or_qss")] == 29200
    assert got[(2024, "head_of_household")] == 21900
    assert got[(2021, "mfj_or_qss")] == 25100      # 2021 wording: "Qualifying widow(er)"
    assert all(f.source_ref in (3, 7) for f in figs)
