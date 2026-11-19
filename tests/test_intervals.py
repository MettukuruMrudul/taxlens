from taxlens.index.intervals import build_units, normalize, size_reduction


def rec(year, text, label=None, rtype="form_line", ref=None):
    return {"form_code": "f1040", "tax_year": year, "record_type": rtype, "line_label": label,
            "heading_path": None, "text": text, "ref": ref or (year, text)}


def test_identical_text_in_consecutive_years_becomes_one_unit():
    units = build_units([rec(y, "Add lines 1a through 1h", "1z") for y in (2022, 2023, 2024)])
    assert len(units) == 1
    assert (units[0].valid_from, units[0].valid_to) == (2022, 2024)
    assert set(units[0].sources) == {2022, 2023, 2024}


def test_changed_amount_closes_the_interval():
    recs = [rec(2021, "Single, $12,550"), rec(2022, "Single, $12,950"), rec(2023, "Single, $12,950")]
    spans = sorted((u.valid_from, u.valid_to) for u in build_units(recs))
    assert spans == [(2021, 2021), (2022, 2023)]


def test_text_that_returns_after_a_gap_gets_two_units():
    recs = [rec(2021, "Reserved"), rec(2022, "Something else"), rec(2023, "Reserved")]
    spans = sorted((u.valid_from, u.valid_to) for u in build_units(recs) if u.text == "Reserved")
    assert spans == [(2021, 2021), (2023, 2023)]


def test_normalize_ignores_cosmetic_differences_only():
    assert normalize("don\u2019t  file") == normalize("don't file")
    assert normalize("$12,550") != normalize("$12,950")


def test_size_reduction():
    assert size_reduction(100, 36) == 0.64
