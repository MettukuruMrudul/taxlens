from datetime import date

from taxlens.retrieve.yeardetect import detect_tax_year


def test_explicit_year_wins():
    assert detect_tax_year("standard deduction for 2021?", explicit=2023).year == 2023


def test_tax_year_phrasings():
    for q in ["standard deduction for 2023", "on my 2022 return", "for tax year 2024",
              "the 2021 Form 1040", "what changed in 2022"]:
        assert detect_tax_year(q).source == "tax_year", q
    assert detect_tax_year("on my 2022 return").year == 2022


def test_filing_year_is_one_year_after_tax_year():
    g = detect_tax_year("I'm filing my return in 2024. What is my standard deduction?")
    assert (g.year, g.source) == (2023, "filing_year")
    assert detect_tax_year("Filing in 2025 as head of household").year == 2024


def test_relative_years_use_today():
    assert detect_tax_year("what was last year's limit?", today=date(2026, 3, 1)).year == 2025


def test_unknown_year():
    assert detect_tax_year("what is the standard deduction?").year is None
