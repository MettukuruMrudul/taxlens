"""Year-specific figures (Week 7, component C4).

Dollar amounts such as the standard deduction change every year and must
never be generated from memory. They are extracted into a table keyed by
(parameter, filing status, tax year) and every value keeps a pointer to the
exact IRS text it came from.

v1 extracts the standard deduction amounts printed in the left margin of
Form 1040, next to line 12.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

FILING_STATUS_PATTERNS = {
    "single_or_mfs": r"Single or Married filing separately,\s*\$([\d,]+)",
    "mfj_or_qss": r"Married filing jointly or Qualifying (?:widow\(er\)|surviving spouse),\s*\$([\d,]+)",
    "head_of_household": r"Head of household,\s*\$([\d,]+)",
}


@dataclass
class Figure:
    parameter: str
    filing_status: str
    tax_year: int
    value: int
    source_ref: object
    source_text: str


def lookup_figure(conn, parameter: str, filing_status: str, tax_year: int) -> int | None:
    """Look up a year-specific figure from the database.

    Returns the integer value (e.g. 13850 for the 2023 single standard deduction)
    or None if no row matches the (parameter, filing_status, tax_year) triple.
    """
    row = conn.execute(
        "SELECT value FROM year_figures WHERE parameter = %s AND filing_status = %s AND tax_year = %s",
        (parameter, filing_status, tax_year)
    ).fetchone()
    return row[0] if row else None


def extract_standard_deduction(form_records: list[dict]) -> list[Figure]:
    """form_records: Form 1040 records with tax_year, record_type, text, ref."""
    figures = []
    for r in form_records:
        if r["record_type"] != "form_text" or "Standard Deduction for" not in r["text"]:
            continue
        for status, pat in FILING_STATUS_PATTERNS.items():
            m = re.search(pat, r["text"])
            if m:
                figures.append(Figure("standard_deduction", status, r["tax_year"],
                                      int(m.group(1).replace(",", "")), r["ref"], m.group(0)))
    return figures
