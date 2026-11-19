"""Validity-interval index (Week 6).

IRS documents repeat most of their text unchanged from one tax year to the
next. Storing every year's copy makes the index bigger and, worse, gives the
retriever several near-identical candidates from different years to confuse.

Instead, each distinct piece of text is stored once as an *index unit* with
the range of tax years it applies to:

    unit: "Add lines 1a through 1h"   valid 2022-2024
    unit: "Wages, salaries, tips..."  valid 2021-2021

A unit is only shared by consecutive years whose text is identical. As soon
as the text changes (a new dollar amount, a renumbered line, new wording),
the interval closes and a new unit begins. Retrieval then filters to units
whose interval contains the year being asked about, so text from the wrong
year can never be returned.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
                         "\u2013": "-", "\u2014": "-", "\u00a0": " "})


def normalize(text: str) -> str:
    """Canonical form used to decide whether two years' text is identical.
    Only cosmetic differences are removed: numbers and words are kept."""
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    return re.sub(r"\s+", " ", text).strip()


_PAGE_FURNITURE = re.compile(
    r"^(form|schedule|publication|pub\.?|instructions|\d{4})\b[^.]{0,60}\bpage \d+\s*$|\bcat\. ?no\.", re.I)


def is_boilerplate(text: str) -> bool:
    """Page headers and footers ("Form 1040 (2021) Page 2", "Cat. No. 11320B Form 1040 (2021)").
    They carry no content and would otherwise match any question naming the form."""
    t = normalize(text)
    return len(t) <= 80 and bool(_PAGE_FURNITURE.search(t))


@dataclass
class Unit:
    form_code: str
    record_type: str
    line_label: str | None
    heading_path: str | None
    text: str
    valid_from: int
    valid_to: int
    # tax_year -> record id (or any reference) of that year's copy, for citations
    sources: dict[int, object] = field(default_factory=dict)

    def applies_to(self, year: int) -> bool:
        return self.valid_from <= year <= self.valid_to


def build_units(records: list[dict]) -> list[Unit]:
    """Collapse per-year records into validity-interval units.

    Each record dict needs: form_code, tax_year, record_type, line_label,
    heading_path, text, and ref (what to cite for that year).
    """
    groups: dict[tuple, dict[int, dict]] = {}
    for r in records:
        key = (r["form_code"], r["record_type"], r.get("line_label"), normalize(r["text"]))
        # if the same text appears twice in one year, keep the first occurrence
        groups.setdefault(key, {}).setdefault(r["tax_year"], r)

    units: list[Unit] = []
    for (form_code, rtype, label, _), by_year in groups.items():
        years = sorted(by_year)
        run = [years[0]]
        for y in years[1:] + [None]:
            if y is not None and y == run[-1] + 1:
                run.append(y)
                continue
            first = by_year[run[0]]
            units.append(Unit(
                form_code=form_code, record_type=rtype, line_label=label,
                heading_path=by_year[run[-1]].get("heading_path"),
                text=first["text"], valid_from=run[0], valid_to=run[-1],
                sources={yr: by_year[yr]["ref"] for yr in run},
            ))
            if y is not None:
                run = [y]
    units.sort(key=lambda u: (u.form_code, u.valid_from, u.line_label or "", u.text))
    return units


def size_reduction(n_records: int, n_units: int) -> float:
    """Share of index entries saved compared with one copy per year."""
    return 1 - n_units / n_records if n_records else 0.0
