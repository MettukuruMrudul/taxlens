"""Tax-year detection (Week 6, rules v1).

Works out which tax year a question is about. An explicit year chosen in
the UI always wins. Otherwise the question text is read, paying special
attention to the classic trap: a "2024 return" covers tax year 2024, but the
return you *file in* 2025 also covers tax year 2024.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

YEAR = r"(20[0-4]\d)"

# Phrases where the year is the filing year, so the tax year is one earlier.
FILING_YEAR_PATTERNS = [
    rf"\b(?:filing|file|filed|submit(?:ting)?)\b[^.?!]{{0,40}}\bin {YEAR}\b",
    rf"\b(?:this|the) (?:filing )?season\b[^.?!]{{0,20}}\b{YEAR}\b",
    rf"\b{YEAR} filing season\b",
]
# Phrases where the year is explicitly the tax year.
TAX_YEAR_PATTERNS = [
    rf"\btax year {YEAR}\b",
    rf"\b{YEAR} tax year\b",
    rf"\b(?:for|in|during|of) {YEAR}\b",
    rf"\b{YEAR} (?:return|taxes|tax return|form 1040|1040|instructions|w-2|income)\b",
    rf"\bform 1040 (?:for )?{YEAR}\b",
]


@dataclass
class YearGuess:
    year: int | None
    source: str          # 'explicit', 'filing_year', 'tax_year', 'relative', 'default', 'none'
    confidence: float = 1.0


def detect_tax_year(question: str, explicit: int | None = None, today: date | None = None,
                    default: int | None = None) -> YearGuess:
    if explicit is not None:
        return YearGuess(explicit, "explicit", confidence=1.0)
    q = question.lower()
    for pat in FILING_YEAR_PATTERNS:
        m = re.search(pat, q)
        if m:
            return YearGuess(int(m.group(1)) - 1, "filing_year", confidence=0.9)
    for pat in TAX_YEAR_PATTERNS:
        m = re.search(pat, q)
        if m:
            return YearGuess(int(m.group(1)), "tax_year", confidence=0.9)
    years = re.findall(rf"\b{YEAR}\b", q)
    if len(set(years)) == 1:
        return YearGuess(int(years[0]), "tax_year", confidence=0.9)
    today = today or date.today()
    if re.search(r"\blast year'?s?\b|\bprevious year\b", q):
        return YearGuess(today.year - 1, "relative", confidence=0.75)
    if re.search(r"\bthis year'?s?\b|\bcurrent year\b", q):
        return YearGuess(today.year, "relative", confidence=0.75)
    if default is not None:
        return YearGuess(default, "default", confidence=0.5)
    return YearGuess(None, "none", confidence=0.0)
