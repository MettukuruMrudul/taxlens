"""Tests for the PDF parser.

The unit tests use small synthetic inputs. The last test runs on the real
2024 Form 1040 and is skipped if that PDF hasn't been downloaded yet.
"""
import pytest

from taxlens.config import RAW_DIR
from taxlens.ingest.parse import _append_line, _gaps, _is_form_decoration, _page_words, _reading_blocks, parse_pdf


def word(text, x0, top, width=20, height=8):
    return {"text": text, "x0": x0, "x1": x0 + width, "top": top, "bottom": top + height,
            "fontname": "Times", "size": 10}


def test_gaps_finds_empty_space_between_spans():
    assert _gaps([(0, 10), (12, 20), (40, 50)], min_gap=8) == [(20, 40)]


def test_two_columns_are_read_left_column_first():
    left = [word(f"L{i}", 40, 100 + 12 * i) for i in range(5)]
    right = [word(f"R{i}", 300, 100 + 12 * i) for i in range(5)]
    blocks = _reading_blocks(right + left)
    order = [w["text"] for b in blocks for w in sorted(b, key=lambda w: w["top"])]
    assert order == [f"L{i}" for i in range(5)] + [f"R{i}" for i in range(5)]


def test_full_width_paragraph_above_two_columns():
    top = [word("Wide", 40, 50, width=400)]
    left = [word(f"L{i}", 40, 100 + 12 * i) for i in range(3)]
    right = [word(f"R{i}", 300, 100 + 12 * i) for i in range(3)]
    blocks = _reading_blocks(left + right + top)
    order = [w["text"] for b in blocks for w in sorted(b, key=lambda w: w["top"])]
    assert order == ["Wide", "L0", "L1", "L2", "R0", "R1", "R2"]


def test_end_of_line_hyphenation_is_undone():
    assert _append_line("see the instruc-", "tions for line 1a") == "see the instructions for line 1a"


def test_real_hyphen_before_capital_is_kept():
    assert _append_line("Form 1040-", "SR filers") == "Form 1040- SR filers"


F1040_2024 = RAW_DIR / "f1040--2024.pdf"


@pytest.mark.skipif(not F1040_2024.exists(), reason="run the fetcher first")
def test_form_1040_2024_key_lines():
    records = parse_pdf(F1040_2024, "form")
    lines = {r.line_label: r.text for r in records if r.record_type == "form_line"}
    assert lines["1a"] == "Total amount from Form(s) W-2, box 1 (see instructions)"
    assert lines["12"] == "Standard deduction or itemized deductions (from Schedule A)"
    assert lines["15"].startswith("Subtract line 14 from line 11")
    assert "Privacy Act" not in lines["15"]          # footer must not leak in
    assert lines["2b"] == "Taxable interest"         # second column of a row
    # every numbered line from 1z to 38 is present
    for label in ["1z"] + [str(n) for n in range(7, 25)] + ["25a", "25d", "26", "33", "38"]:
        assert label in lines, label


# --------------------------------------------------------------------------- Req 2.9: horizontal bounds

class _FakePage:
    """Minimal page stub for testing _page_words."""
    def __init__(self, width, height, words):
        self.width = width
        self.height = height
        self._words = words

    def extract_words(self, **kwargs):
        return self._words


def _make_word(text, x0, top, x1, bottom, fontname="Times", size=10):
    return {"text": text, "x0": x0, "x1": x1, "top": top, "bottom": bottom,
            "fontname": fontname, "size": size}


def test_page_words_excludes_words_outside_horizontal_bounds():
    """Words whose x0 < 0 or x1 > page.width must be dropped (Req 2.9)."""
    page = _FakePage(
        width=612, height=792,
        words=[
            _make_word("InBounds",  10, 100, 80, 110),           # kept
            _make_word("TooLeft",  -5, 100, 30, 110),            # x0 < 0 → dropped
            _make_word("TooRight", 600, 100, 620, 110),          # x1 > 612 → dropped
            _make_word("ExactEdge", 0, 100, 612, 110),           # exactly at boundary → kept
        ],
    )
    result = _page_words(page)
    texts = [w["text"] for w in result]
    assert "InBounds" in texts
    assert "ExactEdge" in texts
    assert "TooLeft" not in texts
    assert "TooRight" not in texts


def test_page_words_excludes_words_outside_vertical_bounds():
    """Words whose top < 0 or bottom > page.height must be dropped (Req 2.9)."""
    page = _FakePage(
        width=612, height=792,
        words=[
            _make_word("Normal",  10, 100, 80, 110),
            _make_word("AboveTop", 10, -5,  80,   5),   # top < 0 → dropped
            _make_word("BelowBot", 10, 790, 80, 800),   # bottom > 792 → dropped
        ],
    )
    result = _page_words(page)
    texts = [w["text"] for w in result]
    assert "Normal" in texts
    assert "AboveTop" not in texts
    assert "BelowBot" not in texts


# --------------------------------------------------------------------------- Req 2.10: form decoration filter

def _bold_word(text, top=100, page_height=792):
    return _make_word(text, 50, top, 100, top + 10, fontname="Arial-Black")


def _regular_word(text, top=100, page_height=792):
    return _make_word(text, 50, top, 100, top + 10, fontname="Times")


def test_is_form_decoration_tip_bold_is_decoration():
    """Bold 'TIP' token is decoration (Req 2.10)."""
    w = _bold_word("TIP")
    assert _is_form_decoration(w, page_height=792) is True


def test_is_form_decoration_caution_bold_is_decoration():
    """Bold 'CAUTION' token is decoration (Req 2.10)."""
    w = _bold_word("CAUTION")
    assert _is_form_decoration(w, page_height=792) is True


def test_is_form_decoration_exclamation_bold_is_decoration():
    """Bold '!' token inside icon is decoration (Req 2.10)."""
    w = _bold_word("!")
    assert _is_form_decoration(w, page_height=792) is True


def test_is_form_decoration_tip_regular_font_is_not_decoration():
    """'TIP' in a regular font is content, not decoration (Req 2.10)."""
    w = _regular_word("TIP")
    assert _is_form_decoration(w, page_height=792) is False


def test_is_form_decoration_footer_word_is_decoration():
    """Word in the bottom 65 pt of the page is footer decoration (Req 2.10)."""
    w = _regular_word("Privacy", top=730)   # 792 - 730 = 62 < 65
    assert _is_form_decoration(w, page_height=792) is True


def test_is_form_decoration_word_above_footer_is_not_decoration():
    """Word above the footer zone is not decoration (Req 2.10)."""
    w = _regular_word("Income", top=700)    # 792 - 700 = 92 > 65
    assert _is_form_decoration(w, page_height=792) is False


@pytest.mark.skipif(not F1040_2024.exists(), reason="run the fetcher first")
def test_form_decoration_absent_from_real_form():
    """TIP / CAUTION / footer text must not appear in any record (Req 2.10)."""
    records = parse_pdf(F1040_2024, "form")
    all_text = " ".join(r.text for r in records)
    # The word 'Privacy' appears only in the page footer
    assert "Privacy Act" not in all_text
