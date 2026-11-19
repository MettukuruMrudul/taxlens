"""Parse IRS PDFs into line-level records with page positions.

Two kinds of documents need two strategies:

* Forms (Form 1040 and its schedules) are organised by numbered lines such
  as "1a Total amount from Form(s) W-2". Each numbered line becomes one
  `form_line` record labelled with its line number. Other text on the form
  (headers, side notes such as the standard deduction amounts printed in
  the left margin) becomes `form_text` records.

* Instructions and publications are prose in two or three columns with
  bold headings. They become `section` records: chunks of body text tagged
  with their heading path, e.g. "Income > Line 1a > Wages".

Every record keeps the page number and the bounding boxes of the text it
came from, so a citation can later highlight the exact spot in the PDF.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pdfplumber

PARSER_VERSION = "2"

LINE_NUM_RE = re.compile(r"^(\d{1,2})([a-z])?$")
LETTER_RE = re.compile(r"^[a-z]$")
LEADER_RE = re.compile(r"^[.\u2026\u25b6\u25ba\u2022]+$")  # dots, ellipses, arrows, bullets
MAX_CHUNK_CHARS = 1200


@dataclass
class Record:
    record_type: str            # 'form_line' | 'form_text' | 'section'
    text: str
    page: int                   # 1-based page number
    boxes: list[dict]           # [{"page", "x0", "top", "x1", "bottom"}]
    line_label: str | None = None
    heading_path: str | None = None


@dataclass
class _Item:
    label: str | None
    words: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- helpers

def _page_words(page) -> list[dict]:
    """Words inside the visible page, with font info. Drops printer marks
    that sit outside the page area (both vertically and horizontally)."""
    words = page.extract_words(
        x_tolerance=1.5, y_tolerance=2, keep_blank_chars=False,
        extra_attrs=["fontname", "size"],
    )
    return [
        w for w in words
        if w["top"] >= 0 and w["bottom"] <= page.height
        and w["x0"] >= 0 and w["x1"] <= page.width
        and w["text"].strip()
    ]


def _rows(words: list[dict], tol: float = 2.5) -> list[list[dict]]:
    """Group words into visual rows (same baseline), each sorted left to right."""
    rows: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if rows and abs(rows[-1][0]["top"] - w["top"]) <= tol:
            rows[-1].append(w)
        else:
            rows.append([w])
    return [sorted(r, key=lambda w: w["x0"]) for r in rows]


def _box(words: list[dict], page_no: int) -> dict:
    return {
        "page": page_no,
        "x0": round(min(w["x0"] for w in words), 1),
        "top": round(min(w["top"] for w in words), 1),
        "x1": round(max(w["x1"] for w in words), 1),
        "bottom": round(max(w["bottom"] for w in words), 1),
    }


def _join(words: list[dict]) -> str:
    text = " ".join(w["text"] for w in words).replace("\x00", "")
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- forms

def _is_form_decoration(w: dict, page_height: float) -> bool:
    """Filter visual decoration tokens that appear on form pages.

    Returns True for:
    - TIP/CAUTION icon lettering and the '!' glyph with bold/black fonts
    - Page-footer text (IRS form footers sit in roughly the bottom 65 pt of the
      page — wider than instruction footers because forms carry the Privacy Act
      notice and catalog number in the last ~62 pt).
    """
    font = w["fontname"]
    if ("Black" in font or "Blk" in font or "Bold" in font) and w["text"] in {"TIP", "CAUTION", "!"}:
        return True
    return w["top"] > page_height - 65


def _number_column_x(rows: list[list[dict]]) -> float | None:
    """Find the x position of the line-number column: the left edge where
    tokens like '1', '12', '2a' start rows and are followed by text."""
    xs = []
    for row in rows:
        for i, w in enumerate(row[:-1]):
            nxt = row[i + 1]
            if LINE_NUM_RE.match(w["text"]) and re.match(r"^[A-Za-z(]", nxt["text"]) and nxt["x0"] - w["x1"] < 20:
                xs.append(round(w["x0"]))
                break
    if not xs:
        return None
    common = [x for x, n in Counter(xs).items() if n >= 2] or xs
    return float(min(common))


def _is_box_label(tok: str, current: str | None) -> bool:
    return current is not None and tok == current


def _parse_form_page(page, page_no: int) -> list[Record]:
    words = [w for w in _page_words(page) if not _is_form_decoration(w, page.height)]
    rows = _rows(words)
    num_x = _number_column_x(rows)
    split_x = (num_x - 3) if num_x else 0.0

    records: list[Record] = []
    items: list[_Item] = []          # numbered items, in order
    current: _Item | None = None
    current_num: str | None = None   # e.g. '1' while inside lines 1a..1z
    side_rows: list[list[dict]] = []
    text_rows: list[list[dict]] = []
    last_item_bottom = -1e9

    for row in rows:
        side = [w for w in row if w["x1"] <= split_x]
        main = [w for w in row if w["x1"] > split_x]
        if side:
            side_rows.append(side)
        if not main:
            continue

        # A row that runs continuously from the margin into the main area
        # (e.g. the footer) is free text, not a margin note plus a line.
        if side and main and main[0]["x0"] - side[-1]["x1"] < 6:
            text_rows.append([w for w in row if not LEADER_RE.match(w["text"])])
            side_rows.pop()
            continue

        started_here = False
        i = 0
        toks = main
        while i < len(toks):
            t = toks[i]["text"]
            nxt = toks[i + 1]["text"] if i + 1 < len(toks) else ""
            nxt2 = toks[i + 2]["text"] if i + 2 < len(toks) else ""
            prev = toks[i - 1]["text"] if i > 0 else ""
            # A new line can only start in the line-number column, or right
            # after leader dots / a box label (the second column of a row).
            at_start = (num_x is not None and abs(toks[i]["x0"] - num_x) <= 10) or i == 0 \
                or LEADER_RE.match(prev) is not None or LINE_NUM_RE.match(prev) is not None

            m = LINE_NUM_RE.match(t) if at_start else None
            # "1 a Total ..." -> label 1a
            if m and not m.group(2) and LETTER_RE.match(nxt) and re.match(r"^[A-Z(]", nxt2):
                current_num = m.group(1)
                current = _Item(label=current_num + nxt)
                items.append(current)
                started_here = True
                i += 2
                continue
            # "12 Standard deduction ..." or "2a Tax-exempt interest ..."
            if m and re.match(r"^[A-Z(\d]", nxt) and not _is_box_label(t, current.label if current else None):
                current_num = m.group(1)
                current = _Item(label=t)
                items.append(current)
                started_here = True
                i += 1
                continue
            # "b Taxable interest ..." -> sub-line of the current number
            if at_start and LETTER_RE.match(t) and current_num and re.match(r"^[A-Z(]", nxt):
                current = _Item(label=current_num + t)
                items.append(current)
                started_here = True
                i += 1
                continue
            # leader dots and the repeated box label at the right edge are not content
            if LEADER_RE.match(t) or _is_box_label(t, current.label if current else None):
                i += 1
                continue
            if started_here and current is not None:
                current.words.append(toks[i])
            else:
                break
            i += 1

        if started_here:
            last_item_bottom = max(w["bottom"] for w in main)
            continue

        # Row with no line number: a wrapped continuation of the previous line
        # if it sits directly below it, otherwise free text on the form.
        content = [w for w in main if not LEADER_RE.match(w["text"])]
        aligned = (current is not None and current.words and content
                   and abs(content[0]["x0"] - current.words[0]["x0"]) <= 4)
        if aligned and 0 <= content[0]["top"] - last_item_bottom <= 6:
            current.words.extend(w for w in content if not _is_box_label(w["text"], current.label))
            last_item_bottom = max(w["bottom"] for w in content)
        elif content:
            text_rows.append(content)

    for item in items:
        if item.words:
            records.append(Record("form_line", _join(item.words), page_no,
                                  [_box(item.words, page_no)], line_label=item.label))
    for block in _blocks(text_rows) + _blocks(side_rows):
        flat = [w for r in block for w in r]
        text = _join(flat)
        if len(text) >= 3:
            records.append(Record("form_text", text, page_no, [_box(flat, page_no)]))
    return records


def _blocks(rows: list[list[dict]], gap: float = 4.0) -> list[list[list[dict]]]:
    """Group rows that sit close together vertically into text blocks."""
    blocks: list[list[list[dict]]] = []
    for row in sorted(rows, key=lambda r: r[0]["top"]):
        if blocks:
            prev = blocks[-1][-1]
            prev_bottom = max(w["bottom"] for w in prev)
            if row[0]["top"] - prev_bottom <= gap and abs(row[0]["x0"] - blocks[-1][0][0]["x0"]) < 60:
                blocks[-1].append(row)
                continue
        blocks.append([row])
    return blocks


def parse_form(pdf) -> list[Record]:
    out: list[Record] = []
    for page_no, page in enumerate(pdf.pages, start=1):
        out.extend(_parse_form_page(page, page_no))
    return out


# --------------------------------------------------------------------------- prose

V_GAP = 8.0   # min empty width (pt) that separates two columns
H_GAP = 4.5   # min empty height (pt) that separates two blocks vertically


def _gaps(spans: list[tuple[float, float]], min_gap: float) -> list[tuple[float, float]]:
    """Empty intervals of at least min_gap between a set of 1-D spans."""
    if not spans:
        return []
    spans = sorted(spans)
    gaps, reach = [], spans[0][1]
    for a, b in spans[1:]:
        if a - reach >= min_gap:
            gaps.append((reach, a))
        reach = max(reach, b)
    return gaps


def _x_gaps(words):
    return _gaps([(w["x0"], w["x1"]) for w in words], V_GAP)


def _respects_gutter(words, gaps) -> bool:
    """True if some gutter in `gaps` is also empty for these words."""
    for g0, g1 in gaps:
        if all(w["x1"] <= g0 + 1 or w["x0"] >= g1 - 1 for w in words) and g1 - g0 >= V_GAP:
            return True
    return False


def _split(words, gaps, key):
    bounds = [(a + b) / 2 for a, b in gaps]
    parts = [[] for _ in range(len(bounds) + 1)]
    for w in words:
        parts[sum(w[key] > b for b in bounds)].append(w)
    return [p for p in parts if p]


def _reading_blocks(words: list[dict], depth: int = 0) -> list[list[dict]]:
    """Recursive XY-cut: split a page into blocks in reading order.

    Columns are split first (left to right). When a page mixes layouts (for
    example a full-width paragraph above two columns), it is split into
    horizontal bands, and neighbouring bands that share the same column
    gutter are merged again so each column is read top to bottom before
    moving to the next one.
    """
    if len(words) < 2 or depth > 30:
        return [words]
    vgaps = _x_gaps(words)
    if vgaps:
        return [b for part in _split(words, vgaps, "x0") for b in _reading_blocks(part, depth + 1)]
    hgaps = _gaps([(w["top"], w["bottom"]) for w in words], H_GAP)
    if not hgaps:
        return [words]
    bands = _split(words, hgaps, "top")
    merged: list[list[dict]] = []
    for band in bands:
        if merged and (_respects_gutter(band, _x_gaps(merged[-1])) or _respects_gutter(merged[-1], _x_gaps(band))):
            candidate = merged[-1] + band
            if _x_gaps(candidate):
                merged[-1] = candidate
                continue
        merged.append(band)
    return [b for band in merged for b in _reading_blocks(band, depth + 1)]


def _is_decoration(w: dict, page_height: float) -> bool:
    """Caution/tip icons and the running page footer are not content."""
    font = w["fontname"]
    if ("Black" in font or "Blk" in font) and w["text"] in {"TIP", "CAUTION", "ENCLOSED", "RECORDS"}:
        return True                             # icon lettering
    if w["text"] == "!" and w["size"] > 14:     # the "!" inside the caution icon
        return True
    return w["top"] > page_height - 42          # footer: page number, "Need more information..."


def _heading_level(row: list[dict]) -> int | None:
    if sum(c.isalpha() for c in _join(row)) < 3:   # page numbers, stray symbols
        return None
    fonts = {w["fontname"].split("+")[-1] for w in row}
    size = max(w["size"] for w in row)
    bold_sans = all(("Bold" in f or "Black" in f or "Blk" in f) and "Times" not in f for f in fonts)
    if not bold_sans:
        return None
    if size >= 15.5:
        return 1
    if size >= 13.5:
        return 2
    if size >= 11.5:
        return 3
    return 4


def _append_line(buf: str, line: str) -> str:
    if not buf:
        return line
    if buf.endswith("-") and line[:1].islower():
        return buf[:-1] + line          # undo end-of-line hyphenation
    return buf + " " + line


def parse_prose(pdf) -> list[Record]:
    records: list[Record] = []
    path: list[tuple[int, str]] = []          # heading stack: (level, text)
    buf, buf_boxes, buf_words = "", [], []
    pending_heading: tuple[int, str] | None = None

    def heading_path() -> str:
        return " > ".join(t for _, t in path)

    def flush_part():
        nonlocal buf_words
        if buf_words:
            buf_boxes.append(_box(buf_words, buf_words[0]["_page"]))
            buf_words = []

    def flush():
        nonlocal buf, buf_boxes
        flush_part()
        text = buf.strip()
        if len(text) >= 20:
            records.append(Record("section", text, buf_boxes[0]["page"], list(buf_boxes),
                                  heading_path=heading_path() or None))
        buf, buf_boxes = "", []

    for page_no, page in enumerate(pdf.pages, start=1):
        words = [w for w in _page_words(page) if not _is_decoration(w, page.height)]
        for w in words:
            w["_page"] = page_no
        for block in _reading_blocks(words):
            flush_part()                       # each block gets its own box
            for row in _rows(block):
                text = _join(row)
                level = _heading_level(row)
                if level is not None:
                    flush()
                    if pending_heading and pending_heading[0] == level:
                        # multi-line heading: merge with the line above
                        pending_heading = (level, pending_heading[1] + " " + text)
                        path[-1] = pending_heading
                    else:
                        while path and path[-1][0] >= level:
                            path.pop()
                        pending_heading = (level, text)
                        path.append(pending_heading)
                    continue
                pending_heading = None
                at_sentence_end = buf.rstrip().endswith((".", ":", ";"))
                # Flush at a sentence boundary once we exceed MAX_CHUNK_CHARS (1200).
                # The second condition is a hard cap (2×1200 = 2400 chars) that forces
                # a flush even mid-sentence, to handle very long sentences that can't
                # be split at a natural boundary.
                if (len(buf) + len(text) > MAX_CHUNK_CHARS and at_sentence_end) \
                        or len(buf) + len(text) > 2 * MAX_CHUNK_CHARS:
                    flush()
                buf = _append_line(buf, text)
                buf_words.extend(row)
    flush()
    return records


# --------------------------------------------------------------------------- entry point

def parse_pdf(path, doc_type: str) -> list[Record]:
    with pdfplumber.open(path) as pdf:
        return parse_form(pdf) if doc_type == "form" else parse_prose(pdf)
