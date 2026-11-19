"""Render a PDF page as a PNG with the cited text highlighted.

The parser stored a bounding box for every record ({page, x0, top, x1,
bottom} in PDF points). Rendering on the server with pdfplumber keeps the
frontend simple: it shows an image of the exact page with a marker-yellow
box over the cited words.
"""
from __future__ import annotations

import io
from functools import lru_cache

RESOLUTION = 110
FILL = (242, 214, 72, 96)
STROKE = (184, 146, 0)


def parse_boxes(spec: str | None) -> tuple[tuple[float, float, float, float], ...]:
    """'x0,top,x1,bottom;x0,top,x1,bottom' -> tuple of boxes (invalid parts skipped)."""
    boxes = []
    for part in (spec or "").split(";"):
        try:
            x0, top, x1, bottom = (float(v) for v in part.split(","))
        except ValueError:
            continue
        if x1 > x0 and bottom > top:
            boxes.append((x0, top, x1, bottom))
    return tuple(boxes[:20])


@lru_cache(maxsize=128)
def render_page(pdf_path: str, page_number: int, boxes: tuple = ()) -> bytes:
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        if not 1 <= page_number <= len(pdf.pages):
            raise ValueError(f"page {page_number} out of range 1-{len(pdf.pages)}")
        im = pdf.pages[page_number - 1].to_image(resolution=RESOLUTION)
        for box in boxes:
            pad = 2
            im.draw_rect((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad),
                         fill=FILL, stroke=STROKE, stroke_width=2)
        buf = io.BytesIO()
        im.annotated.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
