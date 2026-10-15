"""Synthetic PDF builders and a deterministic stub detector for PDF tests.

All documents are generated here with PyMuPDF from invented values; no real
financial documents are used.
"""

from __future__ import annotations

import pymupdf

from finguard.schema import Entity

FONT_SIZE = 11
LEADING = 16
LEFT = 72
TOP = 72


def build_pdf(pages, fontsize=FONT_SIZE, leading=LEADING, setup=None, **save) -> bytes:
    """``pages``: one list per page of lines (str, stacked from the top) or
    ``(x, y, text)`` tuples placed at a baseline position."""
    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page()
        y = TOP
        for item in lines:
            if isinstance(item, tuple):
                page.insert_text(item[:2], item[2], fontsize=fontsize)
            else:
                page.insert_text((LEFT, y), item, fontsize=fontsize)
                y += leading
    if setup is not None:
        setup(doc)
    data = doc.tobytes(**save)
    doc.close()
    return data


def image_only_page(doc: pymupdf.Document) -> None:
    """Append a page holding only an image (a stand-in for a scanned page)."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 20), False)
    pix.set_rect(pix.irect, (200, 200, 200))
    doc.new_page().insert_image(pymupdf.Rect(72, 72, 272, 172), pixmap=pix)


class FindDetector:
    """Returns every occurrence of the given pieces, optionally only on pages
    whose text contains ``only_on``. Records nothing about the text."""

    name = "find"

    def __init__(self, *targets: tuple[str, str], only_on: str | None = None, source: str = "stub"):
        self.targets = targets
        self.only_on = only_on
        self.source = source
        self.calls = 0

    def detect(self, text: str) -> list[Entity]:
        self.calls += 1
        if self.only_on is not None and self.only_on not in text:
            return []
        out = []
        for piece, label in self.targets:
            start = text.find(piece)
            while start != -1:
                out.append(Entity.from_text(text, start, start + len(piece), label, self.source))
                start = text.find(piece, start + len(piece))
        return sorted(out, key=lambda e: (e.start, e.end))


def glyph_text(pdf: bytes, page_number: int, rect) -> str:
    """Text extractable from ``rect`` of a page (independent re-extraction)."""
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        return doc[page_number - 1].get_text("text", clip=pymupdf.Rect(rect)).strip()


def page_text(pdf: bytes, page_number: int) -> str:
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        return doc[page_number - 1].get_text("text")


def stream_bytes(pdf: bytes) -> bytes:
    """Every decompressed stream in the file (content streams included)."""
    out = []
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        for xref in range(1, doc.xref_length()):
            if doc.xref_is_stream(xref):
                out.append(doc.xref_stream(xref) or b"")
    return b"\n".join(out)


def encoded_forms(value: str) -> list[bytes]:
    """Ways a Latin-1 string can appear in a content stream (literal or hex)."""
    raw = value.encode("latin-1")
    return [raw, raw.hex().encode(), raw.hex().upper().encode()]
