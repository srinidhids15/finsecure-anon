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


# --- hand-written PDFs for structures PyMuPDF's high-level API cannot create ---

HELVETICA = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"


def raw_pdf(content: bytes, fonts: dict[str, str], extra: list | None = None, catalog: str = "",
            resources: str = "") -> bytes:
    """A one-page 595x842 PDF written byte by byte.

    Objects 1-4 are catalog, pages, page and content stream; ``extra``
    bodies are numbered from 5 so fonts and other objects can refer to them.
    """
    objs = {
        1: f"<< /Type /Catalog /Pages 2 0 R {catalog}>>".encode(),
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << "
            + " ".join(f"/{k} {v}" for k, v in fonts.items()) + f" >> {resources}>> >>").encode(),
        4: b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
    }
    for n, body in enumerate(extra or [], 5):
        objs[n] = body if isinstance(body, bytes) else body.encode()
    out = bytearray(b"%PDF-1.7\n")
    offsets = {}
    for n in sorted(objs):
        offsets[n] = len(out)
        out += b"%d 0 obj\n" % n + objs[n] + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % offsets[n] for n in sorted(objs))
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


def stream_obj(data: bytes, dictionary: str = "") -> bytes:
    return b"<< /Length %d %s >>\nstream\n" % (len(data), dictionary.encode()) + data + b"\nendstream"


def to_unicode_cmap(pairs: dict[int, str]) -> bytes:
    """A ToUnicode CMap mapping single-byte codes to (possibly several) code points."""
    lines = b"".join(b"<%02X> <%s>\n" % (code, "".join(f"{ord(ch):04X}" for ch in text).encode())
                     for code, text in pairs.items())
    return (b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap /CMapName /FG def\n"
            b"1 begincodespacerange <00> <FF> endcodespacerange\n"
            b"%d beginbfchar\n" % len(pairs) + lines + b"endbfchar\nendcmap CMapName currentdict /CMap "
            b"defineresource pop end end")


def type3_font(to_unicode_ref: str = "") -> str:
    """A Type 3 font whose glyphs (codes 32-90) all draw a filled box."""
    codes = range(32, 91)
    diffs = " ".join(f"{c} /g" for c in codes)
    return ("<< /Type /Font /Subtype /Type3 /FontBBox [0 0 600 700] /FontMatrix [0.001 0 0 0.001 0 0] "
            f"/CharProcs << /g 5 0 R >> /Encoding << /Type /Encoding /Differences [{diffs}] >> "
            f"/FirstChar 32 /LastChar 90 /Widths [{' '.join(['600'] * len(codes))}] /Resources << >> "
            f"{'/ToUnicode ' + to_unicode_ref if to_unicode_ref else ''} >>")


TYPE3_GLYPH = stream_obj(b"600 0 0 0 500 700 d1 0 0 500 700 re f")
