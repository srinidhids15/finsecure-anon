"""Page text with an explicit per-character geometry map.

For each page, ``extract_page`` reads PyMuPDF's ``rawdict`` (character-level
extraction) and builds the string that is passed to the detector, together
with ``geoms``, a tuple of the same length: ``geoms[i]`` is the geometry of
``text[i]``, or None for characters that have no glyph.

How the string is built (in MuPDF's content order, no re-sorting):

* every extracted character is appended as is (Unicode preserved; ligatures
  are expanded by MuPDF into their component letters);
* spans within a line are concatenated directly;
* consecutive lines on the same baseline (separate text runs of one row,
  left to right) are joined by one synthetic space;
* all other line and block boundaries become one synthetic ``"\\n"``.

Whitespace characters, MuPDF's synthetic spaces and the synthetic
separators carry no geometry: redaction regions are built from visible
glyphs only. Every non-whitespace character has a geometry (its ``rawdict``
bounding box, origin, line number and glyph group).

Glyph groups: when one drawn glyph yields several characters (a ligature
such as "ﬁ", or a ``ToUnicode`` entry mapping one code to several code
points), MuPDF gives the first character the glyph's box and each further
character a zero-width box at the glyph's right edge. Those characters
share a ``glyph`` id, so the mapping can refuse a redaction boundary that
falls inside one glyph (MuPDF removes whole glyphs only).

The text is built from the drawn glyphs (``TEXT_IGNORE_ACTUALTEXT``).

``extract_page`` also records page conditions that make the mapping
unreliable or leave content unscanned. ``unsupported`` conditions reject
the document (see ``pipeline``); ``notes`` are reported as warnings.
No text is logged and the returned objects keep text only in memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pymupdf

# Whitespace kept, ligatures expanded, drawn glyphs rather than marked-content
# replacement text. Extraction is not clipped (MuPDF clips to the page unless
# given an unbounded clip), so text outside the visible page is seen and can
# be rejected.
EXTRACT_FLAGS = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_IGNORE_ACTUALTEXT
BASELINE_TOLERANCE = 0.5  # points
INVISIBLE_RENDER_MODES = {3, 7}  # PDF text render modes that draw nothing
COVERING_KINDS = {"fill-image", "fill-imgmask", "fill-shade", "fill-path"}
MIN_COVER_HEIGHT = 0.5  # a filled path must be at least half a glyph high to hide it


def raw_blocks(page: pymupdf.Page, flags: int = EXTRACT_FLAGS) -> list[dict]:
    return page.get_text("rawdict", flags=flags, clip=pymupdf.INFINITE_RECT())["blocks"]


def _visible_chars(page: pymupdf.Page, flags: int) -> str:
    return "".join(c["c"] for b in raw_blocks(page, flags) if b.get("type") == 0 for line in b["lines"]
                   for span in line["spans"] for c in span["chars"] if not c["c"].isspace())


@dataclass(frozen=True)
class CharGeom:
    bbox: tuple[float, float, float, float]
    origin: tuple[float, float]
    line: int  # index of the line on this page
    glyph: int = -1  # characters from the same drawn glyph share this id

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2, (y0 + y1) / 2)


@dataclass(frozen=True)
class PageText:
    number: int  # 1-based
    width: float
    height: float
    text: str = field(default="", repr=False)  # in memory only; never logged, exported or repr'd
    geoms: tuple[CharGeom | None, ...] = field(default=(), repr=False)
    lines: int = 0
    images: int = 0
    drawings: int = 0
    unsupported: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def mapped_chars(self) -> int:
        return sum(g is not None for g in self.geoms)

    def diagnostics(self) -> dict:
        return {
            "page": self.number,
            "width": round(self.width, 2),
            "height": round(self.height, 2),
            "characters": len(self.text),
            "mapped_characters": self.mapped_chars,
            "lines": self.lines,
            "images": self.images,
            "unsupported": list(self.unsupported),
            "notes": list(self.notes),
        }


def _trace_conditions(page: pymupdf.Page, extracted_glyphs: int) -> set[str]:
    """Cross-check with MuPDF's text trace (render mode, opacity, glyph count)."""
    found = set()
    traced = 0
    for span in page.get_texttrace():
        glyphs = sum(1 for c in span.get("chars", ()) if c[0] >= 0 and not chr(c[0]).isspace())
        traced += glyphs
        if glyphs and (span.get("type") in INVISIBLE_RENDER_MODES or span.get("opacity", 1) == 0):
            found.add("invisible text (for example an OCR layer)")
    if traced > extracted_glyphs:  # ligature expansion can only make the extraction longer
        found.add("text that the character-level extraction did not return")
    return found


def _hidden_text_conditions(page: pymupdf.Page) -> set[str]:
    """Text that is extractable but not visible: covered by a later image or
    filled shape, or clipped out. Draw order comes from ``get_bboxlog``, whose
    indices are the ``seqno`` values of ``get_texttrace``."""
    found = set()
    covers = [(i, kind, pymupdf.Rect(b)) for i, (kind, b) in enumerate(page.get_bboxlog()) if kind in COVERING_KINDS]
    if covers:
        for span in page.get_texttrace():
            later = [(kind, r) for i, kind, r in covers if i > span["seqno"]]
            for c in span["chars"] if later else ():
                if c[0] < 0 or chr(c[0]).isspace():
                    continue
                x0, y0, x1, y1 = c[3]
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                if any(r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1
                       and (kind != "fill-path" or r.height >= MIN_COVER_HEIGHT * (y1 - y0)) for kind, r in later):
                    found.add("text covered by an image or filled shape drawn on top of it (hidden but extractable)")
                    break
    if len(_visible_chars(page, EXTRACT_FLAGS | pymupdf.TEXT_CLIP)) < len(_visible_chars(page, EXTRACT_FLAGS)):
        found.add("text clipped out of view (hidden but extractable)")
    return found


def _replacement_text_conditions(page: pymupdf.Page) -> set[str]:
    honoured = _visible_chars(page, EXTRACT_FLAGS & ~pymupdf.TEXT_IGNORE_ACTUALTEXT)
    if honoured != _visible_chars(page, EXTRACT_FLAGS):
        return {"replacement text (ActualText) that differs from the drawn glyphs"}
    return set()


def _font_conditions(page: pymupdf.Page) -> set[str]:
    doc = page.parent
    for xref, _ext, kind, *_ in page.get_fonts(full=True):
        if kind == "Type3" and doc.xref_get_key(xref, "ToUnicode")[0] in ("null", ""):
            return {"a Type 3 font without a Unicode mapping (its glyphs cannot be read reliably)"}
    return set()


def extract_page(page: pymupdf.Page) -> PageText:
    rect = page.rect
    parts: list[str] = []
    geoms: list[CharGeom | None] = []
    unsupported: set[str] = set()
    line_no = glyph_no = 0
    prev_baseline = prev_x1 = None

    for block in raw_blocks(page):
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            chars = [c for span in line["spans"] for c in span["chars"]]
            if not chars:
                continue
            dx, dy = line["dir"]
            if line.get("wmode", 0) != 0 or abs(dy) > 1e-3 or dx <= 0:
                unsupported.add("rotated or vertical text")
            if parts:
                same_row = (prev_baseline is not None and abs(chars[0]["origin"][1] - prev_baseline) <= BASELINE_TOLERANCE
                            and chars[0]["bbox"][0] >= prev_x1 - BASELINE_TOLERANCE)
                parts.append(" " if same_row else "\n")
                geoms.append(None)
            prev = None
            for c in chars:
                ch = c["c"]
                if len(ch) != 1:
                    unsupported.add("characters that do not map to a single Unicode code point")
                    ch = ch[:1] or "�"
                if ch == "�":
                    unsupported.add("characters without a Unicode mapping (text cannot be read reliably)")
                parts.append(ch)
                if ch.isspace() or c.get("synthetic"):
                    geoms.append(None)
                    prev = None
                    continue
                x0, _, x1, _ = c["bbox"]
                continues_glyph = (prev is not None and x1 - x0 <= 1e-6
                                   and abs(c["origin"][0] - prev["bbox"][2]) <= 0.01
                                   and abs(c["origin"][1] - prev["origin"][1]) <= 0.01)
                if not continues_glyph:
                    glyph_no += 1
                prev = c
                g = CharGeom(tuple(c["bbox"]), tuple(c["origin"]), line_no, glyph_no)
                cx, cy = g.center
                if not (rect.x0 <= cx <= rect.x1 and rect.y0 <= cy <= rect.y1):
                    unsupported.add("text outside the visible page area")
                geoms.append(g)
            prev_baseline = chars[-1]["origin"][1]
            prev_x1 = chars[-1]["bbox"][2]
            line_no += 1

    text = "".join(parts)
    images = len(page.get_image_info())
    drawings = len(page.get_drawings())
    notes = []
    if page.rotation:
        unsupported.add(f"page rotation of {page.rotation} degrees")
    unsupported |= _trace_conditions(page, sum(g is not None for g in geoms))
    unsupported |= _hidden_text_conditions(page)
    unsupported |= _replacement_text_conditions(page)
    unsupported |= _font_conditions(page)
    if not text.strip():
        if images:
            unsupported.add("no extractable text; the page is an image (scanned?) and OCR is not supported")
        elif drawings:
            unsupported.add("no extractable text; the page has only vector graphics (text may be outlined)")
        else:
            notes.append("blank page")
    elif images:
        notes.append(f"{images} image(s): any text inside images is not detected or redacted")
    return PageText(
        number=page.number + 1,
        width=rect.width,
        height=rect.height,
        text=text,
        geoms=tuple(geoms),
        lines=line_no,
        images=images,
        drawings=drawings,
        unsupported=tuple(sorted(unsupported)),
        notes=tuple(notes),
    )
