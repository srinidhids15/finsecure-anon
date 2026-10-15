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
bounding box, origin and line number).

``extract_page`` also records page conditions that make the mapping
unreliable or leave content unscanned. ``unsupported`` conditions reject
the document (see ``pipeline``); ``notes`` are reported as warnings.
No text is logged and the returned objects keep text only in memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pymupdf

# Whitespace kept, ligatures expanded. Extraction is not clipped (MuPDF clips
# to the page unless given an unbounded clip), so text outside the visible
# page is seen and can be rejected.
EXTRACT_FLAGS = pymupdf.TEXT_PRESERVE_WHITESPACE


def raw_blocks(page: pymupdf.Page) -> list[dict]:
    return page.get_text("rawdict", flags=EXTRACT_FLAGS, clip=pymupdf.INFINITE_RECT())["blocks"]
BASELINE_TOLERANCE = 0.5  # points
INVISIBLE_RENDER_MODES = {3, 7}  # PDF text render modes that draw nothing


@dataclass(frozen=True)
class CharGeom:
    bbox: tuple[float, float, float, float]
    origin: tuple[float, float]
    line: int  # index of the line on this page

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


def extract_page(page: pymupdf.Page) -> PageText:
    rect = page.rect
    parts: list[str] = []
    geoms: list[CharGeom | None] = []
    unsupported: set[str] = set()
    line_no = 0
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
                    continue
                g = CharGeom(tuple(c["bbox"]), tuple(c["origin"]), line_no)
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
