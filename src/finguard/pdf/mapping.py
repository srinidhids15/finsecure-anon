"""Map detected entities on one page to redaction rectangles.

Validation and canonicalisation are not reimplemented: the page's
entities are passed through ``finguard.anonymization.anonymize`` on the
exact page text, which rejects invalid offsets, mismatched text, unknown
labels and overlaps, and returns each canonical span (trailing whitespace
excluded) with its detector sources.

Each canonical span ``[start, end)`` is mapped through ``PageText.geoms``:
the visible glyphs in the span are grouped by line, and each line gets one
rectangle, the union of those glyphs' bounding boxes. A multiline entity
(an address, a credit-card block) therefore gets one tight rectangle per
line, and nothing between lines is covered.

A mapping is **rejected** (and the document is not returned) when:

* the span contains a visible character without geometry, or no glyph at all;
* a rectangle would also remove an untargeted visible character. MuPDF
  removes a character when the centre of its bounding box lies in a
  redaction rectangle, so any untargeted glyph whose centre falls inside a
  rectangle (enlarged by ``COLLATERAL_MARGIN``) is collateral.

Reasons contain page, label and offsets only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from finguard.anonymization import AnonymizationError, anonymize
from finguard.pdf.extract import CharGeom, PageText
from finguard.schema import Entity

COLLATERAL_MARGIN = 0.5  # points

Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class PdfEntity:
    page: int
    label: str
    start: int  # canonical span in the page text
    end: int
    detected_end: int
    sources: tuple[str, ...]
    rects: tuple[Rect, ...] = ()
    char_indices: tuple[int, ...] = ()  # visible glyphs to be removed
    status: str = "mapped"  # mapped | rejected | redacted
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "detected_end": self.detected_end,
            "regions": [[round(v, 2) for v in r] for r in self.rects],
            "sources": list(self.sources),
            "status": self.status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PageMapping:
    page: int
    entities: tuple[PdfEntity, ...]
    error: str = ""  # whole-page failure (invalid detector output)

    @property
    def rejected(self) -> tuple[PdfEntity, ...]:
        return tuple(e for e in self.entities if e.status == "rejected")

    @property
    def targeted(self) -> frozenset[int]:
        return frozenset(i for e in self.entities if e.status == "mapped" for i in e.char_indices)


def _union(geoms: Sequence[CharGeom]) -> Rect:
    return (min(g.bbox[0] for g in geoms), min(g.bbox[1] for g in geoms),
            max(g.bbox[2] for g in geoms), max(g.bbox[3] for g in geoms))


def _inside(point: tuple[float, float], rect: Rect, margin: float = 0.0) -> bool:
    x, y = point
    return rect[0] - margin <= x <= rect[2] + margin and rect[1] - margin <= y <= rect[3] + margin


def map_page(page: PageText, entities: Sequence[Entity]) -> PageMapping:
    try:
        result = anonymize(page.text, entities)
    except AnonymizationError as exc:  # message holds labels and offsets only
        return PageMapping(page.number, (), f"page {page.number}: detector output rejected: {exc}")

    mapped: list[PdfEntity] = []
    for r in result.redactions:
        base = dict(page=page.number, label=r.label, start=r.start, end=r.end,
                    detected_end=r.detected_end, sources=r.sources)
        idx = [i for i in range(r.start, r.end) if not page.text[i].isspace()]
        missing = [i for i in idx if page.geoms[i] is None]
        if missing or not idx:
            mapped.append(PdfEntity(**base, status="rejected",
                                    reason="characters without page geometry" if idx else "no visible characters"))
            continue
        by_line: dict[int, list[CharGeom]] = {}
        for i in idx:
            by_line.setdefault(page.geoms[i].line, []).append(page.geoms[i])
        rects = tuple(_union(gs) for _, gs in sorted(by_line.items()))
        mapped.append(PdfEntity(**base, rects=rects, char_indices=tuple(idx)))

    # Collateral check against every untargeted visible glyph on the page.
    targeted = {i for e in mapped for i in e.char_indices}
    checked: list[PdfEntity] = []
    for e in mapped:
        if e.status != "mapped":
            checked.append(e)
            continue
        hits = sum(
            1 for i, g in enumerate(page.geoms)
            if g is not None and i not in targeted and any(_inside(g.center, r, COLLATERAL_MARGIN) for r in e.rects)
        )
        if hits:
            e = replace(e, status="rejected", reason=f"region would remove {hits} unrelated character(s)")
        checked.append(e)
    return PageMapping(page.number, tuple(checked))
