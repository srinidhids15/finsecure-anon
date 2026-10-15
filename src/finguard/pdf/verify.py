"""Independent verification of a redacted PDF.

The saved bytes are reopened as a new document and re-extracted
(``rawdict``, independent of the string built for detection). Checks are
geometric, against the source glyphs recorded before redaction:

* **opens**: the output opens, is not encrypted and was not repaired;
* **structure**: same page count and page sizes;
* **redaction completed**: every planned rectangle has an opaque black fill
  at its position, and no redaction annotations remain;
* **targeted text removed**: no glyph of the output lies in a redaction
  rectangle (by centre), and no targeted source glyph is still present at
  its original position;
* **no collateral loss**: every untargeted visible source glyph is still
  present, same character, same position;
* **sanitized**: no annotations, links or form fields; no metadata, XMP,
  embedded files, outlines, structure tree, automatic actions or JavaScript.

The output is **not** searched for original values: a value may
legitimately remain elsewhere (undetected), and its absence elsewhere would
not prove removal. ``complete_pii_removal_guaranteed`` is always False.
Problem messages contain page numbers, labels and counts only.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import pymupdf

from finguard.pdf.extract import PageText, raw_blocks
from finguard.pdf.mapping import PageMapping
from finguard.pdf.redact import FILL

POSITION_TOLERANCE = 0.1  # points
FILL_TOLERANCE = 0.5


@dataclass(frozen=True)
class PdfVerification:
    opened: bool = False
    structure_preserved: bool = False
    redaction_completed: bool = False
    targeted_text_removed: bool = False
    unrelated_text_preserved: bool = False
    hidden_content_removed: bool = False
    regions_checked: int = 0
    glyphs_removed: int = 0
    glyphs_preserved: int = 0
    problems: tuple[str, ...] = field(default=())
    complete_pii_removal_guaranteed: bool = False

    @property
    def verified(self) -> bool:
        return (self.opened and self.structure_preserved and self.redaction_completed
                and self.targeted_text_removed and self.unrelated_text_preserved
                and self.hidden_content_removed and not self.problems)

    def to_dict(self) -> dict:
        return {
            "verified": self.verified,
            "opened": self.opened,
            "structure_preserved": self.structure_preserved,
            "redaction_completed": self.redaction_completed,
            "targeted_text_removed": self.targeted_text_removed,
            "unrelated_text_preserved": self.unrelated_text_preserved,
            "hidden_content_removed": self.hidden_content_removed,
            "regions_checked": self.regions_checked,
            "glyphs_removed": self.glyphs_removed,
            "glyphs_preserved": self.glyphs_preserved,
            "complete_pii_removal_guaranteed": self.complete_pii_removal_guaranteed,
            "problems": list(self.problems),
        }


def _glyphs(page: pymupdf.Page) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
    """(char, origin, bbox centre) for every visible glyph of the output page."""
    out = []
    for block in raw_blocks(page):
        for line in block.get("lines", ()):
            for span in line["spans"]:
                for c in span["chars"]:
                    if c["c"].isspace() or c.get("synthetic"):
                        continue
                    x0, y0, x1, y1 = c["bbox"]
                    out.append((c["c"], tuple(c["origin"]), ((x0 + x1) / 2, (y0 + y1) / 2)))
    return out


def _key(ch: str, origin: tuple[float, float]) -> tuple[str, int, int]:
    return ch, round(origin[0] / POSITION_TOLERANCE), round(origin[1] / POSITION_TOLERANCE)


def _present(index: Counter, ch: str, origin: tuple[float, float]) -> bool:
    c, kx, ky = _key(ch, origin)
    return any(index[(c, kx + dx, ky + dy)] for dx in (-1, 0, 1) for dy in (-1, 0, 1))


def _filled(page: pymupdf.Page, rects) -> int:
    fills = [pymupdf.Rect(d["rect"]) for d in page.get_drawings()
             if d.get("fill") is not None and tuple(round(v, 3) for v in d["fill"]) == tuple(float(v) for v in FILL)]
    missing = 0
    for r in rects:
        target = pymupdf.Rect(r)
        if not any(abs(f.x0 - target.x0) <= FILL_TOLERANCE and abs(f.y0 - target.y0) <= FILL_TOLERANCE
                   and abs(f.x1 - target.x1) <= FILL_TOLERANCE and abs(f.y1 - target.y1) <= FILL_TOLERANCE
                   for f in fills):
            missing += 1
    return missing


def _hidden_content_problems(doc: pymupdf.Document) -> list[str]:
    problems = []
    meta = {k: v for k, v in (doc.metadata or {}).items() if k not in ("format", "encryption") and v}
    if meta:
        problems.append(f"{len(meta)} metadata field(s) remain")
    if doc.get_xml_metadata():
        problems.append("XMP metadata remains")
    if doc.embfile_count():
        problems.append(f"{doc.embfile_count()} embedded file(s) remain")
    if doc.get_toc(simple=True):
        problems.append("outline entries remain")
    catalog = doc.pdf_catalog()
    for key in ("OpenAction", "AA", "StructTreeRoot", "AcroForm", "Names/JavaScript"):
        if doc.xref_get_key(catalog, key)[0] not in ("null", ""):
            problems.append(f"catalog entry {key} remains")
    for page in doc:
        n_annots = sum(1 for _ in page.annots())
        n_links = len(page.get_links())
        n_widgets = sum(1 for _ in page.widgets())
        if n_annots or n_links or n_widgets:
            problems.append(f"page {page.number + 1}: {n_annots} annotation(s), {n_links} link(s), "
                            f"{n_widgets} form field(s) remain")
        if doc.xref_get_key(page.xref, "AA")[0] not in ("null", ""):
            problems.append(f"page {page.number + 1}: automatic action remains")
    return problems


def verify_redacted_pdf(output: bytes, pages: Sequence[PageText], mappings: Sequence[PageMapping]) -> PdfVerification:
    problems: list[str] = []
    try:
        doc = pymupdf.open(stream=output, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        return PdfVerification(problems=(f"output could not be reopened ({type(exc).__name__})",))
    try:
        opened = not (doc.needs_pass or doc.is_encrypted or doc.is_repaired)
        if not opened:
            problems.append("output is encrypted or needed repair")

        structure = doc.page_count == len(pages) and all(
            abs(doc[p.number - 1].rect.width - p.width) < 0.01 and abs(doc[p.number - 1].rect.height - p.height) < 0.01
            for p in pages)
        if not structure:
            problems.append(f"output has {doc.page_count} page(s) or page sizes that differ from the source")
            return PdfVerification(opened=opened, problems=tuple(problems))

        by_page = {m.page: m for m in mappings}
        regions = removed = preserved = 0
        completed = targeted_ok = collateral_ok = True
        for src in pages:
            page = doc[src.number - 1]
            mapping = by_page.get(src.number)
            entities = [e for e in mapping.entities if e.status == "mapped"] if mapping else []
            rects = [r for e in entities for r in e.rects]
            targeted = mapping.targeted if mapping else frozenset()
            regions += len(rects)

            missing_fill = _filled(page, rects) if rects else 0
            redact_annots = sum(1 for a in page.annots() if a.type[0] == pymupdf.PDF_ANNOT_REDACT)
            if missing_fill or redact_annots:
                completed = False
                problems.append(f"page {src.number}: {missing_fill} redaction region(s) without fill, "
                                f"{redact_annots} unapplied redaction(s)")

            glyphs = _glyphs(page)
            index = Counter(_key(c, o) for c, o, _ in glyphs)
            for e in entities:
                inside = sum(1 for _, _, centre in glyphs
                             if any(r[0] <= centre[0] <= r[2] and r[1] <= centre[1] <= r[3] for r in e.rects))
                still = sum(1 for i in e.char_indices if _present(index, src.text[i], src.geoms[i].origin))
                if inside or still:
                    targeted_ok = False
                    problems.append(f"page {src.number}: {e.label} [{e.start}, {e.end}) still has "
                                    f"{max(inside, still)} extractable glyph(s) in its region")
                removed += len(e.char_indices)

            lost = 0
            for i, g in enumerate(src.geoms):
                if g is None or i in targeted:
                    continue
                if _present(index, src.text[i], g.origin):
                    preserved += 1
                else:
                    lost += 1
            if lost:
                collateral_ok = False
                problems.append(f"page {src.number}: {lost} unrelated glyph(s) missing from the output")

        hidden = _hidden_content_problems(doc)
        problems += hidden
        return PdfVerification(
            opened=opened, structure_preserved=True, redaction_completed=completed,
            targeted_text_removed=targeted_ok, unrelated_text_preserved=collateral_ok,
            hidden_content_removed=not hidden, regions_checked=regions, glyphs_removed=removed,
            glyphs_preserved=preserved, problems=tuple(problems),
        )
    finally:
        doc.close()
