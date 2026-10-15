"""Physically redact mapped regions and remove hidden content, in memory.

Redaction (``apply_redactions``):

* one PyMuPDF redaction annotation per mapped rectangle, filled opaque
  black (``FILL``), with no overlay text;
* ``page.apply_redactions`` with text removal, image pixels under the
  rectangle blanked, and vector graphics left untouched (so table rules and
  backgrounds next to a value survive). The removed text is deleted from
  the page content stream, not covered.

Sanitization (``sanitize_document``) runs first, before text extraction,
so the text that is scanned is exactly the page content that remains. It
applies to every page whether or not it has detections:

* all annotations (comments, highlights, file attachments, ...), links
  and form fields are deleted. Their contents were not scanned;
* outlines (bookmarks), the logical-structure tree and catalog/page
  automatic actions are removed;
* ``Document.scrub`` removes metadata, XMP metadata, JavaScript, embedded
  files, thumbnails and cleans page content streams.

``save_pdf`` writes a new PDF to memory with ``garbage=4`` (unreferenced
objects, including replaced content streams, are dropped and duplicates
merged), ``deflate`` and ``clean``. The uploaded bytes are never written.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import pymupdf

from finguard.pdf.mapping import PdfEntity

FILL = (0, 0, 0)
APPLY_OPTIONS = dict(
    images=pymupdf.PDF_REDACT_IMAGE_PIXELS,
    graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
    text=pymupdf.PDF_REDACT_TEXT_REMOVE,
)
SAVE_OPTIONS = dict(garbage=4, deflate=True, clean=True)


@dataclass(frozen=True)
class SanitizationReport:
    annotations_removed: int
    links_removed: int
    form_fields_removed: int
    outline_removed: bool
    structure_tree_removed: bool
    actions_removed: int
    scrubbed: tuple[str, ...] = ("metadata", "xml_metadata", "javascript", "embedded_files",
                                 "attached_files", "thumbnails", "clean_pages")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["scrubbed"] = list(self.scrubbed)
        return d


def _null(doc: pymupdf.Document, xref: int, key: str) -> bool:
    if doc.xref_get_key(xref, key)[0] in ("null", ""):
        return False
    doc.xref_set_key(xref, key, "null")
    return True


def sanitize_document(doc: pymupdf.Document) -> SanitizationReport:
    annots = links = widgets = actions = 0
    for page in doc:
        for w in list(page.widgets()):
            page.delete_widget(w)
            widgets += 1
        for a in list(page.annots()):
            page.delete_annot(a)
            annots += 1
        for link in page.get_links():
            page.delete_link(link)
            links += 1
        actions += _null(doc, page.xref, "AA")
    catalog = doc.pdf_catalog()
    outline = bool(doc.get_toc(simple=True))
    if outline:
        doc.set_toc([])
    structure = _null(doc, catalog, "StructTreeRoot")
    _null(doc, catalog, "MarkInfo")
    actions += _null(doc, catalog, "OpenAction") + _null(doc, catalog, "AA")
    _null(doc, catalog, "AcroForm")
    doc.scrub(attached_files=True, clean_pages=True, embedded_files=True, hidden_text=False,
              javascript=True, metadata=True, redactions=False, remove_links=True, reset_fields=True,
              reset_responses=True, thumbnails=True, xml_metadata=True)
    doc.set_metadata({})
    doc.del_xml_metadata()
    return SanitizationReport(annots, links, widgets, outline, structure, actions)


def apply_redactions(doc: pymupdf.Document, entities: Iterable[PdfEntity]) -> int:
    """Add and apply one black redaction per rectangle; returns the count."""
    by_page: dict[int, list[PdfEntity]] = {}
    for e in entities:
        if e.status != "mapped":
            raise ValueError(f"page {e.page}: refusing to redact an entity with status {e.status!r}")
        by_page.setdefault(e.page, []).append(e)
    count = 0
    for number, items in sorted(by_page.items()):
        page = doc[number - 1]
        for e in items:
            for rect in e.rects:
                page.add_redact_annot(pymupdf.Rect(rect), fill=FILL, cross_out=False)
                count += 1
        page.apply_redactions(**APPLY_OPTIONS)
    return count


def save_pdf(doc: pymupdf.Document) -> bytes:
    return doc.tobytes(**SAVE_OPTIONS)
