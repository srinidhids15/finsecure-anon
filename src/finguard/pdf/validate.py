"""Validate uploaded PDF bytes and inventory content outside the page text.

``open_pdf(data, limits)`` opens the bytes in memory with PyMuPDF after
checking size and file signature, and rejects documents that are
malformed (needed repair), encrypted, empty or over the page limit. The
caller's bytes are never modified.

``inspect_document(doc)`` counts structures that can hold sensitive values
outside the page text: metadata, XMP metadata, annotations, links, form
fields, embedded files, JavaScript and automatic actions, outlines and the
logical-structure tree. These are all removed during redaction
(``redact.sanitize_document``). Optional-content layers and XFA forms
cannot be sanitized reliably here, so their presence rejects the document.

PyMuPDF never executes PDF JavaScript, follows links or opens embedded
files; they are only counted and removed.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import pymupdf

from finguard.pdf.errors import PdfValidationError, UnsupportedPdfError

# MuPDF prints parser warnings to stderr by default; keep the console quiet.
pymupdf.TOOLS.mupdf_display_errors(False)
pymupdf.TOOLS.mupdf_display_warnings(False)


@dataclass(frozen=True)
class PdfLimits:
    """Resource limits, checked before and during processing."""

    max_bytes: int = 20 * 1024 * 1024
    max_pages: int = 100
    max_chars_per_page: int = 100_000
    max_total_chars: int = 500_000


DEFAULT_LIMITS = PdfLimits()


def open_pdf(data: bytes, limits: PdfLimits = DEFAULT_LIMITS) -> pymupdf.Document:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise PdfValidationError("The upload must be file bytes.")
    data = bytes(data)
    if not data:
        raise PdfValidationError("The file is empty.")
    if len(data) > limits.max_bytes:
        raise PdfValidationError(
            f"The file is {len(data) / 1_048_576:.1f} MB; the limit is {limits.max_bytes / 1_048_576:.0f} MB.")
    if b"%PDF-" not in data[:1024]:
        raise PdfValidationError("This is not a PDF file (no PDF header).")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - parser messages are not shown
        raise PdfValidationError(f"The PDF could not be opened; it appears to be damaged ({type(exc).__name__}).") from exc
    try:
        if not doc.is_pdf:
            raise PdfValidationError("This is not a PDF file.")
        if doc.needs_pass or doc.is_encrypted or doc.metadata.get("encryption"):
            raise PdfValidationError("Encrypted or password-protected PDFs are not supported. "
                                     "Remove the protection and upload again.")
        if doc.is_repaired:
            raise PdfValidationError("The PDF is damaged (its structure had to be repaired to open it). "
                                     "Re-save it from the original application and upload again.")
        if doc.page_count == 0:
            raise PdfValidationError("The PDF has no pages.")
        if doc.page_count > limits.max_pages:
            raise PdfValidationError(f"The PDF has {doc.page_count} pages; the limit is {limits.max_pages}.")
    except PdfValidationError:
        doc.close()
        raise
    return doc


@dataclass(frozen=True)
class HiddenContent:
    """Counts of non-page-text structures (no values)."""

    metadata_fields: int
    xml_metadata: bool
    annotations: int
    links: int
    form_fields: int
    embedded_files: int
    javascript: bool
    automatic_actions: int
    outline_entries: int
    structure_tree: bool
    optional_content_groups: int
    xfa_form: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _catalog_has(doc: pymupdf.Document, key: str) -> bool:
    kind, _ = doc.xref_get_key(doc.pdf_catalog(), key)
    return kind not in ("null", "")


def inspect_document(doc: pymupdf.Document) -> HiddenContent:
    annotations = links = widgets = page_actions = 0
    for page in doc:
        widgets += sum(1 for _ in page.widgets())
        annotations += sum(1 for a in page.annots() if a.type[0] != pymupdf.PDF_ANNOT_WIDGET)
        links += len(page.get_links())
        page_actions += doc.xref_get_key(page.xref, "AA")[0] not in ("null", "")
    metadata = {k: v for k, v in (doc.metadata or {}).items() if k not in ("format", "encryption") and v}
    catalog = doc.pdf_catalog()
    names_js = doc.xref_get_key(catalog, "Names/JavaScript")[0] not in ("null", "")
    open_action_js = "JavaScript" in doc.xref_get_key(catalog, "OpenAction")[1]
    return HiddenContent(
        metadata_fields=len(metadata),
        xml_metadata=bool(doc.get_xml_metadata()),
        annotations=annotations,
        links=links,
        form_fields=widgets,
        embedded_files=doc.embfile_count(),
        javascript=names_js or open_action_js,
        automatic_actions=page_actions + _catalog_has(doc, "OpenAction") + _catalog_has(doc, "AA"),
        outline_entries=len(doc.get_toc(simple=True)),
        structure_tree=_catalog_has(doc, "StructTreeRoot"),
        optional_content_groups=len(doc.get_ocgs() or {}),
        xfa_form=doc.xref_get_key(doc.pdf_catalog(), "AcroForm/XFA")[0] not in ("null", ""),
    )


# Marked-content strings that carry text outside the drawn glyphs (replacement,
# alternate and expansion text). Redaction removes glyphs, not these strings.
REPLACEMENT_TEXT = re.compile(rb"/(?:ActualText|Alt|E)\s*[(<]")


def replacement_text_objects(data: bytes) -> int:
    """Number of objects or non-image streams in ``data`` holding ActualText,
    Alt or E strings. Callers pass a garbage-collected save, so only objects
    still referenced (not the removed structure tree) are counted."""
    count = 0
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for xref in range(1, doc.xref_length()):
            source = doc.xref_object(xref, compressed=False).encode("latin-1", "replace")
            stream = b""
            if doc.xref_is_stream(xref) and doc.xref_get_key(xref, "Subtype")[1] != "/Image":
                stream = doc.xref_stream(xref) or b""
            count += bool(REPLACEMENT_TEXT.search(source) or REPLACEMENT_TEXT.search(stream))
    return count


def reject_unsupported_structures(hidden: HiddenContent) -> None:
    if hidden.optional_content_groups:
        raise UnsupportedPdfError(
            f"The PDF uses {hidden.optional_content_groups} optional-content layer(s). Hidden layers can hold "
            "text that is not reliably extracted, so layered PDFs are not supported. Flatten or print the "
            "document to a new PDF and upload that.")
    if hidden.xfa_form:
        raise UnsupportedPdfError("The PDF contains an XFA form, which is not supported.")
