"""In-memory PDF pipeline: validate → sanitize → extract → detect → map → redact → verify.

``process_pdf(data, detector)`` never writes to disk and never modifies
``data``. It raises ``PdfError`` subclasses for documents it cannot accept
(invalid, encrypted, scanned, layered, rotated or invisible text, over the
limits); their messages name pages and features only.

For accepted documents it returns a ``PdfResult``. ``result.output`` holds
the redacted PDF **only** when every detected entity was mapped, the
redactions were applied and saved, and independent verification passed;
otherwise it is None and ``result.status`` says which stage failed.
Detection runs once per page on that page's own text, so offsets are
always page-local and an entity cannot cross a page boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from finguard.pdf.errors import PdfProcessingError, UnsupportedPdfError
from finguard.pdf.extract import PageText, extract_page
from finguard.pdf.mapping import PageMapping, PdfEntity, map_page
from finguard.pdf.redact import SanitizationReport, apply_redactions, sanitize_document, save_pdf
from finguard.pdf.validate import (
    DEFAULT_LIMITS,
    HiddenContent,
    PdfLimits,
    inspect_document,
    open_pdf,
    reject_unsupported_structures,
    replacement_text_objects,
)
from finguard.pdf.verify import PdfVerification, verify_redacted_pdf

STATUS_VERIFIED = "verified"
STATUS_MAPPING_FAILED = "mapping_failed"
STATUS_REDACTION_FAILED = "redaction_failed"
STATUS_VERIFICATION_FAILED = "verification_failed"


@dataclass(frozen=True)
class PdfResult:
    status: str
    pages: tuple[PageText, ...]
    mappings: tuple[PageMapping, ...]
    hidden: HiddenContent
    sanitization: SanitizationReport | None = None
    verification: PdfVerification | None = None
    regions_applied: int = 0
    output: bytes | None = field(default=None, repr=False)
    problems: tuple[str, ...] = ()

    @property
    def verified(self) -> bool:
        return (self.status == STATUS_VERIFIED and self.output is not None
                and self.verification is not None and self.verification.verified)

    @property
    def entities(self) -> tuple[PdfEntity, ...]:
        return tuple(e for m in self.mappings for e in m.entities)


def _unsupported_pages(pages: Sequence[PageText]) -> None:
    bad = [p for p in pages if p.unsupported]
    if not bad:
        return
    if all(not p.text.strip() for p in pages):
        raise UnsupportedPdfError(
            "The PDF has no extractable text. It looks like a scanned or image-only document; "
            "OCR is not supported, so nothing could be detected or redacted.")
    details = "; ".join(f"page {p.number}: {', '.join(p.unsupported)}" for p in bad[:10])
    more = f" (and {len(bad) - 10} more page(s))" if len(bad) > 10 else ""
    raise UnsupportedPdfError(
        f"{len(bad)} of {len(pages)} page(s) cannot be processed safely: {details}{more}. "
        "The document was not redacted, because those pages could hold PII that would be left in place.")


def process_pdf(data: bytes, detector, limits: PdfLimits = DEFAULT_LIMITS) -> PdfResult:
    doc = open_pdf(data, limits)
    try:
        hidden = inspect_document(doc)
        reject_unsupported_structures(hidden)
        # Remove annotations, links, form fields, metadata, ... first, so the
        # extracted text is exactly the page content that remains in the output.
        try:
            sanitization = sanitize_document(doc)
        except Exception as exc:  # noqa: BLE001
            raise PdfProcessingError(f"Removing hidden content failed ({type(exc).__name__}).") from exc
        if replacement_text_objects(doc.tobytes(garbage=4)):
            raise UnsupportedPdfError(
                "The PDF contains marked-content replacement or alternate text (ActualText/Alt). That text is "
                "stored separately from the drawn glyphs and would survive redaction, so the PDF is not supported.")
        pages = tuple(extract_page(page) for page in doc)
        _unsupported_pages(pages)
        for p in pages:
            if len(p.text) > limits.max_chars_per_page:
                raise UnsupportedPdfError(f"Page {p.number} has {len(p.text):,} characters; the limit is "
                                          f"{limits.max_chars_per_page:,} per page.")
        total = sum(len(p.text) for p in pages)
        if total > limits.max_total_chars:
            raise UnsupportedPdfError(f"The PDF has {total:,} characters; the limit is {limits.max_total_chars:,}.")

        mappings = []
        for p in pages:
            try:
                entities = detector.detect(p.text) if p.text.strip() else []
            except Exception as exc:  # noqa: BLE001 - no text in the message
                raise PdfProcessingError(f"Detection failed on page {p.number} ({type(exc).__name__}).") from exc
            mappings.append(map_page(p, entities))
        mappings = tuple(mappings)

        problems = [m.error for m in mappings if m.error]
        problems += [f"page {e.page}: {e.label} [{e.start}, {e.end}) could not be mapped: {e.reason}"
                     for m in mappings for e in m.rejected]
        if problems:
            return PdfResult(STATUS_MAPPING_FAILED, pages, mappings, hidden, sanitization, problems=tuple(problems))

        try:
            applied = apply_redactions(doc, [e for m in mappings for e in m.entities])
            output = save_pdf(doc)
        except Exception as exc:  # noqa: BLE001
            return PdfResult(STATUS_REDACTION_FAILED, pages, mappings, hidden, sanitization,
                             problems=(f"redaction or saving failed ({type(exc).__name__})",))

        verification = verify_redacted_pdf(output, pages, mappings)
        if not verification.verified:
            return PdfResult(STATUS_VERIFICATION_FAILED, pages, mappings, hidden, sanitization, verification,
                             applied, None, verification.problems)
        done = tuple(PageMapping(m.page, tuple(replace(e, status="redacted") for e in m.entities)) for m in mappings)
        return PdfResult(STATUS_VERIFIED, pages, done, hidden, sanitization, verification, applied, output)
    finally:
        doc.close()
