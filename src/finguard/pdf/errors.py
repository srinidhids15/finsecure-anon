"""PDF processing errors.

Every message is safe to show to the user: it names a page number, a
label, offsets, counts or a structural feature, never extracted text or
an entity value.
"""

from __future__ import annotations


class PdfError(Exception):
    """Base class for PDF processing errors with a user-safe message."""


class PdfValidationError(PdfError):
    """Not a usable PDF: empty, too large, malformed, encrypted, too many pages."""


class UnsupportedPdfError(PdfError):
    """A valid PDF with content FinGuard cannot process safely (scanned pages,
    rotated or invisible text, optional-content layers, XFA forms, ...)."""


class PdfProcessingError(PdfError):
    """Detection, redaction or saving failed."""
