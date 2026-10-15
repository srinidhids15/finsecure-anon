"""Errors raised by the anonymizer.

Messages identify a problem by entity index (position in the caller's
list), label and character offsets only. They never contain the entity's
text or any part of the input document, so an exception that ends up in a
log or a traceback does not leak PII.
"""

from __future__ import annotations


class AnonymizationError(ValueError):
    """Base class: the requested redaction cannot be performed safely."""


class UnknownLabelError(AnonymizationError):
    """An entity label has no placeholder (no silent fallback is used)."""


class InvalidSpanError(AnonymizationError):
    """An entity's offsets do not fit the input text, or it is whitespace only."""


class SourceMismatchError(AnonymizationError):
    """``entity.text`` differs from ``text[entity.start:entity.end]``."""


class OverlapError(AnonymizationError):
    """Two entities overlap. Overlaps must be resolved by the detector
    (``detection.hybrid``), not by a second policy in the anonymizer."""


class IntegrityError(AnonymizationError):
    """The produced output failed its own span-replacement verification."""
