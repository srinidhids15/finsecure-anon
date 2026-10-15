"""Anonymization: replace detected PII in text with label placeholders.

Public API::

    from finguard.anonymization import anonymize, detect_and_anonymize

See ``anonymizer`` for the transformation, ``verify`` for the integrity
check and ``result`` for the audit report.
"""

from finguard.anonymization.anonymizer import anonymize, detect_and_anonymize
from finguard.anonymization.errors import (
    AnonymizationError,
    IntegrityError,
    InvalidSpanError,
    OverlapError,
    SourceMismatchError,
    UnknownLabelError,
)
from finguard.anonymization.placeholders import DEFAULT_PLACEHOLDERS, resolve_placeholders
from finguard.anonymization.result import AnonymizationResult, Redaction, VerificationReport
from finguard.anonymization.verify import verify_anonymization

__all__ = [
    "AnonymizationError",
    "AnonymizationResult",
    "DEFAULT_PLACEHOLDERS",
    "IntegrityError",
    "InvalidSpanError",
    "OverlapError",
    "Redaction",
    "SourceMismatchError",
    "UnknownLabelError",
    "VerificationReport",
    "anonymize",
    "detect_and_anonymize",
    "resolve_placeholders",
    "verify_anonymization",
]
