"""Placeholder policy: one fixed replacement string per entity label.

``DEFAULT_PLACEHOLDERS`` covers exactly ``config.ENTITY_LABELS``. Callers
may override individual placeholders, but only for known labels. An entity
whose label has no placeholder raises ``UnknownLabelError``; there is no
generic fallback, because a mislabelled entity is a bug to surface, not to
paper over.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from finguard.anonymization.errors import UnknownLabelError
from finguard.config import ENTITY_LABELS

DEFAULT_PLACEHOLDERS: Mapping[str, str] = MappingProxyType({
    "name": "[NAME REDACTED]",
    "email": "[EMAIL REDACTED]",
    "phone": "[PHONE NUMBER REDACTED]",
    "address": "[ADDRESS REDACTED]",
    "ssn": "[SSN REDACTED]",
    "credit_card": "[CREDIT CARD REDACTED]",
    "company": "[COMPANY NAME REDACTED]",
    "url": "[URL REDACTED]",
})

if set(DEFAULT_PLACEHOLDERS) != set(ENTITY_LABELS):  # keep the policy complete if labels change
    raise RuntimeError("DEFAULT_PLACEHOLDERS must cover exactly config.ENTITY_LABELS")


def resolve_placeholders(overrides: Mapping[str, str] | None = None) -> Mapping[str, str]:
    """Return the defaults with ``overrides`` applied, read-only.

    Raises ``UnknownLabelError`` for a label outside ``ENTITY_LABELS`` and
    ``ValueError`` for an empty or non-string placeholder.
    """
    if overrides is None:
        return DEFAULT_PLACEHOLDERS
    if not isinstance(overrides, Mapping):
        raise TypeError(f"placeholders must be a mapping, got {type(overrides).__name__}")
    unknown = sorted(str(k) for k in overrides if k not in DEFAULT_PLACEHOLDERS)
    if unknown:
        raise UnknownLabelError(f"placeholders given for unknown labels: {unknown}")
    for label, value in overrides.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"placeholder for {label!r} must be a non-empty string")
    return MappingProxyType({**DEFAULT_PLACEHOLDERS, **overrides})
