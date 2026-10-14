"""Rule-based detection: regular expressions (plus checks such as Luhn) for
structured PII like account numbers, card numbers, emails and phone numbers.

Rules are precise on fixed formats; the NER model covers free-form entities
such as names and addresses.
"""

from __future__ import annotations

from finguard.schema import Span


def detect(text: str) -> list[Span]:
    """Return spans found by regex rules, with ``source="rule"``."""
    raise NotImplementedError("Implemented in the rule-based detection phase")
