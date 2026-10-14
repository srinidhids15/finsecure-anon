"""Check that annotations are usable before they reach training/evaluation.

Planned checks: offsets inside the text, no empty labels, explicit handling
of overlapping spans, and a train/dev/test split where the test set stays
untouched.
"""

from __future__ import annotations

from finguard.schema import Span


def validate_example(text: str, spans: list[Span]) -> list[str]:
    """Return a list of human-readable problems (empty list means valid)."""
    raise NotImplementedError("Implemented in the data-validation phase")
