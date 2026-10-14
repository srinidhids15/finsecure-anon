"""Hybrid detection: merge rule spans and NER spans into one clean list.

The merge policy (which source wins on overlap, how ties are broken) is
decided and documented in the hybrid phase.
"""

from __future__ import annotations

from finguard.schema import Span


def merge(rule_spans: list[Span], ner_spans: list[Span]) -> list[Span]:
    """Combine both span lists, resolving overlaps explicitly."""
    raise NotImplementedError("Implemented in the hybrid detection phase")
