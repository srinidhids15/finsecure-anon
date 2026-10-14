"""Load a trained spaCy model and turn its entities into ``Span`` objects."""

from __future__ import annotations

from finguard.schema import Span


def detect(text: str) -> list[Span]:
    """Return spans predicted by the NER model, with ``source="ner"``."""
    raise NotImplementedError("Implemented in the NER inference phase")
