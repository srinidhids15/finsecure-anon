"""Load a trained spaCy model and turn its entities into ``Entity`` objects.

Will implement ``finguard.detection.base.Detector`` (``name`` plus
``detect(text) -> list[Entity]``), the same interface as the rule detector,
so the hybrid merger and the app can treat both detectors alike.
"""

from __future__ import annotations

from finguard.schema import Entity


def detect(text: str) -> list[Entity]:
    """Return entities predicted by the NER model, with ``source="ner"``."""
    raise NotImplementedError("Implemented in the NER inference phase")
