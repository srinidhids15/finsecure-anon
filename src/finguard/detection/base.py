"""The interface every detector implements.

The rule detector (Phase 3), the spaCy detector (Phase 5) and the hybrid
detector (Phase 6) all satisfy ``Detector``. The app and the evaluator can
then use any of them without knowing which one they have.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from finguard.schema import Entity, Span


@runtime_checkable
class Detector(Protocol):
    """Anything with a ``name`` and a ``detect(text) -> list[Entity]`` method.

    Contract for implementations:
      * offsets refer to the exact ``text`` passed in (no normalisation);
      * ``entity.text == text[entity.start:entity.end]`` for every result;
      * results are sorted by ``(start, end, label)``;
      * overlapping results are returned, not dropped. Resolving them is
        the hybrid merger's job, which makes it a visible decision.
    """

    name: str

    def detect(self, text: str) -> list[Entity]: ...


def sort_entities(entities: Sequence[Entity]) -> list[Entity]:
    return sorted(entities, key=lambda e: (e.start, e.end, e.label, e.source))


def overlapping_pairs(spans: Sequence[Span]) -> list[tuple[Span, Span]]:
    """Every pair of spans that share at least one character."""
    ordered = sorted(spans, key=lambda s: (s.start, s.end))
    pairs = []
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            if b.start >= a.end:
                break
            pairs.append((a, b))
    return pairs
