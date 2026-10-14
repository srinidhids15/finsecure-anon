"""Entity-level precision, recall and F1 (overall and per label).

Planned to report exact-match scores and, separately, a partial-overlap
score, so results are not inflated by a lenient definition.
"""

from __future__ import annotations

from finguard.schema import Span


def score(gold: list[list[Span]], predicted: list[list[Span]]) -> dict:
    """Return precision/recall/F1 per label and overall."""
    raise NotImplementedError("Implemented in the evaluation phase")
