"""Replace each detected span with a mask, a label tag, or a Faker value.

Replacement is done right-to-left so earlier offsets stay valid.
"""

from __future__ import annotations

from finguard.schema import Span


def anonymize(text: str, spans: list[Span], strategy: str = "tag") -> str:
    """Return ``text`` with every span replaced according to ``strategy``."""
    raise NotImplementedError("Implemented in the anonymization phase")
