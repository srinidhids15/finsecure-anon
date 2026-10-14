"""Shared data contract used by every FinGuard component.

All detectors (rules, NER, hybrid) *produce* a list of ``Span`` objects, and
the anonymizer and evaluator *consume* them. Keeping this one small type in a
single place means components never need to know about each other's internals.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    """A labelled character range in a piece of text.

    Offsets follow Python slicing: ``text[start:end]`` is the entity, so
    ``end`` is exclusive.

    Attributes:
        start: Index of the first character (>= 0).
        end: Index one past the last character (> start).
        label: Entity type, e.g. ``"PERSON"``. The final label set is decided
            in the data phase, once the dataset has been inspected.
        source: Which detector produced it (``"rule"``, ``"ner"``, ``"gold"``,
            ...). Used by the hybrid merger and for debugging.
    """

    start: int
    end: int
    label: str
    source: str = "unknown"

    def __post_init__(self) -> None:
        # Validate on creation so a bad span fails loudly at its origin,
        # not later inside the anonymizer.
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise TypeError("Span start and end must be integers")
        if self.start < 0:
            raise ValueError(f"Span start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(
                f"Span end ({self.end}) must be greater than start ({self.start})"
            )
        if not self.label or not self.label.strip():
            raise ValueError("Span label must be a non-empty string")

    def text_in(self, text: str) -> str:
        """Return the substring this span covers, checking it fits ``text``."""
        if self.end > len(text):
            raise ValueError(
                f"Span end ({self.end}) exceeds text length ({len(text)})"
            )
        return text[self.start : self.end]

    def overlaps(self, other: Span) -> bool:
        """True if the two spans share at least one character."""
        return self.start < other.end and other.start < self.end
