"""Reproducible corrections for missing gold annotations.

Problem (verified on the training workbook; the official test workbook was
not used): the dataset generator deleted some "." characters from the
finished documents. When a dot inside an inserted PII value was deleted,
the value no longer matched its source column, and the generator's
annotation step (which marked exact copies of the column value) skipped it:

    Email column  perrysteven@example.org   text  perrysteven@exampleorg
    URL column    https://www.keith.com/    text  https://wwwkeith.com/
    Address       ... Apt. 978 ...          text  ... Apt 978 ...
    Name          Dr. Ann Clarke MD         text  Dr Ann Clarke MD

Correction rule ``dot_dropped_variant``: an occurrence is added as a gold span
with the column's label only if **all** of these hold:

  1. it equals the record's own source value with one or more "." removed,
     and every other character identical and in order;
  2. it is not inside a larger token (no letter/digit/"_"/"@" directly
     before or after);
  3. it does not overlap an existing gold span or another correction.

Occurrences failing (2) or (3) are *not* labelled; they are returned as
``ReviewItem``s. The rule reads only the text and the source columns: never
model or rule-detector predictions, and never the official test file.
Original annotations are never edited; corrections are stored separately
with their provenance.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

from finguard.schema import Span

RULE_ID = "dot_dropped_variant"

# Labels whose source values contain "." in this dataset (email, url: always;
# address: "Apt."/"Ste."; name: "Mr."/"Dr."/"Jr.").
DEFAULT_LABELS: tuple[str, ...] = ("email", "url", "address", "name")


@dataclass(frozen=True)
class Correction:
    """One added gold span and why it was added."""

    record_id: str
    start: int
    end: int
    label: str
    text: str  # the corrected entity text, exactly text[start:end]
    source_value: str  # the source-column value it derives from
    dots_removed: int
    rule: str = RULE_ID

    @property
    def reason(self) -> str:
        return (
            f"text contains the record's {self.label} source value with "
            f"{self.dots_removed} '.' removed by the generator's dot deletion, "
            "so the exact-match annotation step left it unlabelled"
        )

    def to_span(self) -> Span:
        return Span(self.start, self.end, self.label, source=f"correction:{self.rule}")

    def to_dict(self) -> dict:
        return {**asdict(self), "reason": self.reason}


@dataclass(frozen=True)
class ReviewItem:
    """A candidate that could not be labelled unambiguously."""

    record_id: str
    start: int
    end: int
    label: str
    problem: str


def dot_variant_pattern(value: str) -> re.Pattern[str] | None:
    """Regex matching ``value`` with any of its dots optionally missing.

    Returns None for values without a dot (they cannot have this defect).
    """
    if "." not in value:
        return None
    return re.compile("".join(r"\.?" if c == "." else re.escape(c) for c in value))


def _inside_token(text: str, start: int, end: int) -> bool:
    def joins(ch: str) -> bool:
        return ch.isalnum() or ch in "_@"

    return (start > 0 and joins(text[start - 1])) or (end < len(text) and joins(text[end]))


def find_dot_drop_corrections(
    record_id: str,
    text: str,
    spans: Sequence[Span],
    source_values: dict[str, str],
    labels: Iterable[str] = DEFAULT_LABELS,
) -> tuple[list[Correction], list[ReviewItem]]:
    """Return corrections (unambiguous) and review items (ambiguous) for one record."""
    candidates: list[Correction] = []
    review: list[ReviewItem] = []

    for label in labels:
        value = source_values.get(label, "")
        pattern = dot_variant_pattern(value)
        if pattern is None:
            continue
        for m in pattern.finditer(text):
            found = m.group(0)
            if found == value:
                continue  # exact form: annotated by the dataset itself (Phase 2 verified)
            start, end = m.span()
            if _inside_token(text, start, end):
                review.append(ReviewItem(record_id, start, end, label, "embedded in a larger token"))
                continue
            clash = [s for s in spans if s.start < end and start < s.end]
            if clash:
                review.append(ReviewItem(
                    record_id, start, end, label,
                    "overlaps gold " + ", ".join(sorted({s.label for s in clash})),
                ))
                continue
            candidates.append(Correction(
                record_id, start, end, label, found, value,
                dots_removed=value.count(".") - found.count("."),
            ))

    # Corrections must not overlap each other either; if they do, none of
    # the overlapping ones is applied.
    candidates.sort(key=lambda c: (c.start, c.end))
    clashing: set[int] = set()
    for i, a in enumerate(candidates):
        for j in range(i + 1, len(candidates)):
            b = candidates[j]
            if b.start >= a.end:
                break
            clashing.update((i, j))
    accepted = []
    for i, c in enumerate(candidates):
        if i in clashing:
            review.append(ReviewItem(record_id, c.start, c.end, c.label, "overlaps another correction"))
        else:
            accepted.append(c)
    return accepted, review


def apply_corrections(spans: Sequence[Span], corrections: Sequence[Correction]) -> tuple[Span, ...]:
    """Return original spans plus corrected spans, sorted. Never edits the originals.

    Corrected spans keep ``source="correction:<rule>"``, so provenance travels
    with them.
    """
    merged = list(spans) + [c.to_span() for c in corrections]
    merged.sort(key=lambda s: (s.start, s.end))
    for a, b in zip(merged, merged[1:]):
        if b.start < a.end:
            raise ValueError(f"correction would create overlap: {a} / {b}")
    return tuple(merged)
