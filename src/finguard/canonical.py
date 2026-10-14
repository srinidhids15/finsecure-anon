"""Span canonicalisation for spaCy conversion and evaluation.

Policy (one rule, applied to gold and predicted spans alike):

    **Trailing whitespace is not part of an entity.** A span's end is moved
    left past any trailing whitespace characters. Its start, its label and
    every other span are left exactly as they are.

Why: the dataset's credit-card blocks end with "\\n" and the gold spans
include it. spaCy cannot align an entity boundary that falls after a
whitespace token, and an evaluator comparing a gold span ending in "\\n" with
a prediction that does not would count a false mismatch.

This function is used *only* at conversion/evaluation time. Stored
annotations (raw workbook, Phase 2 JSONL, NER-ready JSONL) keep their
original offsets.
"""

from __future__ import annotations

from collections.abc import Sequence

from finguard.schema import Entity, Span


def canonical_span(text: str, span: Span) -> Span | None:
    """Return ``span`` with trailing whitespace removed from its end.

    Returns the same object when nothing changes, and None if the span
    consists only of whitespace. An ``Entity`` stays an ``Entity`` with its
    ``text`` updated; ``source`` is preserved.
    """
    if span.end > len(text):
        raise ValueError(f"span end ({span.end}) exceeds text length ({len(text)})")
    end = span.end
    while end > span.start and text[end - 1].isspace():
        end -= 1
    if end == span.end:
        return span
    if end == span.start:
        return None
    if isinstance(span, Entity):
        return Entity(span.start, end, span.label, span.source, text[span.start : end])
    return Span(span.start, end, span.label, span.source)


def canonical_spans(text: str, spans: Sequence[Span]) -> list[Span]:
    """Canonicalise every span; drop none silently.

    Raises ValueError if a span is whitespace only, because that indicates a
    data error that should be fixed, not hidden.
    """
    result = []
    for span in spans:
        canon = canonical_span(text, span)
        if canon is None:
            raise ValueError(f"span {span} contains only whitespace")
        result.append(canon)
    return result
