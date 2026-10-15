"""Replace detected entities in text with label placeholders.

``anonymize(text, entities)`` consumes the entities returned by the frozen
hybrid detector (``detection.hybrid.HybridDetector``); it does no detection
of its own and no conflict resolution.

Steps, all against the original, unmodified ``text``:

1. **Validate** each entity: it is an ``Entity``, its label has a
   placeholder, its offsets fit ``text`` and ``entity.text`` equals
   ``text[start:end]``.
2. **Canonicalise** with ``finguard.canonical.canonical_span``: trailing
   whitespace is not part of an entity. The canonical span is replaced and
   the trailing whitespace stays in the output, so a credit-card block
   returned with its final newline is replaced by ``[CREDIT CARD REDACTED]``
   followed by that newline.
3. **Reject overlaps** between the detected spans. The hybrid detector
   already returns non-overlapping entities; an overlap here means the
   input did not come from it, and inventing a second policy would hide that.
4. **Replace in one left-to-right pass**, copying the text between spans
   unchanged and recording where each placeholder lands in the output. No
   offset is ever adjusted, so earlier replacements cannot shift later ones.
5. **Verify** the output against the original text and the requested
   entities (``verify.verify_anonymization``); a failure raises
   ``IntegrityError``.

Neither the input text nor the entities are modified. Nothing is logged or
written to disk, and error messages contain offsets and labels only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from finguard.anonymization.errors import (
    IntegrityError,
    InvalidSpanError,
    OverlapError,
    SourceMismatchError,
    UnknownLabelError,
)
from finguard.anonymization.placeholders import resolve_placeholders
from finguard.anonymization.result import AnonymizationResult, Redaction
from finguard.anonymization.verify import verify_anonymization
from finguard.canonical import canonical_span
from finguard.schema import Entity

NO_ENTITIES_WARNING = (
    "no entities were supplied; the text is returned unchanged. "
    "This does not show that the text contains no PII."
)


def _sources(entity: Entity) -> tuple[str, ...]:
    return tuple(s for s in entity.source.split("|") if s)


def _validated(text: str, entities: Sequence[Entity], placeholders: Mapping[str, str]):
    """Return ``(input_index, entity, canonical_end)`` sorted by position."""
    planned = []
    for i, e in enumerate(entities):
        if not isinstance(e, Entity):
            raise TypeError(f"entity #{i} is a {type(e).__name__}, expected finguard.schema.Entity")
        where = f"entity #{i} ({e.label!r}, [{e.start}, {e.end}))"
        if e.label not in placeholders:
            raise UnknownLabelError(f"{where}: label has no placeholder")
        if e.end > len(text):
            raise InvalidSpanError(f"{where}: end exceeds text length {len(text)}")
        if text[e.start:e.end] != e.text:
            raise SourceMismatchError(f"{where}: entity text does not match the source text at these offsets")
        canon = canonical_span(text, e)
        if canon is None:
            raise InvalidSpanError(f"{where}: span is whitespace only")
        planned.append((i, e, canon.end))
    planned.sort(key=lambda p: (p[1].start, p[1].end, p[1].label))
    for (i, a, _), (j, b, _) in zip(planned, planned[1:]):
        if b.start < a.end:
            raise OverlapError(
                f"entity #{i} ({a.label!r}, [{a.start}, {a.end})) overlaps entity #{j} "
                f"({b.label!r}, [{b.start}, {b.end})); resolve overlaps in the detector before anonymizing"
            )
    return planned


def anonymize(
    text: str,
    entities: Sequence[Entity],
    placeholders: Mapping[str, str] | None = None,
) -> AnonymizationResult:
    """Replace every entity in ``text`` with its label's placeholder.

    Args:
        text: The original input text; entity offsets refer to it.
        entities: Non-overlapping ``Entity`` objects, normally
            ``HybridDetector().detect(text)``.
        placeholders: Optional overrides of ``DEFAULT_PLACEHOLDERS`` for known
            labels.

    Returns:
        An ``AnonymizationResult`` whose ``verification`` has passed.

    Raises:
        TypeError: ``text`` is not a string or an entity is not an ``Entity``.
        UnknownLabelError, InvalidSpanError, SourceMismatchError, OverlapError:
            the entities cannot be applied to ``text`` as given.
        IntegrityError: the output failed verification (a bug, never expected).
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    entities = tuple(entities)
    mapping = resolve_placeholders(placeholders)
    planned = _validated(text, entities, mapping)

    pieces: list[str] = []
    redactions: list[Redaction] = []
    cursor = out_len = 0
    for _, e, canon_end in planned:
        gap = text[cursor:e.start]
        pieces.append(gap)
        out_len += len(gap)
        placeholder = mapping[e.label]
        pieces.append(placeholder)
        redactions.append(Redaction(
            label=e.label,
            start=e.start,
            end=canon_end,
            detected_end=e.end,
            output_start=out_len,
            output_end=out_len + len(placeholder),
            placeholder=placeholder,
            sources=_sources(e),
        ))
        out_len += len(placeholder)
        cursor = canon_end  # whitespace in [canon_end, e.end) is copied with the next gap
    pieces.append(text[cursor:])

    warnings = []
    if not entities:
        warnings.append(NO_ENTITIES_WARNING)
    kept = [r for r in redactions if r.whitespace_kept]
    if kept:
        labels = sorted({r.label for r in kept})
        warnings.append(
            f"{len(kept)} span(s) ({', '.join(labels)}) ended in whitespace; per the canonical span "
            "policy that whitespace was kept outside the placeholder"
        )

    result = AnonymizationResult(
        sanitized_text="".join(pieces),
        input_length=len(text),
        redactions=tuple(redactions),
        warnings=tuple(warnings),
    )
    report = verify_anonymization(text, entities, result, mapping)
    if not report.detected_spans_replaced:
        raise IntegrityError("output failed span-replacement verification: " + "; ".join(report.problems))
    return replace(result, verification=report)


def detect_and_anonymize(
    text: str,
    detector=None,
    placeholders: Mapping[str, str] | None = None,
) -> AnonymizationResult:
    """Run ``detector`` (default: the frozen ``HybridDetector``) and anonymize its output."""
    if detector is None:
        from finguard.detection.hybrid import HybridDetector

        detector = HybridDetector()
    return anonymize(text, detector.detect(text), placeholders)
