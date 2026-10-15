"""Auditable anonymization results.

None of these types stores an original entity value. A ``Redaction`` says
*what kind* of entity was replaced, *where* (offsets in the original and in
the sanitized text), *with what* placeholder and *which detector(s)* found
it. The sanitized text is held by ``AnonymizationResult`` but kept out of
its ``repr`` and out of ``to_dict()`` unless explicitly requested, because
it may still contain PII the detector missed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from finguard.config import ENTITY_LABELS

COMPLETENESS_NOTE = (
    "Verification covers detected spans only. Complete PII removal cannot be "
    "guaranteed by span verification: the detector can miss sensitive text."
)


@dataclass(frozen=True)
class Redaction:
    """One replacement, in original-text and sanitized-text coordinates.

    ``start``/``end`` is the canonical span that was replaced (trailing
    whitespace excluded, ``finguard.canonical``). ``detected_end`` is the end
    the detector returned; ``text[end:detected_end]`` is whitespace that was
    kept in the output. ``output_start``/``output_end`` locate the
    placeholder in the sanitized text.
    """

    label: str
    start: int
    end: int
    detected_end: int
    output_start: int
    output_end: int
    placeholder: str
    sources: tuple[str, ...]

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def whitespace_kept(self) -> int:
        return self.detected_end - self.end

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "detected_end": self.detected_end,
            "length": self.length,
            "output_start": self.output_start,
            "output_end": self.output_end,
            "placeholder": self.placeholder,
            "sources": list(self.sources),
        }


@dataclass(frozen=True)
class VerificationReport:
    """Outcome of ``verify_anonymization``.

    ``detected_spans_replaced`` is the verified property.
    ``complete_pii_removal_guaranteed`` is always False: span verification
    cannot establish it.
    """

    detected_spans_replaced: bool
    checked_redactions: int
    problems: tuple[str, ...] = ()
    complete_pii_removal_guaranteed: bool = False
    note: str = COMPLETENESS_NOTE

    def to_dict(self) -> dict:
        return {
            "detected_spans_replaced": self.detected_spans_replaced,
            "complete_pii_removal_guaranteed": self.complete_pii_removal_guaranteed,
            "checked_redactions": self.checked_redactions,
            "problems": list(self.problems),
            "note": self.note,
        }


@dataclass(frozen=True)
class AnonymizationResult:
    sanitized_text: str = field(repr=False)
    input_length: int
    redactions: tuple[Redaction, ...]
    warnings: tuple[str, ...] = ()
    verification: VerificationReport | None = None

    @property
    def redaction_count(self) -> int:
        return len(self.redactions)

    @property
    def counts_by_label(self) -> dict[str, int]:
        counts = dict.fromkeys(ENTITY_LABELS, 0)
        for r in self.redactions:
            counts[r.label] += 1
        return counts

    def to_dict(self, include_sanitized_text: bool = False) -> dict:
        """Audit report. Contains no original values; the sanitized text is
        included only on request (it may hold PII the detector missed)."""
        report = {
            "input_length": self.input_length,
            "output_length": len(self.sanitized_text),
            "redaction_count": self.redaction_count,
            "counts_by_label": self.counts_by_label,
            "redactions": [r.to_dict() for r in self.redactions],
            "warnings": list(self.warnings),
            "verification": self.verification.to_dict() if self.verification else None,
        }
        if include_sanitized_text:
            report["sanitized_text"] = self.sanitized_text
        return report
