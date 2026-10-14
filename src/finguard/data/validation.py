"""Quality checks for annotated records.

Each problem becomes an ``Issue`` with a code and a severity:

* ``error``: the record cannot be trusted as training or evaluation data
  (bad offsets, wrong substring, overlapping spans, ...).
* ``warning``: worth knowing about, but the record is still usable
  (e.g. a span that includes a trailing newline).

Issue details contain offsets, labels and lengths, **never the document
text**, so reports can be shared without exposing content.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from finguard.config import ENTITY_LABELS
from finguard.data.annotations import AnnotationParseError, parse_annotation_string
from finguard.data.loader import RawRecord
from finguard.schema import Span

ERROR = "error"
WARNING = "warning"

# Issue codes, kept in one place so reports and tests use the same names.
MISSING_TEXT = "MISSING_TEXT"
PARSE_ERROR = "PARSE_ERROR"
EMPTY_ANNOTATIONS = "EMPTY_ANNOTATIONS"
MALFORMED_SPAN = "MALFORMED_SPAN"
INVALID_LABEL = "INVALID_LABEL"
INVALID_OFFSET = "INVALID_OFFSET"
DUPLICATE_SPAN = "DUPLICATE_SPAN"
SUBSTRING_MISMATCH = "SUBSTRING_MISMATCH"
OVERLAPPING_SPANS = "OVERLAPPING_SPANS"
BOUNDARY_WHITESPACE = "BOUNDARY_WHITESPACE"
UNANNOTATED_VALUE = "UNANNOTATED_VALUE"
DUPLICATE_RECORD = "DUPLICATE_RECORD"
CROSS_SPLIT_DUPLICATE = "CROSS_SPLIT_DUPLICATE"


@dataclass(frozen=True)
class Issue:
    record_id: str
    code: str
    severity: str
    detail: str


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating one record."""

    record_id: str
    spans: tuple[Span, ...]  # spans that passed every per-span check, sorted
    issues: tuple[Issue, ...]

    @property
    def is_valid(self) -> bool:
        return not any(i.severity == ERROR for i in self.issues)


def _is_int(value: object) -> bool:
    # bool is a subclass of int in Python; (True, 5, 'name') is not a span.
    return isinstance(value, int) and not isinstance(value, bool)


def validate_record(
    record_id: str,
    text: str,
    annotation: str,
    source_values: dict[str, str] | None = None,
    labels: Iterable[str] = ENTITY_LABELS,
) -> ValidationResult:
    """Run every per-record check.

    Args:
        record_id: Identifier used in issue reports.
        text: The document text (already repaired, if repair is used).
        annotation: The raw annotation string.
        source_values: Optional ``label -> inserted value`` map. When given,
            each span must extract exactly that value (substring check), and
            any occurrence of the value not covered by a span is flagged.
        labels: Allowed entity labels.
    """
    allowed = set(labels)
    issues: list[Issue] = []

    def add(code: str, severity: str, detail: str) -> None:
        issues.append(Issue(record_id, code, severity, detail))

    if not text or not text.strip():
        add(MISSING_TEXT, ERROR, "text is empty or whitespace only")
        return ValidationResult(record_id, (), tuple(issues))

    try:
        items = parse_annotation_string(annotation)
    except AnnotationParseError as exc:
        add(PARSE_ERROR, ERROR, str(exc))
        return ValidationResult(record_id, (), tuple(issues))

    if not items:
        add(EMPTY_ANNOTATIONS, WARNING, "record has no annotated entities")

    good: list[Span] = []
    seen: set[tuple[int, int, str]] = set()
    for idx, item in enumerate(items):
        # --- structure: (int, int, str) ---
        if not isinstance(item, (tuple, list)) or len(item) != 3:
            add(MALFORMED_SPAN, ERROR, f"item {idx}: expected (start, end, label)")
            continue
        start, end, label = item
        if not (_is_int(start) and _is_int(end) and isinstance(label, str)):
            add(MALFORMED_SPAN, ERROR, f"item {idx}: expected (int, int, str)")
            continue
        # --- label ---
        if label not in allowed:
            add(INVALID_LABEL, ERROR, f"item {idx}: label {label!r} not in allowed set")
            continue
        # --- offsets ---
        if start < 0 or end <= start or end > len(text):
            add(
                INVALID_OFFSET,
                ERROR,
                f"item {idx}: ({start}, {end}) invalid for text length {len(text)}",
            )
            continue
        key = (start, end, label)
        if key in seen:
            add(DUPLICATE_SPAN, ERROR, f"item {idx}: ({start}, {end}, {label}) repeated")
            continue
        seen.add(key)
        # --- substring verification ---
        covered = text[start:end]
        if source_values is not None:
            expected = source_values.get(label, "")
            if covered != expected:
                add(
                    SUBSTRING_MISMATCH,
                    ERROR,
                    f"item {idx}: ({start}, {end}, {label}) does not extract the "
                    f"{label} value (len {len(covered)} vs {len(expected)})",
                )
                continue
        if covered != covered.strip():
            add(
                BOUNDARY_WHITESPACE,
                WARNING,
                f"item {idx}: ({start}, {end}, {label}) starts or ends with whitespace",
            )
        good.append(Span(start, end, label, source="gold"))

    good.sort(key=lambda s: (s.start, s.end))
    issues.extend(
        Issue(record_id, OVERLAPPING_SPANS, ERROR, d) for d in find_overlaps(good)
    )
    if source_values is not None:
        issues.extend(
            Issue(record_id, UNANNOTATED_VALUE, WARNING, d)
            for d in find_unannotated_values(text, good, source_values)
        )
    return ValidationResult(record_id, tuple(good), tuple(issues))


def find_overlaps(spans: Sequence[Span]) -> list[str]:
    """Describe every pair of overlapping spans.

    Policy: overlaps are an *error*. A character can have only one gold
    label for spaCy NER, so we do not silently pick one span over the other.
    """
    ordered = sorted(spans, key=lambda s: (s.start, s.end))
    found = []
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            if b.start >= a.end:
                break  # sorted by start: no later span can overlap a
            found.append(
                f"({a.start}, {a.end}, {a.label}) overlaps ({b.start}, {b.end}, {b.label})"
            )
    return found


def find_unannotated_values(
    text: str, spans: Sequence[Span], source_values: dict[str, str]
) -> list[str]:
    """Find places where an inserted PII value appears but no span covers it.

    Such gaps mean the gold labels are incomplete: a model that correctly
    finds the value would be scored as a false positive.
    """
    found = []
    for label, value in source_values.items():
        if not value:
            continue
        pos = text.find(value)
        while pos != -1:
            end = pos + len(value)
            if not any(s.start <= pos and end <= s.end for s in spans):
                found.append(f"{label} value at ({pos}, {end}) has no annotation")
            pos = text.find(value, pos + 1)
    return found


def validate_raw_record(record: RawRecord, text: str | None = None) -> ValidationResult:
    """Validate a loaded spreadsheet row. ``text`` overrides ``record.text``
    (used to validate the repaired text)."""
    return validate_record(
        record.record_id,
        record.text if text is None else text,
        record.annotation,
        record.source_values,
    )


def find_duplicate_texts(records: Sequence[RawRecord]) -> list[Issue]:
    """Flag every record whose text repeats an earlier record in the same file.

    The first occurrence is kept; later copies are errors, because
    duplicates would count twice in training and inflate scores.
    """
    first_seen: dict[str, str] = {}
    issues = []
    for rec in records:
        if not rec.text:
            continue  # empty texts are already reported as MISSING_TEXT
        if rec.text in first_seen:
            issues.append(
                Issue(rec.record_id, DUPLICATE_RECORD, ERROR,
                      f"same text as {first_seen[rec.text]}")
            )
        else:
            first_seen[rec.text] = rec.record_id
    return issues


def find_cross_split_duplicates(
    train: Sequence[RawRecord], test: Sequence[RawRecord]
) -> list[Issue]:
    """Flag training records whose text also appears in the test file.

    The issue is attached to the *training* record, so that record is removed
    from training and the official test set stays complete and unchanged.
    """
    test_ids = {r.text: r.record_id for r in test if r.text}
    return [
        Issue(r.record_id, CROSS_SPLIT_DUPLICATE, ERROR,
              f"same text as test record {test_ids[r.text]}")
        for r in train
        if r.text in test_ids
    ]
