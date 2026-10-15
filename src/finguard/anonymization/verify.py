"""Output-integrity verification for an anonymization result.

``verify_anonymization(original, entities, result)`` rebuilds what the
output must be from the original text and the recorded redactions, and
checks the requested entities against the recorded redactions. It does not
search the sanitized text for original values: the same string may
legitimately remain elsewhere (an undetected occurrence), and its absence
would not prove that no PII remains.

Checks:

1. Every requested entity (canonicalised) has exactly one recorded
   redaction with the same label and offsets, and nothing extra was recorded.
2. Each redaction uses its label's placeholder, and the placeholder sits at
   the recorded output offsets.
3. Every character outside the redacted spans is copied unchanged, at the
   output position implied by the preceding replacements (no offset shift).
4. The output has nothing before, between or after these pieces.

The result can establish only that *detected* spans were replaced
(``detected_spans_replaced``). ``complete_pii_removal_guaranteed`` is always
False. Problem descriptions contain offsets and labels, never text.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from finguard.anonymization.placeholders import resolve_placeholders
from finguard.anonymization.result import AnonymizationResult, VerificationReport
from finguard.canonical import canonical_span
from finguard.schema import Entity


def _requested(original: str, entities: Sequence[Entity], problems: list[str]) -> Counter:
    keys: Counter = Counter()
    for i, e in enumerate(entities):
        if e.end > len(original) or original[e.start:e.end] != e.text:
            problems.append(f"requested entity #{i} ({e.label!r}, [{e.start}, {e.end})) does not fit the original text")
            continue
        canon = canonical_span(original, e)
        if canon is None:
            problems.append(f"requested entity #{i} ({e.label!r}) is whitespace only")
            continue
        keys[(e.start, canon.end, e.end, e.label)] += 1
    return keys


def verify_anonymization(
    original: str,
    entities: Sequence[Entity],
    result: AnonymizationResult,
    placeholders: Mapping[str, str] | None = None,
) -> VerificationReport:
    """Check that ``result`` is exactly ``original`` with ``entities`` replaced."""
    mapping = resolve_placeholders(placeholders)
    problems: list[str] = []
    out = result.sanitized_text
    redactions = sorted(result.redactions, key=lambda r: (r.start, r.end))

    if result.input_length != len(original):
        problems.append(f"result was made from a text of length {result.input_length}, not {len(original)}")

    # 1. requested entities <-> recorded redactions
    requested = _requested(original, entities, problems)
    recorded = Counter((r.start, r.end, r.detected_end, r.label) for r in redactions)
    for key in sorted(requested - recorded):
        problems.append(f"requested {key[3]!r} span [{key[0]}, {key[1]}) has no matching redaction")
    for key in sorted(recorded - requested):
        problems.append(f"redaction {key[3]!r} [{key[0]}, {key[1]}) was not requested")

    # 2.-4. rebuild the output piece by piece
    cin = cout = floor = 0  # floor: end of the previous detected span, kept whitespace included
    for k, r in enumerate(redactions):
        where = f"redaction #{k} ({r.label!r}, [{r.start}, {r.end}))"
        if not (floor <= r.start < r.end <= r.detected_end <= len(original)):
            problems.append(f"{where}: offsets out of order, overlapping or outside the original text")
            break
        if r.placeholder != mapping.get(r.label):
            problems.append(f"{where}: placeholder is not the one defined for its label")
        gap = r.start - cin
        if out[cout:cout + gap] != original[cin:r.start]:
            problems.append(f"{where}: unredacted text before it was not preserved")
        cout += gap
        if (r.output_start, r.output_end) != (cout, cout + len(r.placeholder)):
            problems.append(f"{where}: recorded output offsets are shifted")
        if out[cout:cout + len(r.placeholder)] != r.placeholder:
            problems.append(f"{where}: placeholder not found at its output position")
        cout += len(r.placeholder)
        cin, floor = r.end, r.detected_end
    if out[cout:] != original[cin:]:
        problems.append(f"unredacted text after offset {cin} was not preserved")

    return VerificationReport(
        detected_spans_replaced=not problems,
        checked_redactions=len(redactions),
        problems=tuple(problems),
    )
