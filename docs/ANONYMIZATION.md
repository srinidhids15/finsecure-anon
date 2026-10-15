# Text anonymization (Phase 6)

`finguard.anonymization` replaces the entities found by the frozen hybrid
detector with label placeholders, returns an audit report that holds no
original values, and verifies that every detected span was replaced
correctly. It works on plain text only. PDF redaction and the UI are later
phases.

## Usage

```python
from finguard.anonymization import anonymize, detect_and_anonymize
from finguard.detection.hybrid import HybridDetector

detector = HybridDetector()                 # frozen rules + NER + merge policy
entities = detector.detect(text)            # list[Entity], non-overlapping
result = anonymize(text, entities)          # or: detect_and_anonymize(text, detector)

result.sanitized_text                       # text with placeholders
result.redaction_count, result.counts_by_label
result.redactions                           # tuple[Redaction]: label, offsets, placeholder, sources
result.verification.detected_spans_replaced # True (otherwise anonymize raises)
result.to_dict()                            # audit report, no original values, no text
```

`anonymize` does no detection and no conflict resolution: it consumes the
hybrid detector's output as is.

## Placeholder policy

| label | placeholder |
|---|---|
| `name` | `[NAME REDACTED]` |
| `email` | `[EMAIL REDACTED]` |
| `phone` | `[PHONE NUMBER REDACTED]` |
| `address` | `[ADDRESS REDACTED]` |
| `ssn` | `[SSN REDACTED]` |
| `credit_card` | `[CREDIT CARD REDACTED]` |
| `company` | `[COMPANY NAME REDACTED]` |
| `url` | `[URL REDACTED]` |

`DEFAULT_PLACEHOLDERS` is read-only. `anonymize(text, entities,
placeholders={"name": "<PERSON>"})` overrides individual placeholders for
known labels. An entity with any other label raises `UnknownLabelError`.
There is no generic fallback placeholder, because a mislabelled entity is
an error to surface.

## Offsets and canonical spans

All offsets are Python string indices (code points) into the original,
unmodified text, as the detectors return them.

1. Each entity is validated. Its offsets must fit the text, and
   `entity.text` must equal `text[start:end]`.
2. The span is canonicalised with `finguard.canonical.canonical_span`, the
   same policy used for training and evaluation: trailing whitespace is not
   part of an entity. Only the canonical span is replaced. The trailing
   whitespace stays in the output.
3. The output is built in one left-to-right pass. The text between spans is
   copied unchanged, and the placeholder's position in the output is
   recorded. No offset is ever adjusted, so a replacement cannot shift a
   later one.

### Credit-card blocks

A credit-card entity is the complete multiline block: network line, holder,
number and expiry, and security code. The rule detector returns it with its
final newline. The block is replaced by a single `[CREDIT CARD REDACTED]`,
and that newline is kept:

```
Card on file:\nVISA 16 digit\nAsha Rao\n4111111111111111 12/27\nCVC: 123\nThank you.
Card on file:\n[CREDIT CARD REDACTED]\nThank you.
```

A block that the NER model returns without its newline produces the same
output. `Redaction.detected_end - Redaction.end` records how much
whitespace was kept, and a warning in the report states it.

## Errors

Every error is a subclass of `AnonymizationError`, which is itself a
`ValueError`. Nothing is redacted unless every entity is valid.

| error | cause |
|---|---|
| `UnknownLabelError` | label has no placeholder |
| `InvalidSpanError` | end beyond the text, or whitespace-only span |
| `SourceMismatchError` | `entity.text` differs from the text at its offsets (e.g. offsets shifted) |
| `OverlapError` | two entities overlap, including identical or nested spans |
| `IntegrityError` | the output failed its own verification (a bug; never expected) |
| `TypeError` | text is not `str`, or an entity is not an `Entity` |

Overlaps are rejected rather than resolved. The hybrid merge policy is the
single documented place where conflicts are resolved, and its output never
overlaps.

## Output verification and its limits

`verify_anonymization(original, entities, result)` runs inside `anonymize`
and can be called independently. Starting from the original text and the
recorded redactions, it checks that:

- every requested entity has exactly one redaction with the same label and
  canonical offsets, and that nothing else was redacted;
- each redaction uses its label's placeholder at the recorded output offsets;
- every character outside the redacted spans, including kept trailing
  whitespace, appears unchanged at the position the preceding replacements
  imply. This rules out offset shifts.

It does **not** search the output for original values. The same string can
legitimately remain elsewhere, as an occurrence the detector did not find,
and a value's absence would not prove that no PII remains.

The report keeps two properties separate:

- `detected_spans_replaced`: verified as described above.
- `complete_pii_removal_guaranteed`: always `False`. The detector can miss
  PII, and then that PII is still in the sanitized text. For example, the
  NER model's `name` F1 on the internal test partition was 0.522
  (docs/EVALUATION.md).

## Privacy and logging

- The module does not log anything and does not write any files.
- `Redaction`, `VerificationReport` and `AnonymizationResult.to_dict()`
  contain labels, offsets, lengths, placeholders and detector sources,
  never original values.
- `sanitized_text` is excluded from `repr(result)` and from `to_dict()`
  unless `to_dict(include_sanitized_text=True)` is called, because it can
  still contain undetected PII.
- Error and verification messages contain entity indices, labels and
  offsets only.
- The input text and the entity objects are not modified.

All tests (`tests/test_anonymization.py`,
`tests/test_anonymization_e2e.py`) use hand-written synthetic text. The
official test workbook was not used.
