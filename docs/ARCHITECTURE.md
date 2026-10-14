# FinGuard Architecture

## The one shared type: `Span`

All components communicate through plain Python strings and lists of
`finguard.schema.Span`:

```python
Span(start=8, end=16, label="PERSON", source="rule")   # text[8:16]
```

- Offsets follow Python slicing (`end` is exclusive).
- Invalid spans (negative start, empty range, blank label) raise an error
  **when they are created**, so bugs surface where they start.
- `source` records which detector produced the span, which the hybrid merger
  and debugging both rely on.

Detectors return `finguard.schema.Entity`, a `Span` subclass that also
carries the matched `text` (`entity.text == text[start:end]`). Gold
annotations stay plain `Span` objects.

```python
Entity(start=6, end=22, label="email", source="rule:email", text="asha@example.com")
```

Every detector implements `finguard.detection.base.Detector`
(`name` + `detect(text) -> list[Entity]`). Because an `Entity` is a `Span`,
every consumer written for `list[Span]` accepts detector output, so any
component can be swapped or tested in isolation.

## Data flow

```
                 ┌──────────────┐
 PDF file ─────► │ pdf.extract  │ ──► text
                 └──────────────┘
                                         ┌────────────────────┐
 text ──────────────────────────────────►│ detection.rules    │──► rule entities ┐
   │                                     └────────────────────┘                │
   │                                     ┌────────────────────┐                ▼
   └────────────────────────────────────►│ ner.inference      │──► ner entities ► detection.hybrid.merge
                                         └────────────────────┘                │
                                                                                ▼ final spans
                         anonymization.anonymize(text, spans) ──► anonymized text
                         pdf.redact(pdf, spans)                ──► redacted PDF

 Training/evaluation path (offline):
 Training workbook ─► data.prepare (Phase 2: repair, validate) ─► data/processed/{train,validation}.jsonl
                   ─► data.build_ner_dataset (corrections, template groups,
                        configs/template_aliases.json → split units, stratified split)
                   ─► data/processed/ner/{train,validation,internal_test}.jsonl
 Test workbook ─► data.prepare ─► data/processed/test.jsonl   (official benchmark; never enters splitting)

 ner/{train,validation}.jsonl ─► ner.convert ─► spacy/{train,validation}.spacy
   ─► ner.train (configs/ner.cfg; weights: train; checkpoint selection: validation) ─► models/finguard_ner/model-best
   ─► ner.inference.NERDetector (same Detector interface as RuleDetector)
 ner/internal_test, then official test (once, at the end) ─► evaluation.metrics(canonical gold, canonical predictions)
```

## Module responsibilities

| Module | Responsibility | Input → Output |
|---|---|---|
| `config` | Paths, random seed, privacy flags | — |
| `schema` | `Span` (gold/offsets) and `Entity` (detector output) contracts | — |
| `data.loader` | Read workbooks (`RawRecord`) and processed JSONL (`Example`) | file → records |
| `data.annotations` | Safe parsing of annotation strings (no `eval`) | str → list |
| `data.repair` | Undo mojibake in text (labels untouched) | text → text |
| `data.validation` | Structure, labels, offsets, substring, overlap, duplicate checks | record → spans + `Issue`s |
| `data.split` | Phase 2 two-way split; NER template key and clustering; alias resolution; stratified three-way split of whole units | groups → units → partitions |
| `data.corrections` | `dot_dropped_variant` rule: add missing gold spans with provenance | record → `Correction`s + review items |
| `data.build_ner_dataset` | Pre-Phase 4 pipeline: verify Phase 2, correct, group, apply reviewed aliases, stratified split (CLI) | training workbook + Phase 2 JSONL + `configs/template_aliases.json` → `data/processed/ner/` |
| `ner.tokenizer` | `finguard.Tokenizer.v1`: English tokenizer + documented boundary rules; registered for `spacy.load` | text → tokens |
| `ner.alignment` | Strict gold-span/token alignment audit, by label and failure pattern (CLI) | examples → report |
| `canonical` | Trailing-whitespace span policy for spaCy conversion and evaluation | text, spans → spans |
| `data.statistics` | Per-class counts and lengths | examples → dict |
| `data.prepare` | Phase 2 pipeline + audit report (CLI) | raw files → JSONL + report |
| `detection.base` | `Detector` protocol, sorting, overlap listing | — |
| `detection.checksums` | Luhn, SSN structure, card-prefix checks | str → bool |
| `detection.rules` | `RuleDetector`: email, url, phone, ssn, credit_card (see docs/RULES.md) | text → `[Entity]` |
| `ner.convert` | JSONL → spaCy DocBin with canonical spans and strict alignment; train/validation only | JSONL → `.spacy` + report |
| `ner.train` | Pre-flight checks, config-driven `spacy train` (`configs/ner.cfg`), provenance manifest (see docs/NER.md) | `.spacy` → `models/finguard_ner/` |
| `ner.inference` | `NERDetector`: trained model behind the `Detector` interface | text → `[Entity]` |
| `detection.hybrid` | Merge rule and NER spans, resolving overlaps | `[Span]`, `[Span]` → `[Span]` |
| `anonymization.anonymizer` | Replace spans (tag, mask or Faker) | text, `[Span]` → text |
| `evaluation.metrics` | Precision / recall / F1 per label | gold, predicted → scores |
| `pdf.extract` | PDF text extraction (PyMuPDF, pdfplumber) | PDF → text |
| `pdf.redact` | Draw redaction boxes over spans | PDF, spans → PDF |
| `app/streamlit_app.py` | User interface; **no logic of its own** | upload → display |

## Design rules

1. **Overlaps are visible.** Detectors return overlapping results instead of
   dropping them. The only exception is a rule that explicitly yields to a
   coarser rule (card number inside a card block), and those candidates are
   reported in `detect_with_report(...).suppressed`. Resolving overlaps
   between detectors is the hybrid merger's documented job.
2. **Dependencies point inward.** The app depends on the package; the package
   never imports Streamlit. Detectors do not import each other; only
   `hybrid` combines them.
3. **The test set stays untouched.** It is never used for training or model
   selection; only the validation split is used for those.
4. **Reviewed configuration is code.** Manual decisions that change data
   handling (template alias families) live in `configs/`, are versioned,
   carry their evidence, and are checked against the data on every build.
5. **One tokenizer everywhere.** Training, evaluation and inference all use
   `finguard.Tokenizer.v1`; a saved model records it in its config.
6. **Privacy by default.** No raw document text in logs; private data stays
   in git-ignored folders.
