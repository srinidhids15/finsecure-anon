# FinGuard Development Checklist

Each phase is implemented, tested and reviewed before the next one starts.
Every phase adds unit tests, and its results are reported exactly as they ran.

## Phase 1: Project setup ✅
- [x] Python 3.11 virtual environment
- [x] `pyproject.toml`, `requirements.txt` (pinned), `.gitignore`
- [x] Package skeleton with documented module stubs
- [x] Shared `Span` data contract with validation
- [x] Smoke tests and `Span` tests passing
- [x] README, architecture notes, this checklist

## Phase 2: Dataset ingestion and quality audit ✅
- [x] Dataset chosen (Mendeley tzrjx692jy v1, CC BY 4.0); attribution in README and docs/DATASET.md
- [x] Format inspected: 2 single-sheet .xlsx files, annotations as Python-literal strings
- [x] Label set confirmed: 8 classes match the dataset exactly
- [x] Safe annotation parsing (`ast.literal_eval`, no `eval`)
- [x] Validation: structure, labels, offsets, substring = source value, overlaps, duplicates
- [x] Root cause of misaligned offsets found (mojibake) and repaired in text only
- [x] Per-class statistics and invalid-record report (before and after repair)
- [x] Template-grouped train/validation split from the training file; test set preserved
- [x] Trailing newline in credit_card spans: `finguard.canonical` (applied in Phases 4-5)

## Phase 3: Rule-based detection ✅
- [x] `Entity` output type and `Detector` protocol (stable interface for NER/hybrid)
- [x] Rules: email, url, phone (documented formats), ssn (SSA structure), credit_card
- [x] Credit-card unit established from training data: full 5-line block incl. trailing newline
- [x] Luhn + network prefix checks; stand-alone number rule yields to block rule
- [x] Overlaps returned, suppressions reported (`detect_with_report`)
- [x] Synthetic unit tests: valid/invalid, punctuation, multiline, adjacency, Unicode, offsets
- [x] docs/RULES.md; dataset issues 6-7 recorded in docs/DATASET.md

## Pre-Phase 4: NER dataset readiness ✅
- [x] URL gap investigated on the training workbook: one symptom of generator dot deletion (16,607 unlabelled values across email/url/address/name)
- [x] `dot_dropped_variant` correction layer with provenance report; 0 ambiguous cases; raw files and Phase 2 outputs unchanged
- [x] Robust template key (29 templates → 27 groups); Phase 2 split found to share templates
- [x] Three-way template-disjoint split (21/3/3 groups; 37,500/3,750/3,750 rows), deterministic
- [x] `finguard.canonical`: trailing-whitespace policy for gold and predictions
- [ ] Decide: apply the correction rule to the official test set at final evaluation, or report it uncorrected
- [x] Paraphrased templates: resolved by reviewed alias families (threshold kept at 0.9)

## Pre-training readiness checkpoint ✅
- [x] All template-group pairs reviewed; 10 alias families with evidence in `configs/template_aliases.json`
- [x] Generator concatenation defect found (issue 12) and aliased with both source templates
- [x] Stratified, seeded split of 14 units: both evaluation partitions hold a high- and a low-frequency unit (35,000 / 5,000 / 5,000)
- [x] Record-weighted and group-level class distributions per partition
- [x] Alignment audit by label and failure pattern; `finguard.Tokenizer.v1` → 0 misaligned gold spans
- [x] Tokenizer persisted (config + entry point); save/reload reproduces token offsets
- [x] Tests: alias groups never cross partitions, reproducibility, high-frequency coverage, strict alignment, card canonicalisation, official test workbook never opened

## Phase 4: spaCy NER training ✅
- [x] `ner.convert`: JSONL → DocBin with `finguard.Tokenizer.v1`, canonical spans, strict alignment; fails with diagnostics, refuses test partitions
- [x] `configs/ner.cfg`: CPU efficiency architecture (tok2vec + ner), no pretrained weights, seed 42
- [x] `ner.train`: pre-flight checks (tokenizer registered, corpora = train/validation and current), `spacy train`, provenance manifest
- [x] Weights from train only; checkpoint selection on validation; internal/official test untouched
- [x] `NERDetector` implements the `Detector` protocol; fresh-process load and tokenizer round-trip verified
- [x] Tests: conversion, alignment, tokenizer serialisation, small-fixture training, loading, inference, labels, empty text

## Phase 5: Evaluation and hybrid detection ✅
- [x] `evaluation.metrics`: exact canonical-span micro/macro/per-label/per-group P/R/F1, typed FP/FN, hand-calculated tests
- [x] `detection.hybrid`: dedupe, card-block precedence, rule priority for structured labels, NER otherwise; suppressions recorded
- [x] Rules, NER and 3 pre-declared hybrid variants on the same validation records; `rule_priority_structured` selected
- [x] System frozen (`configs/evaluation_freeze.json`), then internal test and official benchmark evaluated once each
- [x] Official benchmark: original (diagnostic) and corrected (primary) annotations; workbook unchanged
- [ ] Partial-overlap metric (optional)
- [ ] Address the `name` errors found on the internal test (sentence-initial words, document titles) as a new, validation-driven attempt

## Phase 6: Text anonymization ✅
- [x] Interfaces inspected; anonymizer consumes the frozen hybrid detector's `Entity` output (no detection, no second conflict policy)
- [x] Label placeholders (read-only default, overrides for known labels, unknown label → error)
- [x] Offset validation, source-text match, canonical spans (trailing whitespace kept outside the placeholder)
- [x] Whole credit-card block → one placeholder; trailing newline preserved
- [x] Single left-to-right pass with recorded output offsets; inputs not mutated
- [x] Clear errors for invalid offsets, mismatched text, unknown labels, overlaps
- [x] Audit report without original values; nothing logged or written to disk
- [x] Output verification from original text + recorded operations; completeness explicitly not claimed
- [x] Synthetic unit tests + one end-to-end test with the real hybrid detector; freeze still verifies
- [ ] Mask / Faker pseudonymisation strategies (optional, not started)

## Phase 7: PDF processing
- [ ] Text extraction (PyMuPDF; pdfplumber for tables)
- [ ] Detect scanned PDFs with no text layer
- [ ] Redacted PDF output

## Phase 8: Streamlit frontend
- [ ] Upload text or PDF → show detections → download anonymized output
- [ ] No raw text logging; temporary files cleaned up

## Phase 9: Documentation and final evaluation
- [ ] Final test-set results, limitations, viva notes
