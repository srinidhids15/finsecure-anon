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

## Phase 7: Streamlit app (text) ✅
- [x] `finguard.interface` (no Streamlit import): workflow, escaped highlighting, in-memory exports, synthetic sample
- [x] App calls the frozen hybrid detector and `anonymize()`; no detection or replacement logic in the frontend
- [x] Detector cached once per process (model only), lock-serialised; missing model / init / runtime errors shown without input text
- [x] Scan → counts, highlighted text, entity table (no values) → anonymize → verified output → TXT / CSV / JSON
- [x] Stale results cleared when the input changes; anonymization tied to its scan; sessions isolated
- [x] 100,000-character input limit; localhost-only server, no usage stats, generic browser errors
- [x] Unit tests + `AppTest` tests (stub and real detector); freeze still verifies

## Phase 8: PDF detection and redaction (text-based PDFs) ✅
- [x] PyMuPDF 1.28.2 APIs inspected; redaction removal rule measured (glyph centre in rectangle)
- [x] Validation: signature, size/page/character limits, malformed/repaired, encrypted, empty
- [x] Page conditions: scanned/image-only, vector-only, rotation, rotated/vertical/invisible/off-page text, unmapped glyphs, trace cross-check → rejected; images on text pages noted
- [x] Hidden content: annotations, links, fields, metadata, XMP, embedded files, JavaScript, actions, outlines, structure tree removed before extraction; layers and XFA rejected
- [x] Per-page text with per-character geometry; detection per page with the frozen hybrid detector
- [x] Mapping via `anonymize` (validation + canonicalisation), one rect per line, collateral pre-check; no text search
- [x] Black-fill redactions removing text; `garbage=4` in-memory save; source bytes untouched
- [x] Independent verification (reopen, re-extract); output withheld on any failure
- [x] App PDF tab: summary, entity/page/hidden-content tables, preview, PDF/CSV/JSON downloads; stale-upload invalidation
- [x] Synthetic PDF tests incl. wrong-page regression, overlay-not-redaction, collateral detection; freeze still verifies
- [ ] OCR for scanned pages (not started)
- [ ] Batch processing (not started)

## Phase 9: Quality assurance and release readiness ✅
- [x] Freeze verified before and after; detector, model, tokenizer, policy, metrics untouched; no held-out evaluation re-run
- [x] PDF edge cases examined with byte-level synthetic PDFs. Supported and tested: ligatures, multi-character `ToUnicode`, Type 3 with `ToUnicode`, text over background images. Rejected: ligature-splitting boundaries, Type 3 or embedded fonts without Unicode maps, covered/clipped text, ActualText/Alt/E marked content
- [x] Defects fixed: ActualText pages were scanned with the replacement text; Type 3 fonts without `ToUnicode` were accepted with glyph-name text; a model test failed when the dataset was absent
- [x] Integrity tests: randomised multi-page PDFs (applied redactions, preserved text, no global search), session isolation, verification failure hides output, unsupported cases fail closed, silent MuPDF
- [x] Clean-environment install and tests (with and without the model); app startup and smoke test from the clean copy
- [x] Privacy audit: localhost binding, usage stats off, no app logging, in-memory uploads, fixed filenames, value-free reports, clean git history
- [x] Headless-browser layout check at 1366 px and 390 px (no horizontal overflow); remaining visual checks listed in docs/APP.md
- [ ] Distribute the selected model artifact (needed to verify the freeze and reported scores outside this machine)
- [ ] Evaluation on real documents (performance on real text is unknown)
- [ ] Optional: lock file for transitive dependencies
