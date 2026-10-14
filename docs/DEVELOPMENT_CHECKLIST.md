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
- [ ] Trailing newline in credit_card spans: normalise gold *and* predicted spans in the conversion/evaluation layer (Phases 4-5)

## Phase 3: Rule-based detection ✅
- [x] `Entity` output type and `Detector` protocol (stable interface for NER/hybrid)
- [x] Rules: email, url, phone (documented formats), ssn (SSA structure), credit_card
- [x] Credit-card unit established from training data: full 5-line block incl. trailing newline
- [x] Luhn + network prefix checks; stand-alone number rule yields to block rule
- [x] Overlaps returned, suppressions reported (`detect_with_report`)
- [x] Synthetic unit tests: valid/invalid, punctuation, multiline, adjacency, Unicode, offsets
- [x] docs/RULES.md; dataset issues 6-7 recorded in docs/DATASET.md

## Phase 4: Evaluation
- [ ] Exact-match and partial-overlap P / R / F1, per label and overall
- [ ] Score the rule-based baseline on the validation split

## Phase 5: spaCy NER training and inference
- [ ] Convert to `DocBin`, report spans dropped by token misalignment
- [ ] Train on the train split, select the model on validation
- [ ] `ner.inference` returns `Span` objects

## Phase 6: Hybrid detection
- [ ] Documented overlap-resolution policy
- [ ] Compare rules vs NER vs hybrid on validation; report on test once at the end

## Phase 7: Anonymization
- [ ] Tag / mask / Faker strategies; right-to-left replacement
- [ ] Consistent fake values for repeated entities (optional)

## Phase 8: PDF processing
- [ ] Text extraction (PyMuPDF; pdfplumber for tables)
- [ ] Detect scanned PDFs with no text layer
- [ ] Redacted PDF output

## Phase 9: Streamlit frontend
- [ ] Upload text or PDF → show detections → download anonymized output
- [ ] No raw text logging; temporary files cleaned up

## Phase 10: Documentation and final evaluation
- [ ] Final test-set results, limitations, viva notes
