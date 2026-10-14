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

## Phase 2: Data loading and validation
- [ ] Choose a dataset, record its license and attribution in the README
- [ ] Inspect the label set and decide FinGuard's entity labels
- [ ] `data.loader`: safe annotation parsing (json / `ast.literal_eval`)
- [ ] `data.validation`: bounds, misaligned offsets, overlapping spans
- [ ] Reproducible train / dev / test split (fixed seed); freeze the test set

## Phase 3: Rule-based detection
- [ ] Regex rules per structured entity, plus checksums where applicable (e.g. Luhn)
- [ ] Tests with positive and negative examples for every rule

## Phase 4: Evaluation
- [ ] Exact-match and partial-overlap P / R / F1, per label and overall
- [ ] Score the rule-based baseline on the dev split

## Phase 5: spaCy NER training and inference
- [ ] Convert to `DocBin`, report spans dropped by token misalignment
- [ ] Train on the train split, select the model on dev
- [ ] `ner.inference` returns `Span` objects

## Phase 6: Hybrid detection
- [ ] Documented overlap-resolution policy
- [ ] Compare rules vs NER vs hybrid on dev; report on test once at the end

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
