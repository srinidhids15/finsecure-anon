# FinGuard

FinGuard detects and anonymizes personally identifiable information (PII) in
financial documents such as bank statements, invoices and loan forms. It
combines **rule-based detection** (regular expressions for structured values
like account and card numbers) with a **spaCy NER model** (for free-form
entities like names and addresses), then replaces what it finds.

**Status:** Phases 1 (setup), 2 (dataset ingestion and audit), 3 (rule-based
detector, see [docs/RULES.md](docs/RULES.md)), dataset readiness and 4 (spaCy NER
training, see [docs/NER.md](docs/NER.md)) and 5 (evaluation and hybrid detection,
see [docs/EVALUATION.md](docs/EVALUATION.md)) and 6 (text anonymization, see
[docs/ANONYMIZATION.md](docs/ANONYMIZATION.md)) and 7 (Streamlit app, see
[docs/APP.md](docs/APP.md)) and 8 (text-based PDF redaction, see [docs/PDF.md](docs/PDF.md))
are done. OCR and batch processing are not implemented. See
[docs/DEVELOPMENT_CHECKLIST.md](docs/DEVELOPMENT_CHECKLIST.md).

## Requirements

- Python **3.11** (the project is pinned to 3.11.x)
- macOS, Linux or Windows

## Setup

```bash
# 1. Get Python 3.11 (macOS + Homebrew; on other systems use python.org or your package manager)
brew install python@3.11

# 2. Create and activate a virtual environment in the project folder
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install exact, tested dependencies plus FinGuard itself (editable)
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Check that the installation works
python -c "import finguard; print(finguard.__version__)"
python -m pytest
```

If `python3.11` is not on your PATH after installing it with Homebrew, use
`/opt/homebrew/bin/python3.11` (Apple Silicon) or `/usr/local/bin/python3.11` (Intel).

## Dataset

FinGuard uses the CC BY 4.0 *Synthetic Dataset for PII Detection and
Anonymization in Financial Documents* (Mishra et al., Mendeley Data, 2024,
DOI [10.17632/tzrjx692jy.1](https://doi.org/10.17632/tzrjx692jy.1)).
Download instructions, format, the issues found and every preprocessing
decision are in [docs/DATASET.md](docs/DATASET.md).

```bash
# after extracting the two .xlsx files into data/raw/mendeley_tzrjx692jy_v1/
python -m finguard.data.prepare            # audit + write data/processed/{train,validation,test}.jsonl
python -m finguard.data.build_ner_dataset  # label corrections + template-disjoint train/validation/internal_test
python -m finguard.ner.alignment           # check every gold span aligns with FinGuard's tokenizer
python -m finguard.ner.convert             # train/validation JSONL -> spaCy DocBin corpora
python -m finguard.ner.train               # train the NER model on CPU (see docs/NER.md)
python -m finguard.evaluation.run --dataset validation   # rules vs NER vs hybrid (development)
```

Held-out evaluation (internal test, official benchmark) runs only against the
frozen system in `configs/evaluation_freeze.json`; see [docs/EVALUATION.md](docs/EVALUATION.md).

Training details, settings and the inference API: [docs/NER.md](docs/NER.md).

## Anonymizing text

```python
from finguard.anonymization import detect_and_anonymize

result = detect_and_anonymize(text)   # frozen hybrid detector + placeholder replacement
print(result.sanitized_text)          # "Dear [NAME REDACTED], call [PHONE NUMBER REDACTED] ..."
print(result.to_dict())               # audit report: labels, offsets, placeholders; no original values
```

Each detected entity is replaced by a placeholder for its label. A
credit-card block (holder, number, expiry and CVC lines) becomes a single
`[CREDIT CARD REDACTED]`, and the newline after it is kept. Invalid,
mismatched or overlapping spans raise an error instead of producing doubtful
output. Every result is checked to confirm that each detected span, and
only that span, was replaced at the right offsets. This does **not**
guarantee that all PII was removed, because PII the detector misses stays
in the text. Details: [docs/ANONYMIZATION.md](docs/ANONYMIZATION.md).

## Using the app

```bash
streamlit run app/streamlit_app.py     # from the repository root; opens http://localhost:8501
```

1. Paste text, or click **Load synthetic sample**. The limit is 100,000
   characters.
2. Click **Scan for PII**. You see counts per label, the text with each
   detection highlighted at its exact offsets, and a table of labels,
   offsets and detector sources.
3. Click **Anonymize detected entities**. The sanitized text appears only
   after its verification passes, together with a reminder that missed PII
   may remain.
4. Download the sanitized text (`.txt`), the entity report (`.csv`) or the
   redaction summary (`.json`). The reports hold labels, offsets and
   placeholders, never the detected values.

The app needs the trained model in `models/finguard_ner/model-best`. If it
is missing, the app explains how to train it. Everything runs locally. The
server listens on `localhost` only, and text is kept in the browser
session's memory, not logged or saved. Details: [docs/APP.md](docs/APP.md).

### PDF documents

The **PDF document** tab accepts one text-based PDF of up to 20 MB and 100
pages.
- **Processing:** each page's text is extracted with exact character
  positions, scanned by the same detector, and the detected glyphs are
  **physically removed** and covered by black boxes. Annotations, links,
  form fields, metadata, embedded files and JavaScript are removed too.
- **Verification:** the result is reopened and checked. No text may remain
  in any redacted region, and every other character must be unchanged.
  Only then is it offered for download.
- **Rejected:** with a message explaining why:
  - scanned or image-only pages (no OCR);
  - rotated, invisible, clipped or image-covered text;
  - fonts without a Unicode mapping (including Type 3 fonts without one);
  - ActualText or alternate-text strings;
  - layered PDFs, XFA forms and encrypted files.
- **Not guaranteed:** complete PII removal. Undetected PII and text inside
  images remain.

Details: [docs/PDF.md](docs/PDF.md).

## Running

```bash
python -m pytest                       # all tests; model- and dataset-dependent tests skip if those are absent
python -m pytest tests/test_schema.py  # run one test file
python -m pytest tests/test_app.py     # app tests (Streamlit AppTest, synthetic input)
python -m pytest tests/test_pdf.py tests/test_app_pdf.py   # PDF tests (synthetic PDFs)
streamlit run app/streamlit_app.py     # the FinGuard app at http://localhost:8501
```

## Model availability and reproducibility

The trained model (`models/finguard_ner/model-best`, about 10 MB) and the
dataset are **not in this repository**: both are git-ignored. On
2026-10-22, the setup steps above were checked in a fresh Python 3.11.17
virtual environment, on a copy of the repository without `models/`,
`data/` or `.venv`.

| situation | result |
|---|---|
| install with `pip install -r requirements.txt` | succeeded; `import finguard` works; `finguard.Tokenizer.v1` is registered through the entry point |
| `python -m pytest`, no model, no dataset | 629 passed, 24 skipped (model- and dataset-dependent tests), 0 failed |
| app, no model | starts; explains how to train the model; Scan and PDF actions are disabled |
| freeze check, no model | fails: `changed since freeze: ['models/finguard_ner/model-best']` |
| same, with the selected model directory copied in | freeze verifies (`ff731aed…`); 639 passed, 14 skipped (dataset tests), 0 failed |
| app with the model | text sample and a synthetic PDF processed and verified |

To get a model you must download the dataset (see [docs/DATASET.md](docs/DATASET.md))
and run the pipeline under **Dataset**. A retrained model is a new artifact:

- **It cannot pass the freeze check.** The freeze hash covers every file in
  the model directory, including the training manifest's timestamp.
- **Published scores do not carry over.** They apply only to the original
  model; a new model must be evaluated as a new attempt.

To reproduce the reported results exactly, the original `model-best`
directory has to be distributed separately. That is not done yet.

`requirements.txt` pins only the direct dependencies, so transitive packages
are resolved at install time.

## Results and limitations

All scores come from the synthetic Mendeley dataset, using exact-span
matching ([docs/EVALUATION.md](docs/EVALUATION.md)):

| set | documents | hybrid micro F1 | notes |
|---|---|---|---|
| validation | 5,000 | 0.9928 | used to choose the merge policy |
| internal test (held out, template-disjoint) | 5,000 | 0.9137 | `name` F1 0.5223; single run |
| official benchmark (held out) | 30 | 0.9905 | 16 of 30 documents reuse a training template, so this score is optimistic |

**Real-world performance is unknown.** FinGuard has not been evaluated on
real financial documents. The synthetic data uses a small number of
templates, so expect more missed and spurious detections on real text,
especially for names.

Every output still has to be reviewed:
- **Text:** verification confirms that detected spans were replaced. It does
  not show that all PII is gone.
- **PDF:** physical removal of the detected glyphs is verified, but OCR,
  rotated layouts and several hidden-text structures are not supported and
  are rejected (see [docs/PDF.md](docs/PDF.md)).

FinGuard is a research prototype, not a certified privacy tool.

## Project layout

```
.
├── pyproject.toml          # package metadata, dependency ranges, pytest config
├── requirements.txt        # exact tested versions (reproducible install)
├── src/finguard/           # the importable package (all real logic lives here)
│   ├── schema.py           # Span: the data contract shared by every component
│   ├── config.py           # paths and global constants
│   ├── data/               # loading + validating annotated datasets
│   ├── detection/          # base.py (Detector interface), rules.py (regex), hybrid.py
│   ├── ner/                # spaCy training and inference
│   ├── evaluation/         # precision / recall / F1
│   ├── anonymization/      # placeholder replacement, audit report, output verification
│   ├── interface/          # app workflow, session state, escaped highlighting, exports (no Streamlit)
│   └── pdf/                # validate, geometry-mapped extraction, redaction, verification
├── app/                    # Streamlit app: rendering only (streamlit_app.py)
├── .streamlit/config.toml  # app settings: localhost only, no usage stats, generic errors
├── configs/                # reviewed configuration (template alias families)
├── tests/                  # pytest unit tests
├── docs/                   # architecture notes and development checklist
├── data/                   # datasets: contents git-ignored (see data/README.md)
└── models/                 # trained models: contents git-ignored
```

For how the parts fit together, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Data and privacy

- Downloaded datasets go in `data/raw/`. Real or private documents go in
  `data/private/`. Git ignores both, along with every `*.pdf` outside
  `data/samples/`.
- Only small, fully synthetic examples are committed, in `data/samples/`.
- Raw document text is not logged by default (`config.LOG_RAW_TEXT = False`).
- The anonymizer logs nothing and writes no files; its reports hold labels and
  offsets, never original values.

## License and attribution

Dataset: Mishra, K., Pagare, H., Bidwe, R., Mishra, S. (2024). *Synthetic
Dataset for PII Detection and Anonymization in Financial Documents.* Mendeley
Data, V1. https://doi.org/10.17632/tzrjx692jy.1. Licensed CC BY 4.0. The data
is not redistributed in this repository.
