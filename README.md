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
[docs/ANONYMIZATION.md](docs/ANONYMIZATION.md)) are done. PDF handling and the UI
come in later phases. See
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
in the text. This works on plain text only; there is no PDF redaction or UI
yet. Details: [docs/ANONYMIZATION.md](docs/ANONYMIZATION.md).

## Running

```bash
python -m pytest                       # run all unit tests (real-data tests skip if data is absent)
python -m pytest tests/test_schema.py  # run one test file
streamlit run app/streamlit_app.py     # placeholder UI at http://localhost:8501
```

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
│   └── pdf/                # extract text from PDFs, write redacted PDFs
├── app/                    # Streamlit frontend (calls the package only)
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
