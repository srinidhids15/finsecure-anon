# FinGuard

FinGuard detects and anonymizes personally identifiable information (PII) in
financial documents such as bank statements, invoices and loan forms. It
combines **rule-based detection** (regular expressions for structured values
like account and card numbers) with a **spaCy NER model** (for free-form
entities like names and addresses), then replaces what it finds.

> FinGuard is an original project, built from scratch after studying the
> public FinSecureAnon project. It does not reuse that project's source code,
> trained models or results.

**Status:** Phases 1 (setup), 2 (dataset ingestion and audit) and 3 (rule-based
detector, see [docs/RULES.md](docs/RULES.md)) are done. Evaluation, NER training,
hybrid detection, anonymization, PDF handling and the UI come in later phases. See
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
python -m finguard.data.prepare        # audit + write data/processed/{train,validation,test}.jsonl
```

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
│   ├── anonymization/      # replace detected spans in text
│   └── pdf/                # extract text from PDFs, write redacted PDFs
├── app/                    # Streamlit frontend (calls the package only)
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

## License and attribution

Dataset: Mishra, K., Pagare, H., Bidwe, R., Mishra, S. (2024). *Synthetic
Dataset for PII Detection and Anonymization in Financial Documents.* Mendeley
Data, V1. https://doi.org/10.17632/tzrjx692jy.1. Licensed CC BY 4.0. The data
is not redistributed in this repository.
