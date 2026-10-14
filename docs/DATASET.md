# Dataset

## Attribution

FinGuard is trained and evaluated on:

> Mishra, Kushagra; Pagare, Harsh; Bidwe, Ranjeet; Mishra, Sashikala (2024).
> *Synthetic Dataset for PII Detection and Anonymization in Financial Documents.*
> Mendeley Data, V1. https://doi.org/10.17632/tzrjx692jy.1
> (Symbiosis Institute of Technology, Symbiosis International University)

License: **Creative Commons Attribution 4.0 (CC BY 4.0)**. The data is fully
synthetic and contains no real personal information, according to its authors.
FinGuard does not redistribute the data: the files stay in git-ignored
`data/raw/`, and anyone reproducing the work downloads them from the source.

## Obtaining the data

1. Download the archive from https://data.mendeley.com/datasets/tzrjx692jy/1
2. Extract both workbooks into `data/raw/mendeley_tzrjx692jy_v1/`
3. Check the files are the ones this project was built with:

| File | Rows | SHA-256 |
|---|---|---|
| `Training_Set.xlsx` | 45,000 | `5f960d4f2d17762b75317565dc61b6382d99693983ae7e8a2d2a0070ec96942c` |
| `Testing_Set.xlsx` | 30 | `abf209b309704cda0f6c7f0cab64d8885d0aa07ce78e770343a8130fa57c20c5` |

```bash
shasum -a 256 data/raw/mendeley_tzrjx692jy_v1/*.xlsx
python -m finguard.data.prepare
```

## Format (as inspected, not assumed)

Each file is an Excel workbook with **one sheet** (`Training_Set` / `Testing_Set`)
and 10 text columns:

| Column | Content |
|---|---|
| `Name`, `Credit Card`, `Email`, `URL`, `Phone`, `Address`, `Company`, `SSN` | The synthetic value inserted into this document, one per type |
| `Text` | The document (audit reports, tax filings, notices, confirmations, ...) |
| `True Predictions` | Gold annotations as the **string form of a Python list**: `"[(start, end, 'label'), ...]"` |

- Offsets are character offsets into `Text`, with `end` exclusive (Python slicing).
- Labels: `name, email, phone, address, ssn, credit_card, company, url`.
  These are exactly FinGuard's 8 classes.
- An entity can be annotated several times in one document (e.g. a company
  name repeated in a report).
- `Credit Card` values are multi-line blocks
  (`"VISA 16 digit\n<holder name>\n<number> <MM/YY>\nCVC: <nnn>\n"`) that end
  with a newline, and the gold span includes that newline.

## Issues found

| # | Issue | Scope | Handling |
|---|---|---|---|
| 1 | **Mojibake shifts offsets.** Curly quotes (’ ‘ “ ”) appear as 3-character sequences (`â€™` etc.). The offsets were computed on the original text, so each sequence before an entity moves that entity 2 characters right of its annotation. | Train: 5,000 rows, 31,763 spans, 20,000 sequences. Test: 3 rows, 21 spans, 14 sequences. | Text repaired (4 fixed sequences mapped back). The fix explains **100 %** of misaligned spans. After it, 0 invalid records remain in either file. Annotations are not changed. |
| 2 | Trailing whitespace in spans | Every `credit_card` span (42,500 train, 27 test) | Kept as is; reported as a warning. To be handled during spaCy conversion (Phase 5). |
| 3 | Template repetition | 45,000 training rows come from 10,267 distinct templates; the top templates repeat ~1,700 times. | Train/validation split is **grouped by template**. |
| 4 | Test set is very small | 30 records, 257 spans; 23 to 49 spans (22 to 29 records) per class | Test scores will have wide uncertainty; report them with that caveat. |
| 5 | Test distribution differs | 16 of 30 test documents reuse an exact training template. 3 test documents (`test-000003`, `-000006`, `-000007`; 217, 185 and 111 chars) are shorter than any training document (min 448 chars). | Descriptive only; discussed when results are reported. |

Checked and **not** found: missing text, unparseable annotations, malformed
tuples, unknown labels, out-of-range offsets, overlapping spans, duplicate
spans, duplicate records, and test documents copied into the training file.
(9 test person names also occur somewhere in training; these are common
generated names, not duplicated documents.)

## Preprocessing decisions

1. Raw `.xlsx` files are read-only inputs and are checksummed before and
   after processing.
2. Annotations are parsed with `ast.literal_eval`, never `eval()`.
3. The mojibake repair looks only at text, never labels, and is applied
   identically to both files. Without it, 11 % of training documents and
   3 of 30 test documents would have wrong gold labels.
4. Every span must be `(int, int, str)`, use an allowed label, fit inside the
   text and **extract exactly the value in the matching source column**.
   Overlapping spans are errors (spaCy NER needs one label per character),
   never silently resolved.
5. Training records with any error would be excluded and listed in the report.
   With the real data this excluded 0 records. No annotations are generated or
   guessed.
6. The test file must pass completely, or the pipeline stops. Test records are
   never dropped, re-labelled or used for training or model selection.
7. Validation = 10 % of training templates (seed 42) → 6,595 records (14.7 %
   of rows, because template sizes vary). Train = 38,405 records.
8. Reports contain ids, codes, offsets and counts, never document text.

## Outputs (`data/processed/`, git-ignored, reproducible)

| File | Content |
|---|---|
| `train.jsonl`, `validation.jsonl` | `{"id", "text", "spans": [[start, end, label]], "group"}` |
| `test.jsonl` | `{"id", "text", "spans"}`: all 30 official records, original order |
| `audit_report.md` / `.json` | Issue counts before/after repair, the full invalid-record list, per-class statistics, checksums |

Record ids are `<split>-<Excel row>` (e.g. `train-000002` is the first data
row), so any reported problem can be found in the original file.
