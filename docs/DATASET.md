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
| 2 | Trailing whitespace in spans | Every `credit_card` span (42,500 train, 27 test) | Kept as is in all stored files. `finguard.canonical` trims trailing whitespace at conversion/evaluation time, for gold and predictions alike (see below). |
| 3 | Template repetition | Phase 2 counted 10,267 templates. **Superseded by issue 9:** there are really 27 template groups. | Phase 2 grouped its train/validation split by its (too fine) key; the NER dataset uses the corrected key. |
| 4 | Test set is very small | 30 records, 257 spans; 23 to 49 spans (22 to 29 records) per class | Test scores will have wide uncertainty; report them with that caveat. |
| 5 | Test distribution differs | 16 of 30 test documents reuse an exact training template (measured with the Phase 2 key; not recomputed with the new key, to keep the official test file out of this work). 3 test documents (`test-000003`, `-000006`, `-000007`; 217, 185 and 111 chars) are shorter than any training document (min 448 chars). | Descriptive only; discussed when results are reported. |

Found later, in Phase 3, while checking the rule detector on the **training
partition only**:

| # | Issue | Scope | Handling |
|---|---|---|---|
| 6 | **Unannotated URLs.** In some documents the generator dropped the dot after `www` (`URL` column `https://www.keith.com/`, text `https://wwwkeith.com/`). The annotations mark only exact copies of the column value, so these URLs have no gold span. | 705 in the Phase 2 training partition (785 in the whole training workbook). | Investigated before Phase 4: one symptom of issue 8, corrected there. |
| 7 | **Phone country codes starting with `0`** (`+04 7024846271`): impossible in real numbering, but labelled `phone`. | 2,500 of 39,243 training-partition phone spans | The rule accepts them to follow the dataset's entity definition (docs/RULES.md). |

Found before Phase 4, investigating issue 6 on the **training workbook
only** (all 45,000 rows; the official test workbook was not opened):

| # | Issue | Scope | Handling |
|---|---|---|---|
| 8 | **Dot deletion leaves PII unlabelled.** The generator deleted some `.` characters from finished documents. Where the dot was inside an inserted value, the value no longer matched its column and was not annotated: `perrysteven@example.org` → `perrysteven@exampleorg`, `https://www.keith.com/` → `https://wwwkeith.com/` or `https://wwwkeithcom/`, `Apt. 978` → `Apt 978`, `Dr. Ann Clarke MD` → `Dr Ann Clarke MD`. | **16,607** unlabelled occurrences in 10,109 documents: email 9,136, url 4,605, address 2,517, name 349. No other character is ever deleted (checked: 0 unlabelled single non-dot deletions). | Corrected by the `dot_dropped_variant` rule (below). 0 ambiguous cases. |
| 9 | **Template count inflated by noise.** Unlabelled dot-dropped values, random reference numbers (`INV-…`, `#CLM…`), dates and IP addresses made almost every document look unique: Phase 2's key found 10,267 "templates". After masking corrected spans, deleting dots, masking digits and collapsing whitespace there are **29 templates in 27 groups**. | All 45,000 training rows | Phase 2's train/validation split therefore **shared templates**, so validation scores from it would be optimistic. It is kept for reproducibility but superseded by the NER split. |
| 10 | Paraphrased templates | 9 document types exist in 2–3 reworded versions (word similarity 0.22–0.75, below the 0.9 merge threshold). With template groups alone, two such pairs crossed partitions. | **Resolved:** 10 manually reviewed alias families in `configs/template_aliases.json` join them into split units (see below). The threshold was not lowered. |
| 11 | Tokenizer boundaries | spaCy's default English tokenizer glues a sentence-final period onto the entity's last token: `https://x.com/.`, `Inc.` (gold `Simpson Inc`), `PhD.`, `V.`, and `Jr..` → `Jr` + `..` (gold ends after `Jr.`). 28,539 spans (27,209 url, 1,319 company, 11 name) did not align strictly. | **Resolved** by `finguard.Tokenizer.v1` (below): **0** misaligned spans in train, validation and internal test. Annotations unchanged. |
| 12 | **Template concatenation defect.** Template group `tba5a9c1d21` (1,250 documents) is a complete subscription-renewal template immediately followed (`support team.During`) by a paraphrase of the CFO-audit template, which is typical of a missing comma between two Python string literals in the generator. | 1,250 training-workbook documents | Kept (labels are correct), but aliased with both source templates so neither half can appear on both sides of a split. |

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

## Phase 2 outputs (`data/processed/`, git-ignored, reproducible)

| File | Content |
|---|---|
| `train.jsonl`, `validation.jsonl` | `{"id", "text", "spans": [[start, end, label]], "group"}` |
| `test.jsonl` | `{"id", "text", "spans"}`: all 30 official records, original order |
| `audit_report.md` / `.json` | Issue counts before/after repair, the full invalid-record list, per-class statistics, checksums |

Record ids are `<split>-<Excel row>` (e.g. `train-000002` is the first data
row), so any reported problem can be found in the original file.

## NER-ready dataset (`data/processed/ner/`)

Built from the training workbook and Phase 2's `train.jsonl` +
`validation.jsonl`. The official test workbook and `test.jsonl` are **not
inputs**; the build works with them deleted (tested).

```bash
python -m finguard.data.prepare            # Phase 2 (once)
python -m finguard.data.build_ner_dataset  # corrections + 3-way template split
```

### Annotation correction: `dot_dropped_variant`

An occurrence becomes a gold span with the column's label only if **all** hold:

1. it equals the record's **own** source value with one or more `.` removed,
   all other characters identical and in order;
2. it is a whole token: no letter, digit, `_` or `@` directly before or after;
3. it overlaps neither an existing gold span nor another correction.

Anything failing 2 or 3 is listed for manual review and **not** labelled (0
cases in the real data). The rule reads only the text and the source
columns: no rule-detector or model predictions, no official test data. It is
applied to all 45,000 training-workbook rows **before** splitting, so train,
validation and internal test are corrected identically.

Justification: the variant *is* the inserted PII value (same characters, the
document's own value, at a position where that value was written). The
annotation policy labels every occurrence of an inserted value, and the
generator's dot deletion is noise, like the mojibake of issue 1.

| Label | Spans added | Documents |
|---|---|---|
| email | 9,136 | 9,065 |
| url | 4,605 | 4,576 |
| address | 2,517 | 2,384 |
| name | 349 | 254 |
| **total** | **16,607** | **10,109** |

Gold spans: 397,143 → 413,750. 14,758 corrected spans lost one dot, 1,849
lost two.

Provenance: `corrections.json` lists, per corrected record, the raw
`True Predictions` string exactly as in the workbook, and per added span
the offsets, label, text, source value, dots removed, rule id and reason.
Each JSONL record also lists its added spans under `"corrections"`. Raw
files and Phase 2 outputs are checksummed before and after and are
unchanged. This report contains the corrected entity text (synthetic
data, git-ignored folder) so it can be checked by hand.

**Official test set:** not corrected. It may contain the same defect; it
was deliberately not inspected. See "Outstanding label-quality issues".

### Template groups, alias families and split

**Template groups.** Template key: replace gold spans (including corrections)
with `<label>`, delete `.`, replace digit runs with `0`, collapse
whitespace. Templates with word similarity ≥ 0.9 are merged automatically
(0.986 and 0.984 for two variant pairs; next highest 0.746). Result: 29
templates in 27 template groups.

**Alias families (manual review).** Every pair of groups was compared by
word similarity and entity-slot sequence. Candidates were read in full side
by side. Ten families of variants of the same document were recorded in
`configs/template_aliases.json`, each with its evidence: loan application,
subscription + CFO audit (incl. the concatenation defect, issue 12),
onboarding welcome, service contract, insurance claim, security breach,
invoice payment, account creation, customer feedback, and tax-filing
confirmation (3 letters). Two low-similarity pairs were reviewed and
deliberately *not* aliased (different audit reports; internal-controls vs
tax-compliance review). The build checks every alias id and opening against
the data and stops if one no longer matches. The automatic threshold was
not lowered.

Result: **14 split units**: 1 × 6,250 rows (subscription + CFO audit),
9 × 3,750 (the document-type families) and 4 × 1,250 (stand-alone long
reports). Each regular family has exactly 3,750 documents, which supports
the grouping independently.

**Assignment** (`split.stratified_group_split`, seed 42):

1. Units with at least √(smallest × largest unit size) ≈ 2,795 rows are
   *high-frequency* (10 units), the rest *low-frequency* (4 units). The
   median is not used: it fails when half the units share the smallest size.
2. Each evaluation partition gets `max(1, round(0.1 × n))` units of each
   stratum: one high- and one low-frequency unit.
3. 1,000 stratified candidates are drawn with `random.Random(42)`. Each is
   scored by Σ|row share − target| plus the total-variation distance between
   each evaluation partition's entity-class distribution and the overall
   one, plus 1.0 per class missing from a partition. The first lowest-cost
   candidate wins (cost 0.123).

| Partition | Units (high / low) | Template groups | Records | Row share | Entities |
|---|---|---|---|---|---|
| train | 10 (8 / 2) | 21 | 35,000 | 77.8 % | 327,500 |
| validation | 2 (1 / 1): insurance claim + internal-controls review | 3 | 5,000 | 11.1 % | 46,250 |
| internal_test | 2 (1 / 1): customer feedback + independent auditor's report | 3 | 5,000 | 11.1 % | 40,000 |

The suggested 32,500 / 6,250 / 6,250 assumed 27 independent groups. After
merging variants into 14 units, the closest split with one high- and one
low-frequency unit per evaluation partition is 35,000 / 5,000 / 5,000.
Adding a second unit to each evaluation partition would take ~19 % of rows
each and leave training with ~62 %.

Entity counts:

| Partition | name | email | phone | address | ssn | credit_card | company | url |
|---|---|---|---|---|---|---|---|---|
| train | 62,500 | 35,000 | 35,000 | 38,750 | 41,250 | 33,750 | 46,250 | 35,000 |
| validation | 8,750 | 5,000 | 5,000 | 5,000 | 5,000 | 5,000 | 7,500 | 5,000 |
| internal_test | 6,250 | 3,750 | 5,000 | 5,000 | 5,000 | 3,750 | 6,250 | 5,000 |

Class distribution, **record-weighted** (share of all spans; large units weigh more):

| Partition | name | email | phone | address | ssn | credit_card | company | url |
|---|---|---|---|---|---|---|---|---|
| train | 19.1 % | 10.7 % | 10.7 % | 11.8 % | 12.6 % | 10.3 % | 14.1 % | 10.7 % |
| validation | 18.9 % | 10.8 % | 10.8 % | 10.8 % | 10.8 % | 10.8 % | 16.2 % | 10.8 % |
| internal_test | 15.6 % | 9.4 % | 12.5 % | 12.5 % | 12.5 % | 9.4 % | 15.6 % | 12.5 % |

Class distribution, **group-level** (mean of per-unit shares; every unit counts once):

| Partition | name | email | phone | address | ssn | credit_card | company | url |
|---|---|---|---|---|---|---|---|---|
| train | 16.3 % | 9.6 % | 9.6 % | 14.4 % | 11.8 % | 9.0 % | 19.6 % | 9.6 % |
| validation | 16.1 % | 10.6 % | 10.6 % | 10.6 % | 10.6 % | 10.6 % | 20.6 % | 10.6 % |
| internal_test | 15.2 % | 6.0 % | 13.2 % | 13.2 % | 13.2 % | 6.0 % | 20.3 % | 13.2 % |

Per-unit tables (records, stratum, partition, class shares) are in
`data/processed/ner/dataset_report.md`. Evaluation (Phase 4) should report
group-level (per-unit, macro-averaged) scores next to record-weighted ones,
because each evaluation partition holds only two units.

### Span canonicalisation (`finguard.canonical`)

Policy: **trailing whitespace is not part of an entity.** The end of a span
moves left past trailing whitespace; the start, label and all other spans
stay unchanged. It is applied *only* when converting to spaCy and when
comparing gold with predictions (to both sides). Stored files keep original
offsets. On the real data it fixes all credit-card token misalignments
(8,214 / 1,533 / 694 spans in train / validation / internal test).

### Tokenizer (`finguard.ner.tokenizer`)

Registered as `finguard.Tokenizer.v1` (also exposed as a `spacy_tokenizers`
entry point, so `spacy.load()` finds it for a saved model). Rules added to
spaCy's English defaults, all deterministic and derived from the alignment
audit of the internal partitions, never from model predictions:

| Rule | Fixes |
|---|---|
| suffix `(?<=/)\.` | `https://x.com/.` → `https://x.com/` `.` (27,209 url spans) |
| suffix `.` after `\b`LLC, PLC, MD, DDS, PhD, DVM, I–V | `Meza PhD.`, `Thomas V.` (default `.` rule only splits after lowercase) |
| remove special cases `Inc.`, `Ltd.` | `Simpson Inc.` → `Inc` `.` (1,319 company spans; company values never contain the dot) |
| add `<abbr>.` → `<abbr>` `.` for other `.`-ending exceptions | `Jr..` → `Jr.` `.` (name values include the dot) |

Strict alignment audit (`python -m finguard.ner.alignment`), canonical spans:

| Partition | Spans | Misaligned, default tokenizer | Misaligned, FinGuard tokenizer |
|---|---|---|---|
| train | 327,500 | 19,749 | **0** |
| validation | 46,250 | 3,982 | **0** |
| internal_test | 40,000 | 4,808 | **0** |

(Per-partition default counts differ from issue 11 because units moved
between partitions; the total, 28,539, is unchanged.) No span was dropped,
expanded, contracted or edited. Saving a pipeline and reloading it
reproduces identical token offsets (tested on 3,750 and 500 real documents
and on synthetic edge cases).

Known residual risk (0 occurrences): a dot-dropped `Jr` (correction) before
a sentence period stays one token `Jr.`, because `Jr.` is a valid name
ending in this data. If it ever occurs, the audit reports it, and the
documented resolution is a context-specific special case, not editing the
annotation.

### Outstanding label-quality issues

1. **Official test set not corrected.** If it has the dot-deletion defect,
   gold labels miss some emails/urls/addresses/names, and correct detections
   will count as false positives there. Options: report it as is, or apply
   the same rule (designed without the test file) as a documented,
   prediction-independent step at final evaluation. **Your decision.**
2. Only deleted dots are detected. Other generator noise (e.g. a mangled
   value of a label without dots) would not be found; a search for single
   non-dot deletions found none.
3. Alias families rest on manual review of 27 template groups. If the
   dataset or correction rule changes, group ids change and the build stops
   until the alias file is reviewed again (by design).
4. Each evaluation partition contains only two split units (three
   distinct templates), so scores will vary with the template drawn.
5. Card-holder names inside credit-card blocks are part of the block, not
   separate `name` spans; this follows the dataset's definition.
6. The corpus has only 27 templates; any model trained on it will see
   little linguistic variety. Results on real documents will be lower.
