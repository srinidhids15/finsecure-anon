# FinGuard app (Phase 7)

A Streamlit interface for scanning plain text for PII, inspecting what was
found and anonymizing it. It is a thin layer over existing components:

- **Scanning**: the frozen `HybridDetector` (rules + spaCy NER + merge policy).
- **Redaction and verification**: `finguard.anonymization.anonymize`.
- **Workflow, session state, highlighting and exports**: `finguard.interface`.
  This package does not import Streamlit and is unit-tested directly.
- **Rendering**: `app/streamlit_app.py` only renders. It contains no
  detection, overlap-resolution, canonicalisation or replacement logic.

## Run

```bash
source .venv/bin/activate
pip install -r requirements.txt          # streamlit 1.65.0 is already pinned
python -m finguard.ner.train             # only if models/finguard_ner/model-best is missing
streamlit run app/streamlit_app.py       # from the repository root → http://localhost:8501
```

Start it from the repository root so `.streamlit/config.toml` is applied.
That file sets four things:
- the server listens on `localhost` only;
- usage statistics are not sent;
- the browser shows generic error messages, not exception details;
- developer menu items are hidden.

## Workflow

1. **Input.** Paste text, or click **Load synthetic sample**, which loads an
   invented statement. The limit is 100,000 characters, about 0.85 s of
   detection on CPU in a local measurement. Empty, whitespace-only and
   oversized input is rejected with a message.
2. **Scan for PII.** This runs the hybrid detector once. You then see:
   - the number of entities and one count per label (all eight labels);
   - **Highlighted text**: the original text, HTML-escaped, with each entity
     marked in its label colour at the detector's exact offsets. Every
     character of the input is kept; repeated values are marked only where
     they were detected;
   - **Entity table**: label, start, end (0-based, end exclusive), length and
     detector sources. Detected values are not repeated in the table.
3. **Anonymize detected entities.** This calls `anonymize(text, entities)`.
   Output is shown only if the built-in verification passed:
   - a status line, "Detected-span replacement verified", and a separate
     warning that this does **not** guarantee all PII was removed;
   - the sanitized text, the redaction count and the placeholders used (per
     label);
   - any notes from the anonymizer, such as trailing whitespace kept after a
     card block.

   If anonymization is refused (for example, overlapping spans), an error
   is shown. In that case there is no success status, no sanitized text and
   no download.
4. **Downloads.** These appear only after verified anonymization. They are
   generated in memory and use fixed file names:

   | file | content |
   |---|---|
   | `finguard_sanitized.txt` | the verified sanitized text (UTF-8) |
   | `finguard_entity_report.csv` | `index, label, start, end, replaced_end, whitespace_kept, output_start, output_end, placeholder, sources` |
   | `finguard_redaction_summary.json` | counts, redactions (as in the CSV), warnings, verification status, detector policy and freeze hash |

   Offsets in the CSV and JSON refer to the original input. `[start,
   replaced_end)` is the replaced span, and `output_start`/`output_end`
   locate the placeholder in the TXT file. The CSV and JSON do not contain
   detected values or document text.

## Session state

- Each browser session keeps its own scan and anonymization result in
  `st.session_state`. A scan records the exact text it ran on and a random
  `scan_id`. An anonymization result records the `scan_id` it belongs to.
- **Stale results are cleared.** If the input differs from the scanned text,
  both results are discarded and the user is asked to scan again. A new
  scan discards the previous anonymization result.
- **Only the model is shared.** The detector is loaded once per server
  process (`st.cache_resource`) and shared by all sessions behind a lock. It
  holds only the model. No text or results are cached globally.

## Privacy

- **Local processing.** Text is processed by the local Python process; no
  external service is called.
- **Nothing is persisted.** The app prints, logs and writes nothing.
  Submitted text exists only in the session's memory until the input
  changes, **Clear** is pressed or the session ends.
- **No document text in errors.** Detector and initialisation failures show
  only the exception type. Anonymizer errors contain only labels and offsets.
- **Escaped rendering.** The highlight view escapes all document text
  (`html.escape`) and renders it with `st.html`, so markdown is not applied
  to it.
- **Values only where needed.** Detected values appear only in the
  highlighted view of the user's own input. The table, CSV and JSON do not
  contain them.

## Limitations

- **Plain text only.** There is no PDF upload or redaction and no batch
  processing.
- **The detector makes mistakes,** and missed PII stays in the output. On
  the synthetic sample, the real detector:
  - labels the header "SAMPLE BANK" as a phone number;
  - labels the two balance amounts as addresses (one span runs into the next
    line);
  - misses the final repeated "Ben Lee".

  Internal-test `name` F1 was 0.522 (docs/EVALUATION.md). Always review the
  output.
- **One request at a time.** The shared detector processes one request at a
  time, which is fine for local single-user use, not for serving many users.
- **Not certified.** This is a research prototype, not a certified or
  production-grade privacy tool.

## Tests

```bash
python -m pytest tests/test_interface.py tests/test_app.py
```

- `test_interface.py` covers highlighting, escaping, offsets, input
  validation, safe errors, session-state invalidation and exports.
- `test_app.py` drives the real app script with `streamlit.testing.v1.AppTest`:
  - startup and missing-model handling;
  - scanning, counts, table and highlights;
  - anonymization success and failure, and downloads;
  - stale-result invalidation and session isolation;
  - terminal output.

  Most tests use a stub detector. One uses the real hybrid detector, and one
  checks that the evaluation freeze still verifies. All inputs are synthetic.
