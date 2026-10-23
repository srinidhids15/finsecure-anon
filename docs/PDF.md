# PDF detection and redaction (Phase 8)

`finguard.pdf` redacts detected PII in **text-based** PDFs. Each page's
text is built with a character-level geometry map, passed to the frozen
hybrid detector, mapped back to exact glyph rectangles and physically
removed with PyMuPDF (1.28.2) redactions. The result is then reopened and
verified. Everything happens in memory:

```python
from finguard.pdf.pipeline import process_pdf
from finguard.detection.hybrid import HybridDetector

result = process_pdf(pdf_bytes, HybridDetector())   # raises PdfError for rejected documents
result.verified        # True only if every check below passed
result.output          # redacted PDF bytes, or None
```

## Pipeline

`validate → sanitize → extract → detect → map → redact → save → verify`
(`pdf/pipeline.py`). Each stage is a separate module:

| module | job |
|---|---|
| `validate.py` | size and signature checks, safe open, limits, hidden-content inventory |
| `redact.py` | `sanitize_document` (hidden content), `apply_redactions`, `save_pdf` |
| `extract.py` | page text and per-character geometry (`PageText`), page conditions |
| `mapping.py` | entities → per-line rectangles; collateral pre-check |
| `verify.py` | independent re-extraction checks of the saved output |
| `report.py` | CSV and JSON reports without values |

## Supported input and limits

**Supported:** unencrypted, text-based PDFs (selectable text) with
horizontal, unrotated text. Limits are in `PdfLimits` and are configurable:

| limit | default |
|---|---|
| file size | 20 MB (also `server.maxUploadSize` in `.streamlit/config.toml`) |
| pages | 100 |
| characters per page | 100,000 |
| characters in total | 500,000 |

**Rejected** with a message naming the page and condition. The whole
document is refused, because unprocessed pages could still hold PII:

- **Not a usable PDF:** empty files, files without a PDF header, and files
  that fail to open or need structural repair.
- **Encrypted:** any encryption, including owner-password-only files.
- **Out of scope:** no pages, or over the limits.
- **Image pages:** pages with no extractable text but images (scanned).
  **OCR is not implemented.** A fully image-based PDF gets a dedicated
  "scanned or image-only" message.
- **Vector-only pages:** pages with no text but vector graphics (text may be
  drawn as outlines).
- **Rotation:** rotated pages, and rotated or vertical text.
- **Hidden text:** extractable text that is not visible, which a reviewer
  could not spot in the output:
  - invisible text (render modes 3/7 or opacity 0, e.g. OCR layers);
  - text outside the visible page area;
  - text **covered by an image or filled shape drawn after it**, by draw
    order (`get_bboxlog` indices = `get_texttrace` `seqno`). Filled shapes
    count only if at least half a glyph high, so rules and strike-throughs
    do not trigger it;
  - text clipped out of view (fewer glyphs with `TEXT_CLIP`).
- **Unreadable text:** text whose extracted characters cannot be trusted to
  match what is drawn:
  - glyphs without a Unicode mapping (U+FFFD, e.g. embedded fonts without
    `ToUnicode`);
  - **Type 3 fonts without `ToUnicode`**, whose glyph names give readable but
    wrong text;
  - pages where MuPDF's text trace shows more glyphs than the character
    extraction returned.
- **Replacement and alternate text:** marked-content `ActualText`, `Alt` or
  `E` strings in page content or property lists. They are stored apart from
  the glyphs and would survive glyph redaction. Such strings that exist only
  in the logical-structure tree are removed with the tree, so they are
  accepted.
- **Unsupported structures:** optional-content layers and XFA forms.

**Accepted with a note:** blank pages, and text pages that also contain
images. Text *inside* images is neither detected nor redacted, and the
report says so per page.

## Edge cases (Phase 9 audit)

Each case below was examined with synthetic PDFs written byte by byte
(`tests/pdf_fixtures.raw_pdf`) and is covered by
`tests/test_pdf_edge_cases.py`.

| case | handling | evidence |
|---|---|---|
| Ligatures ("ﬁ", "ﬃ") | **Supported.** MuPDF expands a ligature into its letters: the first gets the glyph box, the rest get zero-width boxes at its right edge. Such characters share a `glyph` id. An entity containing whole ligatures is removed and verified | output text, content streams and region re-extraction |
| Entity boundary inside a ligature | **Rejected** (`mapping_failed`, no output): MuPDF removes whole glyphs only. If the check were bypassed, verification would still fail | mutation check |
| One code mapped to several characters (`ToUnicode`) | **Supported**, handled like a ligature, including boundary rejection | |
| Type 3 font with `ToUnicode` | **Supported**: extraction, mapping, removal and verification | 7 glyphs removed, unrelated text intact |
| Type 3 font without `ToUnicode` | **Rejected** | MuPDF read every glyph as its glyph name, giving readable but wrong text |
| Embedded font without `ToUnicode` | **Rejected** (U+FFFD) | |
| Text drawn over a background image | **Supported**; the image is kept, and pixels under a redaction are blanked | |
| Text covered by a later image or opaque box | **Rejected**: hidden but extractable | |
| Thin rule or strike-through over text | **Supported** (not treated as hiding) | |
| Clipped-out text | **Rejected** | |
| Text only in image pixels | **Not processed** (no OCR). Image-only pages are rejected; text pages carry a per-page note | |
| `ActualText` differing from glyphs | **Rejected**. Extraction uses the drawn glyphs (`TEXT_IGNORE_ACTUALTEXT`) | before this audit such a page was accepted: the replacement string was scanned instead of the drawn name |
| `ActualText`/`Alt`/`E` in content or property lists | **Rejected**, and verification checks the output for them too | |
| `ActualText` only in the structure tree | **Supported**: the tree is removed and the output has no such strings | |

## Geometry mapping

`extract.extract_page` reads `page.get_text("rawdict", flags=TEXT_PRESERVE_WHITESPACE |
TEXT_IGNORE_ACTUALTEXT, clip=INFINITE_RECT())`. MuPDF clips extraction to the page unless it is
given an unbounded clip, so off-page text is still read and can be
rejected. It builds one string and a parallel tuple `geoms` with
`len(geoms) == len(text)`:

- **Characters:** every extracted character is appended as is. Unicode is
  preserved, and ligatures are expanded by MuPDF.
- **Same line:** spans in a line are concatenated directly. Consecutive
  lines on the **same baseline** (separate runs of one row, left to right)
  are joined by one synthetic space.
- **Other boundaries:** every other line or block boundary becomes one
  synthetic `"\n"`. A card block keeps the `"\n"`-separated layout the
  detector was trained on.
- **Geometry:** `geoms[i]` is the glyph's `rawdict` box, its origin and its
  line number. Whitespace, MuPDF synthetic spaces and the separators have
  `None`. Every visible character has geometry.

**Detection is per page.** The same string is passed to the detector, so
offsets are page-local, and an entity can never cross a page boundary. A
value split across two pages is two partial values and may not be
detected.

**Mapping** (`mapping.map_page`) reuses `anonymization.anonymize` on the
page text for validation (offsets, exact text, labels, overlaps) and
canonicalisation (no trailing whitespace). It then builds **one rectangle
per line**: the union of the visible glyph boxes of the canonical span on
that line. So:
- a four-line card block gets four rectangles;
- a wrapped address gets two;
- whitespace between lines is never covered.

There is no text search, so a repeated value is redacted only where it was
detected (regression tests cover the same name on two pages, and twice on
one page).

**Collateral pre-check.** MuPDF removes a character when the centre of its
bounding box lies inside a redaction rectangle (measured on 1.28.2). If the
centre of any untargeted visible glyph on the page lies within a rectangle
enlarged by 0.5 pt, the mapping is rejected and no PDF is produced. Tight
line spacing triggers this. Mappings are also rejected when a span contains
a visible character without geometry.

## Redaction

- One redaction annotation per rectangle, **opaque black fill**, no overlay
  text. Placeholder labels are not drawn into the PDF; the CSV says what
  each region was.
- `page.apply_redactions(images=PDF_REDACT_IMAGE_PIXELS, graphics=PDF_REDACT_LINE_ART_NONE,
  text=PDF_REDACT_TEXT_REMOVE)`. Covered text is deleted from the content
  stream, and image pixels under a rectangle are blanked. Vector graphics
  such as table rules are kept.
- The document is saved to new bytes with `garbage=4, deflate=True,
  clean=True`, so replaced content streams are not retained. The uploaded
  bytes are never modified or written anywhere.

## Hidden content

The sanitize step runs **before** extraction, so the scanned text is
exactly the page content that remains. It removes, on every page:

- annotations and comments (including file attachments), links and form
  fields. Their contents are **removed, not scanned**, and form values are
  lost;
- outlines (bookmarks), the logical-structure tree and `MarkInfo`;
- catalog `OpenAction`/`AA` and page `AA` automatic actions, and `AcroForm`;
- through `Document.scrub`: document metadata, XMP metadata, JavaScript,
  embedded files, thumbnails, and content stream cleaning.

PyMuPDF does not execute JavaScript, follow links or open embedded files;
they are only counted (`HiddenContent`) and removed. Optional-content
layers and XFA forms are rejected rather than partially sanitized.

## Verification

`verify.verify_redacted_pdf` reopens the saved bytes as a new document and
re-extracts every page. It does not reuse the detection string. It
reports separate outcomes:

| check | meaning |
|---|---|
| `opened` | output opens, is not encrypted, needed no repair |
| `structure_preserved` | same page count and page sizes |
| `redaction_completed` | every rectangle has a black fill at its position; no unapplied redactions |
| `targeted_text_removed` | no glyph's centre lies in any redaction rectangle, and no targeted glyph remains at its original position |
| `unrelated_text_preserved` | every untargeted visible source glyph is still present (same character, same position ± 0.1 pt) |
| `hidden_content_removed` | no annotations, links, fields, metadata, XMP, embedded files, outlines, structure tree, actions or JavaScript |
| `complete_pii_removal_guaranteed` | always **false** |

The output is not searched for original values. A value can legitimately
remain elsewhere because it was not detected, and its absence elsewhere
proves nothing. `result.output` is set only if detection mapped every
entity, redaction and saving succeeded, and every check passed. Otherwise
`status` is `mapping_failed`, `redaction_failed` or `verification_failed`,
and there is no output.

## Privacy

- Processing is in memory. Nothing is written to disk, printed or logged,
  and MuPDF's own warning output is switched off.
- `PageText.text` is excluded from `repr`.
- Error messages, problems and reports contain page numbers, labels,
  page-local offsets, rectangles, sources, counts and statuses, never
  extracted text or values.
- Output file names are fixed (`finguard_redacted.pdf`, …) and never
  derived from the upload.

## Limitations

- **No OCR.** Text inside images, scanned pages, and text drawn as vector
  outlines is not processed. Such pages are rejected, or, for images on text
  pages, reported in a note.
- **Rotation and layouts.** Rotated pages, rotated or vertical text, layered
  PDFs and XFA forms are rejected. Multi-column layouts are processed in
  MuPDF's content order, which can join unrelated columns in the detection
  text. This can affect detection but not the geometry mapping.
- **Tight layouts.** In very tight line spacing, glyph boxes overlap the
  neighbouring line, so the mapping is rejected and no PDF is produced.
- **Hidden text and unusual fonts.** These are rejected rather than
  processed: covered, clipped or invisible text, Type 3 fonts without
  `ToUnicode`, and replacement or alternate text (see **Edge cases**).
  Detecting covered text assumes MuPDF's draw-order logs. Soft masks,
  blend modes and transparency groups were not tested.
- **Lying Unicode maps.** A font whose `ToUnicode` map deliberately maps
  glyphs to the wrong characters cannot be detected. The scanned text would
  then differ from what is drawn.
- **Detector quality limits redaction.** Missed PII stays in the PDF, and
  false positives are also redacted. Examples from the synthetic end-to-end
  test: an invoice reference was redacted as an SSN, and a name on page two
  was missed in a trial document.
- **No batch processing.**

## Tests

```bash
python -m pytest tests/test_pdf.py tests/test_pdf_edge_cases.py tests/test_pdf_e2e.py tests/test_app_pdf.py tests/test_integrity.py
```

All PDFs are generated synthetically in `tests/pdf_fixtures.py`.
