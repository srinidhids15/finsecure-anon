"""Tests for PDF validation, geometry mapping, redaction and verification.

Synthetic PDFs only (tests/pdf_fixtures.py). A stub detector makes the
detections deterministic; tests/test_pdf_e2e.py uses the real detector.
"""

import json
from dataclasses import replace

import pymupdf
import pytest
from pdf_fixtures import (
    LEFT,
    TOP,
    FindDetector,
    build_pdf,
    encoded_forms,
    glyph_text,
    image_only_page,
    page_text,
    stream_bytes,
)

from finguard.pdf import pipeline
from finguard.pdf import report as pdf_report
from finguard.pdf.errors import PdfProcessingError, PdfValidationError, UnsupportedPdfError
from finguard.pdf.extract import extract_page
from finguard.pdf.mapping import map_page
from finguard.pdf.pipeline import process_pdf
from finguard.pdf.validate import PdfLimits, open_pdf
from finguard.pdf.verify import verify_redacted_pdf
from finguard.schema import Entity

CARD = ["VISA 16 digit", "Asha Rao", "4111111111111111 12/27", "CVC: 123"]
SECRETS = ("Asha Rao", "asha.rao@example.com", "4111111111111111", "CVC: 123", "048-59-6922", "Ben Lee")


def no_secrets(blob: str) -> bool:
    return not any(s in blob for s in SECRETS)


def result_text(result) -> str:
    return json.dumps(pdf_report.summary(result)) + pdf_report.entity_report_csv(result).decode() + repr(result)


# --- validation -------------------------------------------------------------------

@pytest.mark.parametrize("data, match", [
    (b"", "empty"),
    (b"hello, this is text", "not a PDF"),
    (b"%PDF-1.7\n" + b"\x00garbage" * 50, "damaged|could not be opened"),
])
def test_invalid_bytes_rejected(data, match):
    with pytest.raises(PdfValidationError, match=match):
        process_pdf(data, FindDetector())


def test_truncated_pdf_rejected():
    data = build_pdf([["Dear Asha Rao,"]])
    with pytest.raises(PdfValidationError):
        process_pdf(data[: len(data) // 2], FindDetector())


@pytest.mark.parametrize("user_pw", ["secret", ""])
def test_encrypted_pdf_rejected(user_pw):
    data = build_pdf([["Dear Asha Rao,"]], encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner",
                     user_pw=user_pw)
    with pytest.raises(PdfValidationError, match="Encrypted"):
        process_pdf(data, FindDetector())


def test_size_page_and_character_limits():
    data = build_pdf([["Dear Asha Rao,"]] * 3)
    with pytest.raises(PdfValidationError, match="limit is"):
        process_pdf(data, FindDetector(), PdfLimits(max_bytes=100))
    with pytest.raises(PdfValidationError, match="3 pages; the limit is 2"):
        process_pdf(data, FindDetector(), PdfLimits(max_pages=2))
    with pytest.raises(UnsupportedPdfError, match="per page"):
        process_pdf(data, FindDetector(), PdfLimits(max_chars_per_page=5))
    with pytest.raises(UnsupportedPdfError, match="limit is 20"):
        process_pdf(data, FindDetector(), PdfLimits(max_total_chars=20))


def test_scanned_image_only_pdf_rejected():
    doc = pymupdf.open()
    image_only_page(doc)
    with pytest.raises(UnsupportedPdfError, match="scanned or image-only.*OCR is not supported"):
        process_pdf(doc.tobytes(), FindDetector())


def test_partially_scanned_pdf_rejected_with_page_number():
    data = build_pdf([["Dear Asha Rao,"]], setup=image_only_page)
    with pytest.raises(UnsupportedPdfError, match=r"1 of 2 page\(s\).*page 2: no extractable text; the page is an image"):
        process_pdf(data, FindDetector())


def test_blank_page_is_accepted_and_noted():
    data = build_pdf([["Dear Asha Rao,"], []])
    result = process_pdf(data, FindDetector(("Asha Rao", "name")))
    assert result.verified
    assert result.pages[1].notes == ("blank page",)


def test_text_page_with_image_is_processed_with_a_note():
    def add_image(doc):
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
        pix.set_rect(pix.irect, (255, 255, 255))
        doc[0].insert_image(pymupdf.Rect(400, 400, 450, 450), pixmap=pix)

    result = process_pdf(build_pdf([["Dear Asha Rao,"]], setup=add_image), FindDetector(("Asha Rao", "name")))
    assert result.verified
    assert "text inside images is not detected" in result.pages[0].notes[0]


@pytest.mark.parametrize("setup, match", [
    (lambda d: d[0].set_rotation(90), "page rotation of 90 degrees"),
    (lambda d: d[0].insert_text((300, 300), "Asha Rao", fontsize=11, rotate=90), "rotated or vertical text"),
    (lambda d: d[0].insert_text((72, 300), "Hidden Asha Rao", fontsize=11, render_mode=3), "invisible text"),
    (lambda d: d[0].insert_text((72, -40), "Off page Asha Rao", fontsize=11), "outside the visible page"),
])
def test_unsafe_page_conditions_rejected(setup, match):
    data = build_pdf([["Dear Asha Rao,"]], setup=setup)
    with pytest.raises(UnsupportedPdfError, match=match) as info:
        process_pdf(data, FindDetector(("Asha Rao", "name")))
    assert no_secrets(str(info.value))


def test_optional_content_layers_rejected():
    def layered(doc):
        ocg = doc.add_ocg("hidden layer", on=False)
        doc[0].insert_text((72, 300), "Asha Rao", fontsize=11, oc=ocg)

    with pytest.raises(UnsupportedPdfError, match="optional-content layer"):
        process_pdf(build_pdf([["Dear Asha Rao,"]], setup=layered), FindDetector())


def test_source_bytes_unchanged():
    data = build_pdf([["Dear Asha Rao,"]])
    copy = bytes(data)
    result = process_pdf(data, FindDetector(("Asha Rao", "name")))
    assert data == copy and result.output != data


# --- extraction and geometry -------------------------------------------------------

def extract(data, n=1):
    with open_pdf(data) as doc:
        return extract_page(doc[n - 1])


def test_page_text_and_geometry_map_align():
    lines = ["Dear Asha Rao,", "Balance: 4,500.00 USD"] + CARD
    page = extract(build_pdf([lines]))
    assert page.text == "\n".join(lines)
    assert len(page.geoms) == len(page.text)
    for i, ch in enumerate(page.text):
        assert (page.geoms[i] is None) == ch.isspace()
    first = page.geoms[0]
    assert first.origin == pytest.approx((LEFT, TOP), abs=0.01)
    a = page.text.index("Asha")
    assert page.geoms[a].line == 0 and page.geoms[page.text.index("Balance")].line == 1


def test_same_baseline_runs_joined_by_space_and_lines_by_newline():
    data = build_pdf([[(72, 100, "Name:"), (150, 100, "Asha Rao"), (300, 100, "Total 4,500.00"), (72, 130, "Next")]])
    assert extract(data).text == "Name: Asha Rao Total 4,500.00\nNext"


def test_unicode_characters_preserved():
    # The built-in Helvetica writes Latin-1 only; the text is what the PDF contains.
    page = extract(build_pdf([["Zoë Ångström · £12,50 ¿ok? Müller-Lüdenscheidt"]]))
    assert page.text == "Zoë Ångström · £12,50 ¿ok? Müller-Lüdenscheidt"
    assert all(g is not None for ch, g in zip(page.text, page.geoms) if not ch.isspace())


def test_map_card_block_one_rect_per_line_without_neighbours():
    lines = ["Card on file:"] + CARD + ["Balance: 4,500.00 USD"]
    page = extract(build_pdf([lines]))
    block = "\n".join(CARD)
    start = page.text.index(block)
    mapping = map_page(page, [Entity.from_text(page.text, start, start + len(block), "credit_card", "rule:x")])
    (e,) = mapping.entities
    assert e.status == "mapped" and len(e.rects) == 4
    tops = [r[1] for r in e.rects]
    assert tops == sorted(tops)
    for r in e.rects:  # each rect is one line high
        assert r[3] - r[1] < 16


def test_mapping_rejects_characters_without_geometry():
    page = extract(build_pdf([["Dear Asha Rao,"]]))
    i = page.text.index("Asha")
    broken = replace(page, geoms=page.geoms[:i] + (None,) + page.geoms[i + 1:])
    mapping = map_page(broken, [Entity.from_text(page.text, i, i + 8, "name", "ner")])
    assert mapping.rejected[0].reason == "characters without page geometry"


def test_mapping_rejects_collateral_when_lines_overlap():
    # Lines 4 pt apart: the name's glyph boxes cover the centres of the next line's glyphs.
    data = build_pdf([[(72, 100, "Asha Rao"), (72, 104, "Total 4,500.00")]])
    result = process_pdf(data, FindDetector(("Asha Rao", "name")))
    assert result.status == "mapping_failed" and result.output is None and not result.verified
    assert "would remove" in result.problems[0] and "unrelated" in result.problems[0]
    assert no_secrets(" ".join(result.problems))


def test_invalid_detector_output_fails_without_output():
    class Wrong:
        def detect(self, text):
            return [Entity(0, 4, "name", "ner", "Zzzz")]  # does not match the page text

    result = process_pdf(build_pdf([["Dear Asha Rao,"]]), Wrong())
    assert result.status == "mapping_failed" and result.output is None
    assert "detector output rejected" in result.problems[0] and no_secrets(result.problems[0])


def test_overlapping_detector_output_fails_without_output():
    result = process_pdf(build_pdf([["Dear Asha Rao,"]]), FindDetector(("Asha Rao", "name"), ("Asha", "company")))
    assert result.status == "mapping_failed" and result.output is None
    assert "overlaps" in result.problems[0]


def test_detector_exception_message_is_safe():
    class Boom:
        def detect(self, text):
            raise RuntimeError(text)

    with pytest.raises(PdfProcessingError, match=r"Detection failed on page 1 \(RuntimeError\)") as info:
        process_pdf(build_pdf([["Dear Asha Rao,"]]), Boom())
    assert no_secrets(str(info.value))


# --- redaction and verification ---------------------------------------------------

def test_detected_value_physically_removed_at_its_location():
    data = build_pdf([["Dear Asha Rao, balance 4,500.00"]])
    result = process_pdf(data, FindDetector(("Asha Rao", "name")))
    assert result.verified and result.status == "verified"
    (e,) = result.entities
    (rect,) = e.rects
    assert glyph_text(data, 1, rect) == "Asha Rao"           # present in the source at that location
    assert glyph_text(result.output, 1, rect) == ""           # not extractable from the output there
    assert page_text(result.output, 1).startswith("Dear ")    # neighbours kept
    assert "4,500.00" in page_text(result.output, 1)
    # Physically gone from the file, not covered: no content stream holds the value.
    assert any(f in stream_bytes(data) for f in encoded_forms("Asha Rao"))
    assert not any(f in stream_bytes(result.output) for f in encoded_forms("Asha Rao"))
    v = result.verification
    assert v.redaction_completed and v.targeted_text_removed and v.unrelated_text_preserved
    assert v.complete_pii_removal_guaranteed is False


def test_black_fill_at_region_and_page_structure_preserved():
    data = build_pdf([["Dear Asha Rao,"], ["Page two"]])
    result = process_pdf(data, FindDetector(("Asha Rao", "name")))
    with pymupdf.open(stream=result.output, filetype="pdf") as out, pymupdf.open(stream=data, filetype="pdf") as src:
        assert out.page_count == 2 and [p.rect for p in out] == [p.rect for p in src]
        fills = [d for d in out[0].get_drawings() if d.get("fill") == (0.0, 0.0, 0.0)]
        assert len(fills) == 1
        assert tuple(fills[0]["rect"]) == pytest.approx(result.entities[0].rects[0], abs=0.5)
        assert out[1].get_text().strip() == "Page two"


def test_adjacent_financial_value_on_same_line_preserved():
    data = build_pdf([["Payee Asha Rao 4,500.00 USD due 01/04"]])
    result = process_pdf(data, FindDetector(("Asha Rao", "name")))
    assert result.verified
    assert page_text(result.output, 1).split() == ["Payee", "4,500.00", "USD", "due", "01/04"]


def test_multiline_card_block_every_line_removed():
    lines = ["Card on file:"] + CARD + ["Thank you"]
    data = build_pdf([lines])
    result = process_pdf(data, FindDetector(("\n".join(CARD), "credit_card")))
    assert result.verified and len(result.entities[0].rects) == 4
    text = page_text(result.output, 1)
    for part in ("VISA", "Asha Rao", "4111111111111111", "12/27", "CVC", "123"):
        assert part not in text
    assert text.split() == ["Card", "on", "file:", "Thank", "you"]


def test_wrapped_entity_across_two_lines():
    data = build_pdf([["Mail to: 12 Lake Road,", "Springfield, IL 62704 today"]])
    piece = "12 Lake Road,\nSpringfield, IL 62704"
    result = process_pdf(data, FindDetector((piece, "address")))
    assert result.verified and len(result.entities[0].rects) == 2
    assert page_text(result.output, 1).split() == ["Mail", "to:", "today"]


def test_repeated_values_on_one_page_and_across_pages():
    data = build_pdf([["Asha Rao and Asha Rao"], ["Asha Rao again"]])
    result = process_pdf(data, FindDetector(("Asha Rao", "name")))
    assert result.verified and [e.page for e in result.entities] == [1, 1, 2]
    assert page_text(result.output, 1).split() == ["and"]
    assert page_text(result.output, 2).split() == ["again"]


def test_regression_same_name_two_pages_only_one_detected():
    """Only page 2's occurrence is detected: page 1 must stay untouched (no global search)."""
    data = build_pdf([["Statement for Asha Rao"], ["Signed: Asha Rao (page two)"]])
    result = process_pdf(data, FindDetector(("Asha Rao", "name"), only_on="page two"))
    assert result.verified and [e.page for e in result.entities] == [2]
    assert page_text(result.output, 1).strip() == "Statement for Asha Rao"
    assert "Asha Rao" not in page_text(result.output, 2)
    with pymupdf.open(stream=result.output, filetype="pdf") as out:
        assert not [d for d in out[0].get_drawings() if d.get("fill") == (0.0, 0.0, 0.0)]


def test_regression_same_page_only_second_occurrence_detected():
    class Second:
        def detect(self, text):
            i = text.index("Asha Rao", text.index("Asha Rao") + 1)
            return [Entity.from_text(text, i, i + 8, "name", "ner")]

    data = build_pdf([["Asha Rao paid; Asha Rao signed"]])
    result = process_pdf(data, Second())
    assert result.verified
    assert page_text(result.output, 1).split() == ["Asha", "Rao", "paid;", "signed"]


def test_no_detections_still_sanitizes_and_verifies():
    result = process_pdf(build_pdf([["Nothing here"]]), FindDetector())
    assert result.verified and result.entities == () and result.regions_applied == 0


def test_hidden_content_removed():
    def hidden(doc):
        page = doc[0]
        page.add_text_annot((300, 300), "Note about Asha Rao")
        page.insert_link({"kind": pymupdf.LINK_URI, "from": pymupdf.Rect(72, 60, 200, 80),
                          "uri": "mailto:asha.rao@example.com"})
        w = pymupdf.Widget()
        w.field_type = pymupdf.PDF_WIDGET_TYPE_TEXT
        w.field_name = "holder"
        w.field_value = "Asha Rao"
        w.rect = pymupdf.Rect(72, 400, 300, 420)
        page.add_widget(w)
        doc.embfile_add("notes.txt", b"Asha Rao 048-59-6922")
        doc.set_metadata({"author": "Asha Rao", "title": "Statement 048-59-6922"})
        doc.set_xml_metadata("<x:xmpmeta xmlns:x='adobe:ns:meta/'>Asha Rao</x:xmpmeta>")
        doc.set_toc([[1, "Asha Rao statement", 1]])
        doc.xref_set_key(doc.pdf_catalog(), "OpenAction", "<</S/JavaScript/JS(app.alert('x'))>>")

    data = build_pdf([["Dear customer,"]], setup=hidden)
    result = process_pdf(data, FindDetector())
    found = result.hidden
    assert (found.annotations, found.links, found.form_fields, found.embedded_files) == (1, 1, 1, 1)
    assert found.metadata_fields == 2 and found.xml_metadata and found.outline_entries == 1 and found.javascript
    assert result.verified and result.verification.hidden_content_removed
    s = result.sanitization
    assert (s.annotations_removed, s.links_removed, s.form_fields_removed) == (1, 1, 1)
    out_bytes = stream_bytes(result.output) + result.output
    assert not any(f in out_bytes for f in encoded_forms("Asha Rao") + encoded_forms("048-59-6922"))
    with pymupdf.open(stream=result.output, filetype="pdf") as out:
        assert out.embfile_count() == 0 and out.get_toc() == [] and out.get_xml_metadata() == ""
        assert not any(v for k, v in out.metadata.items() if k not in ("format", "encryption"))
        assert out[0].first_annot is None and not out[0].get_links() and out[0].first_widget is None


def test_overlay_without_removal_fails_verification():
    """A black rectangle drawn over text is not a redaction: verification must say so."""
    data = build_pdf([["Dear Asha Rao,"]])
    detector = FindDetector(("Asha Rao", "name"))
    good = process_pdf(data, detector)
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        doc[0].draw_rect(pymupdf.Rect(good.entities[0].rects[0]), color=None, fill=(0, 0, 0))
        overlay = doc.tobytes()
    mappings = tuple(replace(m, entities=tuple(replace(e, status="mapped") for e in m.entities))
                     for m in good.mappings)
    v = verify_redacted_pdf(overlay, good.pages, mappings)
    assert not v.verified and not v.targeted_text_removed and v.redaction_completed
    assert any("still has" in p for p in v.problems) and no_secrets(" ".join(v.problems))


def test_wrong_geometry_fails_verification():
    data = build_pdf([["Dear Asha Rao, balance 4,500.00"]])
    good = process_pdf(data, FindDetector(("Asha Rao", "name")))
    shifted = tuple(
        replace(m, entities=tuple(replace(e, status="mapped", rects=tuple((r[0] + 60, r[1], r[2] + 60, r[3])
                                                                          for r in e.rects))
                                  for e in m.entities))
        for m in good.mappings)
    v = verify_redacted_pdf(good.output, good.pages, shifted)
    assert not v.verified and not v.redaction_completed


def test_collateral_loss_detected_by_verification():
    data = build_pdf([["Dear Asha Rao, balance 4,500.00"]])
    good = process_pdf(data, FindDetector(("Asha Rao", "name")))
    with pymupdf.open(stream=good.output, filetype="pdf") as doc:
        page = doc[0]
        page.add_redact_annot(page.search_for("4,500.00")[0], fill=(0, 0, 0))
        page.apply_redactions()
        damaged = doc.tobytes()
    mappings = tuple(replace(m, entities=tuple(replace(e, status="mapped") for e in m.entities))
                     for m in good.mappings)
    v = verify_redacted_pdf(damaged, good.pages, mappings)
    assert not v.unrelated_text_preserved and any("unrelated glyph" in p for p in v.problems)


def test_unreadable_output_fails_verification():
    data = build_pdf([["Dear Asha Rao,"]])
    good = process_pdf(data, FindDetector(("Asha Rao", "name")))
    v = verify_redacted_pdf(b"not a pdf", good.pages, good.mappings)
    assert not v.verified and not v.opened


def test_failed_verification_withholds_output(monkeypatch):
    from finguard.pdf.verify import PdfVerification

    monkeypatch.setattr(pipeline, "verify_redacted_pdf",
                        lambda *a: PdfVerification(opened=True, problems=("page 1: forced failure",)))
    result = process_pdf(build_pdf([["Dear Asha Rao,"]]), FindDetector(("Asha Rao", "name")))
    assert result.status == "verification_failed" and result.output is None and not result.verified


def test_save_failure_withholds_output(monkeypatch):
    def boom(doc):
        raise RuntimeError("disk")

    monkeypatch.setattr(pipeline, "save_pdf", boom)
    result = process_pdf(build_pdf([["Dear Asha Rao,"]]), FindDetector(("Asha Rao", "name")))
    assert result.status == "redaction_failed" and result.output is None
    assert result.problems == ("redaction or saving failed (RuntimeError)",)


# --- reports ------------------------------------------------------------------------

def test_reports_contain_no_values_and_expected_columns():
    data = build_pdf([["Dear Asha Rao,", "mail asha.rao@example.com"] + CARD, ["Ben Lee 048-59-6922"]])
    detector = FindDetector(("Asha Rao", "name"), ("asha.rao@example.com", "email"), ("Ben Lee", "name"),
                            ("048-59-6922", "ssn"))
    result = process_pdf(data, detector)
    assert result.verified
    csv_text = pdf_report.entity_report_csv(result).decode()
    header, *rows = csv_text.strip().split("\n")
    assert tuple(header.split(",")) == pdf_report.PDF_CSV_COLUMNS
    assert len(rows) == len(result.entities) == 5
    summary = pdf_report.summary(result)
    assert summary["counts_by_label"]["name"] == 3 and summary["verification"]["verified"] is True
    assert summary["complete_pii_removal_guaranteed"] is False
    assert no_secrets(result_text(result))


def test_failed_result_report_lists_rejections_without_values():
    data = build_pdf([[(72, 100, "Asha Rao"), (72, 104, "Total 4,500.00")]])
    result = process_pdf(data, FindDetector(("Asha Rao", "name")))
    rows = pdf_report.entity_rows(result)
    assert rows[0]["status"] == "rejected" and "unrelated" in rows[0]["reason"]
    assert no_secrets(result_text(result))
