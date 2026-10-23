"""PDF edge cases: ligatures, multi-character mappings, Type 3 fonts, hidden
text (covered, clipped, replacement text) and fonts without Unicode maps.

Each case is either processed end to end (extraction → mapping → physical
removal → independent verification) or rejected with an explicit
unsupported-case error. All PDFs are synthetic (tests/pdf_fixtures.py).
"""

import pymupdf
import pytest
from pdf_fixtures import (
    HELVETICA,
    TYPE3_GLYPH,
    FindDetector,
    build_pdf,
    glyph_text,
    page_text,
    raw_pdf,
    stream_bytes,
    stream_obj,
    to_unicode_cmap,
    type3_font,
)

from finguard.pdf.errors import UnsupportedPdfError
from finguard.pdf.extract import extract_page
from finguard.pdf.pipeline import process_pdf
from finguard.pdf.validate import open_pdf, replacement_text_objects
from finguard.pdf.verify import verify_redacted_pdf

SECRETS = ("fiona", "Griffin", "Asha", "ASHA RAO", "Ref 1234")

# Helvetica with two ligature glyphs: code 1 draws "fi", code 2 draws "ffi".
LIGATURE_FONT = ("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding << /Type /Encoding "
                 "/BaseEncoding /WinAnsiEncoding /Differences [1 /fi 2 /ffi] >> >>")
LIGATURE_PDF = raw_pdf(b"BT /F1 12 Tf 72 700 Td (Dear \x01ona Gri\x02n, total 4,500.00) Tj ET",
                       {"F1": LIGATURE_FONT})


def extract(data):
    with open_pdf(data) as doc:
        return extract_page(doc[0])


def safe(message: str) -> bool:
    return not any(s in message for s in SECRETS)


# --- ligatures ------------------------------------------------------------------------

def test_ligatures_are_expanded_and_grouped_by_glyph():
    page = extract(LIGATURE_PDF)
    assert page.text == "Dear fiona Griffin, total 4,500.00" and page.unsupported == ()
    f, i = page.text.index("fiona"), page.text.index("fiona") + 1
    assert page.geoms[f].glyph == page.geoms[i].glyph                    # "fi" is one glyph
    ffi = page.text.index("ffin")
    assert len({page.geoms[ffi + k].glyph for k in range(3)}) == 1     # "ffi" is one glyph
    assert page.geoms[i].bbox[0] == page.geoms[i].bbox[2]               # continuation: zero width


def test_entity_containing_whole_ligatures_is_removed_and_verified():
    result = process_pdf(LIGATURE_PDF, FindDetector(("fiona Griffin", "name")))
    assert result.verified, result.problems
    assert page_text(result.output, 1) == "Dear \n, total 4,500.00\n"
    assert b"\x01ona" not in stream_bytes(result.output) and b"Gri\x02n" not in stream_bytes(result.output)
    (rect,) = result.entities[0].rects
    assert glyph_text(LIGATURE_PDF, 1, rect) != "" and glyph_text(result.output, 1, rect) == ""


@pytest.mark.parametrize("piece", ["iona", "Grif"])
def test_boundary_inside_a_ligature_is_rejected(piece):
    """"iona" starts inside the "fi" glyph; "Grif" ends inside the "ffi" glyph."""
    result = process_pdf(LIGATURE_PDF, FindDetector((piece, "name")))
    assert result.status == "mapping_failed" and result.output is None
    assert "splits a ligature" in result.problems[0] and safe(result.problems[0])


def test_entity_starting_at_a_ligature_removes_the_whole_glyph():
    result = process_pdf(LIGATURE_PDF, FindDetector(("ffin", "name")))
    assert result.verified
    assert page_text(result.output, 1) == "Dear \ufb01ona Gri\n, total 4,500.00\n"  # untouched "fi" kept


def test_one_code_mapped_to_several_characters():
    """ToUnicode maps one glyph (code 1, drawn as "R") to "Ra": two characters, one glyph."""
    font = ("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding << /Type /Encoding "
            "/BaseEncoding /WinAnsiEncoding /Differences [1 /R] >> /ToUnicode 5 0 R >>")
    cmap = to_unicode_cmap({1: "Ra", **{c: chr(c) for c in range(32, 127)}})
    data = raw_pdf(b"BT /F1 12 Tf 72 700 Td (Dear \x01o paid 12.00) Tj ET", {"F1": font}, [stream_obj(cmap)])
    page = extract(data)
    assert page.text == "Dear Rao paid 12.00"
    r = page.text.index("Rao")
    assert page.geoms[r].glyph == page.geoms[r + 1].glyph != page.geoms[r + 2].glyph
    ok = process_pdf(data, FindDetector(("Rao", "name")))
    assert ok.verified and "Rao" not in page_text(ok.output, 1) and "paid 12.00" in page_text(ok.output, 1)
    split = process_pdf(data, FindDetector(("ao", "name")))
    assert split.status == "mapping_failed" and "splits a ligature" in split.problems[0]


# --- Type 3 fonts ---------------------------------------------------------------------

def type3_pdf(with_to_unicode: bool) -> bytes:
    content = (b"BT /F2 12 Tf 72 700 Td (ASHA RAO PAID) Tj ET "
               b"BT /F1 12 Tf 72 680 Td (Balance 4,500.00) Tj ET")
    extra = [TYPE3_GLYPH, type3_font("7 0 R" if with_to_unicode else "")]
    if with_to_unicode:
        extra.append(stream_obj(to_unicode_cmap({c: chr(c) for c in range(32, 91)})))
    return raw_pdf(content, {"F1": HELVETICA, "F2": "6 0 R"}, extra)


def test_type3_font_with_unicode_map_is_redacted_and_verified():
    data = type3_pdf(True)
    page = extract(data)
    assert page.text == "ASHA RAO PAID\nBalance 4,500.00" and page.unsupported == ()
    result = process_pdf(data, FindDetector(("ASHA RAO", "name")))
    assert result.verified, result.problems
    assert page_text(result.output, 1).split() == ["PAID", "Balance", "4,500.00"]
    v = result.verification
    assert v.targeted_text_removed and v.unrelated_text_preserved and v.glyphs_removed == 7


def test_type3_font_without_unicode_map_is_rejected():
    # MuPDF would read every glyph as "g" (its glyph name): readable but wrong text.
    assert "g" * 5 in extract(type3_pdf(False)).text
    with pytest.raises(UnsupportedPdfError, match="Type 3 font without a Unicode mapping") as info:
        process_pdf(type3_pdf(False), FindDetector(("ASHA RAO", "name")))
    assert safe(str(info.value))


# --- fonts without Unicode information -------------------------------------------------

def test_embedded_font_without_to_unicode_is_rejected():
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_font(fontname="emb", fontbuffer=pymupdf.Font("helv").buffer)
    page.insert_text((72, 100), "Dear Asha Rao", fontname="emb", fontsize=11)
    for xref in range(1, doc.xref_length()):
        if doc.xref_get_key(xref, "ToUnicode")[0] != "null":
            doc.xref_set_key(xref, "ToUnicode", "null")
    with pytest.raises(UnsupportedPdfError, match="without a Unicode mapping"):
        process_pdf(doc.tobytes(), FindDetector(("Asha Rao", "name")))


# --- hidden but extractable text ---------------------------------------------------------

def white_pixmap():
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
    pix.set_rect(pix.irect, (255, 255, 255))
    return pix


def test_text_covered_by_a_later_image_is_rejected():
    def cover(doc):
        doc[0].insert_image(pymupdf.Rect(60, 60, 250, 80), pixmap=white_pixmap())

    data = build_pdf([["Dear Asha Rao,", "Visible line"]], setup=cover)
    with pytest.raises(UnsupportedPdfError, match="covered by an image or filled shape") as info:
        process_pdf(data, FindDetector(("Asha Rao", "name")))
    assert safe(str(info.value))


def test_text_covered_by_a_later_white_box_is_rejected():
    def cover(doc):
        doc[0].draw_rect(pymupdf.Rect(60, 60, 250, 80), color=None, fill=(1, 1, 1))

    with pytest.raises(UnsupportedPdfError, match="covered by an image or filled shape"):
        process_pdf(build_pdf([["Dear Asha Rao,"]], setup=cover), FindDetector())


def test_text_drawn_over_a_background_image_is_redacted():
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(pymupdf.Rect(60, 60, 300, 120), pixmap=white_pixmap())  # background first
    page.insert_text((72, 72), "Dear Asha Rao, balance 4,500.00", fontsize=11)
    data = doc.tobytes()
    result = process_pdf(data, FindDetector(("Asha Rao", "name")))
    assert result.verified and "image(s)" in result.pages[0].notes[0]
    assert "Asha" not in page_text(result.output, 1) and "4,500.00" in page_text(result.output, 1)
    with pymupdf.open(stream=result.output, filetype="pdf") as out:
        assert len(out[0].get_image_info()) == 1  # the background image is kept


def test_thin_rule_over_text_is_not_treated_as_hiding_it():
    def strike(doc):
        doc[0].draw_rect(pymupdf.Rect(70, 68, 160, 69), color=None, fill=(0, 0, 0))  # 1 pt strikethrough

    result = process_pdf(build_pdf([["Dear Asha Rao,"]], setup=strike), FindDetector(("Asha Rao", "name")))
    assert result.verified


def test_clipped_text_is_rejected():
    content = (b"q 0 0 10 10 re W n BT /F1 12 Tf 72 700 Td (Hidden Asha Rao) Tj ET Q "
               b"BT /F1 12 Tf 72 680 Td (Shown) Tj ET")
    with pytest.raises(UnsupportedPdfError, match="clipped out of view"):
        process_pdf(raw_pdf(content, {"F1": HELVETICA}), FindDetector())


def test_text_only_in_image_pixels_is_not_claimed_as_processed():
    """An image of text is not read (no OCR). Pages with text keep a note; image-only pages are rejected."""
    def picture(doc):
        doc[0].insert_image(pymupdf.Rect(300, 300, 400, 340), pixmap=white_pixmap())

    result = process_pdf(build_pdf([["Statement"]], setup=picture), FindDetector())
    assert result.verified
    assert result.pages[0].notes == ("1 image(s): any text inside images is not detected or redacted",)


# --- replacement and alternate text ---------------------------------------------------

def test_actualtext_differing_from_glyphs_is_flagged_and_rejected():
    content = (b"/Span << /ActualText (Ref 1234) >> BDC BT /F1 12 Tf 72 700 Td (Asha Rao) Tj ET EMC "
               b"BT /F1 12 Tf 72 680 Td (Balance 4,500.00) Tj ET")
    data = raw_pdf(content, {"F1": HELVETICA})
    page = extract(data)
    assert page.text == "Asha Rao\nBalance 4,500.00"  # the drawn glyphs, not the replacement
    assert "replacement text (ActualText) that differs from the drawn glyphs" in page.unsupported
    with pytest.raises(UnsupportedPdfError, match="ActualText") as info:
        process_pdf(data, FindDetector(("Asha Rao", "name")))
    assert safe(str(info.value))


@pytest.mark.parametrize("props", [b"<< /ActualText (Asha Rao) >>", b"<< /Alt (Photo of Asha Rao) >>",
                                   b"<< /E (Asha Rao) >>"])
def test_marked_content_strings_that_would_survive_redaction_are_rejected(props):
    content = b"/Span " + props + b" BDC BT /F1 12 Tf 72 700 Td (Asha Rao) Tj ET EMC"
    with pytest.raises(UnsupportedPdfError, match="marked-content replacement or alternate text"):
        process_pdf(raw_pdf(content, {"F1": HELVETICA}), FindDetector(("Asha Rao", "name")))


def test_marked_content_via_property_list_is_rejected():
    content = b"/Span /P1 BDC BT /F1 12 Tf 72 700 Td (Asha Rao) Tj ET EMC"
    data = raw_pdf(content, {"F1": HELVETICA}, ["<< /ActualText (Asha Rao) >>"], resources="/Properties << /P1 5 0 R >> ")
    with pytest.raises(UnsupportedPdfError, match="marked-content"):
        process_pdf(data, FindDetector(("Asha Rao", "name")))


def test_actualtext_only_in_structure_tree_is_removed_and_accepted():
    """Tagged-PDF structure elements are removed during sanitization, so their text cannot survive."""
    content = b"/P << /MCID 0 >> BDC BT /F1 12 Tf 72 700 Td (Dear Asha Rao) Tj ET EMC"
    extra = ["<< /Type /StructTreeRoot /K 6 0 R >>",
             "<< /Type /StructElem /S /P /P 5 0 R /Pg 3 0 R /K 0 /ActualText (Dear Asha Rao) /Alt (Asha) >>"]
    data = raw_pdf(content, {"F1": HELVETICA}, extra, catalog="/StructTreeRoot 5 0 R /MarkInfo << /Marked true >> ")
    assert replacement_text_objects(data) >= 1
    result = process_pdf(data, FindDetector(("Asha Rao", "name")))
    assert result.verified and result.hidden.structure_tree and result.sanitization.structure_tree_removed
    assert replacement_text_objects(result.output) == 0
    assert b"Asha" not in result.output and b"Asha" not in stream_bytes(result.output)


def test_verification_fails_if_replacement_text_remains():
    data = build_pdf([["Dear Asha Rao,"]])
    good = process_pdf(data, FindDetector(("Asha Rao", "name")))
    with pymupdf.open(stream=good.output, filetype="pdf") as doc:
        xref = doc[0].get_contents()[0]
        doc.update_stream(xref, b"/Span << /ActualText (x) >> BDC EMC " + doc.xref_stream(xref))
        tampered = doc.tobytes()
    v = verify_redacted_pdf(tampered, good.pages, good.mappings)
    assert not v.verified and not v.hidden_content_removed
    assert "marked-content replacement or alternate text remains" in v.problems
