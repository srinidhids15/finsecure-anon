"""Cross-cutting integrity invariants (synthetic data only).

* Randomised multi-page PDFs: every accepted redaction is physically
  applied, unrelated text survives, undetected repeats are left alone (no
  global search), and the output is reopened and verified.
* App: sessions are isolated, failed verification hides output and
  downloads, unsupported edge cases fail closed, and nothing is printed.
"""

import random
from dataclasses import replace
from pathlib import Path

import pymupdf
import pytest
import streamlit as st
from pdf_fixtures import FindDetector, build_pdf, glyph_text, page_text, stream_bytes
from streamlit.testing.v1 import AppTest

from finguard.interface import pdf_workflow, workflow
from finguard.pdf.pipeline import process_pdf
from finguard.schema import Entity

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
VALUES = [("Asha Rao", "name"), ("ben.lee@example.com", "email"), ("048-59-6922", "ssn"), ("Acme Ltd", "company")]
FILLER = ["Balance", "4,500.00", "USD", "due", "01/04/2024", "Ref", "7781", "Total:", "paid", "thanks"]
SECRETS = tuple(v for v, _ in VALUES)


class EveryOther:
    """Detects every value occurrence except each value's first one per page."""

    name = "every_other"

    def detect(self, text):
        out = []
        for value, label in VALUES:
            starts, i = [], text.find(value)
            while i != -1:
                starts.append(i)
                i = text.find(value, i + len(value))
            out += [Entity.from_text(text, s, s + len(value), label, "stub") for s in starts[1:]]
        return sorted(out, key=lambda e: e.start)


def random_document(rng):
    pages = []
    for _ in range(rng.randint(1, 3)):
        lines = []
        for _ in range(rng.randint(3, 9)):
            words = rng.sample(FILLER, rng.randint(1, 4))
            if rng.random() < 0.6:
                words.insert(rng.randint(0, len(words)), rng.choice(VALUES)[0])
            lines.append(" ".join(words))
        pages.append(lines)
    return pages


@pytest.mark.parametrize("seed", range(12))
def test_randomised_pdfs_redact_exactly_what_was_detected(seed):
    rng = random.Random(seed)
    pages = random_document(rng)
    data = build_pdf(pages)
    result = process_pdf(data, EveryOther())
    assert result.verified, result.problems
    v = result.verification
    assert v.redaction_completed and v.targeted_text_removed and v.unrelated_text_preserved
    assert v.regions_checked == sum(len(e.rects) for e in result.entities) == result.regions_applied

    for e in result.entities:  # every accepted redaction is physically applied at its location
        for rect in e.rects:
            assert glyph_text(data, e.page, rect) != ""
            assert glyph_text(result.output, e.page, rect) == ""
    for n, lines in enumerate(pages, 1):  # undetected first occurrences stay; detected ones go
        out = page_text(result.output, n)
        source = "\n".join(lines)
        for value, _ in VALUES:
            expected = 1 if value in source else 0
            assert out.count(value) == expected
        fillers_in = sorted(w for line in lines for w in line.split() if w in FILLER)
        assert sorted(w for w in out.split() if w in FILLER) == fillers_in   # unrelated words survive
    with pymupdf.open(stream=result.output, filetype="pdf") as out, pymupdf.open(stream=data, filetype="pdf") as src:
        assert out.page_count == src.page_count and [p.rect for p in out] == [p.rect for p in src]


def test_detected_values_are_absent_from_every_output_stream():
    data = build_pdf([["Pay Acme Ltd for Asha Rao", "ssn 048-59-6922"], ["mail ben.lee@example.com"]])
    result = process_pdf(data, FindDetector(*VALUES))
    assert result.verified
    streams = stream_bytes(result.output) + result.output
    for value in SECRETS:
        assert value.encode() not in streams and value.encode().hex().encode() not in streams


# --- application ---------------------------------------------------------------------

INFO = {"name": "hybrid", "policy": {}, "freeze_verified": True, "freeze_sha256": "f" * 64}
GOOD_PDF = build_pdf([["Dear Asha Rao,"], ["SSN 048-59-6922"]])


@pytest.fixture(autouse=True)
def fresh_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture
def app(monkeypatch):
    shared = {}

    def make(detector=None):
        det = workflow.SharedDetector(detector or FindDetector(*VALUES))
        shared["detector"] = det
        monkeypatch.setattr(workflow, "build_detector", lambda: det)
        monkeypatch.setattr(workflow, "detector_info", lambda: dict(INFO))
        return AppTest.from_file(APP, default_timeout=60).run()

    make.shared = shared
    return make


def downloads(at):
    return [d.proto.label for d in at.get("download_button")]


def run_pdf(at, data):
    at.file_uploader(key="fg_pdf_upload").upload("doc.pdf", data, "application/pdf").run()
    at.button(key="fg_pdf_run").click().run()
    return at


def test_pdf_results_are_not_visible_to_another_session(app):
    first = run_pdf(app(), GOOD_PDF)
    assert "Redacted PDF (.pdf)" in downloads(first)
    second = AppTest.from_file(APP, default_timeout=60).run()  # same process, same cached detector
    assert pdf_workflow.PDF_KEY not in second.session_state and not downloads(second)
    assert not second.metric
    blob = repr(vars(app.shared["detector"]))
    assert not any(s in blob for s in SECRETS)  # the shared detector keeps no document data


def test_text_verification_failure_hides_output_and_downloads(app, monkeypatch):
    from finguard.anonymization import anonymize

    def unverified(text, entities):
        r = anonymize(text, entities)
        return replace(r, verification=replace(r.verification, detected_spans_replaced=False))

    monkeypatch.setattr(workflow, "anonymize", unverified)
    at = app()
    at.text_area(key="fg_input").input("Dear Asha Rao").run()
    at.button(key="fg_scan").click().run()
    at.button(key="fg_anonymize").click().run()
    assert any("failed verification" in e.value for e in at.error)
    assert not at.code and not downloads(at)
    assert not any("verified" in s.value.lower() and "replacement" in s.value.lower() for s in at.success)


def test_failed_pdf_shows_no_preview_or_pdf_download(app):
    tight = build_pdf([[(72, 100, "Asha Rao"), (72, 104, "Total 4,500.00")]])
    at = run_pdf(app(), tight)
    assert not at.get("imgs") and not at.get("image")
    assert any("No preview" in c.value for c in at.caption)
    assert "Redacted PDF (.pdf)" not in downloads(at)


@pytest.mark.parametrize("make_pdf, message", [
    (lambda: build_pdf([["Dear Asha Rao,"]], setup=lambda d: d[0].draw_rect(
        pymupdf.Rect(60, 60, 250, 80), color=None, fill=(1, 1, 1))), "covered by an image or filled shape"),
    (lambda: build_pdf([["Dear Asha Rao,"]], setup=lambda d: d[0].insert_text(
        (72, 300), "Hidden", fontsize=11, render_mode=3)), "invisible text"),
])
def test_unsupported_edge_cases_fail_closed_in_the_app(app, make_pdf, message):
    at = run_pdf(app(), make_pdf())
    assert any(message in e.value for e in at.error)
    assert not downloads(at) and not at.metric
    assert all("Hybrid detector" in s.value for s in at.success)


def test_malformed_pdf_prints_nothing(capfd):
    result = pdf_workflow.run_pdf(FindDetector(), b"%PDF-1.7\n1 0 obj << /Type /Catalog >> endobj\n" + b"\x00" * 64)
    assert result.error and not result.verified
    out, err = capfd.readouterr()
    assert out == "" and "mupdf" not in err.lower()
