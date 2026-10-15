"""Streamlit PDF workspace tests (AppTest, synthetic PDFs, stub detector)."""

from pathlib import Path

import pymupdf
import pytest
import streamlit as st
from pdf_fixtures import FindDetector, build_pdf, image_only_page
from streamlit.testing.v1 import AppTest

from finguard.interface import pdf_workflow, workflow

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
SECRETS = ("Asha Rao", "asha.rao@example.com", "048-59-6922")
INFO = {"name": "hybrid", "policy": {}, "freeze_verified": True, "freeze_sha256": "f" * 64}

GOOD = build_pdf([["Dear Asha Rao,", "mail asha.rao@example.com"], ["SSN 048-59-6922"]])
OTHER = build_pdf([["Hello Asha Rao"]])
DETECTOR = FindDetector(("Asha Rao", "name"), ("asha.rao@example.com", "email"), ("048-59-6922", "ssn"))


@pytest.fixture(autouse=True)
def fresh_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture
def app(monkeypatch):
    def make(detector=DETECTOR):
        monkeypatch.setattr(workflow, "build_detector", lambda: workflow.SharedDetector(detector))
        monkeypatch.setattr(workflow, "detector_info", lambda: dict(INFO))
        return AppTest.from_file(APP, default_timeout=60).run()
    return make


def upload(at, data, name="statement.pdf"):
    at.file_uploader(key="fg_pdf_upload").upload(name, data, "application/pdf").run()
    return at


def run(at):
    at.button(key="fg_pdf_run").click().run()
    return at


def labels(at):
    return [d.proto.label for d in at.get("download_button")]


def visible(at):
    parts = [e.value for kind in ("markdown", "caption", "error", "info", "warning", "success") for e in getattr(at, kind)]
    parts += [df.value.to_csv() for df in at.dataframe]
    return "\n".join(map(str, parts))


def test_pdf_tab_renders_and_button_waits_for_upload(app):
    at = app()
    assert not at.exception
    assert at.button(key="fg_pdf_run").disabled is True
    assert at.file_uploader(key="fg_pdf_upload").allowed_type in (["pdf"], [".pdf"])


def test_verified_pdf_offers_downloads_and_summary(app):
    at = run(upload(app(), GOOD))
    assert not at.exception and not at.error
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Pages"] == "2" and metrics["Entities detected"] == "3" and metrics["Regions redacted"] == "3"
    assert any(s.value.startswith("Redaction applied: 3 region(s)") for s in at.success)
    assert any(s.value.startswith("Targeted text removal verified") for s in at.success)
    assert any("Complete PII removal is not guaranteed" in w.value for w in at.warning)
    assert labels(at) == ["Redacted PDF (.pdf)", "Entity report (.csv)", "Processing summary (.json)"]
    entities = at.dataframe[0].value
    assert list(entities.columns) == ["index", "page", "label", "start", "end", "detected_end", "regions",
                                      "sources", "status", "reason"]
    assert list(entities.page) == [1, 1, 2] and set(entities.status) == {"redacted"}
    assert at.get("imgs") or at.get("image")  # preview of the verified output
    assert not any(s in visible(at) for s in SECRETS)


def test_unsupported_pdf_shows_safe_error_and_no_downloads(app):
    doc = pymupdf.open()
    image_only_page(doc)
    at = run(upload(app(), doc.tobytes()))
    assert any("OCR is not supported" in e.value for e in at.error)
    assert not labels(at)
    assert all("Hybrid detector" in s.value for s in at.success)  # only the sidebar status


def test_invalid_upload_shows_safe_error(app):
    at = run(upload(app(), b"%PDF-1.7\n" + b"\x00" * 200))
    assert any("damaged" in e.value or "could not be opened" in e.value for e in at.error)
    assert not labels(at)


def test_failed_mapping_shows_no_pdf_download(app):
    tight = build_pdf([[(72, 100, "Asha Rao"), (72, 104, "Total 4,500.00")]])
    at = run(upload(app(), tight))
    assert any("could not be mapped" in e.value for e in at.error)
    assert "Redacted PDF (.pdf)" not in labels(at)
    assert labels(at) == ["Entity report (.csv)", "Processing summary (.json)"]
    assert not any(s.value.startswith("Targeted text removal verified") for s in at.success)
    assert not any(s in visible(at) for s in SECRETS)


def test_failed_verification_shows_no_pdf_download(app, monkeypatch):
    from finguard.pdf import pipeline
    from finguard.pdf.verify import PdfVerification

    monkeypatch.setattr(pipeline, "verify_redacted_pdf",
                        lambda *a: PdfVerification(opened=True, problems=("page 1: forced failure",)))
    at = run(upload(app(), GOOD))
    assert any("failed independent verification" in e.value for e in at.error)
    assert "Redacted PDF (.pdf)" not in labels(at)
    assert not any(s.value.startswith("Redaction applied") for s in at.success)


def test_new_upload_invalidates_previous_result(app):
    at = run(upload(app(), GOOD))
    assert "Redacted PDF (.pdf)" in labels(at)
    first = at.session_state[pdf_workflow.PDF_KEY].upload_id
    at = upload(at, OTHER, "other.pdf")
    assert any("uploaded file changed" in i.value for i in at.info)
    assert not labels(at) and pdf_workflow.PDF_KEY not in at.session_state
    at = run(at)
    assert at.session_state[pdf_workflow.PDF_KEY].upload_id != first


def test_pdf_and_text_results_are_independent(app):
    at = run(upload(app(), GOOD))
    assert workflow.SCAN_KEY not in at.session_state
    at.text_area(key="fg_input").input("Dear Asha Rao").run()
    at.button(key="fg_scan").click().run()
    assert workflow.SCAN_KEY in at.session_state and pdf_workflow.PDF_KEY in at.session_state
    assert at.session_state[pdf_workflow.PDF_KEY].result.verified


def test_nothing_sensitive_printed(app, capfd):
    run(upload(app(), GOOD))
    out, err = capfd.readouterr()
    assert not any(s in out + err for s in SECRETS)


# --- pdf_workflow unit tests ----------------------------------------------------------

def test_run_pdf_wraps_errors_and_ids_uploads():
    rec = pdf_workflow.run_pdf(DETECTOR, b"")
    assert rec.error == "The file is empty." and not rec.verified
    good = pdf_workflow.run_pdf(DETECTOR, GOOD)
    assert good.verified and good.upload_id == pdf_workflow.upload_id(GOOD)

    class Broken:
        def detect(self, text):
            raise KeyError(text)

    bad = pdf_workflow.run_pdf(Broken(), GOOD)
    assert bad.error == "Detection failed on page 1 (KeyError)." and not bad.verified


def test_current_pdf_state_transitions():
    state = {}
    rec = pdf_workflow.run_pdf(DETECTOR, GOOD)
    pdf_workflow.store_pdf(state, rec)
    assert pdf_workflow.current_pdf(state, rec.upload_id) == (rec, False)
    assert pdf_workflow.current_pdf(state, "other") == (None, True)
    assert state == {}
    pdf_workflow.store_pdf(state, rec)
    assert pdf_workflow.current_pdf(state, None) == (None, True)


def test_preview_png_renders_output():
    rec = pdf_workflow.run_pdf(DETECTOR, GOOD)
    png = pdf_workflow.preview_png(rec.result.output, 1)
    assert png.startswith(b"\x89PNG")
