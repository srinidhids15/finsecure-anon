"""Streamlit app tests with ``streamlit.testing.v1.AppTest`` (synthetic data only).

Most tests replace the detector with a stub so they run without the trained
model; one test uses the real frozen hybrid detector (skipped without it).
"""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from finguard.interface import workflow
from finguard.ner.inference import DEFAULT_MODEL_DIR, ModelNotFoundError
from finguard.schema import Entity

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
NAMES = {"name": "Name", "email": "Email", "credit_card": "Credit card"}
HAS_MODEL = (DEFAULT_MODEL_DIR / "training_manifest.json").exists()
BLOCK = "VISA 16 digit\nAsha Rao\n4111111111111111 12/27\nCVC: 123\n"
TEXT = f"Dear Asha Rao, mail asha.rao@example.com.\nCard:\n{BLOCK}Bye Asha Rao."
SECRETS = ("Asha Rao", "asha.rao@example.com", "4111111111111111", "CVC: 123")


def find(text, piece, label, source="ner", nth=0):
    start = -1
    for _ in range(nth + 1):
        start = text.index(piece, start + 1)
    return Entity.from_text(text, start, start + len(piece), label, source)


def good_entities(text):
    out = []
    if "Dear Asha Rao" in text:
        out.append(find(text, "Asha Rao", "name"))
    if "asha.rao@example.com" in text:
        out.append(find(text, "asha.rao@example.com", "email", "rule:email|ner"))
    if BLOCK in text:
        out.append(find(text, BLOCK, "credit_card", "rule:credit_card_block"))
    if "Bye Asha Rao" in text:
        out.append(find(text, "Asha Rao", "name", nth=2))
    return out


class Stub:
    name = "stub"

    def __init__(self, fn):
        self.fn = fn

    def detect(self, text):
        return self.fn(text)


INFO = {"name": "hybrid", "policy": {}, "freeze_verified": True, "freeze_sha256": "f" * 64}


@pytest.fixture(autouse=True)
def fresh_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture
def app(monkeypatch):
    def make(fn=good_entities):
        monkeypatch.setattr(workflow, "build_detector", lambda: workflow.SharedDetector(Stub(fn)))
        monkeypatch.setattr(workflow, "detector_info", lambda: dict(INFO))
        return AppTest.from_file(APP, default_timeout=30).run()
    return make


def scan(at, text):
    at.text_area(key="fg_input").input(text).run()
    at.button(key="fg_scan").click().run()
    return at


def all_text(at):
    parts = [e.value for kind in ("markdown", "caption", "error", "info", "warning", "success", "code")
             for e in getattr(at, kind)]
    parts += [h.proto.body for h in at.get("html")]
    parts += [df.value.to_csv() for df in at.dataframe]
    return "\n".join(str(p) for p in parts)


def downloads(at):
    return at.get("download_button")


# --- startup ---------------------------------------------------------------------

def test_app_starts_with_stub_detector(app):
    at = app()
    assert not at.exception
    assert any("Hybrid detector ready" in s.value for s in at.success)
    assert at.button(key="fg_scan").disabled is False
    assert not at.metric and not downloads(at)


def test_missing_model_is_reported_and_scan_disabled(monkeypatch):
    import finguard.ner.inference as inference

    def missing(*args, **kwargs):
        raise ModelNotFoundError("no model")

    monkeypatch.setattr(inference, "NERDetector", missing)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("python -m finguard.ner.train" in e.value for e in at.error)
    assert at.button(key="fg_scan").disabled is True


def test_failed_initialisation_hides_exception_message(monkeypatch):
    import finguard.ner.inference as inference

    def broken(*args, **kwargs):
        raise RuntimeError("Asha Rao 4111111111111111")

    monkeypatch.setattr(inference, "NERDetector", broken)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert any("could not be initialised (RuntimeError)" in e.value for e in at.error)
    assert not any(s in all_text(at) for s in SECRETS)


# --- input validation ---------------------------------------------------------------

@pytest.mark.parametrize("text", ["", "   \n\t "])
def test_empty_and_whitespace_input(app, text):
    at = scan(app(), text)
    assert any("empty or contains only whitespace" in e.value for e in at.error)
    assert not at.metric


def test_sample_button_loads_synthetic_text(app):
    from finguard.interface.samples import SAMPLE_TEXT

    at = app()
    at.button(key="fg_sample").click().run()
    assert at.text_area(key="fg_input").value == SAMPLE_TEXT


# --- scan results ---------------------------------------------------------------------

def test_scan_shows_counts_table_and_highlights(app):
    at = scan(app(), TEXT)
    assert not at.exception and not at.error
    metrics = {m.label: m.value for m in at.metric}
    assert metrics == {"Name": "2", "Email": "1", "Phone": "0", "Address": "0",
                       "SSN": "0", "Credit card": "1", "Company": "0", "URL": "0"}
    table = at.dataframe[0].value
    assert list(table.columns) == ["#", "label", "start", "end", "length", "sources"]
    expected = sorted((e.start, e.end, e.label) for e in good_entities(TEXT))
    assert list(zip(table.start, table.end, table.label)) == expected
    assert list(table.sources) == ["ner", "rule:email, ner", "rule:credit_card_block", "ner"]
    (html,) = at.get("html")
    for start, end, label in expected:
        assert f'title="{NAMES[label]} [{start}, {end})"' in html.proto.body


def test_repeated_value_highlighted_at_detected_offsets_only(app):
    at = scan(app(), TEXT)
    body = at.get("html")[0].proto.body
    # "Asha Rao" occurs 3 times; two are names, the holder line is inside the card block.
    assert body.count('<mark class="fg-name"') == 2
    assert body.count(">Asha Rao</mark>") == 2


def test_user_html_is_escaped(app):
    text = "<script>alert(1)</script> Dear Asha Rao <img src=x onerror=alert(2)>"
    at = scan(app(), text)
    body = at.get("html")[0].proto.body
    assert "<script>" not in body and "<img" not in body
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body
    assert '<mark class="fg-name" data-tag="NAME" title="Name [31, 39)">Asha Rao</mark>' in body


def test_no_detections_does_not_claim_text_is_clean(app):
    at = scan(app(lambda t: []), "Nothing to see here.")
    assert any("does not mean the text contains no PII" in i.value for i in at.info)


def test_detector_runtime_error_is_safe(app):
    def boom(text):
        raise ValueError(f"bad input: {text}")

    at = scan(app(boom), TEXT)
    assert any("Detection failed (ValueError)" in e.value for e in at.error)
    assert not at.metric
    assert not any(s in all_text(at).replace(TEXT, "") for s in SECRETS)


# --- anonymization ---------------------------------------------------------------------

def test_anonymize_shows_verified_output_and_downloads(app):
    at = scan(app(), TEXT)
    at.button(key="fg_anonymize").click().run()
    assert not at.exception and not at.error
    expected = ("Dear [NAME REDACTED], mail [EMAIL REDACTED].\nCard:\n[CREDIT CARD REDACTED]\n"
                "Bye [NAME REDACTED].")
    assert at.code[0].value == expected
    assert any("Detected-span replacement verified: all 4 detected span(s)" in s.value for s in at.success)
    assert any("does not guarantee that all PII was removed" in w.value for w in at.warning)
    assert [m.value for m in at.metric if m.label == "Redactions"] == ["4"]
    placeholders = at.dataframe[1].value
    assert list(placeholders.placeholder) == ["[NAME REDACTED]", "[EMAIL REDACTED]", "[CREDIT CARD REDACTED]"]
    assert any("whitespace" in c.value for c in at.caption)
    assert [d.proto.label for d in downloads(at)] == [
        "Sanitized text (.txt)", "Entity report (.csv)", "Redaction summary (.json)"]


def test_failed_anonymization_shows_no_success_or_downloads(app):
    def overlapping(text):
        return [find(text, "Asha Rao", "name"), find(text, "Asha", "company")]

    at = scan(app(overlapping), TEXT)
    assert not at.exception
    assert any("Highlighting is unavailable" in e.value for e in at.error)
    at.button(key="fg_anonymize").click().run()
    assert any("Anonymization was refused" in e.value and "overlaps" in e.value for e in at.error)
    assert not any("verified" in s.value for s in at.success)
    assert not at.code and not downloads(at)
    assert not at.exception
    assert not any(s in " ".join(e.value for e in at.error) for s in SECRETS)


def test_input_change_invalidates_stale_results(app):
    at = scan(app(), TEXT)
    at.button(key="fg_anonymize").click().run()
    assert at.code and downloads(at)
    at.text_area(key="fg_input").input(TEXT + " P.S.").run()
    assert any("text changed after the last scan" in i.value for i in at.info)
    assert not at.metric and not at.code and not downloads(at)
    assert workflow.SCAN_KEY not in at.session_state and workflow.ANON_KEY not in at.session_state


def test_rescan_discards_previous_anonymization(app):
    at = scan(app(), TEXT)
    at.button(key="fg_anonymize").click().run()
    at.button(key="fg_scan").click().run()
    assert at.metric and not at.code and not downloads(at)


def test_clear_button_resets_everything(app):
    at = scan(app(), TEXT)
    at.button(key="fg_clear").click().run()
    assert at.text_area(key="fg_input").value == ""
    assert not at.metric and workflow.SCAN_KEY not in at.session_state


def test_sessions_do_not_share_results(app):
    first = scan(app(), TEXT)
    assert first.metric
    second = AppTest.from_file(APP, default_timeout=30).run()  # new session, same process and cache
    assert not second.metric and second.text_area(key="fg_input").value == ""
    assert workflow.SCAN_KEY not in second.session_state


def test_nothing_sensitive_printed_to_terminal(app, capfd):
    at = scan(app(), TEXT)
    at.button(key="fg_anonymize").click().run()
    out, err = capfd.readouterr()
    assert not any(s in out + err for s in SECRETS)


# --- real detector -------------------------------------------------------------------

@pytest.mark.skipif(not HAS_MODEL, reason="no finished training run; run python -m finguard.ner.train")
def test_real_hybrid_detector_end_to_end(capfd):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert any("Matches the evaluated freeze" in c.value for c in at.caption)
    at = scan(at, TEXT)
    assert not at.exception and not at.error
    table = at.dataframe[0].value
    card = table[table.label == "credit_card"]
    assert len(card) == 1 and "rule:credit_card_block" in card.sources.iloc[0]
    at.button(key="fg_anonymize").click().run()
    sanitized = at.code[0].value
    assert "[CREDIT CARD REDACTED]\nBye " in sanitized and "[EMAIL REDACTED]" in sanitized
    assert any("Detected-span replacement verified" in s.value for s in at.success)
    out, err = capfd.readouterr()
    assert not any(s in out + err for s in SECRETS)


@pytest.mark.skipif(not HAS_MODEL, reason="no finished training run")
def test_evaluation_freeze_still_valid():
    from finguard.evaluation.freeze import FROZEN_FILES, verify_freeze

    verify_freeze()
    assert not any("interface" in f or "app/" in f for f in FROZEN_FILES)
