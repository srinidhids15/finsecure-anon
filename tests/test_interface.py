"""Tests for the UI-independent app support code (synthetic data only)."""

import csv
import io
import json
from dataclasses import replace
from html.parser import HTMLParser

import pytest

from finguard.anonymization import anonymize
from finguard.config import ENTITY_LABELS
from finguard.interface import exports, workflow
from finguard.interface.highlight import CONTAINER_CLASS, highlight_html
from finguard.interface.samples import SAMPLE_TEXT
from finguard.schema import Entity

BLOCK = "VISA 16 digit\nAsha Rao\n4111111111111111 12/27\nCVC: 123\n"
TEXT = f"Dear Asha Rao,\nmail asha.rao@example.com.\nCard:\n{BLOCK}Thanks, Asha Rao."
SECRETS = ("Asha Rao", "asha.rao@example.com", "4111111111111111", "CVC: 123", "12/27")


def ent(text, piece, label, source="ner", nth=0):
    start = -1
    for _ in range(nth + 1):
        start = text.index(piece, start + 1)
    return Entity.from_text(text, start, start + len(piece), label, source)


ENTITIES = (
    ent(TEXT, "Asha Rao", "name"),
    ent(TEXT, "asha.rao@example.com", "email", "rule:email|ner"),
    ent(TEXT, BLOCK, "credit_card", "rule:credit_card_block"),
    ent(TEXT, "Asha Rao", "name", nth=2),  # nth=1 is the holder inside the block
)


class Stub:
    name = "stub"

    def __init__(self, fn):
        self.fn = fn

    def detect(self, text):
        return self.fn(text)


class DocParser(HTMLParser):
    """Rebuilds the document text from highlight markup and records marks."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.inside = 0
        self.text = ""
        self.marks = []
        self._open = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "div" and attrs.get("class") == CONTAINER_CLASS:
            self.inside += 1
        elif tag == "mark":
            self._open = (len(self.text), attrs["class"].removeprefix("fg-"))
        elif self.inside:
            raise AssertionError(f"unexpected tag {tag!r} inside the document view")

    def handle_endtag(self, tag):
        if tag == "mark":
            start, label = self._open
            self.marks.append((start, len(self.text), label))
            self._open = None

    def handle_data(self, data):
        if self.inside:
            self.text += data


def parse(markup):
    p = DocParser()
    p.feed(markup)
    p.close()
    return p


# --- highlighting ---------------------------------------------------------------

def test_highlight_preserves_every_character_and_offsets():
    p = parse(highlight_html(TEXT, ENTITIES))
    assert p.text == TEXT
    assert p.marks == sorted((e.start, e.end, e.label) for e in ENTITIES)


def test_repeated_values_highlighted_only_where_detected():
    p = parse(highlight_html(TEXT, ENTITIES))
    starts = [s for s, _, label in p.marks if label == "name"]
    occurrences = [i for i in range(len(TEXT)) if TEXT.startswith("Asha Rao", i)]
    assert len(occurrences) == 3
    # occurrences 0 and 2 are detected as names; occurrence 1 lies inside the card block mark
    assert starts == [occurrences[0], occurrences[2]]


@pytest.mark.parametrize("hostile", [
    "<script>alert('x')</script>",
    '"><img src=x onerror=alert(1)>',
    "&lt;already-escaped&gt; & <b>bold</b>",
    "</div><div class='fg-doc'>",
])
def test_user_text_is_escaped(hostile):
    text = f"{hostile} Asha Rao {hostile}"
    markup = highlight_html(text, [ent(text, "Asha Rao", "name")])
    body = markup.split(f'<div class="{CONTAINER_CLASS}">', 1)[1]
    assert "<script" not in body and "<img" not in body and "<b>" not in body
    assert body.count("<div") == 0
    assert parse(markup).text == text


def test_entity_text_inside_mark_is_escaped():
    text = "url: https://x.com/?a=<b>&c=1 end"
    piece = "https://x.com/?a=<b>&c=1"
    markup = highlight_html(text, [ent(text, piece, "url")])
    assert "&lt;b&gt;&amp;c=1</mark>" in markup
    assert parse(markup).text == text


def test_whitespace_unicode_and_crlf_preserved():
    text = "  Zoë Ångström\r\n\t📧 zoë@exämple.com  \n\n"
    es = [ent(text, "Zoë Ångström", "name"), ent(text, "zoë@exämple.com", "email")]
    p = parse(highlight_html(text, es))
    assert p.text == text
    assert p.marks == [(2, 14, "name"), (19, 34, "email")]


def test_empty_and_no_entities():
    assert parse(highlight_html("", [])).text == ""
    assert parse(highlight_html("plain <text>", [])).marks == []


def test_highlight_rejects_bad_entities():
    with pytest.raises(ValueError, match="overlap"):
        highlight_html(TEXT, [ent(TEXT, "Asha Rao", "name"), ent(TEXT, "Asha", "company")])
    with pytest.raises(ValueError, match="does not match"):
        highlight_html(TEXT, [Entity(0, 4, "name", "ner", "Zzzz")])
    with pytest.raises(ValueError, match="unknown label"):
        highlight_html(TEXT, [ent(TEXT, "Asha Rao", "PERSON")])


# --- input validation, scan, anonymize ----------------------------------------

@pytest.mark.parametrize("bad", ["", "   ", "\n\t \n"])
def test_empty_and_whitespace_input_rejected(bad):
    with pytest.raises(workflow.InputError, match="empty or contains only whitespace"):
        workflow.validate_input(bad)


def test_oversized_and_non_text_input_rejected():
    with pytest.raises(workflow.InputError, match="limit is 100,000"):
        workflow.validate_input("a" * (workflow.MAX_INPUT_CHARS + 1))
    assert workflow.validate_input("a" * workflow.MAX_INPUT_CHARS)
    with pytest.raises(workflow.InputError):
        workflow.validate_input(None)


def test_scan_with_stub_detector():
    det = workflow.SharedDetector(Stub(lambda t: list(ENTITIES)))
    record = workflow.scan(det, TEXT)
    assert record.text == TEXT and record.entities == ENTITIES and record.suppressed == 0
    assert len(record.scan_id) == 32 and record.seconds >= 0
    assert workflow.scan(det, TEXT).scan_id != record.scan_id


def test_scan_uses_detect_with_report_when_available():
    class Reporting:
        name = "hybrid"

        def detect_with_report(self, text):
            class R:
                entities = list(ENTITIES)
                suppressed = [object(), object()]
                rule_suppressed = [object()]
            return R()

    record = workflow.scan(workflow.SharedDetector(Reporting()), TEXT)
    assert record.entities == ENTITIES and record.suppressed == 3


def test_detector_failure_message_has_no_input_text():
    def boom(text):
        raise RuntimeError(f"cannot parse {text}")

    with pytest.raises(workflow.ScanError) as info:
        workflow.scan(workflow.SharedDetector(Stub(boom)), TEXT)
    assert "RuntimeError" in str(info.value)
    assert not any(s in str(info.value) for s in SECRETS)


def test_anonymize_scan_success():
    record = workflow.scan(workflow.SharedDetector(Stub(lambda t: list(ENTITIES))), TEXT)
    anon = workflow.anonymize_scan(record)
    assert anon.verified and anon.error is None and anon.scan_id == record.scan_id
    assert anon.result.sanitized_text == (
        "Dear [NAME REDACTED],\nmail [EMAIL REDACTED].\nCard:\n[CREDIT CARD REDACTED]\nThanks, [NAME REDACTED]."
    )


def test_anonymize_scan_failure_is_not_verified_and_leaks_nothing():
    overlapping = (ent(TEXT, "Asha Rao", "name"), ent(TEXT, "Asha", "company"))
    record = workflow.ScanRecord("id1", TEXT, overlapping, 0, 0.0)
    anon = workflow.anonymize_scan(record)
    assert not anon.verified and anon.result is None
    assert "overlaps" in anon.error
    assert not any(s in anon.error for s in SECRETS)


def test_anonymize_scan_unexpected_error(monkeypatch):
    def boom(text, entities):
        raise RuntimeError(text)

    monkeypatch.setattr(workflow, "anonymize", boom)
    anon = workflow.anonymize_scan(workflow.ScanRecord("id", TEXT, ENTITIES, 0, 0.0))
    assert not anon.verified and anon.error == "Anonymization failed (RuntimeError)."


def test_anonymize_scan_rejects_unverified_result(monkeypatch):
    good = anonymize(TEXT, ENTITIES)
    bad = replace(good, verification=replace(good.verification, detected_spans_replaced=False))
    monkeypatch.setattr(workflow, "anonymize", lambda text, entities: bad)
    anon = workflow.anonymize_scan(workflow.ScanRecord("id", TEXT, ENTITIES, 0, 0.0))
    assert not anon.verified and anon.result is None


# --- summaries ------------------------------------------------------------------

def test_counts_and_rows():
    counts = workflow.counts_by_label(ENTITIES)
    assert list(counts) == list(ENTITY_LABELS)
    assert counts["name"] == 2 and counts["email"] == 1 and counts["credit_card"] == 1 and counts["ssn"] == 0
    rows = workflow.entity_rows(reversed(ENTITIES))
    assert [r["start"] for r in rows] == sorted(e.start for e in ENTITIES)
    assert rows[1] == {"#": 2, "label": "email", "start": ENTITIES[1].start, "end": ENTITIES[1].end,
                       "length": 20, "sources": "rule:email, ner"}
    assert not any(s in json.dumps(rows) for s in SECRETS)


def test_placeholder_rows():
    rows = workflow.placeholder_rows(anonymize(TEXT, ENTITIES))
    assert rows == [
        {"label": "name", "placeholder": "[NAME REDACTED]", "count": 2},
        {"label": "email", "placeholder": "[EMAIL REDACTED]", "count": 1},
        {"label": "credit_card", "placeholder": "[CREDIT CARD REDACTED]", "count": 1},
    ]


# --- session state ----------------------------------------------------------------

def test_new_scan_discards_previous_anonymization():
    state = {}
    det = workflow.SharedDetector(Stub(lambda t: list(ENTITIES)))
    first = workflow.scan(det, TEXT)
    workflow.store_scan(state, first)
    workflow.store_anonymization(state, workflow.anonymize_scan(first))
    workflow.store_scan(state, workflow.scan(det, TEXT))
    scan, anon, stale = workflow.current_results(state, TEXT)
    assert scan is not None and anon is None and not stale


def test_input_change_invalidates_results():
    state = {}
    record = workflow.scan(workflow.SharedDetector(Stub(lambda t: list(ENTITIES))), TEXT)
    workflow.store_scan(state, record)
    workflow.store_anonymization(state, workflow.anonymize_scan(record))
    assert workflow.current_results(state, TEXT)[1].verified
    scan, anon, stale = workflow.current_results(state, TEXT + " ")
    assert (scan, anon, stale) == (None, None, True)
    assert state == {}
    assert workflow.current_results(state, TEXT) == (None, None, False)


def test_anonymization_from_another_scan_is_never_returned():
    state = {}
    record = workflow.ScanRecord("scan-a", TEXT, ENTITIES, 0, 0.0)
    workflow.store_scan(state, record)
    workflow.store_anonymization(state, workflow.AnonRecord("scan-b", anonymize(TEXT, ENTITIES), None))
    assert workflow.current_results(state, TEXT)[1] is None
    assert workflow.ANON_KEY not in state


def test_shared_detector_keeps_no_user_data():
    det = workflow.SharedDetector(Stub(lambda t: list(ENTITIES)))
    det.detect_with_count(TEXT)
    blob = repr(vars(det))
    assert not any(s in blob for s in SECRETS)


# --- exports ----------------------------------------------------------------------

@pytest.fixture
def result():
    return anonymize(TEXT, ENTITIES)


def test_txt_export_is_verified_sanitized_text(result):
    assert exports.sanitized_txt(result).decode("utf-8") == result.sanitized_text


def test_csv_export_columns_rows_and_no_values(result):
    data = exports.entity_report_csv(result).decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(data)))
    assert tuple(rows[0]) == exports.CSV_COLUMNS
    assert len(rows) == 4
    card = rows[2]
    block = ENTITIES[2]
    assert card == {
        "index": "3", "label": "credit_card", "start": str(block.start), "end": str(block.end),
        "replaced_end": str(block.end - 1), "whitespace_kept": "1",
        "output_start": str(result.redactions[2].output_start), "output_end": str(result.redactions[2].output_end),
        "placeholder": "[CREDIT CARD REDACTED]", "sources": "rule:credit_card_block",
    }
    for row, r in zip(rows, result.redactions):
        assert result.sanitized_text[int(row["output_start"]):int(row["output_end"])] == row["placeholder"]
    assert not any(s in data for s in SECRETS)


def test_json_export_has_no_values_or_text(result):
    data = exports.redaction_summary_json(result, {"name": "hybrid", "freeze_verified": True})
    summary = json.loads(data)
    assert summary["redaction_count"] == 4 and summary["counts_by_label"]["name"] == 2
    assert summary["verification"]["detected_spans_replaced"] is True
    assert summary["verification"]["complete_pii_removal_guaranteed"] is False
    assert summary["detector"]["name"] == "hybrid"
    assert "sanitized_text" not in summary
    assert not any(s in data.decode() for s in SECRETS + ("Dear", "Thanks"))


def test_exports_refuse_unverified_results(result):
    unverified = replace(result, verification=None)
    for fn in (exports.sanitized_txt, exports.entity_report_csv, exports.redaction_summary_json):
        with pytest.raises(ValueError, match="verified"):
            fn(unverified)


def test_export_filenames_are_fixed():
    for name in (exports.TXT_FILENAME, exports.CSV_FILENAME, exports.JSON_FILENAME):
        assert name.startswith("finguard_") and " " not in name


def test_sample_text_is_within_limits():
    workflow.validate_input(SAMPLE_TEXT)
    assert "4111111111111111" in SAMPLE_TEXT  # standard test card number, not a real account
