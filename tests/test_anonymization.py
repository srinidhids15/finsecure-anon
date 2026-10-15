"""Tests for the text anonymizer and its integrity verifier (synthetic data only)."""

import json
import random
from dataclasses import replace

import pytest

from finguard.anonymization import (
    DEFAULT_PLACEHOLDERS,
    AnonymizationError,
    InvalidSpanError,
    OverlapError,
    SourceMismatchError,
    UnknownLabelError,
    anonymize,
    detect_and_anonymize,
    resolve_placeholders,
    verify_anonymization,
)
from finguard.anonymization.anonymizer import NO_ENTITIES_WARNING
from finguard.config import ENTITY_LABELS
from finguard.schema import Entity, Span

VALUES = {
    "name": "Asha Rao",
    "email": "asha.rao@example.com",
    "phone": "+91 9876543210",
    "address": "12 Lake Road, Pune 411001",
    "ssn": "048-59-6922",
    "credit_card": "VISA 16 digit\nAsha Rao\n4111111111111111 12/27\nCVC: 123\n",
    "company": "Acme Ltd",
    "url": "https://example.com/acct",
}
BLOCK = VALUES["credit_card"]
PH = DEFAULT_PLACEHOLDERS


def ent(text, piece, label, source="ner", nth=0):
    """Entity for the ``nth`` occurrence of ``piece`` in ``text``."""
    start = -1
    for _ in range(nth + 1):
        start = text.index(piece, start + 1)
    return Entity.from_text(text, start, start + len(piece), label, source)


def leaks(blob: str) -> list[str]:
    """Original values (and their distinctive parts) that appear in ``blob``."""
    parts = set(VALUES.values()) | {"Asha", "4111111111111111", "12/27", "CVC: 123", "Lake Road", "9876543210"}
    return [p for p in parts if p.strip() in blob]


# --- placeholders ---------------------------------------------------------

def test_placeholder_policy_covers_every_label():
    assert set(PH) == set(ENTITY_LABELS)
    assert PH["phone"] == "[PHONE NUMBER REDACTED]" and PH["company"] == "[COMPANY NAME REDACTED]"
    with pytest.raises(TypeError):
        PH["name"] = "x"  # read-only


@pytest.mark.parametrize("label", ENTITY_LABELS)
def test_each_label_replaced_with_its_placeholder(label):
    text = f"Before {VALUES[label]} after."
    r = anonymize(text, [ent(text, VALUES[label], label)])
    expected_value = VALUES[label].rstrip()  # canonical span: trailing whitespace stays
    assert r.sanitized_text == text.replace(expected_value, PH[label], 1)
    assert r.redaction_count == 1 and r.counts_by_label[label] == 1
    assert r.redactions[0].placeholder == PH[label]


def test_placeholder_overrides_and_unknown_override_label():
    text = "Hi Asha Rao"
    r = anonymize(text, [ent(text, "Asha Rao", "name")], {"name": "<PERSON>"})
    assert r.sanitized_text == "Hi <PERSON>"
    with pytest.raises(UnknownLabelError):
        resolve_placeholders({"person": "<P>"})
    with pytest.raises(ValueError):
        resolve_placeholders({"name": "  "})
    with pytest.raises(TypeError):
        resolve_placeholders([("name", "x")])


def test_unknown_entity_label_rejected():
    text = "Hi Asha Rao"
    with pytest.raises(UnknownLabelError, match="'PERSON'"):
        anonymize(text, [ent(text, "Asha Rao", "PERSON")])


# --- multiple entities and offsets ----------------------------------------

def test_multiple_entities_in_one_sentence():
    text = "Asha Rao (asha.rao@example.com, +91 9876543210) works at Acme Ltd."
    es = [ent(text, VALUES[k], k) for k in ("name", "email", "phone", "company")]
    r = anonymize(text, es)
    assert r.sanitized_text == ("[NAME REDACTED] ([EMAIL REDACTED], [PHONE NUMBER REDACTED]) "
                                "works at [COMPANY NAME REDACTED].")
    assert [x.label for x in r.redactions] == ["name", "email", "phone", "company"]


def test_unsorted_input_and_output_offsets_are_recorded():
    text = "Mail asha.rao@example.com, then Asha Rao."
    es = [ent(text, "Asha Rao", "name"), ent(text, "asha.rao@example.com", "email")]
    r = anonymize(text, es)
    assert r.sanitized_text == "Mail [EMAIL REDACTED], then [NAME REDACTED]."
    for red in r.redactions:
        assert r.sanitized_text[red.output_start:red.output_end] == red.placeholder
        assert text[red.start:red.end] in VALUES.values()


def test_repeated_values_at_different_offsets():
    text = "Asha Rao paid. Asha Rao signed. Asha Rao left."
    r = anonymize(text, [ent(text, "Asha Rao", "name", nth=i) for i in range(3)])
    assert r.sanitized_text == "[NAME REDACTED] paid. [NAME REDACTED] signed. [NAME REDACTED] left."
    assert [x.start for x in r.redactions] == [0, 15, 32]


def test_only_detected_occurrence_is_replaced_and_verification_does_not_overclaim():
    text = "Asha Rao paid. Asha Rao signed."
    r = anonymize(text, [ent(text, "Asha Rao", "name", nth=1)])
    # The first occurrence was not detected, so it remains; span verification
    # still passes (it verifies requested spans) but makes no completeness claim.
    assert r.sanitized_text == "Asha Rao paid. [NAME REDACTED] signed."
    assert r.verification.detected_spans_replaced is True
    assert r.verification.complete_pii_removal_guaranteed is False


def test_adjacent_entities():
    text = "Asha RaoAcme Ltd+91 9876543210"
    es = [ent(text, "Asha Rao", "name"), ent(text, "Acme Ltd", "company"), ent(text, "+91 9876543210", "phone")]
    r = anonymize(text, es)
    assert r.sanitized_text == "[NAME REDACTED][COMPANY NAME REDACTED][PHONE NUMBER REDACTED]"


def test_entity_at_start_and_end_of_text():
    text = "Asha Rao"
    assert anonymize(text, [ent(text, text, "name")]).sanitized_text == "[NAME REDACTED]"


def test_unicode_and_punctuation():
    text = "«Zoë Ångström» — 📧 zoë@exämple.com; ☎ +91 9876543210!"
    es = [ent(text, "Zoë Ångström", "name"), ent(text, "zoë@exämple.com", "email"),
          ent(text, "+91 9876543210", "phone")]
    r = anonymize(text, es)
    assert r.sanitized_text == "«[NAME REDACTED]» — 📧 [EMAIL REDACTED]; ☎ [PHONE NUMBER REDACTED]!"
    assert r.verification.detected_spans_replaced


# --- credit-card blocks and trailing whitespace ---------------------------

def test_multiline_card_block_fully_replaced_and_newline_kept():
    text = f"Card on file:\n{BLOCK}Thank you."
    r = anonymize(text, [ent(text, BLOCK, "credit_card", "rule:credit_card_block")])
    assert r.sanitized_text == "Card on file:\n[CREDIT CARD REDACTED]\nThank you."
    for part in ("VISA", "Asha Rao", "4111111111111111", "12/27", "CVC"):
        assert part not in r.sanitized_text
    (red,) = r.redactions
    assert red.detected_end - red.end == red.whitespace_kept == 1
    assert any("whitespace" in w for w in r.warnings)


def test_card_block_with_or_without_trailing_newline_gives_same_output():
    text = f"Card:\n{BLOCK}Done"
    with_nl = anonymize(text, [ent(text, BLOCK, "credit_card")])
    without_nl = anonymize(text, [ent(text, BLOCK.rstrip("\n"), "credit_card")])
    assert with_nl.sanitized_text == without_nl.sanitized_text == "Card:\n[CREDIT CARD REDACTED]\nDone"
    assert without_nl.warnings == ()


@pytest.mark.parametrize("tail", ["\n", "\r\n", "\n\n", " \t\n"])
def test_trailing_whitespace_at_end_of_text_preserved(tail):
    block = BLOCK.rstrip("\n") + tail
    text = "Card:\n" + block
    r = anonymize(text, [ent(text, block, "credit_card")])
    assert r.sanitized_text == "Card:\n[CREDIT CARD REDACTED]" + tail


def test_card_block_adjacent_to_following_entity():
    text = f"{BLOCK}Asha Rao"
    r = anonymize(text, [ent(text, BLOCK, "credit_card"), ent(text, "Asha Rao", "name", nth=1)])
    assert r.sanitized_text == "[CREDIT CARD REDACTED]\n[NAME REDACTED]"


# --- empty and zero-redaction inputs --------------------------------------

def test_empty_input():
    r = anonymize("", [])
    assert r.sanitized_text == "" and r.redaction_count == 0
    assert r.warnings == (NO_ENTITIES_WARNING,)


def test_no_entities_returns_text_unchanged_with_zero_report():
    text = "Nothing sensitive here,\nprobably.\n"
    r = anonymize(text, [])
    assert r.sanitized_text == text
    report = r.to_dict()
    assert report["redaction_count"] == 0 and report["redactions"] == []
    assert report["counts_by_label"] == dict.fromkeys(ENTITY_LABELS, 0)
    assert report["verification"]["detected_spans_replaced"] is True
    assert report["verification"]["complete_pii_removal_guaranteed"] is False
    assert "does not show that the text contains no PII" in report["warnings"][0]


# --- invalid input fails clearly ------------------------------------------

def test_offsets_beyond_text_rejected():
    text = "Hi Asha Rao"
    bad = Entity(3, 20, "name", "ner", "x" * 17)
    with pytest.raises(InvalidSpanError, match=r"\[3, 20\).*length 11"):
        anonymize(text, [bad])
    with pytest.raises(InvalidSpanError):
        anonymize("", [Entity(0, 1, "name", "ner", "A")])


def test_shifted_offsets_rejected_as_source_mismatch():
    text = "Hi Asha Rao"
    shifted = Entity(2, 10, "name", "ner", "Asha Rao")  # off by one
    with pytest.raises(SourceMismatchError, match=r"entity #0 \('name', \[2, 10\)\)"):
        anonymize(text, [shifted])


def test_entity_text_not_matching_source_rejected():
    text = "Hi Asha Rao"
    with pytest.raises(SourceMismatchError):
        anonymize(text, [Entity(3, 11, "name", "ner", "Ben  Lee")])


def test_whitespace_only_entity_rejected():
    text = "a   b"
    with pytest.raises(InvalidSpanError, match="whitespace"):
        anonymize(text, [Entity(1, 4, "name", "ner", "   ")])


def test_non_entity_and_non_text_rejected():
    with pytest.raises(TypeError, match="Span"):
        anonymize("Hi Asha Rao", [Span(3, 11, "name")])
    with pytest.raises(TypeError):
        anonymize(None, [])


@pytest.mark.parametrize("second", [
    (5, 12),    # partial overlap
    (0, 8),     # identical span
    (2, 4),     # nested
])
def test_overlapping_spans_rejected(second):
    text = "Asha Rao Kumar"
    a = Entity.from_text(text, 0, 8, "name", "ner")
    b = Entity.from_text(text, *second, "company", "ner")
    with pytest.raises(OverlapError, match="resolve overlaps in the detector"):
        anonymize(text, [a, b])


def test_overlap_with_kept_trailing_whitespace_rejected():
    text = f"{BLOCK}Zed"
    block = ent(text, BLOCK, "credit_card")
    starts_in_newline = Entity.from_text(text, len(BLOCK) - 1, len(text), "name", "ner")
    with pytest.raises(OverlapError):
        anonymize(text, [block, starts_in_newline])


def test_errors_are_all_anonymization_errors():
    for cls in (UnknownLabelError, InvalidSpanError, SourceMismatchError, OverlapError):
        assert issubclass(cls, AnonymizationError) and issubclass(cls, ValueError)


# --- no mutation ----------------------------------------------------------

def test_input_text_and_entities_not_mutated():
    text = f"Dear Asha Rao, card {BLOCK}"
    es = [ent(text, BLOCK, "credit_card", "rule:credit_card_block"), ent(text, "Asha Rao", "name")]
    snapshot = [(e.start, e.end, e.label, e.source, e.text) for e in es]
    original = str(text)
    anonymize(text, es)
    assert text == original
    assert [(e.start, e.end, e.label, e.source, e.text) for e in es] == snapshot
    assert es[0].label == "credit_card"  # list order untouched


def test_accepts_any_iterable_once():
    text = "Hi Asha Rao"
    r = anonymize(text, (e for e in [ent(text, "Asha Rao", "name")]))
    assert r.sanitized_text == "Hi [NAME REDACTED]"
    assert r.verification.detected_spans_replaced


# --- verification ---------------------------------------------------------

def sample():
    text = f"Dear Asha Rao,\ncard {BLOCK}mail asha.rao@example.com now."
    es = [ent(text, "Asha Rao", "name"), ent(text, BLOCK, "credit_card", "rule:credit_card_block"),
          ent(text, "asha.rao@example.com", "email", "rule:email|ner")]
    return text, es, anonymize(text, es)


def test_verification_passes_and_is_attached():
    text, es, r = sample()
    v = verify_anonymization(text, es, r)
    assert v == r.verification
    assert v.detected_spans_replaced and v.problems == () and v.checked_redactions == 3
    assert v.complete_pii_removal_guaranteed is False and "cannot be guaranteed" in v.note


def test_verification_detects_changed_unredacted_text():
    text, es, r = sample()
    tampered = replace(r, sanitized_text=r.sanitized_text.replace("Dear", "Deer"))
    v = verify_anonymization(text, es, tampered)
    assert not v.detected_spans_replaced
    assert any("not preserved" in p for p in v.problems)


def test_verification_detects_lost_trailing_newline():
    text, es, r = sample()
    tampered = replace(r, sanitized_text=r.sanitized_text.replace("[CREDIT CARD REDACTED]\n", "[CREDIT CARD REDACTED]"))
    assert not verify_anonymization(text, es, tampered).detected_spans_replaced


def test_verification_detects_skipped_entity():
    text, es, r = sample()
    skipped = replace(r, redactions=r.redactions[1:])
    v = verify_anonymization(text, es, skipped)
    assert any("has no matching redaction" in p for p in v.problems)


def test_verification_detects_extra_and_wrong_offset_redaction():
    text, es, r = sample()
    v = verify_anonymization(text, es[1:], r)  # name was not requested
    assert any("was not requested" in p for p in v.problems)
    first = r.redactions[0]
    moved = replace(r, redactions=(replace(first, start=first.start + 1),) + r.redactions[1:])
    assert not verify_anonymization(text, es, moved).detected_spans_replaced


def test_verification_detects_output_offset_shift_and_wrong_placeholder():
    text, es, r = sample()
    first = r.redactions[0]
    shifted = replace(r, redactions=(replace(first, output_start=first.output_start + 1,
                                             output_end=first.output_end + 1),) + r.redactions[1:])
    assert any("shifted" in p for p in verify_anonymization(text, es, shifted).problems)
    wrong = replace(r, redactions=(replace(first, placeholder="[EMAIL REDACTED]"),) + r.redactions[1:])
    assert any("placeholder" in p for p in verify_anonymization(text, es, wrong).problems)


def test_verification_detects_wrong_original():
    text, es, r = sample()
    assert not verify_anonymization(text + "!", es, r).detected_spans_replaced


def test_randomised_documents_rebuild_exactly():
    """Independent reconstruction on 200 seeded synthetic documents."""
    rng = random.Random(0)
    fillers = [" ", ", ", "\n", " — ", "é ", "Total: 4,500.00 ", ""]
    for _ in range(200):
        text, expected, entities = "", "", []
        for _ in range(rng.randint(0, 8)):
            filler = rng.choice(fillers)
            text, expected = text + filler, expected + filler
            label = rng.choice(ENTITY_LABELS)
            value = VALUES[label]
            entities.append(Entity.from_text(text + value, len(text), len(text) + len(value), label, "ner"))
            text += value
            canon = value.rstrip()
            expected += PH[label] + value[len(canon):]
        rng.shuffle(entities)
        r = anonymize(text, entities)
        assert r.sanitized_text == expected
        assert r.redaction_count == len(entities)


# --- privacy of reports, reprs and errors ---------------------------------

def test_default_report_and_repr_contain_no_original_values():
    text, es, r = sample()
    blob = json.dumps(r.to_dict()) + repr(r) + repr(r.redactions) + repr(r.verification) + "".join(r.warnings)
    assert leaks(blob) == []
    assert r.to_dict()["redactions"][1] == {
        "label": "credit_card", "start": es[1].start, "end": es[1].end - 1, "detected_end": es[1].end,
        "length": len(BLOCK) - 1, "output_start": r.redactions[1].output_start,
        "output_end": r.redactions[1].output_start + len(PH["credit_card"]),
        "placeholder": "[CREDIT CARD REDACTED]", "sources": ["rule:credit_card_block"],
    }
    assert "sanitized_text" not in r.to_dict()
    assert r.to_dict(include_sanitized_text=True)["sanitized_text"] == r.sanitized_text


def test_error_messages_contain_no_original_values():
    text = f"Asha Rao {BLOCK} asha.rao@example.com"
    a = ent(text, "Asha Rao", "name")
    cases = [
        [Entity(0, 8, "name", "ner", "Asha Rag")],
        [Entity(1, 9, "name", "ner", text[1:9])] + [Entity(0, 8, "nme", "ner", "Asha Rao")],
        [a, ent(text, "Asha Rao", "company", nth=0)],
        [a, Entity.from_text(text, 2, 30, "address", "ner")],
    ]
    messages = []
    for es in cases:
        with pytest.raises(AnonymizationError) as info:
            anonymize(text, es)
        messages.append(str(info.value))
    assert leaks(" ".join(messages)) == []


def test_failed_verification_problems_contain_no_original_values():
    text, es, r = sample()
    tampered = replace(r, sanitized_text="X" + r.sanitized_text, redactions=r.redactions[:1])
    v = verify_anonymization(text, es, tampered)
    assert v.problems and leaks(" ".join(v.problems)) == []


def test_nothing_is_logged(caplog):
    text, es, _ = sample()
    with caplog.at_level("DEBUG"):
        anonymize(text, es)
    assert caplog.records == []


# --- detector integration (stub detector; real hybrid in test_anonymization_e2e) ---

def test_detect_and_anonymize_uses_given_detector():
    class Stub:
        name = "stub"

        def detect(self, text):
            return [ent(text, "Asha Rao", "name")]

    r = detect_and_anonymize("To Asha Rao.", Stub())
    assert r.sanitized_text == "To [NAME REDACTED]."
