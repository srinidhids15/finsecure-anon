"""Tests for span canonicalisation (trailing whitespace policy)."""

import pytest

from finguard.canonical import canonical_span, canonical_spans
from finguard.detection.rules import detect
from finguard.schema import Entity, Span

BLOCK = "VISA 16 digit\nAsha Rao\n4111111111111111 12/27\nCVC: 123\n"
TEXT = f"Paid with card ending in {BLOCK}. Email asha@example.com"
START = TEXT.index("VISA")
GOLD = Span(START, START + len(BLOCK), "credit_card", "gold")


def test_trailing_newline_removed_from_multiline_block():
    canon = canonical_span(TEXT, GOLD)
    assert canon.start == GOLD.start  # start unchanged
    assert canon.end == GOLD.end - 1
    assert TEXT[canon.start : canon.end] == BLOCK.rstrip("\n")
    assert "\n" in TEXT[canon.start : canon.end]  # inner newlines kept
    assert (canon.label, canon.source) == ("credit_card", "gold")


def test_original_span_is_not_modified():
    canonical_span(TEXT, GOLD)
    assert GOLD.end - GOLD.start == len(BLOCK)


def test_unchanged_span_returned_as_is():
    s = Span(0, 4, "name")
    assert canonical_span("Asha Rao", s) is s


@pytest.mark.parametrize("tail", ["\r\n", " \n\t", " "])
def test_all_trailing_whitespace_kinds(tail):
    text = "Asha Rao" + tail + "x"
    canon = canonical_span(text, Span(0, 8 + len(tail), "name"))
    assert (canon.start, canon.end) == (0, 8)


def test_leading_whitespace_is_kept():
    # Policy covers trailing whitespace only; nothing in the data needs more.
    canon = canonical_span(" Asha", Span(0, 5, "name"))
    assert (canon.start, canon.end) == (0, 5)


def test_whitespace_only_span():
    assert canonical_span("a \n b", Span(1, 3, "name")) is None
    with pytest.raises(ValueError):
        canonical_spans("a \n b", [Span(1, 3, "name")])


def test_out_of_range_rejected():
    with pytest.raises(ValueError):
        canonical_span("abc", Span(0, 9, "name"))


def test_entity_keeps_type_text_and_source():
    det = Entity.from_text(TEXT, GOLD.start, GOLD.end, "credit_card", "rule:credit_card_block")
    canon = canonical_span(TEXT, det)
    assert isinstance(canon, Entity)
    assert canon.text == BLOCK.rstrip("\n") == TEXT[canon.start : canon.end]
    assert canon.source == "rule:credit_card_block"


def test_gold_and_prediction_agree_after_the_same_policy():
    (pred,) = [e for e in detect(TEXT) if e.label == "credit_card"]
    g, p = canonical_spans(TEXT, [GOLD])[0], canonical_spans(TEXT, [pred])[0]
    assert (g.start, g.end, g.label) == (p.start, p.end, p.label)


def test_other_spans_untouched_and_order_kept():
    email_start = TEXT.index("asha@")
    spans = [GOLD, Span(email_start, email_start + 16, "email")]
    out = canonical_spans(TEXT, spans)
    assert out[1] is spans[1]
    assert [s.label for s in out] == ["credit_card", "email"]
