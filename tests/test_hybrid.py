"""Tests for the hybrid merge policy (stub NER, real rule detector)."""

import pytest

from finguard.detection.base import Detector, overlapping_pairs
from finguard.detection.hybrid import DEFAULT_POLICY, HybridDetector, HybridPolicy, merge
from finguard.detection.rules import RuleDetector
from finguard.schema import Entity

BLOCK = "VISA 16 digit\nAsha Rao\n4111111111111111 12/27\nCVC: 123\n"
TEXT = f"Dear Ben Lee, card ending in {BLOCK}, mail ben@x.com or call +91 1472004612."
B0 = TEXT.index("VISA")


def ner(text, *items):
    """Entities 'as predicted by NER': (substring, label[, occurrence start])."""
    out = []
    for piece, label, *rest in items:
        start = rest[0] if rest else text.index(piece)
        out.append(Entity.from_text(text, start, start + len(piece), label, "ner"))
    return out


def rules(text):
    return RuleDetector().detect(text)


def keys(entities):
    return [(e.label, e.text) for e in entities]


class StubNER:
    name = "ner"

    def __init__(self, fn):
        self.fn = fn

    def detect(self, text):
        return self.fn(text)


def test_identical_spans_are_merged_and_sources_joined():
    r = merge(TEXT, rules(TEXT), ner(TEXT, ("ben@x.com", "email")))
    (email,) = [e for e in r.entities if e.label == "email"]
    assert email.source == "rule:email|ner"
    assert r.merged_duplicates == 1 and r.suppressed == []


def test_card_block_matches_ner_block_after_canonicalisation():
    # NER predicts the block without its trailing newline; the rule keeps it.
    r = merge(TEXT, rules(TEXT), ner(TEXT, (BLOCK.rstrip("\n"), "credit_card")))
    (card,) = [e for e in r.entities if e.label == "credit_card"]
    assert card.text == BLOCK and card.source == "rule:credit_card_block|ner"


def test_card_block_suppresses_holder_name_and_number_inside():
    n = ner(TEXT, ("Ben Lee", "name"), ("Asha Rao", "name"), ("4111111111111111", "credit_card"))
    r = merge(TEXT, rules(TEXT), n)
    assert ("name", "Asha Rao") not in keys(r.entities)
    assert ("credit_card", "4111111111111111") not in keys(r.entities)
    assert ("name", "Ben Lee") in keys(r.entities)            # outside the block: kept
    reasons = {(s.candidate.text, s.reason, s.winner.label) for s in r.suppressed}
    assert ("Asha Rao", "inside_credit_card_block", "credit_card") in reasons
    assert ("4111111111111111", "inside_credit_card_block", "credit_card") in reasons


def test_partial_overlap_with_card_block_is_suppressed_and_recorded():
    start = TEXT.index("CVC")
    piece = TEXT[start:start + 20]  # runs past the end of the block
    r = merge(TEXT, rules(TEXT), ner(TEXT, (piece, "address", start)))
    assert ("address", piece) not in keys(r.entities)
    assert r.suppressed[0].reason == "inside_credit_card_block"


def test_without_card_precedence_ner_wins_conflicts():
    policy = HybridPolicy(card_block_precedence=False, rule_priority_labels=frozenset())
    r = merge(TEXT, rules(TEXT), ner(TEXT, ("Asha Rao", "name")), policy)
    assert ("name", "Asha Rao") in keys(r.entities)
    assert not any(e.label == "credit_card" for e in r.entities)
    assert r.suppressed[0].reason == "ner_preferred"


def test_rule_priority_for_structured_label_boundary_conflict():
    # NER phone span misses the country code; the validated rule span wins.
    n = ner(TEXT, ("1472004612", "phone"))
    r = merge(TEXT, rules(TEXT), n)
    assert ("phone", "+91 1472004612") in keys(r.entities)
    assert ("phone", "1472004612") not in keys(r.entities)
    assert r.suppressed[0].reason == "rule_priority_label"


def test_ner_wins_for_non_priority_label():
    text = "Contact Asha Rao at asha@x.com"
    policy = HybridPolicy(card_block_precedence=True, rule_priority_labels=frozenset({"phone"}))
    r = merge(text, rules(text), ner(text, ("asha@x.com", "name")), policy)  # contrived NER mistake
    assert keys(r.entities) == [("name", "asha@x.com")]
    assert r.suppressed[0].reason == "ner_preferred"


def test_ner_contextual_entities_kept_when_no_conflict():
    text = "Asha Rao of Acme Ltd lives at 12 Lake Road. Mail a@x.com"
    n = ner(text, ("Asha Rao", "name"), ("Acme Ltd", "company"), ("12 Lake Road", "address"))
    r = merge(text, rules(text), n)
    assert keys(r.entities) == [("name", "Asha Rao"), ("company", "Acme Ltd"),
                                ("address", "12 Lake Road"), ("email", "a@x.com")]


def test_overlapping_rule_entities_longer_span_wins():
    text = "Open https://x.com/?u=a@x.com now"
    r = merge(text, rules(text), [])
    assert keys(r.entities) == [("url", "https://x.com/?u=a@x.com")]
    assert r.suppressed[0].reason == "longer_span"


def test_output_never_overlaps_and_offsets_are_exact():
    n = ner(TEXT, ("Ben Lee", "name"), ("Asha Rao", "name"), ("ben@x", "email"), ("call +91", "phone"))
    r = merge(TEXT, rules(TEXT), n)
    assert overlapping_pairs(r.entities) == []
    for e in r.entities:
        assert TEXT[e.start:e.end] == e.text
    # nothing disappears without a record: every input is kept, merged or suppressed
    total_in = len(rules(TEXT)) + len(n)
    assert len(r.entities) + len(r.suppressed) + r.merged_duplicates == total_in


def test_merge_is_deterministic_and_input_order_independent():
    n = ner(TEXT, ("Asha Rao", "name"), ("1472004612", "phone"))
    a = merge(TEXT, rules(TEXT), n)
    b = merge(TEXT, list(reversed(rules(TEXT))), list(reversed(n)))
    assert a.entities == b.entities
    assert [(s.candidate, s.reason) for s in a.suppressed] == [(s.candidate, s.reason) for s in b.suppressed]


def test_empty_input():
    det = HybridDetector(ner=StubNER(lambda t: []))
    assert det.detect("") == []
    r = merge("", [], [])
    assert r.entities == [] and r.suppressed == []


def test_hybrid_detector_interface():
    det = HybridDetector(ner=StubNER(lambda t: ner(t, ("Ben Lee", "name"))))
    assert isinstance(det, Detector)
    report = det.detect_with_report(TEXT)
    assert ("name", "Ben Lee") in keys(report.entities)
    # the rule detector's own suppression (card number inside the block) is passed through
    assert any(c.source == "rule:credit_card_block" for _, c in report.rule_suppressed)
    with pytest.raises(TypeError):
        det.detect(None)


def test_default_policy_is_documented():
    assert DEFAULT_POLICY.describe() == {
        "card_block_precedence": True,
        "rule_priority_labels": ["credit_card", "email", "phone", "ssn", "url"],
    }
