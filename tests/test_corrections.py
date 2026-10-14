"""Tests for the dot_dropped_variant annotation correction."""

import pytest

from finguard.data.corrections import (
    RULE_ID,
    apply_corrections,
    dot_variant_pattern,
    find_dot_drop_corrections,
)
from finguard.schema import Span

VALUES = {
    "email": "asha@example.org",
    "url": "https://www.acme.com/",
    "address": "12 Lake Road Apt. 4\nPune, MH 411001",
    "name": "Dr. Asha Rao",
    "phone": "+91 1472004612",  # no dot: never corrected
}


def run(text, spans=(), labels=("email", "url", "address", "name")):
    return find_dot_drop_corrections("r1", text, list(spans), VALUES, labels)


def test_pattern_makes_only_dots_optional():
    p = dot_variant_pattern("a.b.c")
    assert p.fullmatch("abc") and p.fullmatch("a.bc") and p.fullmatch("a.b.c")
    assert not p.fullmatch("a-b-c")
    assert dot_variant_pattern("no dots here") is None


@pytest.mark.parametrize(
    "label, variant, dots",
    [
        ("email", "asha@exampleorg", 1),
        ("url", "https://wwwacme.com/", 1),
        ("url", "https://wwwacmecom/", 2),
        ("address", "12 Lake Road Apt 4\nPune, MH 411001", 1),
        ("name", "Dr Asha Rao", 1),
    ],
)
def test_dot_dropped_variants_are_corrected(label, variant, dots):
    text = f"See ({variant}), thanks."
    found, review = run(text)
    assert review == []
    (c,) = found
    assert (c.label, c.text, c.dots_removed, c.rule) == (label, variant, dots, RULE_ID)
    assert text[c.start : c.end] == variant
    assert c.source_value == VALUES[label]


def test_exact_value_is_never_a_correction():
    text = "Mail asha@example.org now"
    assert run(text, [Span(5, 21, "email")]) == ([], [])


def test_labels_without_dots_are_ignored():
    found, _ = run("Call +91 1472004612", labels=("phone",))
    assert found == []


def test_other_mutations_are_not_corrected():
    # Only deleted dots are explained by the defect; any other change is not.
    for text in ("asha@exampl.org", "asha@example-org", "ASHA@exampleorg", "asha@example org"):
        assert run(text) == ([], []), text


def test_repeated_variants_are_all_corrected():
    text = "a asha@exampleorg b asha@exampleorg"
    found, _ = run(text)
    assert [c.start for c in found] == [2, 20]


def test_embedded_variant_goes_to_review():
    found, review = run("xasha@exampleorg")
    assert found == []
    assert [r.problem for r in review] == ["embedded in a larger token"]


def test_overlap_with_gold_goes_to_review():
    text = "Contact Dr Asha Rao now"
    gold = [Span(8, 19, "company")]  # hypothetical conflicting gold span
    found, review = run(text, gold)
    assert found == []
    assert review[0].problem == "overlaps gold company"


def test_overlapping_corrections_go_to_review():
    # Both variants have clean token boundaries but share "Rao".
    values = {"name": "Asha R. Rao", "address": "Rao St. 5"}
    found, review = find_dot_drop_corrections("r", "Asha R Rao St 5", [], values, ("name", "address"))
    assert found == []
    assert [(r.label, r.problem) for r in review] == [
        ("name", "overlaps another correction"), ("address", "overlaps another correction")
    ]


def test_provenance_record():
    (c,), _ = run("x asha@exampleorg")
    d = c.to_dict()
    assert set(d) == {"record_id", "start", "end", "label", "text", "source_value",
                      "dots_removed", "rule", "reason"}
    assert "dot deletion" in d["reason"]


def test_apply_keeps_originals_and_marks_source():
    text = "Dr Asha Rao paid. Email asha@example.org"
    gold = (Span(24, 40, "email", "gold"),)
    found, _ = run(text, gold)
    merged = apply_corrections(gold, found)
    assert merged[1] is gold[0]  # original object untouched
    assert [(s.label, s.source) for s in merged] == [
        ("name", f"correction:{RULE_ID}"), ("email", "gold")
    ]


def test_apply_rejects_overlap():
    (c,), _ = run("x asha@exampleorg")
    with pytest.raises(ValueError):
        apply_corrections([Span(0, 5, "name")], [c])


def test_repeatable():
    text = "Dr Asha Rao: asha@exampleorg, https://wwwacmecom/."
    assert run(text) == run(text)
