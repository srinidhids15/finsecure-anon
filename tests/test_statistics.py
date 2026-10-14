"""Tests for dataset statistics."""

from finguard.data.loader import Example
from finguard.data.statistics import compute_statistics
from finguard.schema import Span


def test_counts_per_class():
    examples = [
        Example("a", "Asha Rao and Asha Rao", (Span(0, 8, "name"), Span(13, 21, "name"))),
        Example("b", "Ben Lee at Acme", (Span(0, 7, "name"), Span(11, 15, "company"))),
    ]
    stats = compute_statistics(examples)
    assert stats["records"] == 2
    assert stats["total_spans"] == 4
    name = stats["per_class"]["name"]
    assert name["spans"] == 3
    assert name["records"] == 2
    assert name["distinct_values"] == 2
    assert name["span_length"]["min"] == 7
    assert stats["per_class"]["company"]["records"] == 1
    # Every configured class is reported, even when absent.
    assert stats["per_class"]["ssn"]["spans"] == 0


def test_empty_input():
    stats = compute_statistics([])
    assert stats["records"] == 0
    assert stats["text_length"]["max"] == 0


def test_record_weighted_vs_group_level_distribution():
    from finguard.data.statistics import class_distribution, group_level_class_distribution

    big = [Example(f"b{i}", "Asha", (Span(0, 4, "name"),), "big") for i in range(9)]
    small = [Example("s0", "a@x.co", (Span(0, 6, "email"),), "small")]
    rec = class_distribution(big + small)
    grp = group_level_class_distribution(big + small)
    assert (rec["name"], rec["email"]) == (0.9, 0.1)  # the big group dominates
    assert (grp["name"], grp["email"]) == (0.5, 0.5)  # each group counts once
    assert grp["ssn"] == 0.0
