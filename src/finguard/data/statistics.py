"""Descriptive statistics for a set of processed examples."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from statistics import mean, median

from finguard.config import ENTITY_LABELS
from finguard.data.loader import Example


def _summary(values: Sequence[int]) -> dict:
    if not values:
        return {"min": 0, "median": 0, "mean": 0.0, "max": 0}
    return {
        "min": min(values),
        "median": median(values),
        "mean": round(mean(values), 1),
        "max": max(values),
    }


def compute_statistics(examples: Sequence[Example]) -> dict:
    """Return record-level and per-class statistics.

    Per class: number of spans, number of records containing the class,
    span length summary and number of distinct surface values (a low count
    warns that a model could memorise values instead of learning context).
    """
    span_lengths: dict[str, list[int]] = defaultdict(list)
    records_with: Counter[str] = Counter()
    distinct: dict[str, set[str]] = defaultdict(set)
    spans_per_record = []

    for ex in examples:
        spans_per_record.append(len(ex.spans))
        for label in {s.label for s in ex.spans}:
            records_with[label] += 1
        for s in ex.spans:
            span_lengths[s.label].append(s.end - s.start)
            distinct[s.label].add(ex.text[s.start : s.end])

    per_class = {
        label: {
            "spans": len(span_lengths[label]),
            "records": records_with[label],
            "distinct_values": len(distinct[label]),
            "span_length": _summary(span_lengths[label]),
        }
        for label in ENTITY_LABELS
    }
    return {
        "records": len(examples),
        "total_spans": sum(spans_per_record),
        "text_length": _summary([len(ex.text) for ex in examples]),
        "spans_per_record": _summary(spans_per_record),
        "per_class": per_class,
    }
