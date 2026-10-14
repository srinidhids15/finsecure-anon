"""Tests for the shared Span data contract."""

import pytest

from finguard.schema import Span


def test_valid_span_extracts_text():
    text = "Paid by Asha Rao on Monday"
    span = Span(8, 16, "PERSON", source="gold")
    assert span.text_in(text) == "Asha Rao"


@pytest.mark.parametrize(
    "start, end, label",
    [
        (-1, 3, "PERSON"),  # negative start
        (5, 5, "PERSON"),  # empty span
        (6, 2, "PERSON"),  # end before start
        (0, 3, ""),  # empty label
        (0, 3, "   "),  # whitespace label
    ],
)
def test_invalid_span_rejected(start, end, label):
    with pytest.raises(ValueError):
        Span(start, end, label)


def test_non_integer_offsets_rejected():
    with pytest.raises(TypeError):
        Span(0.0, 3, "PERSON")


def test_span_past_end_of_text_rejected():
    with pytest.raises(ValueError):
        Span(0, 50, "PERSON").text_in("short text")


def test_overlap_detection():
    a = Span(0, 5, "PERSON")
    assert a.overlaps(Span(4, 8, "ORG"))
    assert not a.overlaps(Span(5, 8, "ORG"))  # touching is not overlapping


def test_span_is_immutable():
    span = Span(0, 3, "PERSON")
    with pytest.raises(AttributeError):
        span.start = 1
