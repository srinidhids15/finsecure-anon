"""Tests for the safe annotation parser."""

import pytest

from finguard.data.annotations import (
    MAX_ANNOTATION_CHARS,
    AnnotationParseError,
    parse_annotation_string,
)


def test_parses_list_of_tuples():
    assert parse_annotation_string("[(5, 15, 'name'), (20, 31, 'ssn')]") == [
        (5, 15, "name"),
        (20, 31, "ssn"),
    ]


def test_empty_list_is_allowed_here():
    # An empty list parses fine; validation reports it as a warning.
    assert parse_annotation_string("[]") == []


def test_tuple_container_becomes_list():
    assert parse_annotation_string("((0, 3, 'name'),)") == [(0, 3, "name")]


def test_code_is_not_executed(tmp_path):
    marker = tmp_path / "pwned"
    malicious = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    with pytest.raises(AnnotationParseError):
        parse_annotation_string(malicious)
    assert not marker.exists()


@pytest.mark.parametrize(
    "raw",
    [
        "",  # empty
        "   ",  # whitespace
        "[(0, 3, 'name')",  # unclosed bracket
        "(0, 3, 'name'), (4",  # truncated
        "{'start': 0}",  # dict, not a list
        "'just a string'",  # string literal
        "42",  # number
        "nan",  # what an empty spreadsheet cell can become
    ],
)
def test_malformed_strings_rejected(raw):
    with pytest.raises(AnnotationParseError):
        parse_annotation_string(raw)


@pytest.mark.parametrize("raw", [None, 3.5, ["(0, 3, 'name')"]])
def test_non_string_input_rejected(raw):
    with pytest.raises(AnnotationParseError):
        parse_annotation_string(raw)


def test_oversized_string_rejected():
    with pytest.raises(AnnotationParseError):
        parse_annotation_string("[" + " " * MAX_ANNOTATION_CHARS + "]")
