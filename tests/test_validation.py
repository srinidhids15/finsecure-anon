"""Tests for record validation: one test per kind of problem."""

import pytest

from finguard.data import validation as v
from finguard.data.loader import RawRecord


def codes(result):
    return sorted(i.code for i in result.issues)


def check(text, annotation, source_values=None):
    return v.validate_record("r1", text, annotation, source_values)


# ------------------------------------------------------------------ valid


def test_valid_example(make_row):
    row = make_row("Dear {name}, your SSN {ssn} is verified by {company}.")
    values = {"name": row["Name"], "ssn": row["SSN"], "company": row["Company"]}
    result = check(row["Text"], row["True Predictions"], values)
    assert result.is_valid
    assert result.issues == ()
    assert [s.label for s in result.spans] == ["name", "ssn", "company"]
    assert [s.text_in(row["Text"]) for s in result.spans] == [
        "Asha Rao", "048-59-6922", "Acme Ltd"
    ]


def test_spans_are_returned_sorted():
    result = check("Asha Rao at Acme Ltd", "[(12, 20, 'company'), (0, 8, 'name')]")
    assert [(s.start, s.label) for s in result.spans] == [(0, "name"), (12, "company")]


# --------------------------------------------------------------- empty text


@pytest.mark.parametrize("text", ["", "   \n\t "])
def test_empty_text(text):
    result = check(text, "[(0, 3, 'name')]")
    assert codes(result) == [v.MISSING_TEXT]
    assert not result.is_valid


def test_empty_annotation_list_is_only_a_warning():
    result = check("Nothing sensitive here.", "[]")
    assert codes(result) == [v.EMPTY_ANNOTATIONS]
    assert result.is_valid


# -------------------------------------------------------- malformed annotations


def test_unparseable_annotation():
    result = check("Asha Rao", "[(0, 8, 'name'")
    assert codes(result) == [v.PARSE_ERROR]
    assert not result.is_valid


@pytest.mark.parametrize(
    "annotation",
    [
        "[(0, 8)]",  # missing label
        "[(0, 8, 'name', 'extra')]",  # too many fields
        "['0, 8, name']",  # a string, not a tuple
        "[('0', 8, 'name')]",  # start is a string
        "[(0.0, 8, 'name')]",  # start is a float
        "[(True, 8, 'name')]",  # bool is not an offset
        "[(0, 8, 3)]",  # label is not a string
    ],
)
def test_malformed_span(annotation):
    result = check("Asha Rao", annotation)
    assert codes(result) == [v.MALFORMED_SPAN]
    assert result.spans == ()


def test_one_bad_span_does_not_hide_good_ones():
    result = check("Asha Rao at Acme Ltd", "[(0, 8, 'name'), (12, 20)]")
    assert codes(result) == [v.MALFORMED_SPAN]
    assert [s.label for s in result.spans] == ["name"]
    assert not result.is_valid


# ----------------------------------------------------------- invalid labels


@pytest.mark.parametrize("label", ["person", "Name", "NAME", " name", ""])
def test_invalid_label(label):
    result = check("Asha Rao", f"[(0, 8, {label!r})]")
    assert codes(result) == [v.INVALID_LABEL]


# ---------------------------------------------------------- invalid offsets


@pytest.mark.parametrize(
    "start, end",
    [
        (-1, 8),  # negative start
        (4, 4),  # empty span
        (6, 2),  # end before start
        (0, 99),  # past the end of the text
    ],
)
def test_invalid_offsets(start, end):
    result = check("Asha Rao", f"[({start}, {end}, 'name')]")
    assert codes(result) == [v.INVALID_OFFSET]


def test_end_exactly_at_text_length_is_valid():
    assert check("Asha Rao", "[(0, 8, 'name')]").is_valid


# ----------------------------------------------------- substring verification


def test_shifted_span_fails_substring_check():
    # Correct length, wrong position: the exact defect found in the raw data.
    text = "Paid to Asha Rao."
    result = check(text, "[(6, 14, 'name')]", {"name": "Asha Rao"})
    assert v.SUBSTRING_MISMATCH in codes(result)
    assert not result.is_valid


def test_substring_check_skipped_without_source_values():
    # Without source values we can only check bounds, not content.
    assert check("Paid to Asha Rao.", "[(6, 14, 'name')]").is_valid


def test_value_present_but_unannotated_is_a_warning():
    text = "Asha Rao paid. Contact Asha Rao."
    result = check(text, "[(0, 8, 'name')]", {"name": "Asha Rao"})
    assert codes(result) == [v.UNANNOTATED_VALUE]
    assert result.is_valid


def test_boundary_whitespace_is_a_warning(make_row):
    row = make_row("Card: {credit_card}done")
    values = {"credit_card": row["Credit Card"]}
    result = check(row["Text"], row["True Predictions"], values)
    assert codes(result) == [v.BOUNDARY_WHITESPACE]
    assert result.is_valid


# --------------------------------------------------------- overlapping spans


@pytest.mark.parametrize(
    "annotation",
    [
        "[(0, 8, 'name'), (5, 12, 'company')]",  # partial overlap
        "[(0, 20, 'company'), (0, 8, 'name')]",  # nested, different labels
        "[(0, 8, 'name'), (0, 4, 'name')]",  # nested, same label
    ],
)
def test_overlapping_spans_are_errors(annotation):
    result = check("Asha Rao at Acme Ltd", annotation)
    assert v.OVERLAPPING_SPANS in codes(result)
    assert not result.is_valid


def test_adjacent_spans_do_not_overlap():
    result = check("Asha RaoAcme Ltd", "[(0, 8, 'name'), (8, 16, 'company')]")
    assert result.is_valid


def test_duplicate_span_is_an_error():
    result = check("Asha Rao", "[(0, 8, 'name'), (0, 8, 'name')]")
    assert codes(result) == [v.DUPLICATE_SPAN]
    assert len(result.spans) == 1


# ------------------------------------------------------ duplicates across rows


def _rec(rid, text):
    return RawRecord(rid, rid.split("-")[0], text, "[]")


def test_duplicate_records_within_a_file():
    recs = [_rec("train-2", "A"), _rec("train-3", "B"), _rec("train-4", "A")]
    issues = v.find_duplicate_texts(recs)
    assert [(i.record_id, i.code) for i in issues] == [("train-4", v.DUPLICATE_RECORD)]
    assert "train-2" in issues[0].detail


def test_empty_texts_are_not_reported_as_duplicates():
    assert v.find_duplicate_texts([_rec("train-2", ""), _rec("train-3", "")]) == []


def test_cross_split_duplicate_is_flagged_on_the_training_record():
    train = [_rec("train-2", "shared"), _rec("train-3", "only train")]
    test = [_rec("test-2", "shared")]
    issues = v.find_cross_split_duplicates(train, test)
    assert [(i.record_id, i.code) for i in issues] == [
        ("train-2", v.CROSS_SPLIT_DUPLICATE)
    ]


def test_issue_details_never_contain_document_text():
    secret = "Asha Rao"
    text = f"Paid to {secret}. Again {secret}."
    result = check(text, "[(6, 14, 'name'), (0, 99, 'name'), (0, 4, 'bogus')]",
                   {"name": secret})
    assert result.issues
    assert all(secret not in i.detail for i in result.issues)
