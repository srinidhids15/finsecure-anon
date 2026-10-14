"""Tests for workbook loading and JSONL round-trips."""

import pandas as pd
import pytest

from finguard.data.loader import (
    DatasetFormatError,
    Example,
    file_sha256,
    load_workbook_records,
    read_jsonl,
    write_jsonl,
)
from finguard.schema import Span


def test_loads_rows_as_strings(tmp_path, make_row, make_workbook):
    row = make_row("Dear {name}, SSN {ssn}.", {"ssn": "012-34-5678", "phone": "12345"})
    path = make_workbook(tmp_path / "t.xlsx", [row, make_row("Hi {name}.")])
    recs = load_workbook_records(path, "train")

    assert [r.record_id for r in recs] == ["train-000002", "train-000003"]
    assert recs[0].text == row["Text"]
    assert recs[0].annotation == row["True Predictions"]
    # Leading zeros and numeric-looking values must stay text.
    assert recs[0].source_values["ssn"] == "012-34-5678"
    assert recs[0].source_values["phone"] == "12345"


def test_empty_cells_become_empty_strings(tmp_path, make_row, make_workbook):
    row = make_row("Hi {name}.")
    row["Text"] = ""
    row["Email"] = None
    recs = load_workbook_records(make_workbook(tmp_path / "t.xlsx", [row]), "test")
    assert recs[0].text == ""
    assert recs[0].source_values["email"] == ""


def test_missing_column_raises(tmp_path, make_row):
    row = make_row("Hi {name}.")
    del row["True Predictions"]
    path = tmp_path / "t.xlsx"
    pd.DataFrame([row]).to_excel(path, index=False)
    with pytest.raises(DatasetFormatError, match="True Predictions"):
        load_workbook_records(path, "train")


def test_multiple_sheets_raise(tmp_path, make_row):
    path = tmp_path / "t.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame([make_row("Hi {name}.")]).to_excel(writer, sheet_name="a", index=False)
        pd.DataFrame([make_row("Hi {name}.")]).to_excel(writer, sheet_name="b", index=False)
    with pytest.raises(DatasetFormatError, match="sheet"):
        load_workbook_records(path, "train")


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_workbook_records(tmp_path / "nope.xlsx", "train")


def test_checksum_detects_change(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    before = file_sha256(p)
    p.write_bytes(b"abd")
    assert file_sha256(p) != before


def test_jsonl_round_trip(tmp_path):
    examples = [
        Example("train-000002", "Asha Rao’s card", (Span(0, 8, "name", "gold"),), "g1"),
        Example("test-000002", "No PII", ()),
    ]
    path = tmp_path / "out.jsonl"
    write_jsonl(examples, path)
    assert read_jsonl(path) == examples


def test_jsonl_with_out_of_range_span_fails(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "x", "text": "abc", "spans": [[0, 10, "name"]]}\n')
    with pytest.raises(DatasetFormatError, match="bad.jsonl:1"):
        read_jsonl(path)
