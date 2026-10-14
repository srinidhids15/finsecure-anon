"""Acceptance tests on the real Mendeley dataset.

Skipped automatically when the raw files are absent (e.g. on a fresh clone,
since data/raw/ is git-ignored). See docs/DATASET.md to obtain the data.
"""

import pytest

from finguard.data.loader import load_workbook_records, read_jsonl
from finguard.data.prepare import DEFAULT_RAW_DIR, TEST_FILE, TRAIN_FILE, run
from finguard.data.validation import validate_raw_record

pytestmark = pytest.mark.skipif(
    not (DEFAULT_RAW_DIR / TRAIN_FILE).exists() or not (DEFAULT_RAW_DIR / TEST_FILE).exists(),
    reason="Mendeley dataset not downloaded to data/raw/mendeley_tzrjx692jy_v1",
)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    out = tmp_path_factory.mktemp("processed")
    return run(DEFAULT_RAW_DIR, out), out


def test_real_data_loads_and_validates(prepared):
    report, _ = prepared
    assert report["files"]["train"]["rows"] == 45000
    assert report["files"]["test"]["rows"] == 30
    assert report["files"]["train"]["final_stage"]["invalid_records"] == 0
    assert report["files"]["test"]["final_stage"]["invalid_records"] == 0
    assert report["raw_files_unchanged"] is True


def test_raw_data_problems_are_reported_explicitly(prepared):
    report, _ = prepared
    raw_train = report["files"]["train"]["raw_stage"]
    assert raw_train["invalid_records"] == 5000
    assert len(raw_train["invalid_record_list"]) == 5000


def test_partitions_do_not_mix(prepared):
    _, out = prepared
    train, val, test = (read_jsonl(out / f"{n}.jsonl") for n in ("train", "validation", "test"))
    ids = [{e.record_id for e in part} for part in (train, val, test)]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    assert all(i.startswith("train-") for i in ids[0] | ids[1])
    assert all(i.startswith("test-") for i in ids[2])
    assert len(ids[0]) + len(ids[1]) == 45000
    assert not {e.group for e in train} & {e.group for e in val}


def test_test_partition_is_the_complete_official_test_file(prepared):
    _, out = prepared
    raw = load_workbook_records(DEFAULT_RAW_DIR / TEST_FILE, "test")
    test = read_jsonl(out / "test.jsonl")
    assert [e.record_id for e in test] == [r.record_id for r in raw]
    for rec, ex in zip(raw, test):
        # Annotations are carried over exactly; only the text encoding was repaired.
        expected = validate_raw_record(rec, ex.text).spans
        assert ex.spans == expected
        for span in ex.spans:
            assert span.text_in(ex.text) == rec.source_values[span.label]
