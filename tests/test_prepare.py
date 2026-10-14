"""End-to-end tests of the Phase 2 pipeline on tiny synthetic workbooks."""

import json

import pytest

from finguard.data.loader import file_sha256, read_jsonl
from finguard.data.prepare import TEST_FILE, TRAIN_FILE, OfficialTestSetError, run

TEMPLATES = [
    "Dear {name}, your SSN {ssn} is verified.",
    "Invoice for {company}, contact {email}.",
    "Call {phone} or visit {url} today.",
    "Ship to {address} for {name}.",
]
NAMES = ["Asha Rao", "Ben Lee", "Chen Wu", "Dev Iyer"]


def corrupt(text: str) -> str:
    """Simulate the dataset's encoding damage (’ -> â€™)."""
    return text.replace("’", "â€™")


@pytest.fixture
def dataset(tmp_path, make_row, make_workbook):
    def values(i: int) -> dict[str, str]:
        # Vary every value so no two generated rows share the same text.
        return {
            "name": NAMES[i % 4], "company": f"Co{i} Ltd", "email": f"u{i}@ex.com",
            "phone": f"+91 90000000{i:02d}", "url": f"https://s{i}.com/",
            "address": f"{i} Main St", "ssn": f"100-00-{i:04d}",
        }

    train = [make_row(t, values(i * 4 + j)) for i, t in enumerate(TEMPLATES) for j in range(4)]
    moji = make_row("The auditor’s note for {company}.")
    moji["Text"] = corrupt(moji["Text"])  # valid only after repair
    bad = make_row("Dear {name}.", {"name": "Eve Das"})
    bad["True Predictions"] = "[(0, 999, 'name')]"  # invalid offset
    shared = make_row("Shared text for {name}.", {"name": "Fay Roy"})
    train += [moji, bad, dict(train[0]), shared]  # rows 18..21 in Excel numbering

    test_moji = make_row("Client’s SSN {ssn} confirmed.")
    test_moji["Text"] = corrupt(test_moji["Text"])
    test = [make_row("Hello {name} from {company}.", {"name": "Gus Pal"}), test_moji, shared]

    raw = tmp_path / "raw"
    raw.mkdir()
    make_workbook(raw / TRAIN_FILE, train)
    make_workbook(raw / TEST_FILE, test)
    return raw, tmp_path / "processed"


def test_pipeline_end_to_end(dataset):
    raw, out = dataset
    before = {p.name: file_sha256(p) for p in raw.iterdir()}

    report = run(raw, out, validation_fraction=0.25, seed=0)

    # Raw files untouched.
    assert {p.name: file_sha256(p) for p in raw.iterdir()} == before
    assert report["raw_files_unchanged"] is True

    # Test set: complete, in order, repaired, valid.
    test = read_jsonl(out / "test.jsonl")
    assert [e.record_id for e in test] == ["test-000002", "test-000003", "test-000004"]
    assert "’" in test[1].text and "â" not in test[1].text
    assert report["files"]["test"]["raw_stage"]["invalid_records"] == 1
    assert report["files"]["test"]["final_stage"]["invalid_records"] == 0

    # Training file: exactly the three bad rows are excluded and listed.
    excluded = {
        e["record_id"]: {x["code"] for x in e["errors"]}
        for e in report["files"]["train"]["final_stage"]["invalid_record_list"]
    }
    assert excluded == {
        "train-000019": {"INVALID_OFFSET"},
        "train-000020": {"DUPLICATE_RECORD"},
        "train-000021": {"CROSS_SPLIT_DUPLICATE"},
    }

    train = read_jsonl(out / "train.jsonl")
    val = read_jsonl(out / "validation.jsonl")
    train_ids = {e.record_id for e in train}
    val_ids = {e.record_id for e in val}
    assert len(train_ids | val_ids) == 17  # 16 valid + the repaired mojibake row
    assert "train-000018" in train_ids | val_ids
    assert not train_ids & val_ids
    assert not (train_ids | val_ids) & {e.record_id for e in test}
    assert not {e.group for e in train} & {e.group for e in val}

    # Reports exist and contain no document text.
    md = (out / "audit_report.md").read_text()
    js = (out / "audit_report.json").read_text()
    for secret in ("Asha Rao", "Gus Pal", "048-59-6922"):
        assert secret not in md and secret not in js
    assert json.loads(js)["dataset"]["doi"] == "10.17632/tzrjx692jy.1"


def test_invalid_test_record_stops_the_pipeline(tmp_path, make_row, make_workbook):
    raw = tmp_path / "raw"
    raw.mkdir()
    make_workbook(raw / TRAIN_FILE, [make_row(t) for t in TEMPLATES])
    broken = make_row("Hello {name}.")
    broken["True Predictions"] = "[(0, 5, 'person')]"
    make_workbook(raw / TEST_FILE, [make_row("Hi {name}."), broken])
    out = tmp_path / "processed"

    with pytest.raises(OfficialTestSetError, match="test-000003"):
        run(raw, out)
    # The failure is documented, but no partition files are produced.
    assert (out / "audit_report.json").exists()
    assert not (out / "train.jsonl").exists()
    assert not (out / "test.jsonl").exists()
