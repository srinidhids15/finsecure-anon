"""End-to-end tests of the NER dataset build on tiny synthetic workbooks."""

import json

import pytest

from finguard.data import build_ner_dataset as build
from finguard.data.loader import file_sha256, read_jsonl
from finguard.data.prepare import TEST_FILE, TRAIN_FILE
from finguard.data.prepare import run as run_phase2

# Synthetic data has no reviewed alias families, so tests pass template_aliases=[]
# (the default alias file names real-data group ids and would be rejected).

# Six clearly different templates (word similarity well below 0.9).
TEMPLATES = [
    "Dear {name}, your loan application with {company} has been received and our underwriting team will contact you at {phone} within five working days to discuss next steps.",
    "Invoice issued by {company} for consulting services has been paid in full; the remittance advice was sent to {email} and a copy is available at {url} for your records.",
    "Security notice: unusual sign-in activity was detected on the account registered under SSN {ssn}; please call {phone} immediately if you did not authorise this access attempt.",
    "Shipping confirmation for order placed by {name}; the parcel will be delivered to {address} by our courier partner and tracking updates are published at {url} daily.",
    "Annual tax filing summary prepared for {name} with taxpayer identification {ssn}; questions regarding deductions should be emailed to {email} before the statutory deadline expires.",
    "Card statement: the card ending in {credit_card}was used for a purchase at {company}; dispute the charge by writing to {email} within sixty days of this statement.",
]
NAMES = ["Asha Rao", "Ben Lee", "Chen Wu"]


def values(i):
    return {
        "name": NAMES[i % 3], "company": f"Co{i} Ltd", "email": f"user{i}@example.org",
        "phone": f"+91 90000000{i:02d}", "url": f"https://www.site{i}.com/",
        "address": f"{i} Main St.\nPune, MH 4110{i:02d}", "ssn": f"100-00-{i + 1:04d}",
    }


@pytest.fixture
def workspace(tmp_path, make_row, make_workbook):
    rows, variants = [], 0
    for t_idx, template in enumerate(TEMPLATES):
        for j in range(3):
            i = t_idx * 3 + j
            row = make_row(template, values(i))
            if j == 0:  # generator defect: a dot-dropped email copy, unannotated
                row["Text"] += f" Alternate contact: user{i}@exampleorg"
                variants += 1
            rows.append(row)
    raw, p2, out = tmp_path / "raw", tmp_path / "phase2", tmp_path / "ner"
    raw.mkdir()
    make_workbook(raw / TRAIN_FILE, rows)
    make_workbook(raw / TEST_FILE, [make_row("Hello {name}.", values(99))])
    run_phase2(raw, p2, validation_fraction=0.3, seed=0)
    # The NER build must not need the official test data at all.
    (raw / TEST_FILE).unlink()
    (p2 / "test.jsonl").unlink()
    return raw, p2, out, rows, variants


def test_build_end_to_end(workspace):
    raw, p2, out, rows, variants = workspace
    before = {p: file_sha256(p) for p in [raw / TRAIN_FILE, *p2.glob("*.json*")]}

    report = build.run(raw, p2, out, template_aliases=[])

    # Inputs (raw workbook + Phase 2 outputs) are unchanged.
    assert {p: file_sha256(p) for p in before} == before
    assert report["inputs_unchanged"] is True

    # Corrections: one per injected variant, with full provenance.
    assert report["corrections"]["spans_added"] == variants
    assert report["corrections"]["review_items"] == 0
    doc = json.loads((out / "corrections.json").read_text())
    raw_annotations = {f"train-{i + 2:06d}": r["True Predictions"] for i, r in enumerate(rows)}
    for entry in doc["records"]:
        assert entry["original_annotation"] == raw_annotations[entry["record_id"]]
        (c,) = entry["corrections"]
        assert c["label"] == "email" and c["text"].endswith("@exampleorg")
        assert c["rule"] == "dot_dropped_variant" and c["reason"]

    # Partitions: complete, disjoint by id and template group.
    parts = {p: read_jsonl(out / f"{p}.jsonl") for p in ("train", "validation", "internal_test")}
    ids = [e.record_id for exs in parts.values() for e in exs]
    assert sorted(ids) == sorted(raw_annotations)
    groups = {p: {e.group for e in exs} for p, exs in parts.items()}
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["internal_test"]
    assert not groups["validation"] & groups["internal_test"]
    assert report["template_groups"] == 6  # the dot variant did not create new templates
    assert [report["partitions"][p]["template_groups"] for p in parts] == [4, 1, 1]

    # Corrected spans are in the JSONL; original spans are still there unchanged.
    for exs in parts.values():
        for ex in exs:
            for s in ex.spans:
                assert 0 <= s.start < s.end <= len(ex.text)
            if ex.text.endswith("@exampleorg"):
                assert ex.spans[-1].label == "email"
                assert ex.text[ex.spans[-1].start : ex.spans[-1].end].endswith("@exampleorg")


def test_build_is_byte_for_byte_repeatable(workspace, tmp_path):
    raw, p2, out, _, _ = workspace
    build.run(raw, p2, out, template_aliases=[])
    build.run(raw, p2, tmp_path / "again", template_aliases=[])
    for name in ("train.jsonl", "validation.jsonl", "internal_test.jsonl",
                 "corrections.json", "split_report.json", "dataset_report.md"):
        assert (out / name).read_bytes() == (tmp_path / "again" / name).read_bytes(), name


def test_tampered_phase2_output_is_rejected(workspace):
    raw, p2, out, _, _ = workspace
    path = p2 / "train.jsonl"
    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    first["text"] += " (edited)"  # any change to Phase 2 text must be caught
    path.write_text("\n".join([json.dumps(first)] + lines[1:]) + "\n")
    with pytest.raises(build.InputMismatchError):
        build.run(raw, p2, out, template_aliases=[])


def test_test_records_cannot_enter(workspace):
    raw, p2, out, _, _ = workspace
    path = p2 / "validation.jsonl"
    with open(path, "a") as fh:
        fh.write(json.dumps({"id": "test-000002", "text": "x", "spans": []}) + "\n")
    with pytest.raises(build.InputMismatchError):
        build.run(raw, p2, out, template_aliases=[])


def test_official_test_files_are_never_opened(tmp_path, make_row, make_workbook, monkeypatch):
    """Spy on every file open while the test workbook and test.jsonl exist."""
    import builtins
    import io

    raw, p2, out = tmp_path / "raw", tmp_path / "phase2", tmp_path / "ner"
    raw.mkdir()
    rows = [make_row(t, values(k * 3 + j)) for k, t in enumerate(TEMPLATES) for j in range(3)]
    make_workbook(raw / TRAIN_FILE, rows)
    make_workbook(raw / TEST_FILE, [make_row("Hello {name}.", values(99))])
    run_phase2(raw, p2, validation_fraction=0.3, seed=0)
    assert (raw / TEST_FILE).exists() and (p2 / "test.jsonl").exists()

    opened = []
    real_open, real_io_open = builtins.open, io.open

    def spy(open_fn):
        def wrapper(file, *args, **kwargs):
            opened.append(str(file))
            return open_fn(file, *args, **kwargs)
        return wrapper

    monkeypatch.setattr(builtins, "open", spy(real_open))
    monkeypatch.setattr(io, "open", spy(real_io_open))
    build.run(raw, p2, out, template_aliases=[])
    monkeypatch.undo()

    assert any(TRAIN_FILE in f for f in opened)  # the spy does see workbook opens
    assert not [f for f in opened if TEST_FILE in f or f.endswith("test.jsonl") and "internal" not in f]
