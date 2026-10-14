"""Tests for the evaluation runner and the freeze guard (synthetic data only).

The real official workbook is never opened here: ``load_official`` is
exercised on a synthetic workbook built in a temp directory.
"""

import json

import pytest

from finguard.data.loader import file_sha256
from finguard.data.prepare import TEST_FILE, TRAIN_FILE
from finguard.data.prepare import run as run_phase2
from finguard.evaluation import freeze as fz
from finguard.evaluation import run as ev
from finguard.schema import Entity, Span

BLOCK = "VISA 16 digit\nAsha Rao\n4111111111111111 12/27\nCVC: 123\n"


def test_held_out_evaluation_refuses_without_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(fz, "FREEZE_FILE", tmp_path / "missing.json")
    for dataset in ("internal_test", "official_test"):
        with pytest.raises(fz.FreezeError, match="freeze the system"):
            ev.run(dataset, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_freeze_detects_any_change(tmp_path):
    path = tmp_path / "freeze.json"
    record = fz.freeze(path, note="test")
    assert fz.verify_freeze(path)["freeze_sha256"] == record["freeze_sha256"]
    tampered = json.loads(path.read_text())
    tampered["files"]["src/finguard/detection/hybrid.py"] = "0" * 64
    path.write_text(json.dumps(tampered))
    with pytest.raises(fz.FreezeError, match="hybrid.py"):
        fz.verify_freeze(path)
    tampered = json.loads(path.read_text())
    tampered["files"]["src/finguard/detection/hybrid.py"] = record["files"]["src/finguard/detection/hybrid.py"]
    tampered["hybrid_policy"] = {"card_block_precedence": False, "rule_priority_labels": []}
    path.write_text(json.dumps(tampered))
    with pytest.raises(fz.FreezeError, match="hybrid_policy"):
        fz.verify_freeze(path)


def test_unknown_dataset_rejected(tmp_path):
    with pytest.raises(ValueError):
        ev.run("train", tmp_path)


def test_policy_variants_are_predeclared():
    assert set(ev.POLICY_VARIANTS) == {"ner_priority", "card_block_only", "rule_priority_structured"}


def test_official_loader_applies_established_correction_without_writing(tmp_path, make_row, make_workbook):
    raw, p2 = tmp_path / "raw", tmp_path / "p2"
    raw.mkdir()
    train = [make_row(t, {"name": n}) for t in ("Dear {name}, SSN {ssn}.", "Mail {email} now.", "Call {phone} today.")
             for n in ("Asha Rao", "Ben Lee", "Chen Wu")]
    make_workbook(raw / TRAIN_FILE, train)
    test_row = make_row("Hello {name}, write to {email}.")
    test_row["Text"] += " Backup: asha@exampleorg"  # generator defect: dot-dropped, unannotated
    test_row["Email"] = "asha@example.org"
    test_row["True Predictions"] = repr([(6, 14, "name")] + [
        (test_row["Text"].index("asha@example.com"), test_row["Text"].index("asha@example.com") + 16, "email")])
    test_row["Text"] = test_row["Text"].replace("asha@example.com", "asha@example.org")
    make_workbook(raw / TEST_FILE, [test_row])
    run_phase2(raw, p2, validation_fraction=0.3, seed=0)
    before = file_sha256(raw / TEST_FILE)

    records, info = ev.load_official(raw / TEST_FILE, p2 / "test.jsonl")

    assert file_sha256(raw / TEST_FILE) == before == info["workbook_sha256"]
    (r,) = records
    assert len(r["gold_original"]) == 2
    assert len(r["gold"]) == 3
    added = [s for s in r["gold"] if s not in r["gold_original"]]
    assert [(s.label, r["text"][s.start:s.end]) for s in added] == [("email", "asha@exampleorg")]
    assert info["corrections_by_label"] == {"email": 1} and info["review_items"] == []


def test_card_diagnostics_classifies_blocks():
    text = f"Card {BLOCK}x"
    gold = (Span(5, 5 + len(BLOCK), "credit_card"),)
    rec = {"text": text, "gold": gold}
    exact = [Entity.from_text(text, 5, 5 + len(BLOCK) - 1, "credit_card", "ner")]
    holder = [Entity.from_text(text, text.index("Asha"), text.index("Asha") + 8, "name", "ner")]
    d = ev.card_diagnostics([rec, rec, rec], [exact, holder, []])
    assert (d["exact_block"], d["only_other_labels_inside"], d["missed"]) == (1, 1, 1)
    assert d["name_predictions_inside_gold_blocks"] == 1
    assert d["block_recall"] == pytest.approx(1 / 3)
