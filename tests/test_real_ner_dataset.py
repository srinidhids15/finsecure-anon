"""Acceptance tests of the NER dataset build on the real training workbook.

Skipped when the raw data is absent. The official test workbook is used only
by Phase 2's own pipeline (to rebuild its inputs in a temp dir); the NER
build never receives it.
"""

import json

import pytest

from finguard.data import build_ner_dataset as build
from finguard.data.loader import file_sha256, load_workbook_records, read_jsonl
from finguard.data.prepare import DEFAULT_RAW_DIR, TEST_FILE, TRAIN_FILE
from finguard.data.prepare import run as run_phase2

pytestmark = pytest.mark.skipif(
    not (DEFAULT_RAW_DIR / TRAIN_FILE).exists() or not (DEFAULT_RAW_DIR / TEST_FILE).exists(),
    reason="Mendeley dataset not downloaded",
)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    p2 = tmp_path_factory.mktemp("phase2")
    out = tmp_path_factory.mktemp("ner")
    raw_sha = {p.name: file_sha256(p) for p in DEFAULT_RAW_DIR.glob("*.xlsx")}
    run_phase2(DEFAULT_RAW_DIR, p2)
    report = build.run(DEFAULT_RAW_DIR, p2, out)
    return report, out, raw_sha, p2


def test_raw_workbooks_preserved(built):
    report, _, raw_sha, _ = built
    assert {p.name: file_sha256(p) for p in DEFAULT_RAW_DIR.glob("*.xlsx")} == raw_sha
    assert report["inputs_unchanged"] is True


def test_corrections_on_real_data(built):
    report, out, *_ = built
    c = report["corrections"]
    assert c["review_items"] == 0
    assert c["spans_added"] == c["gold_spans_after"] - c["gold_spans_before"] > 0
    doc = json.loads((out / "corrections.json").read_text())
    raw = {r.record_id: r for r in load_workbook_records(DEFAULT_RAW_DIR / TRAIN_FILE, "train")}
    for entry in doc["records"][:500]:
        rec = raw[entry["record_id"]]
        assert entry["original_annotation"] == rec.annotation
        for corr in entry["corrections"]:
            value = rec.source_values[corr["label"]]
            assert corr["source_value"] == value
            assert corr["text"] == value.replace(".", "") or corr["text"].replace(".", "") == value.replace(".", "")
            assert corr["text"] != value


def _rows(out, part):
    with open(out / f"{part}.jsonl", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_three_partitions_are_template_disjoint(built):
    report, out, *_ = built
    parts = {p: _rows(out, p) for p in ("train", "validation", "internal_test")}
    assert sum(len(v) for v in parts.values()) == 45000
    assert all(r["id"].startswith("train-") for v in parts.values() for r in v)
    for key, expected in (("group", report["split_units"]), ("template_group", report["template_groups"])):
        seen = {p: {r[key] for r in v} for p, v in parts.items()}
        assert not seen["train"] & seen["validation"], key
        assert not seen["train"] & seen["internal_test"], key
        assert not seen["validation"] & seen["internal_test"], key
        assert sum(len(x) for x in seen.values()) == expected, key


def test_reviewed_alias_families_stay_in_one_partition(built):
    report, out, *_ = built
    assert report["settings"]["template_aliases"].endswith("template_aliases.json")
    where = {}
    for part in ("train", "validation", "internal_test"):
        for r in _rows(out, part):
            where.setdefault(r["template_group"], set()).add(part)
    assert all(len(p) == 1 for p in where.values())
    for family in report["alias_families"]:
        parts = set().union(*(where[g] for g in family["groups"]))
        assert len(parts) == 1, family["name"]


def test_high_frequency_units_reach_both_evaluation_partitions(built):
    report, *_ = built
    for part in ("validation", "internal_test"):
        p = report["partitions"][part]
        assert p["high_frequency_units"] >= 1 and p["low_frequency_units"] >= 1, part
        assert p["missing_classes"] == [], part
    assert report["partitions"]["train"]["missing_classes"] == []


def test_build_is_reproducible(built, tmp_path_factory):
    _, out, _, p2 = built
    again = tmp_path_factory.mktemp("ner_again")
    build.run(DEFAULT_RAW_DIR, p2, again)
    for name in ("train.jsonl", "validation.jsonl", "internal_test.jsonl", "split_report.json"):
        assert (out / name).read_bytes() == (again / name).read_bytes(), name


def test_every_gold_span_aligns_with_the_finguard_tokenizer(built):
    from finguard.ner.alignment import audit
    from finguard.ner.tokenizer import make_nlp

    _, out, *_ = built
    nlp = make_nlp()
    # No documented exceptions remain; if one appears it must be added to
    # docs/DATASET.md together with a resolution, not hidden here.
    for part in ("train", "validation", "internal_test"):
        report = audit(nlp, read_jsonl(out / f"{part}.jsonl"))
        assert report.total_failures == 0, (part, report.to_dict()["failures"])


def test_saved_tokenizer_reproduces_real_document_tokens(built, tmp_path):
    import spacy

    from finguard.ner.tokenizer import make_nlp

    _, out, *_ = built
    nlp = make_nlp()
    nlp.to_disk(tmp_path / "m")
    reloaded = spacy.load(tmp_path / "m")
    texts = [e.text for e in read_jsonl(out / "validation.jsonl")[:500]]
    for a, b in zip(nlp.tokenizer.pipe(texts), reloaded.tokenizer.pipe(texts)):
        assert [(t.idx, t.text) for t in a] == [(t.idx, t.text) for t in b]
