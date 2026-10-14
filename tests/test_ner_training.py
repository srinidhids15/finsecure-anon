"""Train a tiny model on a synthetic fixture and exercise the full workflow.

The fixture is generated here (no dataset files). Quality is not asserted:
40 short documents and ~60 steps only check that conversion, training,
saving, reloading and inference work end to end.
"""

import json
import subprocess
import sys

import pytest

from finguard.config import ENTITY_LABELS
from finguard.data.loader import Example, write_jsonl
from finguard.detection.base import Detector
from finguard.ner import train as ner_train
from finguard.ner.convert import convert_partition
from finguard.ner.inference import ModelNotFoundError, NERDetector
from finguard.ner.tokenizer import make_nlp
from finguard.schema import Span

FIRST = ["Asha", "Ben", "Chen", "Dev", "Eva", "Farah", "Gus", "Hana"]
LAST = ["Rao", "Lee", "Wu", "Iyer", "Klein", "Diaz", "Pal", "Ono"]
CARDS = ["4111111111111111", "5555555555554444", "6011111111111117", "3530111333300000"]
TEMPLATE = ("Dear {name}, your account with {company} is active. Email {email} or call {phone}. "
            "SSN {ssn}. Address: {address}. Card ending in {credit_card} was used. Visit {url} today.")


def make_example(i: int, prefix: str) -> Example:
    name = f"{FIRST[i % 8]} {LAST[(i * 3) % 8]}"
    values = {
        "name": name, "company": f"{LAST[(i + 1) % 8]} Ltd", "email": f"user{i}@example.org",
        "phone": f"+91 98{i:08d}", "ssn": f"{100 + i:03d}-{10 + i % 80:02d}-{1000 + i:04d}",
        "address": f"{i + 10} Lake Road\nPune, MH 4110{i % 100:02d}",
        "credit_card": f"VISA 16 digit\n{name}\n{CARDS[i % 4]} 12/27\nCVC: {100 + i}\n",
        "url": f"https://site{i}.example.com/",
    }
    text, spans, pos = "", [], 0
    for part in TEMPLATE.replace("}", "{").split("{"):
        if part in values:
            spans.append(Span(len(text), len(text) + len(values[part]), part, "gold"))
            text += values[part]
        else:
            text += part
    return Example(f"{prefix}-{i:06d}", text, tuple(spans), group=f"g{i % 4}")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("ner")
    ner_dir, corpus_dir, out = root / "ner", root / "corpus", root / "model"
    write_jsonl([make_example(i, "train") for i in range(40)], ner_dir / "train.jsonl")
    write_jsonl([make_example(i, "train") for i in range(40, 48)], ner_dir / "validation.jsonl")
    (ner_dir / "split_report.json").write_text(json.dumps({
        "settings": {"seed": 42},
        "partitions": {"train": {"units": ["g0", "g1"]}, "validation": {"units": ["g2"]}},
    }))
    nlp = make_nlp()
    report = {"partitions": {p: convert_partition(p, ner_dir, corpus_dir, nlp) for p in ("train", "validation")}}
    (corpus_dir / "conversion_report.json").write_text(json.dumps(report))
    overrides = {"training.max_steps": 60, "training.max_epochs": 0,
                 "training.eval_frequency": 20, "training.patience": 0}
    manifest = ner_train.train(ner_train.DEFAULT_CONFIG, out, corpus_dir, ner_dir, overrides)
    return manifest, out, ner_dir, corpus_dir


def test_training_produces_a_loadable_model_and_manifest(trained):
    manifest, out, _, _ = trained
    assert (out / "model-best" / "config.cfg").exists()
    assert (out / "model-best" / "training_manifest.json").exists()
    assert sorted(manifest["labels"]) == sorted(ENTITY_LABELS)
    assert manifest["tokenizer"]["in_model_config"] == "finguard.Tokenizer.v1"
    assert manifest["corpora"]["train"]["docs"] == 40 and manifest["corpora"]["validation"]["docs"] == 8
    assert [e["step"] for e in manifest["evaluations"]] == [0, 20, 40, 60]
    assert manifest["best_evaluation"]["score"] == max(e["score"] for e in manifest["evaluations"])
    assert "internal_test" not in json.dumps(manifest["corpora"])


def test_model_loads_in_a_fresh_process_with_custom_tokenizer(trained):
    _, out, _, _ = trained
    code = (
        "import json, spacy, sys; nlp = spacy.load(sys.argv[1]); "
        "print(json.dumps([nlp.config['nlp']['tokenizer']['@tokenizers'], "
        "[t.text for t in nlp.make_doc('Visit https://x.com/. Simpson Inc. Meza PhD.')]]))"
    )
    result = subprocess.run([sys.executable, "-c", code, str(out / "model-best")],
                            capture_output=True, text=True, check=True)
    tokenizer, toks = json.loads(result.stdout)
    assert tokenizer == "finguard.Tokenizer.v1"
    assert toks == ["Visit", "https://x.com/", ".", "Simpson", "Inc", ".", "Meza", "PhD", "."]


def test_detector_interface_and_output_contract(trained):
    _, out, _, _ = trained
    det = NERDetector(out / "model-best")
    assert isinstance(det, Detector)
    assert set(det.labels) == set(ENTITY_LABELS)
    text = make_example(3, "x").text
    for e in det.detect(text):
        assert text[e.start : e.end] == e.text
        assert e.label in ENTITY_LABELS and e.source == "ner"
        assert e.text == e.text.rstrip()  # canonical: no trailing whitespace
    assert det.detect("") == [] and det.detect(" \n ") == []
    many = det.detect_many([text, "", "nothing here"])
    assert len(many) == 3 and many[1] == []
    with pytest.raises(TypeError):
        det.detect(None)


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="python -m finguard.ner.train"):
        NERDetector(tmp_path / "nope")


# --------------------------------------------------------------- pre-flight


def test_preflight_rejects_stale_corpus(trained, tmp_path):
    _, _, ner_dir, corpus_dir = trained
    stale = tmp_path / "ner"
    stale.mkdir()
    for f in ner_dir.iterdir():
        (stale / f.name).write_bytes(f.read_bytes())
    with open(stale / "train.jsonl", "a") as fh:
        fh.write(json.dumps({"id": "train-999999", "text": "x", "spans": []}) + "\n")
    with pytest.raises(ner_train.PreflightError, match="changed since conversion"):
        ner_train.check_corpora(corpus_dir, stale)


def test_preflight_rejects_other_corpus_paths(trained, tmp_path):
    _, _, ner_dir, corpus_dir = trained
    with pytest.raises(ner_train.PreflightError, match="paths.dev"):
        ner_train.train(ner_train.DEFAULT_CONFIG, tmp_path / "m", corpus_dir, ner_dir,
                        {"paths.dev": str(corpus_dir / "internal_test.spacy")})


def test_preflight_rejects_default_tokenizer_config(tmp_path):
    cfg = ner_train.DEFAULT_CONFIG.read_text().replace(
        '@tokenizers = "finguard.Tokenizer.v1"', '@tokenizers = "spacy.Tokenizer.v1"')
    (tmp_path / "bad.cfg").write_text(cfg)
    with pytest.raises(ner_train.PreflightError, match="tokenizer"):
        ner_train.check_config(tmp_path / "bad.cfg", {})


def test_tokenizer_is_registered_in_this_environment():
    ner_train.check_tokenizer_registered()
