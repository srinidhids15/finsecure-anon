"""Functional checks of the trained model (skipped until it exists).

Only training- and validation-partition documents are used here. No scores
are computed on the internal test partition or the official test set.
"""

import json
import subprocess
import sys

import pytest

from finguard.config import ENTITY_LABELS, PROCESSED_DATA_DIR
from finguard.data.loader import read_jsonl
from finguard.detection.base import Detector
from finguard.ner.inference import DEFAULT_MODEL_DIR, NERDetector
from finguard.ner.tokenizer import make_nlp

# spaCy writes model-best repeatedly *during* training; the manifest is
# written only once training has finished, so it marks a complete model.
pytestmark = pytest.mark.skipif(
    not (DEFAULT_MODEL_DIR / "training_manifest.json").exists(),
    reason="no finished training run; run python -m finguard.ner.train",
)

BOUNDARY_CASES = [
    "Visit https://www.murphy.info/. Then",
    "Paid by Simpson Inc. The end",
    "Signed Todd Meza PhD. Next",
    "by Scott Thomas V. The",
    "by Peter Roman Jr.. The",
    "Card ending in VISA 16 digit\nAsha Rao\n4111111111111111 12/27\nCVC: 123\n was used.",
]


@pytest.fixture(scope="module")
def detector():
    return NERDetector()


def test_fresh_process_loads_model_with_custom_tokenizer():
    # No "import finguard" in the child: the entry point must resolve the tokenizer.
    code = (
        "import json, spacy, sys; nlp = spacy.load(sys.argv[1]); "
        "print(json.dumps({'tok': nlp.config['nlp']['tokenizer']['@tokenizers'], "
        "'labels': list(nlp.get_pipe('ner').labels), "
        "'offsets': [[(t.idx, t.text) for t in nlp.make_doc(s)] for s in json.loads(sys.argv[2])]}))"
    )
    out = subprocess.run([sys.executable, "-c", code, str(DEFAULT_MODEL_DIR), json.dumps(BOUNDARY_CASES)],
                         capture_output=True, text=True, check=True)
    result = json.loads(out.stdout)
    assert result["tok"] == "finguard.Tokenizer.v1"
    assert sorted(result["labels"]) == sorted(ENTITY_LABELS)
    reference = make_nlp()
    expected = [[[t.idx, t.text] for t in reference.make_doc(s)] for s in BOUNDARY_CASES]
    assert result["offsets"] == expected


def test_detector_protocol_and_labels(detector):
    assert isinstance(detector, Detector)
    assert set(detector.labels) == set(ENTITY_LABELS)


@pytest.mark.parametrize("partition", ["train", "validation"])
def test_inference_on_prepared_documents(detector, partition):
    examples = read_jsonl(PROCESSED_DATA_DIR / "ner" / f"{partition}.jsonl")[:5]
    for ex in examples:
        entities = detector.detect(ex.text)
        assert entities, ex.record_id
        for e in entities:
            assert 0 <= e.start < e.end <= len(ex.text)
            assert ex.text[e.start : e.end] == e.text
            assert e.label in ENTITY_LABELS and e.source == "ner"
            assert e.text == e.text.rstrip()


def test_empty_and_entity_free_text(detector):
    assert detector.detect("") == []
    assert detector.detect("   \n\t") == []
    result = detector.detect("The quarterly review meeting is scheduled for next week.")
    assert isinstance(result, list)
    assert all(e.text == "The quarterly review meeting is scheduled for next week."[e.start:e.end] for e in result)


def test_manifest_records_provenance_without_test_data():
    manifest = json.loads((DEFAULT_MODEL_DIR / "training_manifest.json").read_text())
    assert manifest["tokenizer"]["in_model_config"] == "finguard.Tokenizer.v1"
    assert set(manifest["corpora"]) == {"train", "validation"}
    assert "internal_test" not in json.dumps(manifest["corpora"])
    assert manifest["partitions"]["train_units"] and manifest["partitions"]["validation_units"]
    assert manifest["evaluations"] and manifest["best_evaluation"]["ents_f"] is not None
    reeval = manifest["model_best_validation_scores"]["ents_f"] * 100
    assert reeval == pytest.approx(manifest["best_evaluation"]["ents_f"], abs=1e-6)
