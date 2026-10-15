"""End-to-end: the frozen hybrid detector feeding the anonymizer.

Uses one hand-written synthetic document; no dataset records, and no
official test data. Skipped until a finished NER model exists.
"""

import pytest

from finguard.anonymization import DEFAULT_PLACEHOLDERS as PH
from finguard.anonymization import detect_and_anonymize, verify_anonymization
from finguard.ner.inference import DEFAULT_MODEL_DIR

pytestmark = pytest.mark.skipif(
    not (DEFAULT_MODEL_DIR / "training_manifest.json").exists(),
    reason="no finished training run; run python -m finguard.ner.train",
)

BLOCK = "VISA 16 digit\nAsha Rao\n4111111111111111 12/27\nCVC: 123\n"
TEXT = (
    "Dear Ben Lee,\nYour statement from Acme Ltd is ready. Contact support@acmebank.com "
    "or call +1 (415) 555-0132. SSN on file: 048-59-6922. Visit https://acmebank.com/help.\n"
    f"Card on file:\n{BLOCK}Mailing address: 12 Lake Road, Springfield, IL 62704\n"
)


@pytest.fixture(scope="module")
def detector():
    from finguard.detection.hybrid import HybridDetector

    return HybridDetector()


def test_hybrid_output_anonymized_and_verified(detector):
    entities = detector.detect(TEXT)
    r = detect_and_anonymize(TEXT, detector)

    assert r.verification.detected_spans_replaced
    assert verify_anonymization(TEXT, entities, r).detected_spans_replaced
    assert r.redaction_count == len(entities)
    assert sum(r.counts_by_label.values()) == len(entities)

    # Structured values are found by the deterministic rules; all must be gone.
    for value in ("support@acmebank.com", "+1 (415) 555-0132", "048-59-6922", "https://acmebank.com/help"):
        assert value not in r.sanitized_text
    # The whole card block is replaced, its trailing newline kept.
    assert f"Card on file:\n{PH['credit_card']}\nMailing address: " in r.sanitized_text
    for part in ("VISA", "Asha Rao", "4111111111111111", "12/27", "CVC"):
        assert part not in r.sanitized_text
    card = [x for x in r.redactions if x.label == "credit_card"]
    assert len(card) == 1 and "rule:credit_card_block" in card[0].sources and card[0].whitespace_kept == 1
    # Text around detections is untouched.
    assert r.sanitized_text.startswith("Dear ") and r.sanitized_text.endswith("\n")


def test_detector_and_model_unchanged_by_anonymization(detector):
    from finguard.evaluation.freeze import verify_freeze

    detect_and_anonymize(TEXT, detector)
    verify_freeze()  # raises FreezeError if any frozen file, the model or the policy changed
