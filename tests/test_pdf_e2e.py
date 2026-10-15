"""End-to-end: synthetic PDF → frozen hybrid detector → redaction → verification.

Skipped until a finished NER model exists. Assertions on detections cover
values the deterministic rule detector finds (email, phone, SSN, URL, card
block); NER results are only required to be redacted where detected.
"""

import pytest
from pdf_fixtures import build_pdf, encoded_forms, page_text, stream_bytes

from finguard.ner.inference import DEFAULT_MODEL_DIR
from finguard.pdf.pipeline import process_pdf

pytestmark = pytest.mark.skipif(
    not (DEFAULT_MODEL_DIR / "training_manifest.json").exists(),
    reason="no finished training run; run python -m finguard.ner.train",
)

PAGE1 = [
    "Dear Ben Lee,",
    "Your statement from Acme Ltd is ready. Contact support@acmebank.com",
    "or call +1 (415) 555-0132. SSN on file: 048-59-6922.",
    "Card on file:",
    "VISA 16 digit",
    "Asha Rao",
    "4111111111111111 12/27",
    "CVC: 123",
    "Statement date 31/03/2024",
]
PAGE2 = ["Help: https://acmebank.com/help", "Reference number 7781-22"]
RULE_VALUES = ("support@acmebank.com", "+1 (415) 555-0132", "048-59-6922", "4111111111111111", "CVC: 123",
               "https://acmebank.com/help")


@pytest.fixture(scope="module")
def detector():
    from finguard.interface.workflow import SharedDetector
    from finguard.detection.hybrid import HybridDetector

    return SharedDetector(HybridDetector())


def test_real_detector_pdf_redaction_verified(detector):
    data = build_pdf([PAGE1, PAGE2])
    result = process_pdf(data, detector)
    assert result.status == "verified" and result.verified, result.problems

    labels = {(e.page, e.label) for e in result.entities}
    assert {(1, "email"), (1, "phone"), (1, "ssn"), (1, "credit_card"), (2, "url")} <= labels
    card = [e for e in result.entities if e.label == "credit_card"]
    assert len(card) == 1 and "rule:credit_card_block" in card[0].sources and len(card[0].rects) == 4

    out1, out2 = page_text(result.output, 1), page_text(result.output, 2)
    streams = stream_bytes(result.output)
    for value in RULE_VALUES:
        assert value not in out1 + out2
        assert not any(f in streams for f in encoded_forms(value))
    assert "Card on file:" in out1 and "Statement date 31/03/2024" in out1
    # The NER model labels "7781-22" as an SSN (a false positive, kept as is: the
    # detector is frozen). Only the surrounding text is required to survive.
    assert "Reference number" in out2

    v = result.verification
    assert v.redaction_completed and v.targeted_text_removed and v.unrelated_text_preserved
    assert v.complete_pii_removal_guaranteed is False


def test_freeze_still_valid_after_pdf_processing(detector):
    from finguard.evaluation.freeze import FROZEN_FILES, verify_freeze

    process_pdf(build_pdf([PAGE1]), detector)
    verify_freeze()
    assert not any(f.startswith("src/finguard/pdf") for f in FROZEN_FILES)
