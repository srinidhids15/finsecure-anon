"""Scan → anonymize workflow for the user interface (no Streamlit import).

The Streamlit app calls these functions; they hold no detection or
redaction logic of their own:

* scanning calls the frozen ``HybridDetector`` (``detection.hybrid``);
* anonymization calls ``finguard.anonymization.anonymize``, whose result is
  accepted only if its integrity verification passed.

Errors reach the user as ``UserFacingError`` with a message that never
contains document text: input errors describe sizes, detector failures
report only the exception type, and anonymizer errors are already limited
to labels and offsets by design.

Per-session results are kept in a caller-supplied mapping
(``st.session_state`` in the app). Each scan records the exact text it
was run on and a fresh ``scan_id``; an anonymization result records the
``scan_id`` it belongs to. Results whose text no longer matches the
current input are discarded.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass

from finguard.anonymization import AnonymizationError, AnonymizationResult, anonymize
from finguard.config import ENTITY_LABELS
from finguard.schema import Entity

# Measured on synthetic text (CPU): about 0.85 s to detect 100,000 characters.
MAX_INPUT_CHARS = 100_000

SCAN_KEY = "fg_scan_record"
ANON_KEY = "fg_anon_record"


class UserFacingError(Exception):
    """An error whose message is safe to show (contains no document text)."""


class InputError(UserFacingError):
    pass


class DetectorUnavailableError(UserFacingError):
    pass


class ScanError(UserFacingError):
    pass


class AnonymizeError(UserFacingError):
    pass


# --- detector -------------------------------------------------------------

class SharedDetector:
    """A detector shared by all sessions, serialised by a lock.

    It holds the loaded model only. Text and results are never stored on it.
    """

    def __init__(self, detector) -> None:
        self._detector = detector
        self._lock = threading.Lock()
        self.name = getattr(detector, "name", "detector")

    def detect_with_count(self, text: str) -> tuple[list[Entity], int]:
        """Entities plus the number of candidates the merge policy suppressed."""
        with self._lock:
            if hasattr(self._detector, "detect_with_report"):
                report = self._detector.detect_with_report(text)
                return list(report.entities), len(report.suppressed) + len(report.rule_suppressed)
            return list(self._detector.detect(text)), 0


def build_detector() -> SharedDetector:
    """Load the frozen hybrid detector (rules + trained NER model)."""
    from finguard.detection.hybrid import HybridDetector
    from finguard.ner.inference import ModelNotFoundError

    try:
        return SharedDetector(HybridDetector())
    except ModelNotFoundError as exc:
        raise DetectorUnavailableError(
            "The trained NER model was not found at models/finguard_ner/model-best. "
            "Train it with `python -m finguard.ner.train` (see docs/NER.md), then restart the app."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - reported without details that could carry data
        raise DetectorUnavailableError(
            f"The detector could not be initialised ({type(exc).__name__}). "
            "Check the installation (`pip install -r requirements.txt`) and the model files, then restart."
        ) from exc


def detector_info() -> dict:
    """Hybrid policy and evaluation-freeze status (non-sensitive metadata)."""
    from finguard.detection.hybrid import DEFAULT_POLICY
    from finguard.evaluation.freeze import FreezeError, verify_freeze

    info = {"name": "hybrid", "policy": DEFAULT_POLICY.describe()}
    try:
        record = verify_freeze()
        info.update(freeze_verified=True, freeze_sha256=record["freeze_sha256"])
    except FreezeError as exc:
        info.update(freeze_verified=False, freeze_sha256=None, freeze_problem=str(exc))
    except Exception as exc:  # noqa: BLE001
        info.update(freeze_verified=False, freeze_sha256=None, freeze_problem=type(exc).__name__)
    return info


# --- input, scan, anonymize ---------------------------------------------

def validate_input(text) -> str:
    if not isinstance(text, str):
        raise InputError("Input must be text.")
    if not text.strip():
        raise InputError("Enter or paste some text to scan. The input is empty or contains only whitespace.")
    if len(text) > MAX_INPUT_CHARS:
        raise InputError(f"The input has {len(text):,} characters; the limit is {MAX_INPUT_CHARS:,}. "
                         "Split the document into smaller parts.")
    return text


@dataclass(frozen=True)
class ScanRecord:
    scan_id: str
    text: str  # held in this session's memory only, to tie results to their input
    entities: tuple[Entity, ...]
    suppressed: int
    seconds: float


@dataclass(frozen=True)
class AnonRecord:
    scan_id: str
    result: AnonymizationResult | None
    error: str | None

    @property
    def verified(self) -> bool:
        return (self.error is None and self.result is not None and self.result.verification is not None
                and self.result.verification.detected_spans_replaced)


def scan(detector: SharedDetector, text: str) -> ScanRecord:
    validate_input(text)
    start = time.perf_counter()
    try:
        entities, suppressed = detector.detect_with_count(text)
    except Exception as exc:  # noqa: BLE001
        raise ScanError(f"Detection failed ({type(exc).__name__}). Nothing was stored; "
                        "try again or reduce the input size.") from exc
    return ScanRecord(uuid.uuid4().hex, text, tuple(entities), suppressed, time.perf_counter() - start)


def anonymize_scan(record: ScanRecord) -> AnonRecord:
    """Anonymize a scan's entities; never returns an unverified result."""
    try:
        result = anonymize(record.text, record.entities)
    except AnonymizationError as exc:  # messages hold labels and offsets only
        return AnonRecord(record.scan_id, None, f"Anonymization was refused: {exc}")
    except Exception as exc:  # noqa: BLE001
        return AnonRecord(record.scan_id, None, f"Anonymization failed ({type(exc).__name__}).")
    if result.verification is None or not result.verification.detected_spans_replaced:
        return AnonRecord(record.scan_id, None, "Anonymization output failed verification and was discarded.")
    return AnonRecord(record.scan_id, result, None)


# --- summaries --------------------------------------------------------------

def counts_by_label(entities: Sequence[Entity]) -> dict[str, int]:
    counts = dict.fromkeys(ENTITY_LABELS, 0)
    for e in entities:
        counts[e.label] += 1
    return counts


def entity_rows(entities: Sequence[Entity]) -> list[dict]:
    """Table rows without entity text: label, offsets, length, sources."""
    return [
        {"#": i, "label": e.label, "start": e.start, "end": e.end, "length": e.end - e.start,
         "sources": ", ".join(s for s in e.source.split("|") if s)}
        for i, e in enumerate(sorted(entities, key=lambda e: (e.start, e.end)), 1)
    ]


def placeholder_rows(result: AnonymizationResult) -> list[dict]:
    seen: dict[str, dict] = {}
    for r in result.redactions:
        row = seen.setdefault(r.label, {"label": r.label, "placeholder": r.placeholder, "count": 0})
        row["count"] += 1
    return [seen[label] for label in ENTITY_LABELS if label in seen]


# --- session state ----------------------------------------------------------

def store_scan(state: MutableMapping, record: ScanRecord) -> None:
    state[SCAN_KEY] = record
    state.pop(ANON_KEY, None)


def store_anonymization(state: MutableMapping, record: AnonRecord) -> None:
    state[ANON_KEY] = record


def clear_results(state: MutableMapping) -> None:
    state.pop(SCAN_KEY, None)
    state.pop(ANON_KEY, None)


def current_results(state: MutableMapping, current_text: str) -> tuple[ScanRecord | None, AnonRecord | None, bool]:
    """``(scan, anonymization, stale)`` for the current input.

    If the input differs from the scanned text, both results are discarded
    and ``stale`` is True. An anonymization result belonging to another scan
    is never returned.
    """
    record = state.get(SCAN_KEY)
    if record is None:
        return None, None, False
    if record.text != current_text:
        clear_results(state)
        return None, None, True
    anon = state.get(ANON_KEY)
    if anon is not None and anon.scan_id != record.scan_id:
        state.pop(ANON_KEY, None)
        anon = None
    return record, anon, False
