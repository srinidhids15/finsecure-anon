"""In-memory downloads built from a verified ``AnonymizationResult``.

Nothing is written to disk. File names are fixed and never derived from
the document. The CSV and JSON reports hold labels, offsets, placeholders
and detector sources; they contain no original entity values and no
document text. Only the TXT export holds text: the verified sanitized output.
"""

from __future__ import annotations

import csv
import io
import json

import finguard
from finguard.anonymization import AnonymizationResult

TXT_FILENAME = "finguard_sanitized.txt"
CSV_FILENAME = "finguard_entity_report.csv"
JSON_FILENAME = "finguard_redaction_summary.json"

CSV_COLUMNS = (
    "index",
    "label",
    "start",
    "end",
    "replaced_end",
    "whitespace_kept",
    "output_start",
    "output_end",
    "placeholder",
    "sources",
)


def _require_verified(result: AnonymizationResult) -> None:
    if result.verification is None or not result.verification.detected_spans_replaced:
        raise ValueError("exports are only built from a verified anonymization result")


def sanitized_txt(result: AnonymizationResult) -> bytes:
    _require_verified(result)
    return result.sanitized_text.encode("utf-8")


def entity_report_csv(result: AnonymizationResult) -> bytes:
    """One row per redaction.

    ``start``/``end`` are the detector's offsets in the original text;
    ``[start, replaced_end)`` was replaced (canonical span) and
    ``whitespace_kept`` trailing characters were left in place.
    ``output_start``/``output_end`` locate the placeholder in the TXT export.
    """
    _require_verified(result)
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for i, r in enumerate(result.redactions, 1):
        writer.writerow([i, r.label, r.start, r.detected_end, r.end, r.whitespace_kept,
                         r.output_start, r.output_end, r.placeholder, "|".join(r.sources)])
    return buf.getvalue().encode("utf-8")


def redaction_summary_json(result: AnonymizationResult, detector_info: dict | None = None) -> bytes:
    """``result.to_dict()`` (no text) plus tool and detector metadata."""
    _require_verified(result)
    summary = {
        "tool": "FinGuard",
        "version": finguard.__version__,
        "scope": "plain-text anonymization of detected spans",
        "detector": detector_info or {},
        **result.to_dict(include_sanitized_text=False),
    }
    return (json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
