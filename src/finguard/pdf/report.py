"""Non-sensitive PDF reports: per-entity CSV and a JSON summary.

Both are built in memory from a ``PdfResult`` and contain page numbers,
labels, page-local offsets, region rectangles, detector sources, statuses,
counts and verification results. They never contain extracted text or
entity values.
"""

from __future__ import annotations

import csv
import io
import json

import finguard
from finguard.config import ENTITY_LABELS
from finguard.pdf.pipeline import PdfResult

PDF_FILENAME = "finguard_redacted.pdf"
PDF_CSV_FILENAME = "finguard_pdf_entity_report.csv"
PDF_JSON_FILENAME = "finguard_pdf_summary.json"

PDF_CSV_COLUMNS = ("index", "page", "label", "start", "end", "detected_end", "regions", "sources", "status", "reason")


def counts_by_label(result: PdfResult) -> dict[str, int]:
    counts = dict.fromkeys(ENTITY_LABELS, 0)
    for e in result.entities:
        counts[e.label] += 1
    return counts


def entity_rows(result: PdfResult) -> list[dict]:
    return [
        {"index": i, "page": e.page, "label": e.label, "start": e.start, "end": e.end,
         "detected_end": e.detected_end, "regions": len(e.rects), "sources": "|".join(e.sources),
         "status": e.status, "reason": e.reason}
        for i, e in enumerate(result.entities, 1)
    ]


def entity_report_csv(result: PdfResult) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=PDF_CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(entity_rows(result))
    return buf.getvalue().encode("utf-8")


def summary(result: PdfResult) -> dict:
    return {
        "tool": "FinGuard",
        "version": finguard.__version__,
        "scope": "text-based PDF: detected spans physically redacted; hidden content removed",
        "status": result.status,
        "pages": [p.diagnostics() for p in result.pages],
        "entity_count": len(result.entities),
        "counts_by_label": counts_by_label(result),
        "regions_applied": result.regions_applied,
        "entities": [e.to_dict() for e in result.entities],
        "hidden_content_found": result.hidden.to_dict(),
        "sanitization": result.sanitization.to_dict() if result.sanitization else None,
        "verification": result.verification.to_dict() if result.verification else None,
        "problems": list(result.problems),
        "complete_pii_removal_guaranteed": False,
    }


def summary_json(result: PdfResult) -> bytes:
    return (json.dumps(summary(result), indent=2) + "\n").encode("utf-8")
