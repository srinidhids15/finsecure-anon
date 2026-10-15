"""PDF workflow for the user interface (no Streamlit import).

Wraps ``finguard.pdf.pipeline.process_pdf`` with user-safe errors and keeps
per-session PDF results separate from text-scan results. A result is tied to
the SHA-256 digest of the exact uploaded bytes; when the upload changes or
is removed, the result is discarded. The digest and the result live only
in the session's memory.
"""

from __future__ import annotations

import hashlib
from collections.abc import MutableMapping
from dataclasses import dataclass

from finguard.pdf.errors import PdfError
from finguard.pdf.pipeline import PdfResult, process_pdf
from finguard.pdf.validate import DEFAULT_LIMITS, PdfLimits

PDF_KEY = "fg_pdf_record"


@dataclass(frozen=True)
class PdfRecord:
    upload_id: str
    result: PdfResult | None
    error: str | None

    @property
    def verified(self) -> bool:
        return self.error is None and self.result is not None and self.result.verified


def upload_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_pdf(detector, data: bytes, limits: PdfLimits = DEFAULT_LIMITS) -> PdfRecord:
    uid = upload_id(data)
    try:
        return PdfRecord(uid, process_pdf(data, detector, limits), None)
    except PdfError as exc:  # messages name pages and features only
        return PdfRecord(uid, None, str(exc))
    except Exception as exc:  # noqa: BLE001
        return PdfRecord(uid, None, f"PDF processing failed ({type(exc).__name__}). No output was produced.")


def store_pdf(state: MutableMapping, record: PdfRecord) -> None:
    state[PDF_KEY] = record


def clear_pdf(state: MutableMapping) -> None:
    state.pop(PDF_KEY, None)


def current_pdf(state: MutableMapping, current_upload_id: str | None) -> tuple[PdfRecord | None, bool]:
    """``(record, stale)``: a record is returned only for the current upload."""
    record = state.get(PDF_KEY)
    if record is None:
        return None, False
    if record.upload_id != current_upload_id:
        clear_pdf(state)
        return None, True
    return record, False


def preview_png(output: bytes, page_number: int, zoom: float = 1.5) -> bytes:
    """PNG of one page of the verified output (rendered in memory)."""
    import pymupdf

    with pymupdf.open(stream=output, filetype="pdf") as doc:
        return doc[page_number - 1].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False).tobytes("png")
