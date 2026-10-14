"""Extract text from PDFs (PyMuPDF primary, pdfplumber for tables/fallback).

Scanned (image-only) PDFs have no text layer; they are detected and
reported, not silently returned as empty text.
"""

from __future__ import annotations

from pathlib import Path


def extract_text(pdf_path: Path) -> str:
    """Return the text content of every page, in reading order."""
    raise NotImplementedError("Implemented in the PDF processing phase")
