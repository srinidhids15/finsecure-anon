"""Shared test helpers.

All data here is tiny, hand-written and synthetic. It exercises the code
paths; it is never mixed with, or a substitute for, the real dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from finguard.data.loader import EXPECTED_COLUMNS, SOURCE_COLUMN_FOR_LABEL

VALUES = {
    "name": "Asha Rao",
    "credit_card": "VISA 16 digit\nAsha Rao\n4111111111111111 01/30\nCVC: 123\n",
    "email": "asha@example.com",
    "url": "https://example.com/",
    "phone": "+91 9876543210",
    "address": "12 Lake Road\nPune, MH 411001",
    "company": "Acme Ltd",
    "ssn": "048-59-6922",
}


def build_row(template: str, values: dict[str, str] | None = None) -> dict[str, str]:
    """Fill ``{label}`` placeholders and compute correct gold offsets.

    Returns a spreadsheet row (column name -> cell text).
    """
    values = {**VALUES, **(values or {})}
    text, spans, i = "", [], 0
    while i < len(template):
        if template[i] == "{":
            j = template.index("}", i)
            label = template[i + 1 : j]
            spans.append((len(text), len(text) + len(values[label]), label))
            text += values[label]
            i = j + 1
        else:
            text += template[i]
            i += 1
    row = {col: values[label] for label, col in SOURCE_COLUMN_FOR_LABEL.items()}
    row["Text"] = text
    row["True Predictions"] = repr(spans)
    return row


def write_workbook(path: Path, rows: list[dict[str, str]], sheet: str = "Sheet1") -> Path:
    df = pd.DataFrame(rows, columns=list(EXPECTED_COLUMNS))
    df.to_excel(path, sheet_name=sheet, index=False)
    return path


@pytest.fixture
def make_row():
    return build_row


@pytest.fixture
def make_workbook():
    return write_workbook
