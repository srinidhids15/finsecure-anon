"""Read the Mendeley workbooks and the processed JSONL files.

Raw files are opened read-only and never modified. Every row becomes a
``RawRecord`` holding the cell values *as text*, before any parsing, so that
the validation step can report exactly what was wrong with each record.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from finguard.schema import Span

TEXT_COLUMN = "Text"
ANNOTATION_COLUMN = "True Predictions"

# Each entity label has a column holding the exact value inserted into the
# text. Validation uses it to check that every span extracts that value.
SOURCE_COLUMN_FOR_LABEL: dict[str, str] = {
    "name": "Name",
    "credit_card": "Credit Card",
    "email": "Email",
    "url": "URL",
    "phone": "Phone",
    "address": "Address",
    "company": "Company",
    "ssn": "SSN",
}

EXPECTED_COLUMNS: tuple[str, ...] = (
    *SOURCE_COLUMN_FOR_LABEL.values(),
    TEXT_COLUMN,
    ANNOTATION_COLUMN,
)


class DatasetFormatError(ValueError):
    """The file does not have the structure the loader expects."""


@dataclass(frozen=True)
class RawRecord:
    """One spreadsheet row, unparsed.

    ``record_id`` is ``"<split>-<excel row number>"``, for example
    ``"train-000002"`` for the first data row. It lets anyone open the
    original file and find the exact row a report talks about.
    """

    record_id: str
    split: str
    text: str
    annotation: str
    source_values: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Example:
    """A validated, processed example: text plus gold spans."""

    record_id: str
    text: str
    spans: tuple[Span, ...]
    group: str = ""  # template group (train/validation only), see split.py


def file_sha256(path: Path) -> str:
    """Checksum used to prove the raw files were not changed."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_to_str(value: object) -> str:
    """Normalise an empty or missing cell to ``""``; keep strings as they are."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return value if isinstance(value, str) else str(value)


def load_workbook_records(path: Path, split: str) -> list[RawRecord]:
    """Load every row of a single-sheet dataset workbook.

    Args:
        path: The ``.xlsx`` file.
        split: Short split name used in record ids (``"train"`` or ``"test"``).

    Raises:
        FileNotFoundError: if the file does not exist.
        DatasetFormatError: if there is more than one sheet or a column is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with pd.ExcelFile(path, engine="openpyxl") as book:
        if len(book.sheet_names) != 1:
            raise DatasetFormatError(
                f"{path.name}: expected 1 sheet, found {book.sheet_names}"
            )
        # dtype=str + keep_default_na=False: never let pandas turn values
        # into numbers or NaN (e.g. an SSN like "048-59-6922" stays a string).
        df = pd.read_excel(book, sheet_name=0, dtype=str, keep_default_na=False)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetFormatError(f"{path.name}: missing columns {missing}")

    records = []
    for position, row in enumerate(df.itertuples(index=False)):
        values = dict(zip(df.columns, row))
        excel_row = position + 2  # +1 for 1-based rows, +1 for the header row
        records.append(
            RawRecord(
                record_id=f"{split}-{excel_row:06d}",
                split=split,
                text=_cell_to_str(values[TEXT_COLUMN]),
                annotation=_cell_to_str(values[ANNOTATION_COLUMN]),
                source_values={
                    label: _cell_to_str(values[col])
                    for label, col in SOURCE_COLUMN_FOR_LABEL.items()
                },
            )
        )
    return records


def write_jsonl(examples: list[Example], path: Path) -> None:
    """Write processed examples, one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            obj = {
                "id": ex.record_id,
                "text": ex.text,
                "spans": [[s.start, s.end, s.label] for s in ex.spans],
            }
            if ex.group:
                obj["group"] = ex.group
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Example]:
    """Read processed examples written by ``write_jsonl``.

    Spans are rebuilt as ``Span`` objects and checked against the text again,
    so a hand-edited or corrupted file fails here instead of during training.
    """
    examples = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            try:
                obj = json.loads(line)
                spans = tuple(
                    Span(s, e, label, source="gold") for s, e, label in obj["spans"]
                )
                for span in spans:
                    span.text_in(obj["text"])  # raises if out of bounds
            except (ValueError, TypeError, KeyError) as exc:
                raise DatasetFormatError(f"{path}:{line_no}: {exc}") from exc
            examples.append(
                Example(obj["id"], obj["text"], spans, obj.get("group", ""))
            )
    return examples
