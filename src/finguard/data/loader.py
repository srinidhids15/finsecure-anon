"""Load annotated examples from disk into ``(text, list[Span])`` pairs.

Planned for the data phase. Annotation strings are parsed with ``json`` /
``ast.literal_eval`` only -- never ``eval()``.
"""

from __future__ import annotations

from pathlib import Path

from finguard.schema import Span


def load_examples(path: Path) -> list[tuple[str, list[Span]]]:
    """Read a dataset file and return ``(text, gold_spans)`` pairs."""
    raise NotImplementedError("Implemented in the data-loading phase")
