"""Safe parsing of the dataset's annotation strings.

The "True Predictions" column stores annotations as the *text* of a Python
list of tuples, for example::

    "[(5, 15, 'name'), (79, 138, 'credit_card')]"

We parse it with ``ast.literal_eval``, which only accepts Python literals
(numbers, strings, tuples, lists, ...). Unlike ``eval()`` it cannot call
functions or run code, so a malicious cell such as
``"__import__('os').system('rm -rf ~')"`` raises an error instead of executing.

This module checks only the *structure* (a list of items). Whether each item
is a well-formed ``(start, end, label)`` triple, and whether the offsets fit
the text, is checked in ``validation.py`` so that every problem can be
reported per span instead of rejecting the whole record at once.
"""

from __future__ import annotations

import ast

# Real annotation strings are at most a few hundred characters; anything far
# larger is almost certainly corrupt and is refused before parsing.
MAX_ANNOTATION_CHARS = 100_000


class AnnotationParseError(ValueError):
    """The annotation string is not a list literal we can safely read."""


def parse_annotation_string(raw: object) -> list:
    """Parse an annotation cell into a Python list of (unchecked) items.

    Args:
        raw: The cell value from the spreadsheet.

    Returns:
        The list of items exactly as written in the cell.

    Raises:
        AnnotationParseError: if the value is not a string, is empty, is too
            long, is not a valid literal, or is not a list/tuple.
    """
    if not isinstance(raw, str):
        raise AnnotationParseError(f"expected a string, got {type(raw).__name__}")
    stripped = raw.strip()
    if not stripped:
        raise AnnotationParseError("annotation string is empty")
    if len(stripped) > MAX_ANNOTATION_CHARS:
        raise AnnotationParseError(f"annotation string longer than {MAX_ANNOTATION_CHARS} chars")
    try:
        value = ast.literal_eval(stripped)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as exc:
        raise AnnotationParseError(f"not a valid literal: {type(exc).__name__}") from exc
    if not isinstance(value, (list, tuple)):
        raise AnnotationParseError(f"expected a list of spans, got {type(value).__name__}")
    return list(value)
