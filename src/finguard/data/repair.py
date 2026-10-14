"""Undo the text-encoding damage found in the Mendeley dataset.

What happened (verified in Phase 2): some documents contained curly quotes
(’ ‘ “ ”). Their UTF-8 bytes were later decoded as Windows-1252, so each
1-character quote became 3 characters ("’" -> "â€™"). This is called mojibake.

The annotation offsets were computed on the *original* text. Every mojibake
sequence before an entity therefore pushes the entity 2 characters to the
right of where its annotation says it is. In the raw files this misaligns
31,763 training spans and 21 test spans.

The repair maps each 3-character sequence back to its single character. It
looks only at the text, never at the labels, so the same function is
applied unchanged to training and test data.
"""

from __future__ import annotations

# Exactly the four sequences present in the dataset. Each is the
# Windows-1252 reading of the UTF-8 bytes of one curly quote.
# (0x9D is undefined in Windows-1252 and survives as the control char U+009D.)
MOJIBAKE_MAP: dict[str, str] = {
    "â€™": "’",  # â€™ -> ’  right single quote / apostrophe
    "â€˜": "‘",  # â€˜ -> ‘  left single quote
    "â€œ": "“",  # â€œ -> “  left double quote
    "â€\u009d": "”",  # â€\x9d -> ”  right double quote
}


def repair_mojibake(text: str) -> tuple[str, int]:
    """Replace known mojibake sequences with the characters they stand for.

    Returns:
        ``(repaired_text, number_of_replacements)``. Text without mojibake is
        returned unchanged with a count of 0.
    """
    count = 0
    for broken, fixed in MOJIBAKE_MAP.items():
        n = text.count(broken)
        if n:
            text = text.replace(broken, fixed)
            count += n
    return text, count
