"""Tests for the mojibake text repair."""

import pytest

from finguard.data.repair import MOJIBAKE_MAP, repair_mojibake
from finguard.data.validation import validate_record


@pytest.mark.parametrize("char", ["’", "‘", "“", "”"])
def test_each_curly_quote_round_trips(char):
    # Reproduce the damage the way it happened: UTF-8 bytes read as cp1252.
    # (0x9D has no cp1252 character, so it survives as U+009D.)
    broken = char.encode("utf-8").decode("cp1252", errors="ignore")
    broken = broken if len(broken) == 3 else broken + "\u009d"
    assert broken in MOJIBAKE_MAP
    assert repair_mojibake(f"x{broken}y") == (f"x{char}y", 1)


def test_clean_text_is_unchanged():
    text = "Plain ASCII text, with 'straight' quotes."
    assert repair_mojibake(text) == (text, 0)


def test_counts_every_replacement():
    text = "Auditorâ€™s â€œreportâ€\u009d"
    repaired, n = repair_mojibake(text)
    assert repaired == "Auditor’s “report”"
    assert n == 3


def test_repair_realigns_offsets_computed_on_the_original_text():
    original = "The Auditor’s report for Acme Ltd"
    start = original.index("Acme Ltd")
    annotation = repr([(start, start + 8, "company")])
    corrupted = original.replace("’", "â€™")  # 2 chars longer

    before = validate_record("r", corrupted, annotation, {"company": "Acme Ltd"})
    assert not before.is_valid

    repaired, _ = repair_mojibake(corrupted)
    after = validate_record("r", repaired, annotation, {"company": "Acme Ltd"})
    assert after.is_valid
    assert repaired == original
