"""Tests for Luhn, SSN and card-prefix checks."""

import pytest

from finguard.detection.checksums import (
    card_prefix_plausible,
    luhn_valid,
    ssn_structurally_valid,
)


def with_check_digit(body: str) -> str:
    """Append the digit that makes ``body`` Luhn-valid (test helper)."""
    for d in "0123456789":
        if luhn_valid(body + d):
            return body + d
    raise AssertionError("unreachable")


# Publicly documented test card numbers (never issued to anyone).
@pytest.mark.parametrize(
    "number",
    ["4111111111111111", "5555555555554444", "378282246310005",
     "6011111111111117", "3530111333300000", "30569309025904", "4222222222222"],
)
def test_luhn_accepts_valid_numbers(number):
    assert luhn_valid(number)


@pytest.mark.parametrize(
    "number",
    ["4111111111111112", "1234567812345678", "", "4111-1111-1111-1111",
     "41111111111111a1", "４１１１"],  # last: fullwidth digits
)
def test_luhn_rejects_invalid_or_non_ascii(number):
    assert not luhn_valid(number)


def test_check_digit_helper():
    assert with_check_digit("411111111111111") == "4111111111111111"


@pytest.mark.parametrize(
    "parts, ok",
    [
        (("575", "74", "7674"), True),
        (("000", "12", "3456"), False),
        (("666", "12", "3456"), False),
        (("900", "12", "3456"), False),
        (("123", "00", "4567"), False),
        (("123", "45", "0000"), False),
    ],
)
def test_ssn_structural_rules(parts, ok):
    assert ssn_structurally_valid(*parts) is ok


@pytest.mark.parametrize(
    "digits, ok",
    [("4111111111111111", True), ("2445443555642900", True), ("213117812841877", True),
     ("378282246310005", True), ("0000000000000000", False), ("9111111111111111", False),
     ("1234567812345670", False)],
)
def test_card_prefix(digits, ok):
    assert card_prefix_plausible(digits) is ok
