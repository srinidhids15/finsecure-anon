"""Checksum and plausibility checks used to reject look-alike numbers.

These checks only reduce false positives. Passing them proves **nothing**
about whether a number is real:

* Luhn is a typo-detection checksum. Roughly 1 in 10 random digit strings
  pass it, and a Luhn-valid number may never have been issued, may be
  expired or cancelled, or may be a test number. Only the card network or
  issuer can say whether a card exists or is active.
* The SSN check applies the SSA's published structural rules (no area 000,
  666 or 900-999; no group 00; no serial 0000). It cannot tell whether an
  SSN was ever assigned.
"""

from __future__ import annotations


def luhn_valid(digits: str) -> bool:
    """Return True if ``digits`` (ASCII 0-9 only) passes the Luhn checksum."""
    if not digits or not digits.isascii() or not digits.isdigit():
        return False
    total = 0
    # Walk from the rightmost digit; double every second digit.
    for position, char in enumerate(reversed(digits)):
        value = ord(char) - 48
        if position % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def ssn_structurally_valid(area: str, group: str, serial: str) -> bool:
    """Apply the SSA structural rules to an SSN split as AAA-GG-SSSS."""
    if area in {"000", "666"} or area.startswith("9"):
        return False
    return group != "00" and serial != "0000"


# Leading digits (issuer identification number ranges) used by the major
# networks, including the legacy JCB 15-digit prefixes 2131 and 1800.
# The check is deliberately broad: its job is only to reject numbers that no
# card network uses, e.g. ones starting with 0 or 9.
_CARD_PREFIXES: tuple[str, ...] = (
    "4",  # Visa
    "51", "52", "53", "54", "55",  # Mastercard
    *(str(p) for p in range(2221, 2721)),  # Mastercard 2-series
    "34", "37",  # American Express
    "300", "301", "302", "303", "304", "305", "36", "38", "39",  # Diners Club
    "35", "2131", "1800",  # JCB
    "6",  # Discover, Maestro, UnionPay, RuPay
    "50", "56", "57", "58",  # Maestro
)


def card_prefix_plausible(digits: str) -> bool:
    """True if the number starts with a prefix a card network uses."""
    return digits.startswith(_CARD_PREFIXES)
