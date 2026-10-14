"""Rule-based detection of structured PII: email, url, phone, ssn, credit_card.

Names, companies and addresses are deliberately *not* covered. They have no
fixed format, and regexes for them would either miss most cases or match
ordinary words. The spaCy model handles them (Phase 5).

Every rule reads only the text it is given: no gold labels, no
dataset-specific values. Patterns use ``[0-9]`` and ``[A-Za-z]`` rather than
``\\d`` and ``\\w``, because in Python those also match non-ASCII digits and
letters (e.g. fullwidth "１２３"), which would create false positives.

Supported formats are documented on each pattern and in docs/RULES.md.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

from finguard.config import ENTITY_LABELS
from finguard.detection.base import sort_entities
from finguard.detection.checksums import (
    card_prefix_plausible,
    luhn_valid,
    ssn_structurally_valid,
)
from finguard.schema import Entity

# --------------------------------------------------------------------- rules


@dataclass(frozen=True)
class RegexRule:
    """One pattern for one label, plus optional checks.

    Attributes:
        name: Unique id; entities get ``source="rule:<name>"``.
        label: Entity label the rule produces.
        pattern: Compiled regex. Matches are found left to right and never
            overlap each other (``re.finditer`` semantics).
        accept: Extra check on a match (checksum, value ranges, ...).
        trim: Adjusts the match end (e.g. drop sentence punctuation after a URL).
        yield_to: Names of rules whose matches already cover this entity at
            the right granularity. A candidate lying entirely inside such a
            match is not emitted; it is recorded in
            ``DetectionReport.suppressed`` instead, so it is never lost silently.
    """

    name: str
    label: str
    pattern: re.Pattern[str]
    accept: Callable[[re.Match[str]], bool] = lambda m: True
    trim: Callable[[str, int, int], int] | None = None
    yield_to: tuple[str, ...] = ()

    def find(self, text: str) -> Iterator[tuple[int, int]]:
        for match in self.pattern.finditer(text):
            if not self.accept(match):
                continue
            start, end = match.span()
            if self.trim is not None:
                end = self.trim(text, start, end)
            if end > start:
                yield start, end


# ---- email ------------------------------------------------------------------
# local-part@domain.tld. ASCII only: internationalised addresses (e.g. with
# accented letters) are not supported. Dots may not start, end or repeat in
# the local part.
EMAIL_PATTERN = re.compile(
    r"""
    (?<![A-Za-z0-9._%+-])                          # not mid-token
    [A-Za-z0-9](?:[A-Za-z0-9._%+-]{0,62}[A-Za-z0-9])?   # local part
    @
    (?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+  # domain labels
    [A-Za-z]{2,24}                                 # top-level domain
    (?![A-Za-z0-9-]|\.[A-Za-z0-9])                 # domain really ended
    """,
    re.VERBOSE,
)


def _email_ok(match: re.Match[str]) -> bool:
    return ".." not in match.group(0).split("@", 1)[0]


# ---- url ----------------------------------------------------------------------
# http:// or https:// URLs, and bare hosts starting with "www.". A bare
# domain such as "example.com" is NOT matched: it is indistinguishable from
# the domain part of an email or from ordinary abbreviations.
URL_PATTERN = re.compile(
    r"""
    (?<![A-Za-z0-9@/._-])                          # not inside an email/path
    (?:https?://|www\.)
    (?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,24}   # host
    (?::[0-9]{1,5})?                               # optional port
    (?![A-Za-z0-9-])
    (?:[/?#][^\s<>"\u2018\u2019\u201c\u201d]*)?    # optional path/query/fragment
    """,
    re.VERBOSE | re.IGNORECASE,
)

_URL_TRAILING = ".,;:!?'\"’”)]}>"
_PAIRS = {")": "(", "]": "[", "}": "{"}


def _trim_url(text: str, start: int, end: int) -> int:
    """Drop trailing sentence punctuation, but keep balanced brackets.

    "see https://x.com/." -> "https://x.com/"
    "https://en.wikipedia.org/wiki/A_(b)" keeps its closing parenthesis.
    """
    url = text[start:end]
    # Count brackets once and update while trimming (linear time even for
    # adversarial input such as thousands of trailing ")").
    open_count = {c: url.count(c) for c in _PAIRS.values()}
    close_count = {c: url.count(c) for c in _PAIRS}
    while end > start and text[end - 1] in _URL_TRAILING:
        char = text[end - 1]
        if char in _PAIRS:
            if open_count[_PAIRS[char]] >= close_count[char]:
                break  # this bracket closes one opened inside the URL
            close_count[char] -= 1
        end -= 1
    return end


# ---- phone --------------------------------------------------------------------
# Supported formats:
#   A. International: "+" country code (1-3 digits), optional space/hyphen,
#      then either 6-12 contiguous digits ("+91 1472004612") or 2-5 groups
#      separated by single spaces, hyphens or dots, the first optionally in
#      parentheses ("+1 (555) 234-5678", "+44 20 7946 0958", "+49-30-901820").
#      Total digit count must be 8-15 (the E.164 maximum is 15).
#      The country code may start with 0 ("+04 7024846271"). Real E.164
#      codes never do, but the dataset's generator produces them and labels
#      them as phone (2,500 of 39,243 phone spans in the training partition),
#      so the rule follows the dataset's entity definition.
#   B. North American without "+": (NXX) NXX-XXXX, NXX-NXX-XXXX, NXX.NXX.XXXX
#      where N is 2-9. Separators are required.
# Bare digit runs (e.g. "9876543210") are NOT matched: without a "+" or
# phone-style punctuation they cannot be told apart from account, invoice
# or reference numbers.
PHONE_PATTERN = re.compile(
    r"""
    (?<![A-Za-z0-9+_.-])
    (?:
        \+[0-9]{1,3}[ -]?
        (?:
            [0-9]{6,12}
          | (?:\([0-9]{1,5}\)|[0-9]{1,5})(?:[ .-][0-9]{2,8}){1,4}
        )
      | (?:\([2-9][0-9]{2}\)[ ]?|[2-9][0-9]{2}[-.])[2-9][0-9]{2}[-.][0-9]{4}
    )
    (?![0-9]|[.-][0-9])
    """,
    re.VERBOSE,
)


def _phone_ok(match: re.Match[str]) -> bool:
    digits = sum(c.isdigit() for c in match.group(0))
    return 8 <= digits <= 15 if match.group(0).startswith("+") else digits == 10


# ---- ssn ----------------------------------------------------------------------
# US SSN in the hyphenated AAA-GG-SSSS form (the only form in the dataset).
# Unhyphenated 9-digit numbers are NOT matched: too many other identifiers
# have 9 digits.
SSN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9-])([0-9]{3})-([0-9]{2})-([0-9]{4})(?![A-Za-z0-9]|-[0-9])"
)


def _ssn_ok(match: re.Match[str]) -> bool:
    return ssn_structurally_valid(*match.groups())


# ---- credit card: information block (the dataset's entity definition) ---------
# In the dataset a credit_card entity is a whole block, verified on all
# 36,897 credit_card spans of the training partition:
#
#     <network>\n<holder name>\n<number> <MM/YY>\n<CVC|CVV|CID>: <code>\n
#
# The gold span starts at the network line and ends *after* the final
# newline. This rule reproduces exactly that unit, including the trailing
# newline when present (a block at the very end of a text has none).
# Trimming that whitespace is a span-normalisation step that will be
# applied to gold and predicted spans alike in the conversion/evaluation
# layer, not here.
_NETWORKS = (
    "Visa", "Mastercard", "American Express", "Amex", "Discover",
    "Diners Club / Carte Blanche", "Diners Club International", "Diners Club",
    "JCB", "Maestro", "UnionPay", "RuPay",
)
_NETWORK_RE = "(?:" + "|".join(re.escape(n) for n in _NETWORKS) + ")(?:\\ (?P<declared>[0-9]{2})\\ digit)?"

CARD_BLOCK_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    rf"(?P<network>{_NETWORK_RE})\r?\n"
    r"(?P<holder>[^\r\n]{1,100})\r?\n"
    r"(?P<number>[0-9]{12,19})[ ](?:0[1-9]|1[0-2])/[0-9]{2}\r?\n"
    r"(?:CVC|CVV2?|CID|CSC):[ ][0-9]{3,4}(?![0-9])"
    r"(?:\r?\n)?",
    re.IGNORECASE,
)


def _card_block_ok(match: re.Match[str]) -> bool:
    number = match.group("number")
    declared = match.group("declared")
    if declared is not None and int(declared) != len(number):
        return False  # "VISA 16 digit" must be followed by 16 digits
    return luhn_valid(number)


# ---- credit card: standalone number (outside any block) ---------------------
# For documents that print just a card number. 13-19 digits, contiguous or in
# 4-4-4-4(-) / Amex 4-6-5 groups, Luhn-valid, with a known network prefix.
# It yields to the block rule: a number inside a detected block is part of
# that larger entity, so it is reported as suppressed and not emitted.
CARD_NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9-])"
    r"(?:[0-9]{4}(?:[ -][0-9]{4}){2,3}|[0-9]{4}[ -][0-9]{6}[ -][0-9]{5}|[0-9]{13,19})"
    r"(?![A-Za-z0-9]|-[0-9])"
)


def _card_number_ok(match: re.Match[str]) -> bool:
    digits = re.sub(r"[ -]", "", match.group(0))
    return 13 <= len(digits) <= 19 and card_prefix_plausible(digits) and luhn_valid(digits)


DEFAULT_RULES: tuple[RegexRule, ...] = (
    RegexRule("email", "email", EMAIL_PATTERN, _email_ok),
    RegexRule("url", "url", URL_PATTERN, trim=_trim_url),
    RegexRule("phone", "phone", PHONE_PATTERN, _phone_ok),
    RegexRule("ssn", "ssn", SSN_PATTERN, _ssn_ok, yield_to=("phone",)),
    RegexRule("credit_card_block", "credit_card", CARD_BLOCK_PATTERN, _card_block_ok),
    RegexRule(
        "credit_card_number", "credit_card", CARD_NUMBER_PATTERN, _card_number_ok,
        yield_to=("credit_card_block",),
    ),
)

# Labels the rule detector does not attempt (see module docstring).
LABELS_NOT_COVERED: tuple[str, ...] = ("name", "company", "address")


# ------------------------------------------------------------------ detector


@dataclass(frozen=True)
class DetectionReport:
    """Full result of one detection run.

    ``entities`` is what ``detect`` returns. ``suppressed`` lists every
    candidate a ``yield_to`` rule held back, paired with the entity that
    covers it, so nothing disappears without a trace.
    """

    entities: list[Entity]
    suppressed: list[tuple[Entity, Entity]] = field(default_factory=list)


class RuleDetector:
    """Runs a set of ``RegexRule`` objects; implements ``base.Detector``."""

    name = "rules"

    def __init__(self, rules: Sequence[RegexRule] = DEFAULT_RULES) -> None:
        names = [r.name for r in rules]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate rule names: {names}")
        for rule in rules:
            if rule.label not in ENTITY_LABELS:
                raise ValueError(f"rule {rule.name!r} has unknown label {rule.label!r}")
            missing = set(rule.yield_to) - set(names)
            if missing:
                raise ValueError(f"rule {rule.name!r} yields to unknown rules {missing}")
        self.rules = tuple(rules)

    def detect(self, text: str) -> list[Entity]:
        """Return all rule matches, sorted, overlaps included."""
        return self.detect_with_report(text).entities

    def detect_with_report(self, text: str) -> DetectionReport:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")

        found: dict[str, list[Entity]] = {
            rule.name: [
                Entity.from_text(text, s, e, rule.label, f"rule:{rule.name}")
                for s, e in rule.find(text)
            ]
            for rule in self.rules
        }

        entities: list[Entity] = []
        suppressed: list[tuple[Entity, Entity]] = []
        for rule in self.rules:
            covers = [c for name in rule.yield_to for c in found[name]]
            for ent in found[rule.name]:
                cover = next(
                    (c for c in covers if c.start <= ent.start and ent.end <= c.end), None
                )
                if cover is None:
                    entities.append(ent)
                else:
                    suppressed.append((ent, cover))
        return DetectionReport(sort_entities(entities), suppressed)


_DEFAULT_DETECTOR = RuleDetector()


def detect(text: str) -> list[Entity]:
    """Detect structured PII in ``text`` with the default rules."""
    return _DEFAULT_DETECTOR.detect(text)
