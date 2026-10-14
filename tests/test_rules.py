"""Tests for the rule-based detector.

Every fixture here is synthetic and written for these tests. None is taken
from the official test workbook.
"""

import re

import pytest

from finguard.config import ENTITY_LABELS
from finguard.detection.base import Detector, overlapping_pairs
from finguard.detection.checksums import luhn_valid
from finguard.detection.rules import (
    DEFAULT_RULES,
    EMAIL_PATTERN,
    RegexRule,
    RuleDetector,
    detect,
)
from finguard.schema import Entity, Span


def check_boundaries(text, entities):
    """Invariants every detector result must satisfy."""
    for e in entities:
        assert isinstance(e, Entity)
        assert text[e.start : e.end] == e.text
        assert e.label in ENTITY_LABELS
        assert e.source.startswith("rule:")
    keys = [(e.start, e.end, e.label, e.source) for e in entities]
    assert keys == sorted(keys)


def found(text, label=None):
    entities = detect(text)
    check_boundaries(text, entities)
    return [e.text for e in entities if label is None or e.label == label]


def make_luhn(body: str) -> str:
    return next(body + d for d in "0123456789" if luhn_valid(body + d))


# ===================================================================== email


@pytest.mark.parametrize(
    "email",
    ["asha@example.com", "asha.rao+tax@mail.example.co.in", "a_b-c%d@my-bank.org",
     "x1@sub.domain.museum", "UPPER.Case@Example.COM"],
)
def test_valid_emails(email):
    assert found(f"Contact {email} today.", "email") == [email]


@pytest.mark.parametrize(
    "wrapped, email",
    [("<asha@example.com>", "asha@example.com"),
     ("(asha@example.com)", "asha@example.com"),
     ("asha@example.com.", "asha@example.com"),
     ("asha@example.com,", "asha@example.com"),
     ('"asha@example.com"', "asha@example.com"),
     ("asha@example.com’s inbox", "asha@example.com")],
)
def test_email_punctuation_is_not_included(wrapped, email):
    assert found(f"Write to {wrapped}", "email") == [email]


@pytest.mark.parametrize(
    "text",
    ["asha@", "@example.com", "asha@example", "asha..rao@example.com",
     "asha@-bank.com", "asha@bank..com", "name at example dot com",
     "josé@example.com"],  # non-ASCII local part: unsupported, no partial match
)
def test_invalid_emails(text):
    assert found(text, "email") == []


def test_repeated_and_adjacent_emails():
    text = "a@x.com,b@y.org a@x.com"
    entities = [e for e in detect(text) if e.label == "email"]
    assert [(e.start, e.text) for e in entities] == [(0, "a@x.com"), (8, "b@y.org"), (16, "a@x.com")]


def test_email_pattern_does_not_hang_on_long_input():
    # Guard against catastrophic backtracking on a long non-matching string.
    assert EMAIL_PATTERN.search("a" * 5000 + "@") is None


# ======================================================================= url


@pytest.mark.parametrize(
    "url",
    ["http://example.com/", "https://www.example.com/", "https://my-bank.co.in/a/b?x=1&y=2#top",
     "www.example.org", "https://example.com:8443/path", "HTTPS://EXAMPLE.COM/",
     "https://en.wikipedia.org/wiki/Bank_(finance)"],
)
def test_valid_urls(url):
    assert found(f"Visit {url} now", "url") == [url]


@pytest.mark.parametrize(
    "text, url",
    [("Visit https://example.com/.", "https://example.com/"),
     ("(see https://example.com/a)", "https://example.com/a"),
     ("Visit https://example.com/, then", "https://example.com/"),
     ("Try www.example.com!", "www.example.com"),
     ("“https://example.com/”", "https://example.com/"),
     ("site https://example.com/’s page", "https://example.com/")],
)
def test_url_trailing_punctuation_trimmed(text, url):
    assert found(text, "url") == [url]


@pytest.mark.parametrize(
    "text",
    ["example.com", "Email asha@example.com", "ftp://example.com/file",
     "https://", "https://localhost/", "version 1.2.3", "www.", "e.g. this"],
)
def test_not_urls(text):
    assert found(text, "url") == []


def test_url_with_email_inside_returns_both_overlapping():
    text = "Open https://example.com/?u=asha@example.com now"
    entities = detect(text)
    assert [(e.label, e.text) for e in entities] == [
        ("url", "https://example.com/?u=asha@example.com"),
        ("email", "asha@example.com"),
    ]
    # Overlap is reported, not silently resolved.
    assert len(overlapping_pairs(entities)) == 1


# ===================================================================== phone


@pytest.mark.parametrize(
    "phone",
    ["+91 1472004612", "+242 3283446338", "+04 7024846271", "+911472004612",
     "+1 (555) 234-5678", "+44 20 7946 0958", "+49-30-901820",
     "(555) 234-5678", "555-234-5678", "555.234.5678"],
)
def test_supported_phone_formats(phone):
    assert found(f"Call {phone} today", "phone") == [phone]


@pytest.mark.parametrize(
    "text, phone",
    [("number (+91 1472004612), and", "+91 1472004612"),
     ("at +91 1472004612.", "+91 1472004612"),
     ("line1\n+56 8366018544\nline3", "+56 8366018544")],
)
def test_phone_punctuation_and_multiline(text, phone):
    assert found(text, "phone") == [phone]


@pytest.mark.parametrize(
    "text",
    [
        "Account 9876543210 is active",  # bare digit run
        "Invoice INV-123456789012 issued",  # reference number
        "Claim #CLM9076109877 opened",  # reference number glued to letters
        "From IP address 1234567 we saw",  # arbitrary number
        "Score +12345 points",  # too few digits
        "Value +91 14720046123456789",  # too many digits
        "Dated 2023-10-14",  # date
        "Code 123-456-7890",  # area code cannot start with 1
        "Ref 555-234-56789",  # trailing extra digit
        "+９１ １４７２００４６１２",  # fullwidth digits
    ],
)
def test_arbitrary_numbers_are_not_phones(text):
    assert found(text, "phone") == []


# ======================================================================= ssn


@pytest.mark.parametrize(
    "text",
    ["SSN 575-74-7674.", "(SSN) (048-59-6922).", "TIN: 141-32-3590\n", "575-74-7674"],
)
def test_valid_ssn(text):
    assert len(found(text, "ssn")) == 1


@pytest.mark.parametrize(
    "text",
    ["000-12-3456", "666-12-3456", "900-12-3456", "123-00-4567", "123-45-0000",  # SSA rules
     "1575-74-7674", "575-74-76745", "ABC575-74-7674", "575-74-7674X",  # glued to more chars
     "575747674", "575 74 7674", "57-574-7674",  # other shapes
     "５７５-７４-７６７４"],  # fullwidth digits
)
def test_not_ssn(text):
    assert found(text, "ssn") == []


def test_ssn_inside_phone_is_reported_as_suppressed():
    report = RuleDetector().detect_with_report("Call +1 555-12-3456 now")
    assert [(e.label, e.text) for e in report.entities] == [("phone", "+1 555-12-3456")]
    assert [(s.label, s.text, c.label) for s, c in report.suppressed] == [
        ("ssn", "555-12-3456", "phone")
    ]


# =============================================================== credit card

BLOCK = "VISA 16 digit\nAsha Rao\n4111111111111111 12/27\nCVC: 123\n"


def test_card_block_is_the_entity_including_trailing_newline():
    text = f"Paid with card ending in {BLOCK}. Thank you."
    entities = [e for e in detect(text) if e.label == "credit_card"]
    assert len(entities) == 1
    card = entities[0]
    assert card.text == BLOCK
    assert card.text.endswith("\n")
    assert card.source == "rule:credit_card_block"
    assert card.start == text.index("VISA")


def test_number_inside_block_is_suppressed_not_lost():
    report = RuleDetector().detect_with_report(f"Card {BLOCK}")
    assert [e.source for e in report.entities] == ["rule:credit_card_block"]
    (number, cover), = report.suppressed
    assert number.text == "4111111111111111"
    assert number.source == "rule:credit_card_number"
    assert cover.text == BLOCK


@pytest.mark.parametrize(
    "block",
    [
        "American Express\nAsha Rao\n378282246310005 05/26\nCID: 0471\n",
        "Mastercard\nAsha Rao\n5555555555554444 11/26\nCVV: 959\n",
        "Diners Club / Carte Blanche\nAsha Rao\n30569309025904 01/29\nCVC: 111\n",
        "JCB 16 digit\nAsha Rao\n3530111333300000 05/34\nCVC: 603\n",
        "VISA 13 digit\nAsha Rao\n4222222222222 06/27\nCVC: 419\n",
        "Discover\nDr. Asha Rao Jr.\n6011111111111117 08/30\nCVC: 850\n",
    ],
)
def test_card_block_variants(block):
    assert found(f"card {block}rest", "credit_card") == [block]


def test_card_block_at_end_of_text_without_newline():
    text = "Card " + BLOCK.rstrip("\n")
    assert found(text, "credit_card") == [BLOCK.rstrip("\n")]


def test_card_block_with_windows_line_endings():
    block = BLOCK.replace("\n", "\r\n")
    assert found(f"Card {block}x", "credit_card") == [block]


def test_two_adjacent_blocks():
    other = "Mastercard\nBen Lee\n5555555555554444 11/26\nCVV: 959\n"
    assert found(BLOCK + other, "credit_card") == [BLOCK, other]


@pytest.mark.parametrize(
    "block",
    [
        BLOCK.replace("12/27", "13/27"),  # month 13
        BLOCK.replace("CVC: 123", "CVC: 12"),  # code too short
        BLOCK.replace("Asha Rao\n", ""),  # holder line missing
    ],
)
def test_malformed_block_is_not_a_block_but_its_valid_number_is_reported(block):
    # No block entity is produced. The Luhn-valid number inside is still
    # card PII, so the number rule reports it under its own source id.
    assert [(e.source, e.text) for e in detect(block)] == [
        ("rule:credit_card_number", "4111111111111111")
    ]


def test_block_with_luhn_invalid_number_yields_nothing():
    assert detect(BLOCK.replace("4111111111111111", "4111111111111112")) == []


def test_declared_length_mismatch_falls_back_to_number_rule():
    # "VISA 16 digit" followed by a 13-digit number is not a valid block. The
    # number is still card-like, so the number rule reports it, with its
    # own source id so the change of granularity is visible.
    text = BLOCK.replace("4111111111111111", "4222222222222")
    entities = detect(text)
    assert [(e.source, e.text) for e in entities] == [
        ("rule:credit_card_number", "4222222222222")
    ]


@pytest.mark.parametrize(
    "number",
    ["4111111111111111", "4111 1111 1111 1111", "4111-1111-1111-1111",
     "378282246310005", "3782 822463 10005"],
)
def test_standalone_card_numbers(number):
    entities = detect(f"Card no. {number}, exp 12/27")
    assert [(e.source, e.text) for e in entities] == [("rule:credit_card_number", number)]


@pytest.mark.parametrize(
    "text",
    [
        "Card 4111111111111112",  # Luhn fails
        "Invoice INV-4111111111111111",  # reference number
        "Ref A4111111111111111",  # glued to letters
        "Card " + make_luhn("911111111111111"),  # Luhn-valid, no network uses prefix 9
        "Card " + make_luhn("41111111111"),  # 12 digits: too short standalone
        "Card " + make_luhn("4111111111111111111"),  # 20 digits
        "Card 4111 1111 1111 111",  # bad grouping
    ],
)
def test_not_card_numbers(text):
    assert found(text, "credit_card") == []


# ===================================================================== general


def test_not_attempted_labels():
    text = "Asha Rao of Acme Ltd lives at 12 Lake Road\nPune, MH 411001."
    assert detect(text) == []


def test_full_synthetic_document_offsets():
    text = (
        "Dear Asha Rao, your card ending in " + BLOCK + " was charged.\n"
        "Questions? Email asha@example.com or call +91 1472004612.\n"
        "Our records list SSN 575-74-7674. Details: https://example.com/acct."
    )
    entities = detect(text)
    check_boundaries(text, entities)
    assert [(e.label, e.text) for e in entities] == [
        ("credit_card", BLOCK),
        ("email", "asha@example.com"),
        ("phone", "+91 1472004612"),
        ("ssn", "575-74-7674"),
        ("url", "https://example.com/acct"),
    ]
    for e in entities:
        assert text.find(e.text, e.start) == e.start


def test_unicode_before_entities_keeps_character_offsets():
    text = "Café ☕ \U0001F4B3 — résumé: asha@example.com"
    (email,) = detect(text)
    assert email.start == text.index("asha")
    assert text[email.start : email.end] == "asha@example.com"


def test_input_is_not_modified_and_results_are_repeatable():
    text = f"Mail a@x.com. {BLOCK}"
    copy = str(text)
    first = detect(text)
    assert text == copy
    assert detect(text) == first


def test_empty_and_whitespace_text():
    assert detect("") == []
    assert detect(" \n\t") == []


def test_non_string_input_rejected():
    with pytest.raises(TypeError):
        detect(b"asha@example.com")


def test_entity_output_format():
    (e,) = detect("x asha@example.com")
    assert e.to_dict() == {
        "start": 2, "end": 18, "label": "email",
        "text": "asha@example.com", "source": "rule:email",
    }
    assert isinstance(e, Span)  # usable anywhere a Span is expected


def test_rule_detector_implements_detector_protocol():
    assert isinstance(RuleDetector(), Detector)


def test_detector_validates_rule_configuration():
    rule = DEFAULT_RULES[0]
    with pytest.raises(ValueError, match="duplicate"):
        RuleDetector([rule, rule])
    with pytest.raises(ValueError, match="unknown label"):
        RuleDetector([RegexRule("x", "PERSON", re.compile("x"))])
    with pytest.raises(ValueError, match="yields to unknown"):
        RuleDetector([RegexRule("x", "email", re.compile("x"), yield_to=("nope",))])


def test_subset_of_rules():
    det = RuleDetector([r for r in DEFAULT_RULES if r.label == "email"])
    assert [e.label for e in det.detect("a@x.com +91 1472004612")] == ["email"]


def test_rules_do_not_produce_duplicates():
    text = ("a@x.com https://x.com/ +91 1472004612 575-74-7674 " + BLOCK) * 3
    entities = detect(text)
    keys = [(e.start, e.end, e.label) for e in entities]
    assert len(keys) == len(set(keys)) == 15


def test_url_trimming_is_linear_on_many_unbalanced_brackets():
    import time

    text = "https://x.com/" + ")" * 100_000
    started = time.perf_counter()
    assert found(text, "url") == ["https://x.com/"]
    assert time.perf_counter() - started < 1.0
