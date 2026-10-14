"""Tests for FinGuard's spaCy tokenizer and the alignment audit."""

import pytest
import spacy

from finguard.canonical import canonical_spans
from finguard.data.loader import Example
from finguard.ner.alignment import audit, failure_pattern
from finguard.ner.tokenizer import TOKENIZER_NAME, make_nlp
from finguard.schema import Span


@pytest.fixture(scope="module")
def nlp():
    return make_nlp()


def tokens(nlp, text):
    return [t.text for t in nlp(text)]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Visit https://x.com/. Then", ["Visit", "https://x.com/", ".", "Then"]),
        ("at http://wwwxcom/.", ["at", "http://wwwxcom/", "."]),
        ("Simpson Inc. The", ["Simpson", "Inc", ".", "The"]),
        ("Cook Ltd.", ["Cook", "Ltd", "."]),
        ("Andrade LLC. x", ["Andrade", "LLC", ".", "x"]),
        ("Todd Meza PhD. x", ["Todd", "Meza", "PhD", "."] + ["x"]),
        ("Ann Clarke MD.", ["Ann", "Clarke", "MD", "."]),
        ("Scott Thomas V. The", ["Scott", "Thomas", "V", ".", "The"]),
        ("by Peter Roman Jr.. The", ["by", "Peter", "Roman", "Jr.", ".", "The"]),
    ],
)
def test_sentence_period_is_split_from_entity_end(nlp, text, expected):
    assert tokens(nlp, text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("see https://x.com/a.b/c ok", ["see", "https://x.com/a.b/c", "ok"]),  # inner dots kept
        ("Mr. Smith and Dr. Rao", ["Mr.", "Smith", "and", "Dr.", "Rao"]),  # abbreviations kept
        ("Peter Roman Jr. said", ["Peter", "Roman", "Jr.", "said"]),  # value with its dot
        ("mail a.b@example.com now", ["mail", "a.b@example.com", "now"]),
        ("XV. CMD.", ["XV", ".", "CMD", "."]),  # \b guard: no partial suffix match
    ],
)
def test_legitimate_tokens_unchanged(nlp, text, expected):
    assert tokens(nlp, text) == expected


def test_default_tokenizer_shows_the_original_problem():
    default = spacy.blank("en")
    assert tokens(default, "Visit https://x.com/. Then")[1] == "https://x.com/."
    assert tokens(default, "Simpson Inc. The")[1] == "Inc."


def test_config_names_the_registered_tokenizer(nlp):
    assert nlp.config["nlp"]["tokenizer"]["@tokenizers"] == TOKENIZER_NAME


def test_save_and_reload_reproduces_token_boundaries(nlp, tmp_path):
    nlp.to_disk(tmp_path / "model")
    reloaded = spacy.load(tmp_path / "model")
    texts = [
        "Visit https://x.com/. Simpson Inc. Peter Roman Jr.. Meza PhD. Andrade LLC.",
        "Card ending in VISA 16 digit\nAsha Rao\n4111111111111111 12/27\nCVC: 123\n, thanks.",
    ]
    for text in texts:
        a, b = nlp(text), reloaded(text)
        assert [(t.idx, t.text) for t in a] == [(t.idx, t.text) for t in b]


# --------------------------------------------------------------- alignment

BLOCK = "VISA 16 digit\nAsha Rao\n4111111111111111 12/27\nCVC: 123\n"


def _example(text, *pieces):
    """Build an Example with spans for each (substring, label) given."""
    spans = []
    for piece, label in pieces:
        start = text.index(piece)
        spans.append(Span(start, start + len(piece), label, "gold"))
    return Example("r1", text, tuple(spans))


@pytest.mark.parametrize(
    "tail, raw_aligns",
    [
        (" was charged.", False),  # "\n " is one whitespace token: raw end falls inside it
        (". Thanks.", True),
        (", please call.", True),
        ("", True),
    ],
)
def test_credit_card_block_aligns_after_canonicalisation(nlp, tail, raw_aligns):
    text = f"Paid with card ending in {BLOCK}{tail}"
    (gold,) = _example(text, (BLOCK, "credit_card")).spans
    doc = nlp(text)
    assert (doc.char_span(gold.start, gold.end, alignment_mode="strict") is not None) is raw_aligns
    (canon,) = canonical_spans(text, [gold])
    span = doc.char_span(canon.start, canon.end, alignment_mode="strict")
    assert span is not None and span.text == BLOCK.rstrip("\n")
    assert canon.start == gold.start  # only the end moved


def test_audit_counts_failures_without_dropping_spans(nlp):
    text = "Visit https://x.com/. Simpson Inc. Peter Roman Jr.. Card " + BLOCK + "."
    ex = _example(text, ("https://x.com/", "url"), ("Simpson Inc", "company"),
                  ("Peter Roman Jr.", "name"), (BLOCK, "credit_card"))
    default_report = audit(spacy.blank("en"), [ex])
    finguard_report = audit(nlp, [ex])
    assert sum(default_report.spans.values()) == sum(finguard_report.spans.values()) == 4
    assert finguard_report.total_failures == 0
    assert {l for l, _ in default_report.failures} == {"url", "company", "name"}


def test_failure_pattern_describes_glued_suffix():
    doc = spacy.blank("en")("Visit https://x.com/. Then")
    start = 6
    assert failure_pattern(doc, start, start + len("https://x.com/")) == "end+'.'"
