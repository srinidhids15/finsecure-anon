"""FinGuard's spaCy tokenizer: English defaults plus documented boundary fixes.

Why: with spaCy's default English tokenizer, 28,539 gold entities in the
internal partitions do not align with token boundaries. Every one of them is
an entity followed by a sentence-final period that the tokenizer glued onto
the entity's last token (audit in docs/DATASET.md, issue 11):

    url      https://www.murphy.info/.   URL rule keeps a trailing "/." together
    company  Simpson Inc.                special-case exception "Inc." is one token
    name     Todd Meza PhD.              "." after an uppercase letter is never split

Fixes (the smallest set that aligns every legitimate boundary):

1. Suffix rule ``(?<=/)\\.``: split a final "." that directly follows "/".
   It applies only at the end of a token, so dots inside a URL are untouched.
2. Suffix rules splitting a final "." after the uppercase entity suffixes the
   data generator (Faker en_US) uses: company LLC, PLC; name MD, DDS, PhD,
   DVM, I, II, III, IV, V. spaCy's default "." suffix rule only fires after
   lowercase letters, so these would otherwise keep the sentence period.
3. Remove the special-case exceptions "Inc." and "Ltd.": company values in
   this data never contain the dot (Faker suffixes are "Inc", "Ltd"), so a
   following "." is always sentence punctuation.
4. For every other abbreviation exception ending in "." (e.g. "Jr.", "Dr."),
   add a special case "<abbr>." -> "<abbr>" "." . Name values such as
   "Peter Roman Jr." include the dot, and a sentence period after them gives
   "Jr.."; the default ".." suffix rule then splits it as "Jr" + "..",
   cutting through the gold boundary (3 spans in the training partition).

Abbreviations other than "Inc."/"Ltd." keep their exception because their
values include the dot. Known cost, documented in docs/DATASET.md: a
dot-dropped "Jr" followed by a sentence period stays one token "Jr."
(0 occurrences in the internal partitions).

Splitting a token never breaks the alignment of an existing span (it only
adds boundaries).

The tokenizer is registered as ``finguard.Tokenizer.v1``. A pipeline created
with ``make_nlp()`` records it in its config, and spaCy also serialises the
compiled patterns and rules with the model, so saving and reloading gives
the same tokens. ``import finguard.ner.tokenizer`` before ``spacy.load``.
"""

from __future__ import annotations

import spacy
from spacy.language import Language
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer
from spacy.util import compile_suffix_regex, registry

TOKENIZER_NAME = "finguard.Tokenizer.v1"

UPPERCASE_ENTITY_SUFFIXES: tuple[str, ...] = (
    "LLC", "PLC",  # company
    "MD", "DDS", "PhD", "DVM", "I", "II", "III", "IV", "V",  # name
)

EXTRA_SUFFIXES: tuple[str, ...] = (
    r"(?<=/)\.",  # "https://x.com/." -> "https://x.com/" "."
    # "\b" keeps e.g. "XV." or "CMD." from matching the shorter suffixes.
    *(rf"(?<=\b{suffix})\." for suffix in UPPERCASE_ENTITY_SUFFIXES),
)

REMOVED_SPECIAL_CASES: tuple[str, ...] = ("Inc.", "Ltd.")


def customize(nlp: Language, tokenizer: Tokenizer) -> Tokenizer:
    """Apply FinGuard's rules to an English tokenizer (in place) and return it."""
    tokenizer.suffix_search = compile_suffix_regex(
        list(nlp.Defaults.suffixes) + list(EXTRA_SUFFIXES)
    ).search
    rules = {
        text: attrs for text, attrs in tokenizer.rules.items()
        if text not in REMOVED_SPECIAL_CASES
    }
    for text, attrs in list(rules.items()):
        doubled = text + "."
        if text.endswith(".") and len(text) > 1 and " " not in text and doubled not in rules:
            rules[doubled] = [*attrs, {ORTH: "."}]
    tokenizer.rules = rules
    return tokenizer


@registry.tokenizers(TOKENIZER_NAME)
def create_finguard_tokenizer():
    default_factory = registry.tokenizers.get("spacy.Tokenizer.v1")()

    def make(nlp: Language) -> Tokenizer:
        return customize(nlp, default_factory(nlp))

    return make


def make_nlp() -> Language:
    """A blank English pipeline that uses FinGuard's tokenizer."""
    return spacy.blank("en", config={"nlp": {"tokenizer": {"@tokenizers": TOKENIZER_NAME}}})
