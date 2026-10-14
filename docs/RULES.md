# Rule-Based Detector

`finguard.detection.rules` finds the five *structured* entity types with
validated regular expressions. It is deterministic, needs no training and
reads only the text it is given: no gold labels and no row-specific values.

```python
from finguard.detection.rules import detect, RuleDetector

detect("Email asha@example.com or call +91 1472004612.")
# [Entity(start=6, end=22, label='email', source='rule:email', text='asha@example.com'),
#  Entity(start=31, end=45, label='phone', source='rule:phone', text='+91 1472004612')]
```

## Output

Each result is an `Entity` (a `Span` subclass from `finguard.schema`):

| Field | Meaning |
|---|---|
| `start` | inclusive character offset into the input text |
| `end` | exclusive character offset |
| `label` | one of the project labels (`email`, `url`, `phone`, `ssn`, `credit_card`) |
| `text` | exactly `input[start:end]` |
| `source` | `rule:<rule name>`, e.g. `rule:credit_card_block` |

Offsets count Python characters (Unicode code points) of the unmodified
input. Results are sorted by `(start, end, label, source)`.

## Coverage

| Label | Rule(s) | Covered? |
|---|---|---|
| `email` | `email` | yes |
| `url` | `url` | yes |
| `phone` | `phone` | yes, in the formats below |
| `ssn` | `ssn` | yes, hyphenated US form |
| `credit_card` | `credit_card_block`, `credit_card_number` | yes |
| `name`, `company`, `address` | none | **no**: free-form text without a reliable pattern; handled by the spaCy model (Phase 5) |

## Supported formats

**email:** `local@domain.tld`, ASCII only. The local part may contain
letters, digits and `. _ % + -`, may not start or end with a dot, and may not
contain `..`. Surrounding punctuation is not included.

**url:** must start with `http://`, `https://` or `www.`. Optional port,
path, query and fragment. Trailing sentence punctuation (`. , ; : ! ?`,
quotes) is trimmed, and closing brackets are trimmed only when unbalanced, so
`https://en.wikipedia.org/wiki/Bank_(finance)` keeps its `)`. Bare domains
(`example.com`) are **not** matched: they are indistinguishable from email
domains and abbreviations.

**phone:**
- International: `+` + 1-3 digit country code + optional space or hyphen,
  then 6-12 contiguous digits, or 2-5 groups separated by single spaces,
  hyphens or dots (first group optionally in parentheses). 8-15 digits in
  total. Examples: `+91 1472004612`, `+1 (555) 234-5678`, `+44 20 7946 0958`.
- North American: `(NXX) NXX-XXXX`, `NXX-NXX-XXXX`, `NXX.NXX.XXXX` (N = 2-9).
- Country codes starting with `0` are accepted. Real ones never start with
  `0`, but the dataset labels such numbers as phones (6.4 % of training-partition
  phone spans).
- **Not matched:** bare digit runs (`9876543210`), which look the same as account,
  invoice and reference numbers.

**ssn:** `AAA-GG-SSSS`, rejected if the area is `000`, `666` or `9xx`, the
group is `00` or the serial is `0000` (SSA structural rules). Unhyphenated
9-digit numbers are not matched.

**credit_card:** see the next section.

## Credit-card detection unit

Inspecting all 36,897 `credit_card` spans in the **training partition**
showed that the annotated entity is always a complete 5-line *information
block*, never just the number:

```
VISA 16 digit\n          <- network (10 variants in the data)
Asha Rao\n               <- card holder
4111111111111111 12/27\n <- number (12-19 digits, all Luhn-valid) + expiry MM/YY
CVC: 123\n               <- CVC / CVV (3 digits) or CID (4 digits, Amex)
```

The gold span starts at the network name and ends **after the final
newline**. So:

1. **`credit_card_block`** is the primary rule and reproduces that unit
   exactly, trailing newline included (when the block ends the text, there is
   none). Checks: known network name, valid month, 3-4 digit code, Luhn-valid
   number, and the number length must agree with a declared length such as
   `16 digit`.
2. **`credit_card_number`** finds stand-alone card numbers (13-19 digits,
   contiguous or in `4-4-4-4` / Amex `4-6-5` groups, known network prefix,
   Luhn-valid) for documents that print only a number. It **yields to** the
   block rule: a number inside a detected block is not emitted as a second,
   smaller entity. Instead it is listed in
   `RuleDetector.detect_with_report(text).suppressed`, together with the
   block that covers it.

When a block is malformed (for example month `13`, a missing holder line or a
wrong declared length) no block entity is produced. If the number inside is
still a valid card number, it is reported by `credit_card_number`, with that
source id, so the smaller granularity is never hidden.

The source annotations are **not** modified. Removing the trailing newline
(needed for spaCy token alignment) is a span-normalisation step for the
conversion/evaluation layer, applied to gold and predicted spans alike.

> **Luhn is not verification.** The checksum only catches typos: about 1 in
> 10 random digit strings pass it. A Luhn-valid number may never have been
> issued, or may be expired, cancelled or a test number. Only the card
> network or issuer can confirm that a card exists or is active.

## Overlaps and suppression

- Each rule's matches never overlap one another (`re.finditer` semantics).
- Matches from *different* rules may overlap, e.g. an email inside a URL's
  query string. **Both are returned.** `detection.base.overlapping_pairs`
  lists such pairs, and resolving them is the hybrid merger's job (Phase 6).
- The only results held back are those of a rule that explicitly *yields
  to* another rule (`credit_card_number` → `credit_card_block`; `ssn` →
  `phone`, for an SSN-shaped part inside a phone number). They are not lost:
  `detect_with_report` returns them in `suppressed`.

## Interface for later phases

`RuleDetector` implements `finguard.detection.base.Detector`:

```python
class Detector(Protocol):
    name: str
    def detect(self, text: str) -> list[Entity]: ...
```

The spaCy detector (Phase 5) will implement the same protocol. Because
`Entity` is a `Span`, the evaluator, anonymizer and hybrid merger accept
results from either detector. A subset of rules can be used with
`RuleDetector([r for r in DEFAULT_RULES if r.label != "url"])`.

## Development check (training partition only)

Run once on the 38,405 **training-partition** documents to check the rules
against the entity definition, using exact `(start, end, label)` matching.
It is not the Phase 4 evaluation: validation and test were not used.

| Label | Gold spans | Matched | Missed | Extra |
|---|---|---|---|---|
| email | 29,894 | 29,894 | 0 | 0 |
| url | 35,066 | 35,066 | 0 | 705 |
| phone | 39,243 | 39,243 | 0 | 0 |
| ssn | 43,664 | 43,664 | 0 | 0 |
| credit_card | 36,897 | 36,897 | 0 | 0 |

- All 705 extra URLs are real URLs that the gold labels miss (see
  docs/DATASET.md, issues 6 and 8). The rule was deliberately **not**
  changed to avoid them. The NER-ready dataset (`data/processed/ner/`)
  adds these as gold spans via a correction rule that uses only the source
  columns, not the detector.
- Exactly one rule change came from this check: accepting phone country
  codes that start with `0` (2,500 phone spans were missed before it).
  Other changes (excluding curly quotes from URL paths, allowing longer phone
  groups, linear-time URL trimming) came from the synthetic unit tests.
- These numbers reflect a templated synthetic corpus. They do not show
  that the rules would perform as well on real-world documents.

## Known limitations

- Names, companies and addresses are not detected.
- Phone: national formats without `+` (other than North American),
  extensions (`x123`), and vanity numbers are not supported. A grouped
  international number followed by a space and a short number (e.g. a year)
  may absorb it if the total stays within 15 digits.
- Email: internationalised (non-ASCII) addresses and quoted local parts are
  not supported. IP-address hosts and `localhost` are not URLs here.
- Card blocks with a network name outside the known list, or a different
  line layout, are only caught by the number rule (number only).
- Stand-alone Maestro numbers with 12 digits are not matched (12 digits is
  too short to tell apart safely from other identifiers); inside a block
  they are.
- SSNs written without hyphens are not matched.
- Text with mojibake must be repaired first (`data.repair`), as the Phase 2
  pipeline does; the detector itself never changes its input.
