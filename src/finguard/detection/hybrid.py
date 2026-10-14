"""Hybrid detection: combine rule-based and NER entities with an explicit policy.

Both detectors run on the same, unmodified input. The merge is
deterministic and uses no gold data and no confidence scores (the rule
detector has none, and none are invented).

Merge policy (``HybridPolicy``), applied in this order:

1. **Deduplicate.** Candidates with the same canonical ``(start, end, label)``
   (see ``finguard.canonical``) are one entity. The rule entity's original
   offsets are kept, so a credit-card block keeps its trailing newline, and
   the sources are joined (``"rule:credit_card_block|ner"``).
2. **Credit-card block precedence.** A credit-card block validated by the
   rule detector (``rule:credit_card_block``) is the dataset's entity unit.
   Every other candidate that overlaps it (a holder name, a number, an
   address fragment) is suppressed, and the reason is recorded.
3. **Rule-priority labels.** When a rule entity whose label is in
   ``rule_priority_labels`` overlaps an NER entity, the rule entity wins.
4. **Everything else.** For any other overlap the NER entity wins (it is
   the only source for name, address and company, and is context-aware).
   Among rule entities that overlap each other (an email inside a URL), the
   longer span wins, ties going to the earlier start.

Every suppressed candidate is returned in ``HybridResult.suppressed`` with
the entity that won and the rule that applied. Non-overlapping candidates
from either source are all kept. The final entities never overlap, which
the redaction step needs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from finguard.canonical import canonical_span
from finguard.detection.base import sort_entities
from finguard.detection.rules import RuleDetector
from finguard.schema import Entity

CARD_BLOCK_SOURCE = "rule:credit_card_block"
STRUCTURED_LABELS = frozenset({"email", "url", "phone", "ssn", "credit_card"})


@dataclass(frozen=True)
class HybridPolicy:
    card_block_precedence: bool = True
    rule_priority_labels: frozenset[str] = STRUCTURED_LABELS

    def describe(self) -> dict:
        return {
            "card_block_precedence": self.card_block_precedence,
            "rule_priority_labels": sorted(self.rule_priority_labels),
        }


# Selected on the validation partition (docs/EVALUATION.md).
DEFAULT_POLICY = HybridPolicy()


@dataclass(frozen=True)
class Suppression:
    candidate: Entity
    winner: Entity
    reason: str


@dataclass
class HybridResult:
    entities: list[Entity]
    suppressed: list[Suppression] = field(default_factory=list)
    merged_duplicates: int = 0
    rule_suppressed: list[tuple[Entity, Entity]] = field(default_factory=list)  # from RuleDetector


def _is_rule(e: Entity) -> bool:
    return e.source.startswith("rule:")


def _canon_key(text: str, e: Entity) -> tuple[int, int, str]:
    c = canonical_span(text, e)
    c = c or e
    return (c.start, c.end, c.label)


def _overlap(a: Entity, b: Entity) -> bool:
    return a.start < b.end and b.start < a.end


def merge(
    text: str,
    rule_entities: Sequence[Entity],
    ner_entities: Sequence[Entity],
    policy: HybridPolicy = DEFAULT_POLICY,
) -> HybridResult:
    """Combine the two detectors' outputs for ``text`` (see module policy)."""
    result = HybridResult(entities=[])

    # 1. deduplicate identical canonical spans (rule entity kept, sources joined)
    by_key: dict[tuple[int, int, str], Entity] = {}
    for e in list(rule_entities) + list(ner_entities):
        key = _canon_key(text, e)
        if key in by_key:
            kept = by_key[key]
            if e.source not in kept.source.split("|"):
                by_key[key] = Entity(kept.start, kept.end, kept.label, f"{kept.source}|{e.source}", kept.text)
            result.merged_duplicates += 1
        else:
            by_key[key] = e
    candidates = sort_entities(list(by_key.values()))

    def suppress(loser: Entity, winner: Entity, reason: str) -> None:
        result.suppressed.append(Suppression(loser, winner, reason))
        alive.discard(id(loser))

    alive = {id(e) for e in candidates}

    # 2. card-block precedence
    if policy.card_block_precedence:
        blocks = [e for e in candidates if e.source.split("|")[0] == CARD_BLOCK_SOURCE]
        for block in blocks:
            for e in candidates:
                if e is not block and id(e) in alive and _overlap(e, block):
                    suppress(e, block, "inside_credit_card_block")

    # 3./4. remaining pairwise overlaps, resolved in a fixed order
    def winner_of(a: Entity, b: Entity) -> tuple[Entity, Entity, str]:
        ra, rb = _is_rule(a), _is_rule(b)
        if ra and not rb:
            return (a, b, "rule_priority_label") if a.label in policy.rule_priority_labels else (b, a, "ner_preferred")
        if rb and not ra:
            return (b, a, "rule_priority_label") if b.label in policy.rule_priority_labels else (a, b, "ner_preferred")
        # same source type: longer span wins, then earlier start
        la, lb = a.end - a.start, b.end - b.start
        if la != lb:
            return (a, b, "longer_span") if la > lb else (b, a, "longer_span")
        return (a, b, "earlier_start") if a.start <= b.start else (b, a, "earlier_start")

    changed = True
    while changed:
        changed = False
        live = [e for e in candidates if id(e) in alive]
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                if b.start >= a.end:
                    break
                if id(a) in alive and id(b) in alive and _overlap(a, b):
                    win, lose, reason = winner_of(a, b)
                    suppress(lose, win, reason)
                    changed = True

    result.entities = sort_entities([e for e in candidates if id(e) in alive])
    return result


class HybridDetector:
    """Rules + NER behind the ``Detector`` interface."""

    name = "hybrid"

    def __init__(self, rules: RuleDetector | None = None, ner=None, policy: HybridPolicy = DEFAULT_POLICY):
        from finguard.ner.inference import NERDetector

        self.rules = rules or RuleDetector()
        self.ner = ner if ner is not None else NERDetector()
        self.policy = policy

    def detect_with_report(self, text: str) -> HybridResult:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        rule_report = self.rules.detect_with_report(text)
        result = merge(text, rule_report.entities, self.ner.detect(text), self.policy)
        result.rule_suppressed = list(rule_report.suppressed)
        return result

    def detect(self, text: str) -> list[Entity]:
        return self.detect_with_report(text).entities
