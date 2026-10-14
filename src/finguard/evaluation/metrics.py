"""Exact-span evaluation shared by every FinGuard detector.

Policy
------
* **Canonicalisation.** Before comparison, gold and predicted spans of a
  document both pass through ``finguard.canonical.canonical_spans``
  (trailing whitespace is not part of an entity). This only affects
  credit-card blocks, whose stored gold spans end in "\\n". Stored
  annotations are never modified.
* **Exact match.** A prediction is a true positive only if a gold entity
  has the same ``(start, end, label)``. Different offsets or a different
  label is an error: no partial credit.
* **Multisets.** Matching uses counts, so a prediction emitted twice
  counts once as TP and once as FP. Gold annotations contain no duplicates
  (validated in Phase 2).
* **Zero denominators.** Precision is undefined (``None``) when there are
  no predictions, recall when there is no gold. F1 is ``None`` only when
  both are undefined; if exactly one is undefined, F1 is 0.0 (the system
  failed on that side).
* **Micro** averages sum TP/FP/FN over all labels. **Macro** averages are
  unweighted means over the labels that have at least one gold *or*
  predicted entity in the evaluated set, with undefined precision/recall
  counted as 0. Labels with neither are listed in ``absent_labels`` and
  excluded.
* **Error types.** FP: ``boundary`` (overlaps gold of the same label),
  ``label`` (overlaps gold of another label) or ``spurious`` (overlaps no
  gold). FN: ``boundary``, ``label`` or ``missed`` analogously.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from finguard.canonical import canonical_spans
from finguard.config import ENTITY_LABELS
from finguard.schema import Span

Key = tuple[int, int, str]


@dataclass(frozen=True)
class EvalDocument:
    """One document to score: its text, gold spans, predictions and group."""

    doc_id: str
    text: str
    gold: Sequence[Span]
    predicted: Sequence[Span]
    group: str = ""


@dataclass(frozen=True)
class ErrorExample:
    doc_id: str
    kind: str  # "fp" or "fn"
    error_type: str
    start: int
    end: int
    label: str
    text: str
    other: str = ""  # the overlapping span on the other side, "label:start-end"


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: Counts) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn


def safe_div(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def prf(c: Counts) -> dict:
    """Precision, recall and F1 for one count triple (see module policy)."""
    p = safe_div(c.tp, c.tp + c.fp)
    r = safe_div(c.tp, c.tp + c.fn)
    if p is None and r is None:
        f = None
    elif p is None or r is None:
        f = 0.0
    else:
        f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return {
        "precision": p, "recall": r, "f1": f,
        "tp": c.tp, "fp": c.fp, "fn": c.fn,
        "gold": c.tp + c.fn, "predicted": c.tp + c.fp,
    }


def _classify(span: Key, others: Iterable[Key]) -> tuple[str, str]:
    s, e, label = span
    overlapping = [o for o in others if o[0] < e and s < o[1]]
    same = [o for o in overlapping if o[2] == label]
    if same:
        o = same[0]
        return "boundary", f"{o[2]}:{o[0]}-{o[1]}"
    if overlapping:
        o = overlapping[0]
        return "label", f"{o[2]}:{o[0]}-{o[1]}"
    return "", ""


def compare_document(doc: EvalDocument) -> tuple[dict[str, Counts], list[ErrorExample]]:
    """Per-label counts and error records for one document."""
    gold = Counter((s.start, s.end, s.label) for s in canonical_spans(doc.text, doc.gold))
    pred = Counter((s.start, s.end, s.label) for s in canonical_spans(doc.text, doc.predicted))
    counts: dict[str, Counts] = defaultdict(Counts)
    errors: list[ErrorExample] = []
    for key in gold.keys() | pred.keys():
        matched = min(gold[key], pred[key])
        label = key[2]
        counts[label].tp += matched
        extra_fp = pred[key] - matched
        extra_fn = gold[key] - matched
        counts[label].fp += extra_fp
        counts[label].fn += extra_fn
        for kind, n, others, default in (("fp", extra_fp, gold, "spurious"), ("fn", extra_fn, pred, "missed")):
            if n:
                etype, other = _classify(key, others)
                for _ in range(n):
                    errors.append(ErrorExample(doc.doc_id, kind, etype or default, key[0], key[1],
                                               label, doc.text[key[0]:key[1]], other))
    return counts, errors


@dataclass
class EvaluationResult:
    per_label: dict[str, dict]
    micro: dict
    macro: dict
    absent_labels: list[str]
    documents: int
    errors: list[ErrorExample] = field(default_factory=list)
    per_group: dict[str, dict] = field(default_factory=dict)
    group_macro: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """Everything except the individual error records."""
        error_counts = Counter((e.kind, e.label, e.error_type) for e in self.errors)
        return {
            "documents": self.documents,
            "micro": self.micro,
            "macro": self.macro,
            "absent_labels": self.absent_labels,
            "per_label": self.per_label,
            "error_counts": [
                {"kind": k, "label": l, "error_type": t, "count": n}
                for (k, l, t), n in sorted(error_counts.items())
            ],
            "per_group": self.per_group,
            "group_macro": self.group_macro,
        }


def _aggregate(per_label_counts: dict[str, Counts], labels: Sequence[str]) -> tuple[dict, dict, dict, list]:
    per_label = {l: prf(per_label_counts.get(l, Counts())) for l in labels}
    total = Counts()
    for l in labels:
        total.add(per_label_counts.get(l, Counts()))
    micro = prf(total)
    present = [l for l in labels if per_label[l]["gold"] or per_label[l]["predicted"]]
    absent = [l for l in labels if l not in present]
    if present:
        macro = {
            m: sum((per_label[l][m] or 0.0) for l in present) / len(present)
            for m in ("precision", "recall", "f1")
        }
    else:
        macro = {"precision": None, "recall": None, "f1": None}
    macro["labels_averaged"] = len(present)
    return per_label, micro, macro, absent


def evaluate(documents: Sequence[EvalDocument], labels: Sequence[str] = ENTITY_LABELS) -> EvaluationResult:
    """Score all documents; also per group (``EvalDocument.group``) if set."""
    unknown = {s.label for d in documents for s in (*d.gold, *d.predicted)} - set(labels)
    if unknown:
        raise ValueError(f"labels not in the evaluated label set: {sorted(unknown)}")

    totals: dict[str, Counts] = defaultdict(Counts)
    by_group: dict[str, dict[str, Counts]] = defaultdict(lambda: defaultdict(Counts))
    group_docs: Counter = Counter()
    errors: list[ErrorExample] = []
    for doc in documents:
        counts, errs = compare_document(doc)
        errors.extend(errs)
        for label, c in counts.items():
            totals[label].add(c)
            if doc.group:
                by_group[doc.group][label].add(c)
        if doc.group:
            group_docs[doc.group] += 1

    per_label, micro, macro, absent = _aggregate(totals, labels)
    per_group = {}
    for g in sorted(by_group):
        g_label, g_micro, g_macro, g_absent = _aggregate(by_group[g], labels)
        per_group[g] = {"documents": group_docs[g], "micro": g_micro, "macro": g_macro,
                        "per_label": g_label, "absent_labels": g_absent}
    group_macro = {}
    if per_group:
        for m in ("precision", "recall", "f1"):
            values = [v["micro"][m] for v in per_group.values() if v["micro"][m] is not None]
            group_macro[m] = sum(values) / len(values) if values else None
        group_macro["groups"] = len(per_group)
    errors.sort(key=lambda e: (e.doc_id, e.start, e.end, e.kind))
    return EvaluationResult(per_label, micro, macro, absent, len(documents), errors, per_group, group_macro)
