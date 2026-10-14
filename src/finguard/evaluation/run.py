"""Evaluate the rule-based, NER and hybrid detectors on one dataset.

    python -m finguard.evaluation.run --dataset validation       # development
    python -m finguard.evaluation.run --dataset internal_test    # held out, needs freeze
    python -m finguard.evaluation.run --dataset official_test    # held out, needs freeze

All systems are scored on the same documents with the same function
(``metrics.evaluate``). Rule and NER predictions are computed once per
document; hybrid results are derived from them with ``hybrid.merge``, the
same code ``HybridDetector`` uses.

* ``validation`` also scores every pre-declared hybrid policy variant
  (``POLICY_VARIANTS``); this is where the policy is chosen.
* ``internal_test`` and ``official_test`` score only the frozen
  ``DEFAULT_POLICY`` and refuse to run unless ``verify_freeze()`` passes.
* ``official_test`` is scored twice: against the original annotations
  (diagnostic) and against annotations corrected with the established
  ``dot_dropped_variant`` rule (primary). The workbook is opened read-only
  for its source columns only; it is never written.

Reports: ``reports/evaluation/<dataset>/`` -> ``metrics.json``,
``per_label.csv``, ``per_group.csv``, ``errors.csv``, ``report.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from finguard.canonical import canonical_spans
from finguard.config import ENTITY_LABELS, PROCESSED_DATA_DIR, PROJECT_ROOT
from finguard.data.loader import file_sha256
from finguard.detection.hybrid import DEFAULT_POLICY, HybridPolicy, STRUCTURED_LABELS, merge
from finguard.detection.rules import RuleDetector
from finguard.evaluation.metrics import EvalDocument, EvaluationResult, evaluate
from finguard.schema import Entity, Span

REPORTS_DIR = PROJECT_ROOT / "reports" / "evaluation"
NER_DIR = PROCESSED_DATA_DIR / "ner"
HELD_OUT = ("internal_test", "official_test")
EXAMPLES_PER_LABEL = 5

# Declared before looking at any validation result; evaluated on validation only.
POLICY_VARIANTS: dict[str, HybridPolicy] = {
    "ner_priority": HybridPolicy(card_block_precedence=False, rule_priority_labels=frozenset()),
    "card_block_only": HybridPolicy(card_block_precedence=True, rule_priority_labels=frozenset()),
    "rule_priority_structured": HybridPolicy(card_block_precedence=True, rule_priority_labels=STRUCTURED_LABELS),
}


# ------------------------------------------------------------------ loading


def _spans(raw: list) -> tuple[Span, ...]:
    return tuple(Span(s, e, l, "gold") for s, e, l in raw)


def load_prepared(partition: str) -> list[dict]:
    """Records of a prepared NER partition, with original and corrected gold."""
    path = NER_DIR / f"{partition}.jsonl"
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            corrected = _spans(o["spans"])
            added = {(s, e, l) for s, e, l, _ in o["corrections"]}
            original = tuple(sp for sp in corrected if (sp.start, sp.end, sp.label) not in added)
            records.append({"id": o["id"], "text": o["text"], "gold": corrected, "gold_original": original,
                            "unit": o["group"], "template_group": o["template_group"],
                            "corrections": o["corrections"]})
    return records


def load_official(workbook: Path | None = None, test_jsonl: Path | None = None) -> tuple[list[dict], dict]:
    """Official test documents with original and corrected gold (evaluation copy).

    Text and original spans come from Phase 2's ``test.jsonl`` (mojibake
    repaired, validated). Source columns are read from the workbook,
    read-only, and the frozen correction rule is applied in memory.
    """
    from finguard.data.corrections import find_dot_drop_corrections
    from finguard.data.loader import load_workbook_records, read_jsonl
    from finguard.data.prepare import DEFAULT_RAW_DIR, TEST_FILE

    workbook = Path(workbook) if workbook is not None else DEFAULT_RAW_DIR / TEST_FILE
    test_jsonl = Path(test_jsonl) if test_jsonl is not None else PROCESSED_DATA_DIR / "test.jsonl"
    sha_before = file_sha256(workbook)
    raw = {r.record_id: r for r in load_workbook_records(workbook, "test")}
    records, review, n_corr = [], [], Counter()
    for ex in read_jsonl(test_jsonl):
        found, ambiguous = find_dot_drop_corrections(ex.record_id, ex.text, ex.spans, raw[ex.record_id].source_values)
        review.extend(ambiguous)
        n_corr.update(c.label for c in found)
        corrected = tuple(sorted(list(ex.spans) + [c.to_span() for c in found], key=lambda s: (s.start, s.end)))
        records.append({"id": ex.record_id, "text": ex.text, "gold": corrected, "gold_original": tuple(ex.spans),
                        "unit": "", "template_group": "",
                        "corrections": [[c.start, c.end, c.label, c.rule] for c in found]})
    if file_sha256(workbook) != sha_before:
        raise RuntimeError("official workbook changed during evaluation")
    info = {"workbook": str(workbook), "workbook_sha256": sha_before,
            "corrections_by_label": dict(n_corr), "corrections_total": sum(n_corr.values()),
            "review_items": [vars(r) for r in review]}
    return records, info


# --------------------------------------------------------------- predicting


def predict(records: list[dict], variants: dict[str, HybridPolicy]) -> dict[str, list[list[Entity]]]:
    from finguard.ner.inference import NERDetector

    rules = RuleDetector()
    ner = NERDetector()
    texts = [r["text"] for r in records]
    rule_out = [rules.detect(t) for t in texts]
    ner_out = ner.detect_many(texts)
    preds = {"rules": rule_out, "ner": ner_out}
    for name, policy in variants.items():
        preds[f"hybrid:{name}"] = [merge(t, r, n, policy).entities for t, r, n in zip(texts, rule_out, ner_out)]
    preds["_suppression"] = [
        Counter(s.reason for s in merge(t, r, n, DEFAULT_POLICY).suppressed)
        for t, r, n in zip(texts, rule_out, ner_out)
    ]
    return preds


# -------------------------------------------------------------- diagnostics


def card_diagnostics(records: list[dict], preds: list[list[Entity]]) -> dict:
    """How each gold credit-card block was handled, and names inside blocks."""
    kinds: Counter = Counter()
    names_inside = 0
    for r, ps in zip(records, preds):
        canon_preds = canonical_spans(r["text"], ps)
        for g in canonical_spans(r["text"], [s for s in r["gold"] if s.label == "credit_card"]):
            inside = [p for p in canon_preds if p.start < g.end and g.start < p.end]
            names_inside += sum(p.label == "name" for p in inside)
            if any((p.start, p.end, p.label) == (g.start, g.end, g.label) for p in inside):
                kinds["exact_block"] += 1
            elif not inside:
                kinds["missed"] += 1
            elif any(p.label == "credit_card" for p in inside):
                kinds["partial_credit_card"] += 1
            else:
                kinds["only_other_labels_inside"] += 1
    total = sum(kinds.values())
    return {"gold_blocks": total, **kinds,
            "block_recall": kinds["exact_block"] / total if total else None,
            "name_predictions_inside_gold_blocks": names_inside}


def score(records: list[dict], predictions: list[list[Entity]], gold_key: str, group_key: str) -> EvaluationResult:
    docs = [EvalDocument(r["id"], r["text"], r[gold_key], p, r[group_key]) for r, p in zip(records, predictions)]
    return evaluate(docs)


# ------------------------------------------------------------------ reports


def _fmt(x):
    return "–" if x is None else f"{x:.4f}"


def write_reports(out: Path, payload: dict, results: dict[str, dict[str, EvaluationResult]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    with open(out / "per_label.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["gold", "system", "label", "precision", "recall", "f1", "tp", "fp", "fn", "gold_count", "predicted"])
        for gold, systems in results.items():
            for system, res in systems.items():
                rows = list(res.per_label.items()) + [("MICRO", res.micro)]
                for label, m in rows:
                    w.writerow([gold, system, label, m["precision"], m["recall"], m["f1"],
                                m["tp"], m["fp"], m["fn"], m["gold"], m["predicted"]])
                w.writerow([gold, system, "MACRO", res.macro["precision"], res.macro["recall"],
                            res.macro["f1"], "", "", "", "", ""])

    with open(out / "per_group.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["gold", "system", "grouping", "group", "documents", "precision", "recall", "f1", "tp", "fp", "fn"])
        for gold, systems in results.items():
            for system, res in systems.items():
                for grouping, groups in payload["per_group"].get(gold, {}).get(system, {}).items():
                    for g, v in groups.items():
                        m = v["micro"]
                        w.writerow([gold, system, grouping, g, v["documents"], m["precision"], m["recall"],
                                    m["f1"], m["tp"], m["fp"], m["fn"]])

    with open(out / "errors.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["gold", "system", "doc_id", "kind", "error_type", "label", "start", "end", "text", "overlapping"])
        for gold, systems in results.items():
            for system, res in systems.items():
                for e in res.errors:
                    w.writerow([gold, system, e.doc_id, e.kind, e.error_type, e.label, e.start, e.end, e.text, e.other])

    (out / "report.md").write_text(render_markdown(payload, results), encoding="utf-8")


def render_markdown(payload: dict, results: dict[str, dict[str, EvaluationResult]]) -> str:
    L = [f"# Evaluation report: {payload['dataset']}", "",
         f"Documents: {payload['documents']}. Generated by `python -m finguard.evaluation.run --dataset {payload['dataset']}`.",
         "", "Exact match on canonical `(start, end, label)`; see docs/EVALUATION.md for the policy.", ""]
    if payload.get("freeze"):
        L += [f"Frozen system: `freeze_sha256 {payload['freeze']['freeze_sha256']}` (frozen {payload['freeze']['frozen_at']}).", ""]
    for gold, systems in results.items():
        L += [f"## Gold: {gold}", "", "| System | Micro P | Micro R | Micro F1 | Macro P | Macro R | Macro F1 | Gold | Predicted | TP | FP | FN |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for system, res in systems.items():
            m, M = res.micro, res.macro
            L.append(f"| {system} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | {_fmt(m['f1'])} | "
                     f"{_fmt(M['precision'])} | {_fmt(M['recall'])} | {_fmt(M['f1'])} | {m['gold']} | {m['predicted']} | "
                     f"{m['tp']} | {m['fp']} | {m['fn']} |")
        L += ["", "Per-label F1 (precision / recall):", "",
              "| System | " + " | ".join(ENTITY_LABELS) + " |", "|---|" + "---|" * len(ENTITY_LABELS)]
        for system, res in systems.items():
            cells = []
            for l in ENTITY_LABELS:
                v = res.per_label[l]
                cells.append(f"{_fmt(v['f1'])} ({_fmt(v['precision'])} / {_fmt(v['recall'])})")
            L.append(f"| {system} | " + " | ".join(cells) + " |")
        L.append("")
        groups = payload["per_group"].get(gold, {})
        for system, by in groups.items():
            for grouping, gs in by.items():
                if not gs:
                    continue
                L += [f"Per {grouping} ({system}): " + "; ".join(
                    f"`{g}` F1 {_fmt(v['micro']['f1'])} (n={v['documents']})" for g, v in gs.items()), ""]
        L += ["Credit-card blocks:", ""]
        for system, d in payload["card_diagnostics"].get(gold, {}).items():
            L.append(f"- {system}: {d}")
        L.append("")
    L += ["## Representative errors", ""]
    for gold, systems in payload["examples"].items():
        for system, by_label in systems.items():
            for label, exs in by_label.items():
                for e in exs:
                    L.append(f"- [{gold}/{system}] {e['kind'].upper()} {e['error_type']} {label} "
                             f"`{e['doc_id']}` {e['start']}-{e['end']}: {e['text'][:60]!r}" +
                             (f" (overlaps {e['other']})" if e["other"] else ""))
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------- run


def model_identifiers() -> dict:
    from finguard.evaluation.freeze import MODEL_DIR, sha256_dir

    manifest = json.loads((PROJECT_ROOT / MODEL_DIR / "training_manifest.json").read_text())
    return {"model_dir": MODEL_DIR, "model_sha256": sha256_dir(PROJECT_ROOT / MODEL_DIR),
            "config_sha256": manifest["config_sha256"], "trained_git_commit": manifest["git"]["commit"],
            "best_step": manifest["best_evaluation"]["step"]}


def run(dataset: str, out_root: Path = REPORTS_DIR) -> dict:
    from finguard.evaluation.freeze import verify_freeze

    if dataset not in ("validation", *HELD_OUT):
        raise ValueError(f"unknown dataset {dataset!r}")
    freeze_record = verify_freeze() if dataset in HELD_OUT else None
    variants = POLICY_VARIANTS if dataset == "validation" else {"selected": DEFAULT_POLICY}

    if dataset == "official_test":
        records, official_info = load_official()
        golds = {"corrected": "gold", "original": "gold_original"}
        groupings = {}
    else:
        records, official_info = load_prepared(dataset), None
        golds = {"corrected": "gold", "original": "gold_original"}
        groupings = {"unit": "unit", "template_group": "template_group"}

    preds = predict(records, variants)
    suppression = sum(preds.pop("_suppression"), Counter())
    systems = [s for s in preds]

    results: dict[str, dict[str, EvaluationResult]] = {}
    per_group: dict = {}
    card: dict = {}
    examples: dict = {}
    for gold_name, gold_key in golds.items():
        results[gold_name] = {}
        per_group[gold_name] = {}
        card[gold_name] = {}
        examples[gold_name] = {}
        for system in systems:
            res = score(records, preds[system], gold_key, "unit")
            results[gold_name][system] = res
            per_group[gold_name][system] = {
                g: score(records, preds[system], gold_key, key).per_group for g, key in groupings.items()
            }
            per_group[gold_name][system] = {g: {k: {"documents": v["documents"], "micro": v["micro"],
                                                    "macro": v["macro"]} for k, v in gs.items()}
                                            for g, gs in per_group[gold_name][system].items()}
            card[gold_name][system] = card_diagnostics([{**r, "gold": r[gold_key]} for r in records], preds[system])
            by_label: dict = {}
            for e in res.errors:
                bucket = by_label.setdefault(e.label, [])
                if len(bucket) < EXAMPLES_PER_LABEL:
                    bucket.append(vars(e))
            examples[gold_name][system] = by_label

    source = (PROCESSED_DATA_DIR / "test.jsonl") if dataset == "official_test" else (NER_DIR / f"{dataset}.jsonl")
    payload = {
        "dataset": dataset,
        "documents": len(records),
        "source_file": str(source),
        "source_sha256": file_sha256(source),
        "split_report_sha256": file_sha256(NER_DIR / "split_report.json"),
        "official": official_info,
        "model": model_identifiers(),
        "freeze": freeze_record and {"freeze_sha256": freeze_record["freeze_sha256"],
                                     "frozen_at": freeze_record["frozen_at"]},
        "hybrid_policies": {name: p.describe() for name, p in variants.items()},
        "selected_policy": DEFAULT_POLICY.describe(),
        "hybrid_suppressions_selected_policy": dict(suppression),
        "policies": {
            "match": "exact canonical (start, end, label); multiset counts",
            "canonicalisation": "finguard.canonical: trailing whitespace removed from span ends (gold and predictions)",
            "macro": "unweighted mean over labels with >=1 gold or predicted entity; undefined P/R count as 0",
            "zero_division": "P undefined without predictions, R undefined without gold; F1 0.0 if exactly one undefined",
        },
        "results": {g: {s: r.summary() for s, r in sysres.items()} for g, sysres in results.items()},
        "per_group": per_group,
        "card_diagnostics": card,
        "examples": examples,
    }
    for g in payload["results"].values():
        for s in g.values():
            s.pop("per_group", None)
    write_reports(out_root / dataset, payload, results)
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=["validation", *HELD_OUT])
    parser.add_argument("--out", type=Path, default=REPORTS_DIR)
    args = parser.parse_args(argv)
    payload = run(args.dataset, args.out)
    for gold, systems in payload["results"].items():
        print(f"== gold: {gold}")
        for system, r in systems.items():
            m = r["micro"]
            print(f"  {system:34s} P={_fmt(m['precision'])} R={_fmt(m['recall'])} F1={_fmt(m['f1'])} "
                  f"macroF1={_fmt(r['macro']['f1'])}  (gold {m['gold']}, pred {m['predicted']})")
    print(f"reports: {args.out / args.dataset}")


if __name__ == "__main__":
    main()
