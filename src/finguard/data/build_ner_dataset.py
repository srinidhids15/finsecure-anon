"""Build the NER-ready dataset: corrected labels + template-disjoint 3-way split.

Run (after ``python -m finguard.data.prepare``):
    python -m finguard.data.build_ner_dataset

Inputs, all read-only:
  * ``Training_Set.xlsx``: source columns and the raw annotation strings;
  * Phase 2 ``train.jsonl`` and ``validation.jsonl``: repaired, validated text.
The official ``Testing_Set.xlsx`` and Phase 2 ``test.jsonl`` are **not**
inputs: this module never opens them.

Steps
  1. Checksum every input (re-checked at the end).
  2. Check that the Phase 2 records match the workbook: same ids, text equal
     to the repaired workbook text, spans equal to the validated raw annotation.
  3. Add missing gold spans with the ``dot_dropped_variant`` rule
     (corrections.py). Ambiguous candidates are listed for review, never
     labelled.
  4. Compute each document's template group from its *corrected* spans,
     then join manually reviewed variants into split units using
     ``configs/template_aliases.json``.
  5. Assign whole units to train / validation / internal_test with the
     stratified, seeded algorithm in ``split.stratified_group_split``.
  6. Write ``data/processed/ner/`` (Phase 2 files are left untouched): three
     JSONL partitions, ``corrections.json``, ``split_report.json`` and a
     human-readable ``dataset_report.md``.

Stored spans keep their original offsets (credit-card blocks still end in
"\\n"); ``finguard.canonical`` trims that at conversion/evaluation time.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path

from finguard.config import CONFIGS_DIR, PROCESSED_DATA_DIR, RANDOM_SEED
from finguard.data.corrections import (
    DEFAULT_LABELS,
    Correction,
    ReviewItem,
    apply_corrections,
    find_dot_drop_corrections,
)
from finguard.data.loader import Example, file_sha256, load_workbook_records, read_jsonl
from finguard.data.prepare import DEFAULT_RAW_DIR, TRAIN_FILE
from finguard.data.repair import repair_mojibake
from finguard.data.split import (
    PARTITIONS,
    cluster_templates,
    normalized_template,
    oversized_groups,
    resolve_template_aliases,
    stratified_group_split,
    template_similarity,
)
from finguard.data.statistics import (
    class_distribution,
    compute_statistics,
    group_level_class_distribution,
)
from finguard.data.validation import validate_raw_record

DEFAULT_OUT_DIR = PROCESSED_DATA_DIR / "ner"
DEFAULT_ALIASES = CONFIGS_DIR / "template_aliases.json"
PHASE2_INPUTS = ("train.jsonl", "validation.jsonl")
SIMILARITY_THRESHOLD = 0.9
RELATED_TEMPLATE_FLOOR = 0.7  # report (not merge) template pairs at least this similar
ROW_SHARE_TOLERANCE = 0.05  # flag partitions whose row share is off target by more


class InputMismatchError(RuntimeError):
    """Phase 2 outputs and the training workbook disagree."""


def _load_inputs(raw_dir: Path, phase2_dir: Path):
    records = load_workbook_records(raw_dir / TRAIN_FILE, "train")
    phase2: dict[str, Example] = {}
    for name in PHASE2_INPUTS:
        for ex in read_jsonl(phase2_dir / name):
            phase2[ex.record_id] = ex
    ids = [r.record_id for r in records]
    if set(ids) != set(phase2):
        raise InputMismatchError(
            f"Phase 2 ids ({len(phase2)}) differ from training workbook ids ({len(ids)}); "
            "re-run python -m finguard.data.prepare"
        )
    if any(not i.startswith("train-") for i in ids):
        raise InputMismatchError("only training-workbook records may enter this pipeline")
    return records, phase2


def _verify_record(rec, ex: Example) -> None:
    text, _ = repair_mojibake(rec.text)
    if text != ex.text:
        raise InputMismatchError(f"{rec.record_id}: Phase 2 text differs from repaired workbook text")
    result = validate_raw_record(rec, text)
    if not result.is_valid:
        raise InputMismatchError(f"{rec.record_id}: raw annotation no longer validates")
    if [(s.start, s.end, s.label) for s in result.spans] != [(s.start, s.end, s.label) for s in ex.spans]:
        raise InputMismatchError(f"{rec.record_id}: Phase 2 spans differ from the raw annotation")


def _write_partition(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(
    raw_dir: Path = DEFAULT_RAW_DIR,
    phase2_dir: Path = PROCESSED_DATA_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = RANDOM_SEED,
    threshold: float = SIMILARITY_THRESHOLD,
    correction_labels: Sequence[str] = DEFAULT_LABELS,
    template_aliases: Path | Sequence[dict] = DEFAULT_ALIASES,
) -> dict:
    """Build the dataset. ``template_aliases`` is a path to an alias file or
    a list of alias families (``[]`` for none)."""
    raw_dir, phase2_dir, out_dir = Path(raw_dir), Path(phase2_dir), Path(out_dir)
    if isinstance(template_aliases, (str, Path)):
        alias_source = str(template_aliases)
        families = json.loads(Path(template_aliases).read_text(encoding="utf-8"))["families"]
    else:
        alias_source, families = "argument", list(template_aliases)
    inputs = [raw_dir / TRAIN_FILE] + [phase2_dir / n for n in PHASE2_INPUTS]
    checksums_before = {p.name: file_sha256(p) for p in inputs}

    records, phase2 = _load_inputs(raw_dir, phase2_dir)

    # ---- 2-3: verify, then correct ------------------------------------
    corrected: dict[str, tuple] = {}
    corrections: dict[str, list[Correction]] = {}
    review: list[ReviewItem] = []
    for rec in records:
        ex = phase2[rec.record_id]
        _verify_record(rec, ex)
        found, ambiguous = find_dot_drop_corrections(
            rec.record_id, ex.text, ex.spans, rec.source_values, correction_labels
        )
        review.extend(ambiguous)
        if found:
            corrections[rec.record_id] = found
        corrected[rec.record_id] = apply_corrections(ex.spans, found)

    # ---- 4: template groups (from corrected spans) -> split units ------
    ids = [r.record_id for r in records]
    templates = {i: normalized_template(phase2[i].text, corrected[i]) for i in ids}
    group_of_template = cluster_templates(list(templates.values()), threshold)
    representative: dict[str, str] = {}
    for t, g in sorted(group_of_template.items()):
        representative.setdefault(g, t)  # the group's smallest template (its id is hashed from it)
    unit_of_group = resolve_template_aliases(representative, families)
    groups = [group_of_template[templates[i]] for i in ids]
    units = [unit_of_group[g] for g in groups]

    # ---- 5: stratified split of whole units -----------------------------
    unit_sizes = Counter(units)
    unit_label_counts: dict[str, Counter] = defaultdict(Counter)
    for i, u in zip(ids, units):
        unit_label_counts[u].update(s.label for s in corrected[i])
    split = stratified_group_split(dict(unit_sizes), unit_label_counts, fractions, seed)
    partition_of_unit = {u: p for p, us in split.groups.items() for u in us}
    indices = {p: [k for k, u in enumerate(units) if partition_of_unit[u] == p] for p in PARTITIONS}

    all_ids = [ids[k] for p in PARTITIONS for k in indices[p]]
    assert len(all_ids) == len(set(all_ids)) == len(records), "a record is missing or duplicated"
    for a, b in combinations(PARTITIONS, 2):
        for keys in (units, groups):
            sa = {keys[k] for k in indices[a]}
            sb = {keys[k] for k in indices[b]}
            assert not sa & sb, f"unit or template group shared by {a} and {b}"

    # ---- 6: write ------------------------------------------------------
    for p in PARTITIONS:
        rows = []
        for k in indices[p]:
            rid = ids[k]
            rows.append({
                "id": rid,
                "text": phase2[rid].text,
                "spans": [[s.start, s.end, s.label] for s in corrected[rid]],
                "group": units[k],
                "template_group": groups[k],
                "corrections": [[c.start, c.end, c.label, c.rule] for c in corrections.get(rid, [])],
            })
        _write_partition(out_dir / f"{p}.jsonl", rows)

    report = _build_report(
        ids, groups, units, templates, representative, families, alias_source, split,
        indices, partition_of_unit, corrections, review, corrected, phase2, fractions, seed,
        threshold, correction_labels, checksums_before,
    )

    checksums_after = {p.name: file_sha256(p) for p in inputs}
    report["inputs_unchanged"] = checksums_before == checksums_after
    if not report["inputs_unchanged"]:
        raise RuntimeError("an input file changed during processing")

    raw_annotation = {r.record_id: r.annotation for r in records}
    correction_doc = {
        "rule": "dot_dropped_variant",
        "labels_considered": list(correction_labels),
        "summary": report["corrections"],
        "records": [
            {
                "record_id": rid,
                "original_annotation": raw_annotation[rid],
                "corrections": [c.to_dict() for c in corrections[rid]],
            }
            for rid in sorted(corrections)
        ],
        "review": [vars(r) for r in sorted(review, key=lambda r: (r.record_id, r.start))],
    }
    with open(out_dir / "corrections.json", "w", encoding="utf-8") as fh:
        json.dump(correction_doc, fh, indent=1, ensure_ascii=False)
    with open(out_dir / "split_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    (out_dir / "dataset_report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def _build_report(
    ids, groups, units, templates, representative, families, alias_source, split, indices,
    partition_of_unit, corrections, review, corrected, phase2, fractions, seed, threshold,
    correction_labels, checksums_before,
) -> dict:
    n_rows = len(ids)
    unit_sizes = Counter(units)
    targets = dict(zip(PARTITIONS, fractions))

    flat = [c for cs in corrections.values() for c in cs]
    by_label = Counter(c.label for c in flat)
    docs_by_label = defaultdict(set)
    for c in flat:
        docs_by_label[c.label].add(c.record_id)

    def examples_for(keys):
        return [Example(ids[k], phase2[ids[k]].text, corrected[ids[k]], units[k]) for k in keys]

    unit_table = {}
    for u in sorted(unit_sizes, key=lambda u: (-unit_sizes[u], u)):
        keys = [k for k, x in enumerate(units) if x == u]
        exs = examples_for(keys)
        unit_table[u] = {
            "records": unit_sizes[u],
            "partition": partition_of_unit[u],
            "stratum": split.strata[u],
            "template_groups": sorted({groups[k] for k in keys}),
            "entities": dict(Counter(s.label for ex in exs for s in ex.spans)),
            "class_distribution": class_distribution(exs),
        }

    partitions = {}
    for p in PARTITIONS:
        exs = examples_for(indices[p])
        row_share = len(exs) / n_rows
        stats = compute_statistics(exs)
        partitions[p] = {
            "units": split.groups[p],
            "unit_count": len(split.groups[p]),
            "high_frequency_units": sum(split.strata[u] == "high" for u in split.groups[p]),
            "low_frequency_units": sum(split.strata[u] == "low" for u in split.groups[p]),
            "template_groups": len({groups[k] for k in indices[p]}),
            "distinct_templates": len({templates[ids[k]] for k in indices[p]}),
            "records": len(exs),
            "row_share": round(row_share, 4),
            "target_share": targets[p],
            "entities": {l: c["spans"] for l, c in stats["per_class"].items()},
            "total_entities": stats["total_spans"],
            "class_distribution_record_weighted": class_distribution(exs),
            "class_distribution_group_level": group_level_class_distribution(exs),
            "missing_classes": [l for l, c in stats["per_class"].items() if c["spans"] == 0],
            "corrected_spans": sum(len(corrections.get(ids[k], [])) for k in indices[p]),
            "statistics": stats,
        }

    related = []
    for a, b in combinations(sorted(representative), 2):
        sim = template_similarity(representative[a], representative[b])
        if RELATED_TEMPLATE_FLOOR <= sim < threshold:
            ua, ub = units[groups.index(a)], units[groups.index(b)]
            related.append({
                "groups": [a, b],
                "similarity": round(sim, 3),
                "same_unit": ua == ub,
                "partitions": [partition_of_unit[ua], partition_of_unit[ub]],
                "opening_words": [representative[a][:60], representative[b][:60]],
            })
    related.sort(key=lambda r: -r["similarity"])

    return {
        "inputs": {"sha256": checksums_before},
        "settings": {
            "fractions": list(fractions), "seed": seed,
            "similarity_threshold": threshold, "correction_labels": list(correction_labels),
            "template_aliases": alias_source,
            "algorithm": (
                "template = gold spans (incl. corrections) -> <label>, delete '.', digit runs -> '0', "
                "collapse whitespace; templates with word similarity >= threshold form template groups; "
                "reviewed alias families join groups into split units; units >= median size are "
                "'high' frequency; each evaluation partition gets max(1, round(f*n)) units per stratum; "
                "random.Random(seed) draws candidate assignments scored by row-share deviation plus "
                "class-distribution distance of the evaluation partitions; lowest cost wins"
            ),
            "candidates_evaluated": split.candidates_evaluated,
            "selected_cost": split.cost,
        },
        "records": n_rows,
        "distinct_templates": len(set(templates.values())),
        "template_groups": len(set(groups)),
        "split_units": len(unit_sizes),
        "alias_families": [
            {"name": f["name"], "groups": [m["group"] for m in f["members"]], "reason": f["reason"]}
            for f in families
        ],
        "oversized_units": {
            u: {"rows": n, "partition": partition_of_unit[u]} for u, n in oversized_groups(units).items()
        },
        "units": unit_table,
        "related_template_pairs": related,
        "partitions": partitions,
        "corrections": {
            "records_corrected": len(corrections),
            "spans_added": len(flat),
            "by_label": {l: {"spans": by_label[l], "records": len(docs_by_label[l])} for l in correction_labels},
            "dots_removed": dict(sorted(Counter(c.dots_removed for c in flat).items())),
            "review_items": len(review),
            "review_by_problem": dict(Counter(r.problem for r in review)),
            "gold_spans_before": sum(len(phase2[i].spans) for i in ids),
            "gold_spans_after": sum(len(corrected[i]) for i in ids),
        },
    }


def render_markdown(r: dict) -> str:
    c = r["corrections"]
    labels = list(next(iter(r["partitions"].values()))["entities"])
    lines = [
        "# NER Dataset Report",
        "",
        "Generated by `python -m finguard.data.build_ner_dataset` from the training workbook",
        "and Phase 2 outputs. The official test workbook is not an input.",
        "",
        f"Inputs unchanged after processing: **{r.get('inputs_unchanged')}**",
        "",
        "## Annotation corrections (`dot_dropped_variant`)",
        "",
        f"Gold spans: {c['gold_spans_before']} → {c['gold_spans_after']} "
        f"(+{c['spans_added']} in {c['records_corrected']} records). "
        f"Review items (not labelled): {c['review_items']}.",
        "",
        "| Label | Spans added | Records |",
        "|---|---|---|",
    ]
    lines += [f"| {l} | {v['spans']} | {v['records']} |" for l, v in c["by_label"].items()]
    lines += ["", f"Dots removed per corrected span: {c['dots_removed']}",
              "", "Full provenance (original annotation, span, text, reason) in `corrections.json`.",
              "", "## Template-disjoint split", "",
              f"{r['records']} records · {r['distinct_templates']} normalised templates · "
              f"{r['template_groups']} template groups (similarity ≥ {r['settings']['similarity_threshold']}) · "
              f"{len(r['alias_families'])} reviewed alias families → **{r['split_units']} split units**. "
              f"Seed {r['settings']['seed']}, {r['settings']['candidates_evaluated']} candidates, "
              f"selected cost {r['settings']['selected_cost']}.", "",
              "| Partition | Units (high / low freq.) | Template groups | Templates | Records | Row share | Target | Entities | Corrected spans |",
              "|---|---|---|---|---|---|---|---|---|"]
    for p, v in r["partitions"].items():
        lines.append(
            f"| {p} | {v['unit_count']} ({v['high_frequency_units']} / {v['low_frequency_units']}) | "
            f"{v['template_groups']} | {v['distinct_templates']} | {v['records']} | {v['row_share']:.1%} | "
            f"{v['target_share']:.0%} | {v['total_entities']} | {v['corrected_spans']} |"
        )
    lines += ["", "### Entity counts", "", "| Partition | " + " | ".join(labels) + " |",
              "|---|" + "---|" * len(labels)]
    for p, v in r["partitions"].items():
        lines.append(f"| {p} | " + " | ".join(str(v["entities"][l]) for l in labels) + " |")
    for kind, title in (("class_distribution_record_weighted", "record-weighted (share of all spans)"),
                        ("class_distribution_group_level", "group-level (mean of per-unit shares; each unit counts once)")):
        lines += ["", f"### Class distribution, {title}", "", "| Partition | " + " | ".join(labels) + " |",
                  "|---|" + "---|" * len(labels)]
        for p, v in r["partitions"].items():
            lines.append(f"| {p} | " + " | ".join(f"{v[kind][l]:.1%}" for l in labels) + " |")
    lines += ["", "### Split units", "", "| Unit | Records | Stratum | Partition | Template groups | name | company | address | url |",
              "|---|---|---|---|---|---|---|---|---|"]
    for u, v in r["units"].items():
        d = v["class_distribution"]
        lines.append(
            f"| `{u}` | {v['records']} | {v['stratum']} | {v['partition']} | {len(v['template_groups'])} | "
            f"{d['name']:.0%} | {d['company']:.0%} | {d['address']:.0%} | {d['url']:.0%} |"
        )
    lines += ["", "### Oversized units (> 1.5 × median size)", ""]
    if r["oversized_units"]:
        lines += [f"- `{u}`: {v['rows']} rows, {v['partition']}" for u, v in r["oversized_units"].items()]
    else:
        lines.append("None.")
    lines += ["", f"### Related templates (similarity {RELATED_TEMPLATE_FLOOR}–{r['settings']['similarity_threshold']})", ""]
    if r["related_template_pairs"]:
        lines += ["| Similarity | Same unit | Partitions | Openings |", "|---|---|---|---|"]
        for pair in r["related_template_pairs"]:
            lines.append(
                f"| {pair['similarity']} | {'yes' if pair['same_unit'] else '**no**'} | {' / '.join(pair['partitions'])} | "
                f"{pair['opening_words'][0][:40]!r} / {pair['opening_words'][1][:40]!r} |"
            )
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--phase2-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    args = parser.parse_args(argv)
    r = run(args.raw_dir, args.phase2_dir, args.out_dir, seed=args.seed, threshold=args.threshold,
            template_aliases=args.aliases)
    c = r["corrections"]
    print(f"corrections: +{c['spans_added']} spans in {c['records_corrected']} records; review items: {c['review_items']}")
    print(f"templates: {r['distinct_templates']} -> {r['template_groups']} groups -> {r['split_units']} split units")
    for p, v in r["partitions"].items():
        print(f"{p:13s}: {v['unit_count']:2d} units ({v['high_frequency_units']} high / {v['low_frequency_units']} low), "
              f"{v['template_groups']:2d} template groups, {v['records']:6d} records ({v['row_share']:.1%}), "
              f"{v['total_entities']} entities")
    print(f"inputs unchanged: {r['inputs_unchanged']}  ->  {args.out_dir / 'dataset_report.md'}")


if __name__ == "__main__":
    main()
