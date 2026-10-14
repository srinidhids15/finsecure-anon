"""Phase 2 pipeline: raw Mendeley workbooks -> audited, split JSONL files.

Run:
    python -m finguard.data.prepare

Steps
  1. Checksum the raw files (they are re-checked at the end to prove they
     were not modified).
  2. Load both workbooks and validate the *raw* text. This records how
     many annotations are broken as downloaded.
  3. Repair the text-encoding damage (see repair.py) and validate again.
  4. Flag duplicate records, within each file and between train and test.
  5. Official test file: every record must be valid. If not, stop. The
     test set is never filtered or edited to make it pass.
  6. Training file: drop invalid records (each one is listed in the
     report), then split the rest into train/validation by template.
  7. Write train / validation / test JSONL plus a JSON and a Markdown report.

The report contains ids, issue codes, offsets and counts, never document text.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from finguard.config import PROCESSED_DATA_DIR, RANDOM_SEED, RAW_DATA_DIR
from finguard.data.loader import (
    Example,
    RawRecord,
    file_sha256,
    load_workbook_records,
    write_jsonl,
)
from finguard.data.repair import repair_mojibake
from finguard.data.split import grouped_split, template_key
from finguard.data.statistics import compute_statistics
from finguard.data.validation import (
    ERROR,
    Issue,
    find_cross_split_duplicates,
    find_duplicate_texts,
    validate_raw_record,
)

DEFAULT_RAW_DIR = RAW_DATA_DIR / "mendeley_tzrjx692jy_v1"
TRAIN_FILE = "Training_Set.xlsx"
TEST_FILE = "Testing_Set.xlsx"

DATASET_INFO = {
    "title": "Synthetic Dataset for PII Detection and Anonymization in Financial Documents",
    "authors": ["Kushagra Mishra", "Harsh Pagare", "Ranjeet Bidwe", "Sashikala Mishra"],
    "publisher": "Mendeley Data",
    "version": 1,
    "published": "2024-10-22",
    "doi": "10.17632/tzrjx692jy.1",
    "url": "https://data.mendeley.com/datasets/tzrjx692jy/1",
    "license": "CC BY 4.0",
}

DECISIONS = [
    "Raw .xlsx files are read-only inputs and are checksummed before and after processing.",
    "Annotations are parsed with ast.literal_eval (literals only), never eval().",
    "Text repair: four mojibake sequences (UTF-8 curly quotes mis-decoded as "
    "Windows-1252) are mapped back to the original character. Annotations are "
    "not modified. The repair reads only the text, not the labels, and is "
    "applied identically to train and test.",
    "Validation: every span must be (int, int, str), use an allowed label, fit "
    "inside the text and extract exactly the value in the matching source "
    "column. Overlapping and duplicate spans are errors.",
    "Spans with leading/trailing whitespace (credit_card values end in a "
    "newline in the source data) are kept unchanged and reported as warnings; "
    "trimming is deferred to the spaCy conversion phase.",
    "Records with any error are excluded from training and listed in the report. "
    "No annotations are generated or guessed to replace them.",
    "Test set: all records must pass validation; otherwise the pipeline stops. "
    "Test records are never dropped, edited (beyond the shared text repair) or "
    "used for training or model selection.",
    "Train/validation split is made from the training file only, grouped by "
    "template so no template appears in both partitions.",
]


class OfficialTestSetError(RuntimeError):
    """The official test file has records that fail validation."""


def _audit(records: Sequence[RawRecord], repair: bool) -> dict:
    """Validate raw text, repair it, validate again. Returns per-record results."""
    raw_results = [validate_raw_record(r) for r in records]
    texts, repaired_rows, replacements = [], 0, 0
    for r in records:
        text, n = repair_mojibake(r.text) if repair else (r.text, 0)
        texts.append(text)
        repaired_rows += n > 0
        replacements += n
    final_results = [validate_raw_record(r, t) for r, t in zip(records, texts)]
    return {
        "raw": raw_results,
        "final": final_results,
        "texts": texts,
        "repaired_rows": repaired_rows,
        "replacements": replacements,
    }


def _issue_counts(issues: Sequence[Issue]) -> dict:
    by_code = Counter(i.code for i in issues)
    records = defaultdict(set)
    for i in issues:
        records[i.code].add(i.record_id)
    return {
        code: {"issues": n, "records": len(records[code])}
        for code, n in sorted(by_code.items())
    }


def _invalid_listing(issues: Sequence[Issue]) -> list[dict]:
    """One entry per invalid record: id plus its error codes and details."""
    errors = defaultdict(list)
    for i in issues:
        if i.severity == ERROR:
            errors[i.record_id].append({"code": i.code, "detail": i.detail})
    return [{"record_id": rid, "errors": errs} for rid, errs in sorted(errors.items())]


def run(
    raw_dir: Path = DEFAULT_RAW_DIR,
    out_dir: Path = PROCESSED_DATA_DIR,
    validation_fraction: float = 0.1,
    seed: int = RANDOM_SEED,
    repair: bool = True,
) -> dict:
    """Execute the pipeline and return the report dictionary."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    paths = {"train": raw_dir / TRAIN_FILE, "test": raw_dir / TEST_FILE}
    checksums_before = {k: file_sha256(p) for k, p in paths.items()}

    records = {k: load_workbook_records(p, k) for k, p in paths.items()}
    audits = {k: _audit(recs, repair) for k, recs in records.items()}

    # Record-level checks that need the whole file (or both files).
    extra = {
        "train": find_duplicate_texts(records["train"])
        + find_cross_split_duplicates(records["train"], records["test"]),
        "test": find_duplicate_texts(records["test"]),
    }

    report: dict = {
        "dataset": DATASET_INFO,
        "decisions": DECISIONS,
        "settings": {
            "repair": repair,
            "validation_fraction": validation_fraction,
            "seed": seed,
        },
        "files": {},
    }
    examples: dict[str, list[Example]] = {}

    for split in ("train", "test"):
        recs, audit = records[split], audits[split]
        raw_issues = [i for res in audit["raw"] for i in res.issues] + extra[split]
        final_issues = [i for res in audit["final"] for i in res.issues] + extra[split]
        invalid_ids = {i.record_id for i in final_issues if i.severity == ERROR}

        report["files"][split] = {
            "file": paths[split].name,
            "sha256": checksums_before[split],
            "rows": len(recs),
            "raw_stage": {
                "invalid_records": len({i.record_id for i in raw_issues if i.severity == ERROR}),
                "issues": _issue_counts(raw_issues),
                "invalid_record_list": _invalid_listing(raw_issues),
            },
            "repair": {
                "rows_repaired": audit["repaired_rows"],
                "sequences_replaced": audit["replacements"],
            },
            "final_stage": {
                "invalid_records": len(invalid_ids),
                "issues": _issue_counts(final_issues),
                "invalid_record_list": _invalid_listing(final_issues),
            },
        }

        groups_needed = split == "train"
        examples[split] = [
            Example(
                rec.record_id,
                text,
                res.spans,
                template_key(text, rec.source_values) if groups_needed else "",
            )
            for rec, res, text in zip(recs, audit["final"], audit["texts"])
            if rec.record_id not in invalid_ids
        ]

    if report["files"]["test"]["final_stage"]["invalid_records"]:
        _write_reports(report, out_dir)
        bad = [e["record_id"] for e in report["files"]["test"]["final_stage"]["invalid_record_list"]]
        raise OfficialTestSetError(
            f"{len(bad)} official test records failed validation: {bad[:10]}. "
            "See audit_report.json. The test set is not filtered automatically."
        )

    # --- train / validation from the training file only ---
    train_all = examples["train"]
    train_idx, val_idx = grouped_split(
        [ex.group for ex in train_all], validation_fraction, seed
    )
    partitions = {
        "train": [train_all[i] for i in train_idx],
        "validation": [train_all[i] for i in val_idx],
        "test": examples["test"],
    }

    # --- integrity checks before anything is written ---
    test_ids = {ex.record_id for ex in partitions["test"]}
    train_ids = {ex.record_id for ex in partitions["train"]}
    val_ids = {ex.record_id for ex in partitions["validation"]}
    assert not (train_ids | val_ids) & test_ids, "test ids leaked into train/validation"
    assert not train_ids & val_ids, "train and validation overlap"
    assert all(i.startswith("test-") for i in test_ids)
    assert len(partitions["test"]) == len(records["test"]), "test records were dropped"
    assert not ({e.group for e in partitions["train"]} & {e.group for e in partitions["validation"]})

    for name, exs in partitions.items():
        write_jsonl(exs, out_dir / f"{name}.jsonl")

    # Descriptive only: how many test documents reuse a training template.
    # Computed from text and source columns, not labels; not used for any decision.
    train_templates = {ex.group for ex in train_all}
    test_templates = [
        template_key(text, rec.source_values)
        for rec, text in zip(records["test"], audits["test"]["texts"])
    ]
    report["partitions"] = {
        name: {"file": f"{name}.jsonl", "statistics": compute_statistics(exs)}
        for name, exs in partitions.items()
    }
    report["partitions"]["train"]["templates"] = len({e.group for e in partitions["train"]})
    report["partitions"]["validation"]["templates"] = len({e.group for e in partitions["validation"]})
    report["test_template_reuse"] = {
        "test_records": len(test_templates),
        "test_records_with_template_seen_in_training_file": sum(
            t in train_templates for t in test_templates
        ),
    }

    checksums_after = {k: file_sha256(p) for k, p in paths.items()}
    report["raw_files_unchanged"] = checksums_before == checksums_after
    if not report["raw_files_unchanged"]:
        raise RuntimeError("raw dataset files changed during processing")

    _write_reports(report, out_dir)
    return report


# ----------------------------------------------------------------- reports


def _write_reports(report: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "audit_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    (out_dir / "audit_report.md").write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict, max_listed: int = 20) -> str:
    ds = report["dataset"]
    lines = [
        "# FinGuard Dataset Audit Report",
        "",
        f"**Dataset:** {ds['title']} (v{ds['version']}), {', '.join(ds['authors'])}. "
        f"{ds['publisher']}, {ds['published']}. DOI {ds['doi']}. License {ds['license']}.",
        "",
        "Generated by `python -m finguard.data.prepare`. Contains no document text.",
        "",
        "## Files",
        "",
        "| Split | File | Rows | SHA-256 |",
        "|---|---|---|---|",
    ]
    for split, f in report["files"].items():
        lines.append(f"| {split} | {f['file']} | {f['rows']} | `{f['sha256'][:16]}…` |")
    if "raw_files_unchanged" in report:
        lines += ["", f"Raw files unchanged after processing: **{report['raw_files_unchanged']}**"]

    for split, f in report["files"].items():
        for stage, title in (("raw_stage", "as downloaded"), ("final_stage", "after text repair")):
            st = f[stage]
            lines += [
                "",
                f"## {split}: validation {title}",
                "",
                f"Invalid records: **{st['invalid_records']}** of {f['rows']}",
                "",
            ]
            if st["issues"]:
                lines += ["| Issue | Issues | Records |", "|---|---|---|"]
                lines += [
                    f"| {code} | {c['issues']} | {c['records']} |"
                    for code, c in st["issues"].items()
                ]
            else:
                lines.append("No issues.")
            listing = st["invalid_record_list"]
            if listing:
                lines += ["", f"Invalid records (first {min(max_listed, len(listing))} of {len(listing)}; full list in audit_report.json):", ""]
                for entry in listing[:max_listed]:
                    codes = sorted({e["code"] for e in entry["errors"]})
                    lines.append(f"- `{entry['record_id']}`: {', '.join(codes)} ({len(entry['errors'])} errors)")
            if stage == "raw_stage":
                rep = f["repair"]
                lines += ["", f"Text repair: {rep['sequences_replaced']} mojibake sequences replaced in {rep['rows_repaired']} rows."]

    if "partitions" in report:
        lines += ["", "## Partitions", "", "| Partition | Records | Spans | Templates |", "|---|---|---|---|"]
        for name, p in report["partitions"].items():
            s = p["statistics"]
            lines.append(f"| {name} | {s['records']} | {s['total_spans']} | {p.get('templates', '-')} |")
        reuse = report["test_template_reuse"]
        lines += [
            "",
            f"Test records whose template also occurs in the training file: "
            f"{reuse['test_records_with_template_seen_in_training_file']} of {reuse['test_records']} "
            "(descriptive only).",
        ]
        for name, p in report["partitions"].items():
            s = p["statistics"]
            lines += [
                "",
                f"### Per-class statistics: {name}",
                "",
                f"Text length (chars) min/median/max: {s['text_length']['min']} / "
                f"{s['text_length']['median']} / {s['text_length']['max']}; "
                f"spans per record min/median/max: {s['spans_per_record']['min']} / "
                f"{s['spans_per_record']['median']} / {s['spans_per_record']['max']}",
                "",
                "| Class | Spans | Records | Distinct values | Span length min/median/max |",
                "|---|---|---|---|---|",
            ]
            for label, c in s["per_class"].items():
                sl = c["span_length"]
                lines.append(
                    f"| {label} | {c['spans']} | {c['records']} | {c['distinct_values']} | "
                    f"{sl['min']} / {sl['median']} / {sl['max']} |"
                )

    lines += ["", "## Preprocessing decisions", ""]
    lines += [f"{n}. {d}" for n, d in enumerate(report["decisions"], start=1)]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--no-repair", action="store_true",
        help="validate the text exactly as downloaded (for comparison)",
    )
    args = parser.parse_args(argv)
    report = run(args.raw_dir, args.out_dir, args.validation_fraction, args.seed, not args.no_repair)

    for split, f in report["files"].items():
        print(
            f"{split:5s}: {f['rows']} rows | invalid as downloaded: {f['raw_stage']['invalid_records']}"
            f" | repaired rows: {f['repair']['rows_repaired']}"
            f" | invalid after repair: {f['final_stage']['invalid_records']}"
        )
    for name, p in report["partitions"].items():
        print(f"{name:10s}: {p['statistics']['records']} records -> {args.out_dir / p['file']}")
    print(f"raw files unchanged: {report['raw_files_unchanged']}")
    print(f"report: {args.out_dir / 'audit_report.md'}")


if __name__ == "__main__":
    main()
