"""Train FinGuard's spaCy NER model (config-driven, CPU).

Run (after ``python -m finguard.ner.convert``):
    python -m finguard.ner.train                       # settings from configs/ner.cfg
    python -m finguard.ner.train --max-epochs 1 --output models/try1

What it does:
  1. Pre-flight checks, failing loudly:
     * ``finguard.Tokenizer.v1`` is registered in *this* environment and the
       config uses it;
     * the corpora are exactly ``train.spacy`` (weights) and
       ``validation.spacy`` (model selection), never the internal test
       partition;
     * the corpora were converted from the current JSONL partitions
       (checksums in ``conversion_report.json``).
  2. ``spacy train`` with the config plus command-line overrides. spaCy
     evaluates on the validation corpus every ``eval_frequency`` steps and
     keeps the best-scoring pipeline (validation entity F1) as ``model-best``.
  3. Writes ``training_manifest.json`` (versions, git commit, tokenizer
     rules, labels, architecture, settings, corpus checksums, partition
     identifiers, per-evaluation losses and validation scores) into the
     output directory and into ``model-best``.

Outputs go to ``models/`` (git-ignored).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import warnings
from collections.abc import Sequence
from pathlib import Path

from finguard import __version__ as FINGUARD_VERSION
from finguard.config import CONFIGS_DIR, ENTITY_LABELS, MODELS_DIR, PROJECT_ROOT
from finguard.data.loader import file_sha256
from finguard.ner.convert import DEFAULT_NER_DIR, DEFAULT_OUT_DIR as DEFAULT_CORPUS_DIR
from finguard.ner.tokenizer import (
    EXTRA_SUFFIXES,
    REMOVED_SPECIAL_CASES,
    TOKENIZER_NAME,
)

DEFAULT_CONFIG = CONFIGS_DIR / "ner.cfg"
DEFAULT_OUTPUT = MODELS_DIR / "finguard_ner"


class PreflightError(RuntimeError):
    """Training would not be reproducible or would use the wrong data."""


def check_tokenizer_registered() -> None:
    from spacy.util import registry

    if TOKENIZER_NAME not in registry.tokenizers.get_all():
        raise PreflightError(f"{TOKENIZER_NAME} is not registered in this environment")


def check_config(config_path: Path, overrides: dict) -> dict:
    from spacy.util import load_config

    config = load_config(config_path, overrides=overrides, interpolate=True)
    tok = config["nlp"]["tokenizer"].get("@tokenizers")
    if tok != TOKENIZER_NAME:
        raise PreflightError(f"config uses tokenizer {tok!r}, expected {TOKENIZER_NAME!r}")
    if set(config["nlp"]["pipeline"]) != {"tok2vec", "ner"}:
        raise PreflightError(f"unexpected pipeline {config['nlp']['pipeline']}")
    return config


def check_corpora(corpus_dir: Path, ner_dir: Path) -> dict:
    """Corpora must be train/validation and match the current JSONL files."""
    report_path = corpus_dir / "conversion_report.json"
    if not report_path.exists():
        raise PreflightError(f"{report_path} missing; run python -m finguard.ner.convert")
    report = json.loads(report_path.read_text())
    if set(report["partitions"]) != {"train", "validation"}:
        raise PreflightError(f"corpora must be exactly train + validation, got {sorted(report['partitions'])}")
    corpora = {}
    for part, info in report["partitions"].items():
        corpus = corpus_dir / f"{part}.spacy"
        source = ner_dir / f"{part}.jsonl"
        if "internal_test" in corpus.name or "internal_test" in source.name:
            raise PreflightError("the internal test partition must not be used for training")
        if file_sha256(source) != info["source_sha256"]:
            raise PreflightError(f"{source} changed since conversion; re-run python -m finguard.ner.convert")
        if file_sha256(corpus) != info["output_sha256"]:
            raise PreflightError(f"{corpus} does not match conversion_report.json")
        corpora[part] = {
            "corpus": str(corpus), "corpus_sha256": info["output_sha256"],
            "source": str(source), "source_sha256": info["source_sha256"],
            "docs": info["output_docs"], "entities": info["entities"],
        }
    return corpora


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def read_training_log(path: Path) -> list[dict]:
    """Parse spaCy's ConsoleLogger.v3 JSONL file into evaluation records."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def partition_identifiers(ner_dir: Path) -> dict:
    split = json.loads((ner_dir / "split_report.json").read_text())
    return {
        "split_report_sha256": file_sha256(ner_dir / "split_report.json"),
        "train_units": split["partitions"]["train"]["units"],
        "validation_units": split["partitions"]["validation"]["units"],
        "seed": split["settings"]["seed"],
    }


def build_overrides(args: argparse.Namespace, corpus_dir: Path, log_path: Path) -> dict:
    overrides = {
        "paths.train": str(corpus_dir / "train.spacy"),
        "paths.dev": str(corpus_dir / "validation.spacy"),
        "paths.train_log": str(log_path),
    }
    optional = {
        "training.max_epochs": args.max_epochs,
        "training.max_steps": args.max_steps,
        "training.eval_frequency": args.eval_frequency,
        "training.patience": args.patience,
        "training.dropout": args.dropout,
        "training.optimizer.learn_rate": args.learn_rate,
        "training.batcher.size.start": args.batch_start,
        "training.batcher.size.stop": args.batch_size,
        "system.seed": args.seed,
    }
    overrides.update({k: v for k, v in optional.items() if v is not None})
    return overrides


def train(
    config_path: Path = DEFAULT_CONFIG,
    output: Path = DEFAULT_OUTPUT,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    ner_dir: Path = DEFAULT_NER_DIR,
    overrides: dict | None = None,
) -> dict:
    """Run pre-flight checks, train, and write the manifest. Returns it."""
    from spacy.cli.train import train as spacy_train

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    overrides = dict(overrides or {})
    # The corpora always come from corpus_dir (checked below); a caller may
    # not point training or model selection at any other file.
    for key, name in (("paths.train", "train.spacy"), ("paths.dev", "validation.spacy")):
        expected = str(Path(corpus_dir) / name)
        if overrides.get(key, expected) != expected:
            raise PreflightError(f"{key} must be {expected}, got {overrides[key]}")
        overrides[key] = expected
    log_path = Path(overrides.setdefault("paths.train_log", str(output / "training_log.jsonl")))
    if log_path.exists():
        log_path.unlink()  # the logger appends; start clean

    check_tokenizer_registered()
    config = check_config(config_path, overrides)
    corpora = check_corpora(Path(corpus_dir), Path(ner_dir))

    started = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spacy_train(config_path, output, use_gpu=-1, overrides=overrides)
    elapsed = time.time() - started

    manifest = summarize(
        output, config_path, config, overrides, corpora, Path(ner_dir), log_path,
        elapsed, sorted({f"{w.category.__name__}: {w.message}" for w in caught}),
    )
    return manifest


def evaluate_on_validation(nlp, dev_corpus: Path) -> dict:
    """Score a pipeline on the validation corpus (overall and per label)."""
    from spacy.training import Corpus

    examples = list(Corpus(str(dev_corpus), max_length=0)(nlp))
    scores = nlp.evaluate(examples)
    return {
        "docs": len(examples),
        "ents_p": scores["ents_p"], "ents_r": scores["ents_r"], "ents_f": scores["ents_f"],
        "ents_per_type": scores["ents_per_type"],
    }


def summarize(
    output: Path, config_path: Path, config: dict, overrides: dict, corpora: dict,
    ner_dir: Path, log_path: Path, elapsed: float | None, warnings_seen: list[str],
) -> dict:
    """Write ``training_manifest.json`` for a finished run (output + model-best)."""
    import spacy
    import thinc

    best = spacy.load(output / "model-best")
    labels = list(best.get_pipe("ner").labels)
    evaluations = [
        {
            "epoch": row.get("epoch"), "step": row.get("step"),
            "losses": row.get("losses"), "score": row.get("score"),
            **{k: row.get("scores", {}).get(k) for k in ("ents_f", "ents_p", "ents_r")},
        }
        for row in read_training_log(log_path)
    ]
    best_eval = max(evaluations, key=lambda e: e["score"] or 0.0) if evaluations else None

    manifest = {
        "finguard_version": FINGUARD_VERSION,
        "spacy_version": spacy.__version__,
        "thinc_version": thinc.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "device": "cpu",
        "git": git_state(),
        "config_file": str(config_path),
        "config_sha256": file_sha256(config_path),
        "overrides": overrides,
        "architecture": {
            "pipeline": config["nlp"]["pipeline"],
            "tok2vec_embed": dict(config["components"]["tok2vec"]["model"]["embed"]),
            "tok2vec_encode": dict(config["components"]["tok2vec"]["model"]["encode"]),
            "ner": {k: v for k, v in config["components"]["ner"]["model"].items() if k != "tok2vec"},
            "pretrained_weights": None,
            "static_vectors": None,
        },
        "training_settings": {k: config["training"][k] for k in
                              ("seed", "dropout", "max_epochs", "max_steps", "eval_frequency", "patience")}
                             | {"batch_size_words": dict(config["training"]["batcher"]["size"]),
                                "learn_rate": config["training"]["optimizer"]["learn_rate"]},
        "tokenizer": {
            "name": TOKENIZER_NAME,
            "extra_suffixes": list(EXTRA_SUFFIXES),
            "removed_special_cases": list(REMOVED_SPECIAL_CASES),
            "in_model_config": best.config["nlp"]["tokenizer"]["@tokenizers"],
        },
        "labels": labels,
        "configured_labels": list(ENTITY_LABELS),
        "corpora": corpora,
        "partitions": partition_identifiers(ner_dir),
        "evaluations": evaluations,
        "best_evaluation": best_eval,
        "model_best_validation_scores": evaluate_on_validation(best, Path(corpora["validation"]["corpus"])),
        "training_seconds": None if elapsed is None else round(elapsed, 1),
        "warnings": warnings_seen,
        "model_best": str(output / "model-best"),
        "model_last": str(output / "model-last"),
    }
    for target in (output / "training_manifest.json", output / "model-best" / "training_manifest.json"):
        target.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--ner-dir", type=Path, default=DEFAULT_NER_DIR)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--eval-frequency", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--learn-rate", type=float)
    parser.add_argument("--batch-start", type=int, help="initial batch size in words")
    parser.add_argument("--batch-size", type=int, help="final (max) batch size in words")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)

    log_path = args.output / "training_log.jsonl"
    overrides = build_overrides(args, args.corpus_dir, log_path)
    manifest = train(args.config, args.output, args.corpus_dir, args.ner_dir, overrides)
    best = manifest["best_evaluation"] or {}
    print(f"best validation ents_f={best.get('ents_f')} at step {best.get('step')} "
          f"(epoch {best.get('epoch')}); {manifest['training_seconds']} s")
    print(f"model: {manifest['model_best']}")


if __name__ == "__main__":
    main()
