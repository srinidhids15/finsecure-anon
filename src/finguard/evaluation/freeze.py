"""Freeze the evaluated system before any held-out evaluation.

``python -m finguard.evaluation.freeze`` records SHA-256 hashes of every
file that determines predictions or scores (detector code, canonicalisation,
tokenizer, correction rule, metrics, runner, NER config and the trained
model) together with the selected hybrid policy, in
``configs/evaluation_freeze.json``.

Held-out evaluations (internal test, official benchmark) call
``verify_freeze()`` first and refuse to run if anything changed. A change
after freezing therefore requires a new freeze, which must be disclosed as
a new evaluation attempt.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from finguard.config import CONFIGS_DIR, PROJECT_ROOT

FREEZE_FILE = CONFIGS_DIR / "evaluation_freeze.json"

FROZEN_FILES = (
    "src/finguard/schema.py",
    "src/finguard/canonical.py",
    "src/finguard/config.py",
    "src/finguard/detection/base.py",
    "src/finguard/detection/checksums.py",
    "src/finguard/detection/rules.py",
    "src/finguard/detection/hybrid.py",
    "src/finguard/ner/tokenizer.py",
    "src/finguard/ner/inference.py",
    "src/finguard/data/corrections.py",
    "src/finguard/data/repair.py",
    "src/finguard/evaluation/metrics.py",
    "src/finguard/evaluation/run.py",
    "configs/ner.cfg",
)
MODEL_DIR = "models/finguard_ner/model-best"


class FreezeError(RuntimeError):
    """The system differs from what was frozen."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def sha256_dir(path: Path) -> str:
    """Hash of every file in a directory: relative paths and contents, sorted."""
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(path)).encode())
        h.update(sha256_file(f).encode())
    return h.hexdigest()


def current_state(root: Path = PROJECT_ROOT) -> dict:
    from finguard.detection.hybrid import DEFAULT_POLICY

    return {
        "files": {f: sha256_file(root / f) for f in FROZEN_FILES},
        "model": {MODEL_DIR: sha256_dir(root / MODEL_DIR)},
        "hybrid_policy": DEFAULT_POLICY.describe(),
    }


def freeze(path: Path = FREEZE_FILE, note: str = "") -> dict:
    state = current_state()
    record = {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": note,
        **state,
        "freeze_sha256": hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest(),
    }
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def verify_freeze(path: Path | None = None) -> dict:
    """Return the freeze record, or raise ``FreezeError`` listing differences."""
    path = Path(path) if path is not None else FREEZE_FILE
    if not path.exists():
        raise FreezeError(f"{path} not found: freeze the system before held-out evaluation")
    record = json.loads(path.read_text(encoding="utf-8"))
    state = current_state()
    diffs = []
    for section in ("files", "model"):
        for name, digest in record[section].items():
            if state[section].get(name) != digest:
                diffs.append(name)
        diffs += [n for n in state[section] if n not in record[section]]
    if state["hybrid_policy"] != record["hybrid_policy"]:
        diffs.append("hybrid_policy")
    if diffs:
        raise FreezeError(f"changed since freeze: {sorted(set(diffs))}")
    return record


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Freeze the evaluated system")
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    record = freeze(note=args.note)
    print(f"frozen {len(record['files'])} files + model; freeze_sha256 {record['freeze_sha256']}")


if __name__ == "__main__":
    main()
