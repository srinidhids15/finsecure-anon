"""Central project settings: filesystem paths and global constants.

Every module imports paths from here instead of hard-coding them, so moving a
folder means changing one line.
"""

from __future__ import annotations

from pathlib import Path

# src/finguard/config.py -> parents[2] is the repository root.
# This assumes an editable install (`pip install -e .`), which is how the
# project is meant to be developed.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"  # downloaded datasets (git-ignored)
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"  # cleaned splits (git-ignored)
PRIVATE_DATA_DIR: Path = DATA_DIR / "private"  # real documents (git-ignored)
SAMPLES_DIR: Path = DATA_DIR / "samples"  # small synthetic examples (committed)

MODELS_DIR: Path = PROJECT_ROOT / "models"  # trained spaCy models (git-ignored)

# Entity classes FinGuard detects. These match the labels used in the
# Mendeley dataset's "True Predictions" column (verified in Phase 2).
ENTITY_LABELS: tuple[str, ...] = (
    "name",
    "email",
    "phone",
    "address",
    "ssn",
    "credit_card",
    "company",
    "url",
)

# Fixed seed so data splits and synthetic generation are reproducible.
RANDOM_SEED: int = 42

# Privacy default: raw document text is never written to logs.
LOG_RAW_TEXT: bool = False
