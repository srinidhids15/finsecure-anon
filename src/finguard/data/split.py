"""Split the *training file* into train and validation partitions.

Why a grouped split: the dataset is generated from templates. 45,000
training rows come from about 10,000 distinct templates, and some templates
are used over 1,700 times with only the PII values changed. With a random
row split, the same sentence would appear in both partitions, and the
validation score would measure memorisation, not generalisation.

So each record gets a *template key* (its text with every inserted PII value
replaced by a placeholder), and all records sharing a key go to the same
partition.

The official test file is never passed to anything in this module.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from sklearn.model_selection import GroupShuffleSplit

from finguard.config import RANDOM_SEED


def template_key(text: str, source_values: dict[str, str]) -> str:
    """Return a short hash identifying the record's template.

    Longer values are replaced first so that a name inside a credit-card
    block, for example, cannot break up the longer value.
    """
    template = text
    for label, value in sorted(source_values.items(), key=lambda kv: -len(kv[1])):
        if value:
            template = template.replace(value, f"<{label}>")
    return hashlib.sha1(template.encode("utf-8")).hexdigest()[:12]


def grouped_split(
    groups: Sequence[str],
    validation_fraction: float = 0.1,
    seed: int = RANDOM_SEED,
) -> tuple[list[int], list[int]]:
    """Return ``(train_indices, validation_indices)`` with no shared group.

    ``validation_fraction`` is the fraction of *groups* (templates), so the
    number of validation rows is approximate.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(set(groups)) < 2:
        raise ValueError("need at least two distinct groups to split")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=validation_fraction, random_state=seed
    )
    indices = list(range(len(groups)))
    train_idx, val_idx = next(splitter.split(indices, groups=list(groups)))
    train_idx, val_idx = sorted(train_idx.tolist()), sorted(val_idx.tolist())
    # Defensive check: the guarantee this module exists for.
    assert not {groups[i] for i in train_idx} & {groups[i] for i in val_idx}
    return train_idx, val_idx
