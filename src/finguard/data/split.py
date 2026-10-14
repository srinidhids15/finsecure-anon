"""Template-aware splitting of the *training file*.

Why grouped splits: the dataset is generated from templates, with only the
PII values (and a few numbers) changing between documents. If the same
template appeared in both training and evaluation partitions, scores would
measure memorisation, not generalisation.

Two generations of template keys live here:

* ``template_key`` + ``grouped_split`` (Phase 2, two-way). Kept so Phase 2
  outputs stay reproducible. Its key is too fine-grained: generator noise
  (randomly deleted dots, random reference numbers) made 45,000 documents
  look like 10,267 templates, while there are really 27 template groups.
* ``normalized_template`` + ``cluster_templates`` + ``three_way_split``
  (NER dataset). Used for training / validation / internal test.

The official test file is never passed to anything in this module.
"""

from __future__ import annotations

import difflib
import hashlib
import math
import random
import re
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.model_selection import GroupShuffleSplit

from finguard.config import RANDOM_SEED
from finguard.schema import Span


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


# ---------------------------------------------------------------- NER split


def normalized_template(text: str, spans: Sequence[Span]) -> str:
    """Reduce a document to its template.

    Steps, in order (the order matters):
      1. replace every gold span with ``<label>``;
      2. delete every "." (the generator deletes dots at random, so
         "190.22.99" may appear as "19022.99");
      3. replace each run of digits with "0" (invoice numbers, dates, IPs);
      4. collapse whitespace.
    """
    out = text
    for span in sorted(spans, key=lambda s: s.start, reverse=True):
        out = out[: span.start] + f"<{span.label}>" + out[span.end :]
    out = out.replace(".", "")
    out = re.sub(r"[0-9]+", "0", out)
    return re.sub(r"\s+", " ", out).strip()


def template_similarity(a: str, b: str) -> float:
    """Word-level similarity in [0, 1] (difflib ratio over word lists)."""
    return difflib.SequenceMatcher(a=a.split(), b=b.split(), autojunk=False).ratio()


def cluster_templates(templates: Sequence[str], threshold: float = 0.9) -> dict[str, str]:
    """Map each distinct template to a group id.

    Templates whose word similarity is >= ``threshold`` are linked, and
    linked templates form one group (single linkage, i.e. connected
    components). This merges near-identical variants, e.g. a sentence that
    differs only by an added date. On the training workbook the pairwise
    similarities are 0.986 and 0.984 for the two variant pairs, then at most
    0.746, so 0.9 falls in a clear gap.

    The group id is a hash of the group's lexicographically first template,
    so it does not depend on input order.
    """
    distinct = sorted(set(templates))
    parent = list(range(len(distinct)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            if root(i) != root(j) and template_similarity(distinct[i], distinct[j]) >= threshold:
                parent[root(j)] = root(i)

    members: dict[int, list[str]] = {}
    for i, t in enumerate(distinct):
        members.setdefault(root(i), []).append(t)
    group_of = {}
    for group in members.values():
        gid = "t" + hashlib.sha1(min(group).encode("utf-8")).hexdigest()[:10]
        for t in group:
            group_of[t] = gid
    return group_of


PARTITIONS = ("train", "validation", "internal_test")


@dataclass(frozen=True)
class ThreeWaySplit:
    """Indices per partition plus the group assignment that produced them."""

    indices: dict[str, list[int]]
    groups: dict[str, list[str]]  # partition -> sorted group ids


def three_way_split(
    groups: Sequence[str],
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = RANDOM_SEED,
) -> ThreeWaySplit:
    """Assign whole groups to train / validation / internal_test.

    Algorithm (documented so it can be reproduced by hand):
      1. take the distinct group ids and sort them;
      2. shuffle that list with ``random.Random(seed)``;
      3. the first ``round(n * f_internal_test)`` groups (at least 1) go to
         internal_test, the next ``round(n * f_validation)`` (at least 1) go
         to validation, and the rest to train.

    Fractions apply to *groups*, never rows; a group is never broken up, so
    row shares follow the sizes of the groups drawn.
    """
    if len(fractions) != 3 or any(f <= 0 for f in fractions) or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("fractions must be three positive numbers summing to 1")
    ordered = sorted(set(groups))
    n = len(ordered)
    n_test = max(1, round(n * fractions[2]))
    n_val = max(1, round(n * fractions[1]))
    if n - n_test - n_val < 1:
        raise ValueError(f"need more groups: {n} groups cannot fill three partitions")
    random.Random(seed).shuffle(ordered)
    assigned = {
        "internal_test": sorted(ordered[:n_test]),
        "validation": sorted(ordered[n_test : n_test + n_val]),
        "train": sorted(ordered[n_test + n_val :]),
    }
    partition_of = {g: p for p, gs in assigned.items() for g in gs}
    indices: dict[str, list[int]] = {p: [] for p in PARTITIONS}
    for i, g in enumerate(groups):
        indices[partition_of[g]].append(i)
    return ThreeWaySplit(indices, {p: assigned[p] for p in PARTITIONS})


def oversized_groups(groups: Sequence[str], factor: float = 1.5) -> dict[str, int]:
    """Groups with more than ``factor`` x the median group size.

    One such group landing in a small partition shifts that partition's row
    share far from its template share, so these are reported with the split.
    """
    sizes = Counter(groups)
    median = statistics.median(sizes.values())
    return {g: n for g, n in sorted(sizes.items()) if n > factor * median}


# --------------------------------------------- alias families + stratified split


def resolve_template_aliases(
    representative: dict[str, str], families: Sequence[dict]
) -> dict[str, str]:
    """Map every template group to its split unit.

    ``representative`` maps group id -> the group's representative normalised
    template. ``families`` comes from ``configs/template_aliases.json``: each
    family lists member groups with the opening text they must start with.
    Members of a family share the unit id ``f-<family name>``; every other
    group is its own unit (unit id = group id).

    Raises ValueError if a member id is unknown, its opening does not match,
    or a group appears in two families, so a stale alias file fails loudly
    instead of silently not merging.
    """
    unit = {g: g for g in representative}
    seen: dict[str, str] = {}
    for family in families:
        name = family["name"]
        for member in family["members"]:
            gid = member["group"]
            if gid not in representative:
                raise ValueError(f"alias family {name!r}: unknown template group {gid!r}")
            if not representative[gid].startswith(member["opening"]):
                raise ValueError(
                    f"alias family {name!r}: group {gid!r} no longer starts with {member['opening']!r}"
                )
            if gid in seen:
                raise ValueError(f"group {gid!r} is in families {seen[gid]!r} and {name!r}")
            seen[gid] = name
            unit[gid] = f"f-{name}"
    return unit


@dataclass(frozen=True)
class StratifiedSplit:
    groups: dict[str, list[str]]  # partition -> sorted unit ids
    strata: dict[str, str]  # unit -> "high" | "low"
    cost: float
    candidates_evaluated: int


def _size_strata(unit_sizes: dict[str, int]) -> dict[str, str]:
    """Units at least the geometric mean of the smallest and largest unit size
    are "high" frequency.

    (The median is not used: when half the units share the smallest size,
    the median equals it and every unit would count as "high".)
    """
    threshold = math.sqrt(min(unit_sizes.values()) * max(unit_sizes.values()))
    return {u: "high" if n >= threshold else "low" for u, n in unit_sizes.items()}


def _split_cost(
    assignment: dict[str, list[str]],
    unit_sizes: dict[str, int],
    unit_label_counts: dict[str, dict[str, int]],
    targets: dict[str, float],
) -> float:
    total_rows = sum(unit_sizes.values())
    overall: Counter = Counter()
    for counts in unit_label_counts.values():
        overall.update(counts)
    overall_total = sum(overall.values())
    labels = sorted(overall)

    cost = 0.0
    for part, units in assignment.items():
        rows = sum(unit_sizes[u] for u in units)
        cost += abs(rows / total_rows - targets[part])
        counts: Counter = Counter()
        for u in units:
            counts.update(unit_label_counts[u])
        # Each class missing from a partition costs 1.0, more than any
        # row-share or distribution difference can, so such candidates lose
        # whenever a complete one exists.
        cost += float(sum(counts[l] == 0 for l in labels))
        if part != "train":
            part_total = sum(counts.values())
            cost += 0.5 * sum(abs(counts[l] / part_total - overall[l] / overall_total) for l in labels)
    return cost


def stratified_group_split(
    unit_sizes: dict[str, int],
    unit_label_counts: dict[str, dict[str, int]],
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = RANDOM_SEED,
    n_candidates: int = 1000,
) -> StratifiedSplit:
    """Assign whole split units to train / validation / internal_test.

    Algorithm (deterministic for a given seed):
      1. Strata: units with at least sqrt(smallest * largest unit size) rows
         are "high" frequency, the rest "low" (all "high" if sizes are equal).
      2. Quotas: for each stratum with n >= 3 units, each evaluation
         partition p gets max(1, round(f_p * n)) units, so both evaluation
         partitions contain high- *and* low-frequency templates. Strata with
         fewer than 3 units go entirely to train.
      3. Candidates: ``random.Random(seed)`` draws ``n_candidates`` stratified
         assignments (shuffle each stratum's sorted units, fill internal_test,
         then validation, then train).
      4. Each candidate is scored, lower is better:
           sum over partitions |row share - target fraction|
         + for validation and internal_test, the total-variation distance
           between their entity-class distribution and the overall one.
         + 1.0 for every (partition, entity class) pair with no examples,
           which outweighs everything else, so a candidate with every class
           everywhere is preferred whenever one exists.
         The first candidate with the lowest cost wins.
    """
    if len(fractions) != 3 or any(f <= 0 for f in fractions) or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError("fractions must be three positive numbers summing to 1")
    targets = dict(zip(PARTITIONS, fractions))
    strata = _size_strata(unit_sizes)
    by_stratum: dict[str, list[str]] = {}
    for u in sorted(unit_sizes):
        by_stratum.setdefault(strata[u], []).append(u)

    quotas = {}
    for name, units in by_stratum.items():
        n = len(units)
        quotas[name] = (
            (max(1, round(fractions[2] * n)), max(1, round(fractions[1] * n))) if n >= 3 else (0, 0)
        )
    if not any(q[0] for q in quotas.values()):
        raise ValueError("need at least 3 units in some stratum to fill three partitions")

    rng = random.Random(seed)
    best: dict[str, list[str]] = {}
    best_cost = float("inf")
    for _ in range(n_candidates):
        assignment = {p: [] for p in PARTITIONS}
        for name in sorted(by_stratum):
            units = list(by_stratum[name])
            rng.shuffle(units)
            n_test, n_val = quotas[name]
            assignment["internal_test"] += units[:n_test]
            assignment["validation"] += units[n_test : n_test + n_val]
            assignment["train"] += units[n_test + n_val :]
        cost = _split_cost(assignment, unit_sizes, unit_label_counts, targets)
        if cost < best_cost:
            best, best_cost = assignment, cost
    return StratifiedSplit({p: sorted(best[p]) for p in PARTITIONS}, strata, round(best_cost, 6), n_candidates)
