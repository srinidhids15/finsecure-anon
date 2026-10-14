"""Tests for template keys and the grouped train/validation split."""

import pytest

from finguard.data.split import grouped_split, template_key


def test_same_template_different_values_share_a_key():
    a = template_key("Dear Asha Rao, welcome to Acme Ltd.", {"name": "Asha Rao", "company": "Acme Ltd"})
    b = template_key("Dear Ben Lee, welcome to Zeta Inc.", {"name": "Ben Lee", "company": "Zeta Inc"})
    assert a == b


def test_different_templates_get_different_keys():
    a = template_key("Dear Asha Rao.", {"name": "Asha Rao"})
    b = template_key("Hello Asha Rao.", {"name": "Asha Rao"})
    assert a != b


def test_longer_values_replaced_first():
    # The name sits inside the card value; replacing the name first would
    # split the card into pieces and give two records different keys.
    a = template_key("Card VISA\nAsha Rao\n4111", {"name": "Asha Rao", "credit_card": "VISA\nAsha Rao\n4111"})
    b = template_key("Card VISA\nBen Lee\n4222", {"name": "Ben Lee", "credit_card": "VISA\nBen Lee\n4222"})
    assert a == b


def test_no_group_in_both_partitions():
    groups = [f"g{i % 7}" for i in range(70)]
    train, val = grouped_split(groups, validation_fraction=0.3, seed=1)
    assert set(train) | set(val) == set(range(70))
    assert not set(train) & set(val)
    assert not {groups[i] for i in train} & {groups[i] for i in val}


def test_split_is_reproducible():
    groups = [f"g{i % 10}" for i in range(100)]
    assert grouped_split(groups, 0.2, seed=42) == grouped_split(groups, 0.2, seed=42)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.5])
def test_bad_fraction_rejected(fraction):
    with pytest.raises(ValueError):
        grouped_split(["a", "b"], fraction)


def test_single_group_cannot_be_split():
    with pytest.raises(ValueError):
        grouped_split(["a", "a", "a"])


# ------------------------------------------------------------ NER 3-way split

from finguard.data.split import (
    cluster_templates,
    normalized_template,
    oversized_groups,
    three_way_split,
)
from finguard.schema import Span


def test_normalized_template_masks_spans_dots_digits_whitespace():
    text = "Dear Asha Rao,  invoice INV-2023 from IP 190.22.99.\nThanks."
    t = normalized_template(text, [Span(5, 13, "name")])
    assert t == "Dear <name>, invoice INV-0 from IP 0 Thanks"


def test_dot_deleted_copies_share_a_template():
    a = normalized_template("Visit 190.22.99. Bye.", [])
    b = normalized_template("Visit 19022.99 Bye", [])
    assert a == b


def test_cluster_merges_near_identical_but_not_different_templates():
    base = "This Service Contract between <company> and <name> was entered at <address> for services rendered over the year"
    variant = base.replace("was entered at", "was entered into 0 at")
    other = "Dear <name> your subscription with <company> is up for renewal next month please confirm"
    groups = cluster_templates([base, variant, other, base], threshold=0.9)
    assert groups[base] == groups[variant] != groups[other]
    assert len(set(groups.values())) == 2


def test_cluster_ids_do_not_depend_on_input_order():
    ts = ["alpha beta gamma", "delta epsilon zeta", "alpha beta gamma delta"]
    assert cluster_templates(ts) == cluster_templates(list(reversed(ts)))


def _groups(n_groups, rows_per_group=3):
    return [f"g{g:02d}" for g in range(n_groups) for _ in range(rows_per_group)]


def test_three_way_split_is_group_disjoint_and_complete():
    groups = _groups(20)
    split = three_way_split(groups, seed=7)
    all_idx = sorted(i for p in split.indices.values() for i in p)
    assert all_idx == list(range(len(groups)))
    seen = [{groups[i] for i in idx} for idx in split.indices.values()]
    assert not (seen[0] & seen[1] or seen[0] & seen[2] or seen[1] & seen[2])
    assert [len(split.groups[p]) for p in ("train", "validation", "internal_test")] == [16, 2, 2]


def test_three_way_split_is_deterministic_and_seed_dependent():
    groups = _groups(30)
    assert three_way_split(groups, seed=1) == three_way_split(groups, seed=1)
    assert three_way_split(groups, seed=1).groups != three_way_split(groups, seed=2).groups


def test_three_way_split_does_not_depend_on_row_order():
    groups = _groups(10)
    a = three_way_split(groups, seed=3).groups
    b = three_way_split(list(reversed(groups)), seed=3).groups
    assert a == b


def test_groups_never_broken_even_when_sizes_differ():
    groups = ["big"] * 50 + _groups(9, 1)
    split = three_way_split(groups, seed=0)
    holders = [p for p, gs in split.groups.items() if "big" in gs]
    assert len(holders) == 1
    assert len(split.indices[holders[0]]) >= 50


@pytest.mark.parametrize("fractions", [(0.8, 0.2), (0.8, 0.1, 0.2), (1.0, 0.0, 0.0), (0.9, -0.1, 0.2)])
def test_bad_fractions(fractions):
    with pytest.raises(ValueError):
        three_way_split(_groups(10), fractions)


def test_too_few_groups():
    with pytest.raises(ValueError):
        three_way_split(_groups(2))


def test_oversized_groups_flagged():
    groups = ["a"] * 10 + ["b"] * 10 + ["c"] * 10 + ["huge"] * 40
    assert oversized_groups(groups) == {"huge": 40}
    assert oversized_groups(_groups(5)) == {}


# ------------------------------------------- alias families + stratified split

from finguard.data.split import resolve_template_aliases, stratified_group_split

REPS = {"tA": "Dear <name> loan one", "tB": "Your loan two <name>", "tC": "Unrelated report"}
FAMILY = [{"name": "loan", "members": [{"group": "tA", "opening": "Dear <name>"},
                                       {"group": "tB", "opening": "Your loan"}], "reason": "paraphrase"}]


def test_alias_family_becomes_one_unit():
    units = resolve_template_aliases(REPS, FAMILY)
    assert units == {"tA": "f-loan", "tB": "f-loan", "tC": "tC"}


def test_no_aliases_means_one_unit_per_group():
    assert resolve_template_aliases(REPS, []) == {g: g for g in REPS}


@pytest.mark.parametrize(
    "families, message",
    [
        ([{"name": "x", "members": [{"group": "tZ", "opening": ""}]}], "unknown template group"),
        ([{"name": "x", "members": [{"group": "tA", "opening": "Hello"}]}], "no longer starts with"),
        ([{"name": "x", "members": [{"group": "tA", "opening": "Dear"}]},
          {"name": "y", "members": [{"group": "tA", "opening": "Dear"}]}], "is in families"),
    ],
)
def test_stale_or_conflicting_aliases_fail_loudly(families, message):
    with pytest.raises(ValueError, match=message):
        resolve_template_aliases(REPS, families)


LABELS = ("name", "email", "url")


def _units(sizes):
    """sizes: {unit: rows}; every unit contains every label."""
    return dict(sizes), {u: {l: n for l in LABELS} for u, n in sizes.items()}


def test_evaluation_partitions_get_high_and_low_frequency_units():
    sizes, labels = _units({**{f"big{i}": 2500 for i in range(9)}, **{f"small{i}": 1250 for i in range(18)}})
    split = stratified_group_split(sizes, labels, seed=42)
    for part in ("validation", "internal_test"):
        strata = {split.strata[u] for u in split.groups[part]}
        assert strata == {"high", "low"}, part
    all_units = [u for us in split.groups.values() for u in us]
    assert sorted(all_units) == sorted(sizes)  # every unit exactly once, never split


def test_stratified_split_is_reproducible_and_seeded():
    sizes, labels = _units({f"u{i}": (3750 if i < 9 else 1250) for i in range(14)})
    a = stratified_group_split(sizes, labels, seed=42)
    assert a == stratified_group_split(sizes, labels, seed=42)
    assert any(stratified_group_split(sizes, labels, seed=s).groups != a.groups for s in range(1, 6))


def test_small_strata_stay_in_train():
    # One huge unit is a stratum of its own (< 3 units): it cannot be shared
    # by three partitions, so it goes to train; the low stratum fills evaluation.
    sizes, labels = _units({"huge": 9000, **{f"u{i}": 1000 for i in range(6)}})
    split = stratified_group_split(sizes, labels, seed=0)
    assert split.strata["huge"] == "high"
    assert {split.strata[u] for u in sizes if u != "huge"} == {"low"}
    assert "huge" in split.groups["train"]
    assert len(split.groups["validation"]) == len(split.groups["internal_test"]) == 1


def test_equal_sizes_form_a_single_stratum():
    sizes, labels = _units({f"u{i}": 10 for i in range(10)})
    split = stratified_group_split(sizes, labels, seed=0)
    assert set(split.strata.values()) == {"high"}
    assert [len(split.groups[p]) for p in ("train", "validation", "internal_test")] == [8, 1, 1]


def test_prefers_assignments_with_every_class_everywhere():
    sizes = {f"u{i}": 100 for i in range(6)}
    labels = {u: {"name": 10} for u in sizes}
    labels["u0"] = {"name": 10, "url": 5}  # only u0 and u1 contain url
    labels["u1"] = {"name": 10, "url": 5}
    split = stratified_group_split(sizes, labels, fractions=(0.6, 0.2, 0.2), seed=3)
    # Only 2 url units: train cannot also have one, so the best achievable is
    # url in both evaluation partitions (cost includes 1.0 for train) ...
    has_url = {p: any("url" in labels[u] for u in us) for p, us in split.groups.items()}
    assert sum(has_url.values()) == 2


def test_stratified_split_rejects_bad_fractions():
    sizes, labels = _units({f"u{i}": 10 for i in range(5)})
    with pytest.raises(ValueError):
        stratified_group_split(sizes, labels, fractions=(0.9, 0.1))
