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
