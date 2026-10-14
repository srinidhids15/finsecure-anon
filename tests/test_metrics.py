"""Hand-calculated tests for exact-span evaluation."""

import pytest

from finguard.evaluation.metrics import EvalDocument, evaluate, prf, Counts
from finguard.schema import Entity, Span

TEXT = "Asha Rao paid Acme Ltd via a@x.com today"
#       0       8     14      22  26  30     37


def S(s, e, label):
    return Span(s, e, label)


GOLD = [S(0, 8, "name"), S(14, 22, "company"), S(27, 34, "email")]


def test_hand_calculated_example():
    pred = [
        S(0, 8, "name"),        # TP
        S(14, 18, "company"),   # boundary error: FP + FN for company
        S(27, 34, "email"),     # TP
        S(23, 26, "phone"),     # spurious FP ("via")
    ]
    r = evaluate([EvalDocument("d1", TEXT, GOLD, pred)])
    # micro: TP=2, FP=2, FN=1 -> P=2/4, R=2/3, F1=2PR/(P+R)=4/7
    assert (r.micro["tp"], r.micro["fp"], r.micro["fn"]) == (2, 2, 1)
    assert r.micro["precision"] == pytest.approx(0.5)
    assert r.micro["recall"] == pytest.approx(2 / 3)
    assert r.micro["f1"] == pytest.approx(4 / 7)
    # per label
    assert r.per_label["name"]["f1"] == 1.0
    assert (r.per_label["company"]["precision"], r.per_label["company"]["recall"]) == (0.0, 0.0)
    assert r.per_label["phone"]["precision"] == 0.0
    assert r.per_label["phone"]["recall"] is None          # no phone gold
    assert r.per_label["phone"]["f1"] == 0.0                # one side undefined -> 0
    assert r.per_label["ssn"]["f1"] is None                 # absent label
    # macro over present labels {name, company, email, phone}
    # P = (1 + 0 + 1 + 0)/4, R = (1 + 0 + 1 + 0[undefined->0])/4, F1 = (1+0+1+0)/4
    assert r.macro["labels_averaged"] == 4
    assert r.macro["precision"] == pytest.approx(0.5)
    assert r.macro["recall"] == pytest.approx(0.5)
    assert r.macro["f1"] == pytest.approx(0.5)
    assert set(r.absent_labels) == {"address", "ssn", "credit_card", "url"}
    # error typing
    kinds = sorted((e.kind, e.label, e.error_type) for e in r.errors)
    assert kinds == [("fn", "company", "boundary"), ("fp", "company", "boundary"), ("fp", "phone", "spurious")]


def test_wrong_label_same_offsets_is_not_a_match():
    r = evaluate([EvalDocument("d", TEXT, [S(0, 8, "name")], [S(0, 8, "company")])])
    assert (r.micro["tp"], r.micro["fp"], r.micro["fn"]) == (0, 1, 1)
    assert {(e.kind, e.error_type) for e in r.errors} == {("fp", "label"), ("fn", "label")}


def test_off_by_one_offset_is_not_a_match():
    r = evaluate([EvalDocument("d", TEXT, [S(0, 8, "name")], [S(0, 7, "name")])])
    assert r.micro["tp"] == 0 and r.micro["f1"] == 0.0


def test_duplicate_predictions_count_as_false_positives():
    r = evaluate([EvalDocument("d", TEXT, [S(0, 8, "name")], [S(0, 8, "name"), S(0, 8, "name")])])
    assert (r.micro["tp"], r.micro["fp"], r.micro["fn"]) == (1, 1, 0)
    assert r.micro["precision"] == 0.5 and r.micro["recall"] == 1.0


def test_canonicalisation_applies_to_gold_and_predictions():
    block = "VISA 16 digit\nAsha Rao\n4111111111111111 12/27\nCVC: 123\n"
    text = "Card " + block + "was used"
    gold = [S(5, 5 + len(block), "credit_card")]            # ends with "\n"
    pred_trimmed = [S(5, 5 + len(block) - 1, "credit_card")]  # model-style, no "\n"
    rule_style = [Entity.from_text(text, 5, 5 + len(block), "credit_card", "rule:credit_card_block")]
    for pred in (pred_trimmed, rule_style):
        r = evaluate([EvalDocument("d", text, gold, pred)])
        assert r.micro["tp"] == 1 and r.micro["f1"] == 1.0


def test_empty_inputs():
    r = evaluate([])
    assert r.documents == 0
    assert r.micro["precision"] is None and r.micro["recall"] is None and r.micro["f1"] is None
    assert r.macro["f1"] is None and r.macro["labels_averaged"] == 0
    r = evaluate([EvalDocument("d", "nothing here", [], [])])
    assert r.micro["f1"] is None and r.errors == []


def test_no_predictions_but_gold():
    r = evaluate([EvalDocument("d", TEXT, GOLD, [])])
    assert r.micro["precision"] is None and r.micro["recall"] == 0.0 and r.micro["f1"] == 0.0
    assert {e.error_type for e in r.errors} == {"missed"}


def test_predictions_but_no_gold():
    r = evaluate([EvalDocument("d", TEXT, [], [S(0, 8, "name")])])
    assert r.micro["precision"] == 0.0 and r.micro["recall"] is None and r.micro["f1"] == 0.0


def test_counts_are_summed_over_documents_not_averaged():
    d1 = EvalDocument("d1", TEXT, [S(0, 8, "name")], [S(0, 8, "name")])          # 1 TP
    d2 = EvalDocument("d2", TEXT, [S(0, 8, "name"), S(27, 34, "email")], [])     # 2 FN
    r = evaluate([d1, d2])
    assert r.micro["recall"] == pytest.approx(1 / 3)  # not mean(1.0, 0.0) = 0.5


def test_per_group_and_group_macro():
    d1 = EvalDocument("d1", TEXT, [S(0, 8, "name")], [S(0, 8, "name")], group="A")
    d2 = EvalDocument("d2", TEXT, [S(0, 8, "name")], [], group="B")
    d3 = EvalDocument("d3", TEXT, [S(0, 8, "name")], [], group="B")
    r = evaluate([d1, d2, d3])
    assert r.per_group["A"]["micro"]["f1"] == 1.0
    assert r.per_group["B"]["micro"]["f1"] == 0.0
    assert r.per_group["B"]["documents"] == 2
    assert r.group_macro["f1"] == 0.5            # each group counts once
    assert r.micro["f1"] == pytest.approx(0.5)   # record-weighted: P=1, R=1/3 -> 0.5


def test_unknown_label_rejected():
    with pytest.raises(ValueError):
        evaluate([EvalDocument("d", TEXT, [S(0, 8, "PERSON")], [])])


def test_prf_zero_tp():
    assert prf(Counts(0, 3, 4))["f1"] == 0.0
