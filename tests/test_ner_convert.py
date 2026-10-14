"""Tests for converting prepared examples into spaCy training docs."""

import json

import pytest
from spacy.tokens import DocBin

from finguard.data.loader import Example, write_jsonl
from finguard.ner.convert import (
    ConversionError,
    convert_examples,
    convert_partition,
    example_to_doc,
)
from finguard.ner.tokenizer import make_nlp
from finguard.schema import Span

BLOCK = "VISA 16 digit\nAsha Rao\n4111111111111111 12/27\nCVC: 123\n"


@pytest.fixture(scope="module")
def nlp():
    return make_nlp()


def ex(text, *pieces, rid="train-000002"):
    spans = []
    for piece, label in pieces:
        start = text.index(piece)
        spans.append(Span(start, start + len(piece), label, "gold"))
    return Example(rid, text, tuple(spans))


def test_multiple_entities_convert_exactly(nlp):
    text = "Dear Asha Rao, visit https://x.com/. Simpson Inc. paid. SSN 575-74-7674."
    e = ex(text, ("Asha Rao", "name"), ("https://x.com/", "url"),
           ("Simpson Inc", "company"), ("575-74-7674", "ssn"))
    doc, failures, moved = example_to_doc(nlp, e)
    assert failures == [] and moved == 0
    assert [(x.text, x.label_) for x in doc.ents] == [
        ("Asha Rao", "name"), ("https://x.com/", "url"), ("Simpson Inc", "company"), ("575-74-7674", "ssn")
    ]
    # every other token is explicitly outside ("O"), not missing
    assert all(t.ent_iob_ in ("B", "I", "O") for t in doc)


def test_credit_card_block_is_canonicalised(nlp):
    text = f"Card ending in {BLOCK} was used."
    e = ex(text, (BLOCK, "credit_card"))
    doc, failures, moved = example_to_doc(nlp, e)
    assert failures == [] and moved == 1
    (ent,) = doc.ents
    assert ent.text == BLOCK.rstrip("\n")
    assert ent.start_char == e.spans[0].start
    assert ent.end_char == e.spans[0].end - 1


def test_misaligned_span_fails_with_diagnostic(nlp):
    e = ex("Contact AshaRao today", ("Asha", "name"), rid="train-000099")
    with pytest.raises(ConversionError) as info:
        convert_examples(nlp, [e], "train")
    (msg,) = info.value.failures
    assert "train-000099" in msg and "(8, 12, name)" in msg and "end+'Rao'" in msg


def test_unknown_label_fails(nlp):
    e = ex("Contact Asha Rao today", ("Asha Rao", "PERSON"))
    with pytest.raises(ConversionError, match="label not in ENTITY_LABELS"):
        convert_examples(nlp, [e], "train")


def test_overlapping_spans_fail(nlp):
    text = "Contact Asha Rao today"
    e = Example("r", text, (Span(8, 16, "name"), Span(13, 16, "company")))
    with pytest.raises(ConversionError):
        convert_examples(nlp, [e], "train")


def test_all_failures_are_reported_not_just_the_first(nlp):
    bad = [ex("AshaRao", ("Asha", "name"), rid=f"r{i}") for i in range(3)]
    with pytest.raises(ConversionError) as info:
        convert_examples(nlp, bad, "train")
    assert len(info.value.failures) == 3


def test_report_counts_every_label(nlp):
    text = "Asha Rao at Acme Ltd. Mail a@x.com"
    docbin, report = convert_examples(nlp, [ex(text, ("Asha Rao", "name"), ("Acme Ltd", "company"),
                                                   ("a@x.com", "email"))], "train")
    d = report.to_dict()
    assert (d["input_records"], d["output_docs"], d["input_spans"], d["output_entities"]) == (1, 1, 3, 3)
    assert d["entities"]["company"] == 1 and d["entities"]["ssn"] == 0
    assert d["rejected"] == []


def test_docbin_round_trip_keeps_entities(nlp, tmp_path):
    text = f"Dear Asha Rao, card {BLOCK}thanks."
    docbin, _ = convert_examples(nlp, [ex(text, ("Asha Rao", "name"), (BLOCK, "credit_card"))], "train")
    docbin.to_disk(tmp_path / "x.spacy")
    (doc,) = DocBin().from_disk(tmp_path / "x.spacy").get_docs(nlp.vocab)
    assert [(e.label_, e.start_char, e.end_char) for e in doc.ents] == [
        ("name", 5, 13), ("credit_card", text.index("VISA"), text.index("VISA") + len(BLOCK) - 1)
    ]


def test_convert_partition_writes_report_and_file(tmp_path, nlp):
    text = "Dear Asha Rao, SSN 575-74-7674."
    write_jsonl([ex(text, ("Asha Rao", "name"), ("575-74-7674", "ssn"))], tmp_path / "train.jsonl")
    info = convert_partition("train", tmp_path, tmp_path / "out", nlp)
    assert (tmp_path / "out" / "train.spacy").exists()
    assert info["output_docs"] == 1 and len(info["source_sha256"]) == 64


def test_failed_conversion_writes_nothing(tmp_path, nlp):
    write_jsonl([ex("AshaRao", ("Asha", "name"))], tmp_path / "validation.jsonl")
    with pytest.raises(ConversionError):
        convert_partition("validation", tmp_path, tmp_path / "out", nlp)
    assert not (tmp_path / "out" / "validation.spacy").exists()


@pytest.mark.parametrize("partition", ["internal_test", "test", "official_test"])
def test_test_partitions_are_refused(tmp_path, partition):
    with pytest.raises(ValueError, match="refusing"):
        convert_partition(partition, tmp_path, tmp_path / "out")
