"""Convert prepared JSONL partitions into spaCy training corpora (.spacy).

For every record:
  1. tokenise the text with ``finguard.Tokenizer.v1`` (``make_nlp``);
  2. canonicalise the gold spans (``finguard.canonical``: trailing whitespace,
     i.e. the newline that ends every credit-card block, is not part of an
     entity);
  3. create each entity with ``Doc.char_span(..., alignment_mode="strict")``;
  4. set the entities, marking every other token explicitly as outside ("O").

Nothing is expanded, contracted or skipped. If any span fails (misaligned,
unknown label, overlap), conversion stops with a diagnostic listing every
failure, and no output file is written.

Only the ``train`` and ``validation`` partitions can be converted; the
internal test partition is refused, so it cannot leak into training.

Run:  python -m finguard.ner.convert
      -> data/processed/ner/spacy/{train,validation}.spacy + conversion_report.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from spacy.language import Language
from spacy.tokens import Doc, DocBin

from finguard.canonical import canonical_spans
from finguard.config import ENTITY_LABELS, PROCESSED_DATA_DIR
from finguard.data.loader import Example, file_sha256, read_jsonl
from finguard.ner.alignment import failure_pattern
from finguard.ner.tokenizer import TOKENIZER_NAME, make_nlp

CONVERTIBLE_PARTITIONS = ("train", "validation")
DEFAULT_NER_DIR = PROCESSED_DATA_DIR / "ner"
DEFAULT_OUT_DIR = DEFAULT_NER_DIR / "spacy"


class ConversionError(ValueError):
    """At least one gold span could not be converted exactly."""

    def __init__(self, failures: list[str]):
        self.failures = failures
        shown = "\n  ".join(failures[:20])
        more = f"\n  ... and {len(failures) - 20} more" if len(failures) > 20 else ""
        super().__init__(f"{len(failures)} span(s) could not be converted:\n  {shown}{more}")


@dataclass
class ConversionReport:
    partition: str
    input_records: int = 0
    output_docs: int = 0
    input_spans: int = 0
    entities: Counter = field(default_factory=Counter)
    canonicalised_spans: int = 0  # spans whose end moved (trailing whitespace)
    rejected: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "partition": self.partition,
            "input_records": self.input_records,
            "output_docs": self.output_docs,
            "input_spans": self.input_spans,
            "output_entities": sum(self.entities.values()),
            "entities": {l: self.entities[l] for l in ENTITY_LABELS},
            "canonicalised_spans": self.canonicalised_spans,
            "rejected": self.rejected,
        }


def example_to_doc(nlp: Language, ex: Example) -> tuple[Doc, list[str], int]:
    """Return ``(doc, failures, n_canonicalised)`` for one record.

    ``failures`` is empty when every span converted exactly.
    """
    doc = nlp.make_doc(ex.text)
    failures: list[str] = []
    ents = []
    canon = canonical_spans(ex.text, ex.spans)
    moved = sum(c.end != s.end for c, s in zip(canon, ex.spans))
    for span in canon:
        where = f"{ex.record_id} ({span.start}, {span.end}, {span.label})"
        if span.label not in ENTITY_LABELS:
            failures.append(f"{where}: label not in ENTITY_LABELS")
            continue
        ent = doc.char_span(span.start, span.end, label=span.label, alignment_mode="strict")
        if ent is None:
            failures.append(f"{where}: not aligned to tokens, {failure_pattern(doc, span.start, span.end)}")
            continue
        ents.append(ent)
    if not failures:
        try:
            doc.set_ents(ents, default="outside")
        except ValueError as exc:  # overlapping entities
            failures.append(f"{ex.record_id}: {exc}")
    return doc, failures, moved


def convert_examples(
    nlp: Language, examples: Sequence[Example], partition: str
) -> tuple[DocBin, ConversionReport]:
    """Convert all examples or raise ``ConversionError`` listing every failure."""
    report = ConversionReport(partition)
    docbin = DocBin(attrs=["ORTH", "ENT_IOB", "ENT_TYPE"], store_user_data=False)
    failures: list[str] = []
    for ex in examples:
        report.input_records += 1
        report.input_spans += len(ex.spans)
        doc, errs, moved = example_to_doc(nlp, ex)
        report.canonicalised_spans += moved
        if errs:
            failures.extend(errs)
            report.rejected.append(ex.record_id)
            continue
        report.entities.update(e.label_ for e in doc.ents)
        docbin.add(doc)
        report.output_docs += 1
    if failures:
        raise ConversionError(failures)
    assert report.output_docs == report.input_records
    assert sum(report.entities.values()) == report.input_spans
    return docbin, report


def convert_partition(
    partition: str, ner_dir: Path = DEFAULT_NER_DIR, out_dir: Path = DEFAULT_OUT_DIR,
    nlp: Language | None = None,
) -> dict:
    """Convert one prepared partition to ``<out_dir>/<partition>.spacy``."""
    if partition not in CONVERTIBLE_PARTITIONS:
        raise ValueError(
            f"refusing to convert {partition!r}: only {CONVERTIBLE_PARTITIONS} may be used for training"
        )
    nlp = nlp or make_nlp()
    source = Path(ner_dir) / f"{partition}.jsonl"
    docbin, report = convert_examples(nlp, read_jsonl(source), partition)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{partition}.spacy"
    docbin.to_disk(target)
    return {
        **report.to_dict(),
        "source": str(source),
        "source_sha256": file_sha256(source),
        "output": str(target),
        "output_sha256": file_sha256(target),
    }


def main(argv: Sequence[str] | None = None) -> None:
    import spacy

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ner-dir", type=Path, default=DEFAULT_NER_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    nlp = make_nlp()
    result = {
        "spacy_version": spacy.__version__,
        "tokenizer": TOKENIZER_NAME,
        "labels": list(ENTITY_LABELS),
        "partitions": {p: convert_partition(p, args.ner_dir, args.out_dir, nlp) for p in CONVERTIBLE_PARTITIONS},
    }
    with open(Path(args.out_dir) / "conversion_report.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    for p, r in result["partitions"].items():
        print(f"{p:10s}: {r['input_records']} records -> {r['output_docs']} docs | "
              f"{r['input_spans']} spans -> {r['output_entities']} entities "
              f"({r['canonicalised_spans']} canonicalised) | rejected {len(r['rejected'])}")
        print("            " + ", ".join(f"{l}={n}" for l, n in r["entities"].items()))


if __name__ == "__main__":
    main()
