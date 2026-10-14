"""Audit: do gold entity boundaries coincide with token boundaries?

For every gold span (after ``finguard.canonical``), the document is tokenised
with the intended tokenizer and the span is looked up with
``Doc.char_span(..., alignment_mode="strict")``. Strict mode returns None
unless both ends fall exactly on token boundaries. Nothing is expanded,
contracted or dropped: failures are counted and reported.

Run:  python -m finguard.ner.alignment     (writes data/processed/ner/alignment_report.*)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from spacy.language import Language

from finguard.canonical import canonical_spans
from finguard.data.loader import Example, read_jsonl


@dataclass
class AlignmentReport:
    spans: Counter = field(default_factory=Counter)  # per label
    failures: Counter = field(default_factory=Counter)  # per (label, pattern)
    examples: dict = field(default_factory=lambda: defaultdict(list))  # (label, pattern) -> ids

    @property
    def total_failures(self) -> int:
        return sum(self.failures.values())

    def to_dict(self) -> dict:
        return {
            "spans": dict(self.spans),
            "total_spans": sum(self.spans.values()),
            "total_failures": self.total_failures,
            "failures": [
                {"label": l, "pattern": p, "count": n, "examples": self.examples[(l, p)]}
                for (l, p), n in self.failures.most_common()
            ],
        }


def failure_pattern(doc, start: int, end: int) -> str:
    """Describe how a span misses token boundaries.

    ``end+'.'`` means the token holding the span's last character continues
    with "." beyond the span; ``start+'x'`` means the first token starts with
    "x" before the span.
    """
    parts = []
    for tok in doc:
        t_start, t_end = tok.idx, tok.idx + len(tok)
        if t_start < start < t_end:
            parts.append(f"start+{doc.text[t_start:start]!r}")
        if t_start < end < t_end:
            parts.append(f"end+{doc.text[end:t_end]!r}")
    return " ".join(parts) or "unknown"


def audit(nlp: Language, examples: Sequence[Example], max_examples: int = 5) -> AlignmentReport:
    report = AlignmentReport()
    docs = nlp.tokenizer.pipe(ex.text for ex in examples)
    for ex, doc in zip(examples, docs):
        for span in canonical_spans(ex.text, ex.spans):
            report.spans[span.label] += 1
            if doc.char_span(span.start, span.end, alignment_mode="strict") is None:
                key = (span.label, failure_pattern(doc, span.start, span.end))
                report.failures[key] += 1
                if len(report.examples[key]) < max_examples:
                    report.examples[key].append(f"{ex.record_id}:{span.start}-{span.end}")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    from finguard.config import PROCESSED_DATA_DIR
    from finguard.ner.tokenizer import make_nlp

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR / "ner")
    parser.add_argument("--default-tokenizer", action="store_true",
                        help="audit spaCy's unmodified English tokenizer (for comparison)")
    args = parser.parse_args(argv)

    import spacy

    nlp = spacy.blank("en") if args.default_tokenizer else make_nlp()
    result = {}
    for part in ("train", "validation", "internal_test"):
        rep = audit(nlp, read_jsonl(args.data_dir / f"{part}.jsonl"))
        result[part] = rep.to_dict()
        print(f"{part:13s}: {rep.total_failures} of {sum(rep.spans.values())} spans misaligned"
              + "".join(f"\n    {l} {p}: {n}" for (l, p), n in rep.failures.most_common()))
    name = "alignment_report_default" if args.default_tokenizer else "alignment_report"
    with open(args.data_dir / f"{name}.json", "w", encoding="utf-8") as fh:
        json.dump({"tokenizer": "spacy default en" if args.default_tokenizer else "finguard.Tokenizer.v1",
                   "partitions": result}, fh, indent=2)


if __name__ == "__main__":
    main()
