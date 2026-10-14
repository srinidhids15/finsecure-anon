"""Run a trained FinGuard NER model and return ``Entity`` objects.

``NERDetector`` implements ``finguard.detection.base.Detector`` (``name`` and
``detect(text) -> list[Entity]``), the same interface as ``RuleDetector``.
It does not combine its output with the rules; that is the hybrid
detector's job (Phase 6).

Loading checks that the model uses ``finguard.Tokenizer.v1`` (spaCy finds it
through the ``spacy_tokenizers`` entry point) and that its labels are
FinGuard labels. Predictions follow the same span policy as gold labels
(``finguard.canonical``: no trailing whitespace). Offsets refer to the exact
input text.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from finguard.canonical import canonical_span
from finguard.config import ENTITY_LABELS, MODELS_DIR
from finguard.detection.base import sort_entities
from finguard.ner.tokenizer import TOKENIZER_NAME
from finguard.schema import Entity

DEFAULT_MODEL_DIR = MODELS_DIR / "finguard_ner" / "model-best"


class ModelNotFoundError(FileNotFoundError):
    pass


class NERDetector:
    """spaCy NER model behind the ``Detector`` interface."""

    name = "ner"

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_DIR) -> None:
        import spacy

        path = Path(model_path)
        if not (path / "config.cfg").exists():
            raise ModelNotFoundError(
                f"no spaCy model at {path}; train one with python -m finguard.ner.train"
            )
        self.nlp = spacy.load(path)
        tokenizer = self.nlp.config["nlp"]["tokenizer"].get("@tokenizers")
        if tokenizer != TOKENIZER_NAME:
            raise ValueError(f"model at {path} uses tokenizer {tokenizer!r}, expected {TOKENIZER_NAME!r}")
        self.labels: tuple[str, ...] = tuple(self.nlp.get_pipe("ner").labels)
        unknown = set(self.labels) - set(ENTITY_LABELS)
        if unknown:
            raise ValueError(f"model has labels outside ENTITY_LABELS: {sorted(unknown)}")
        self.model_path = path
        # Predictions consisting only of whitespace cannot be entities; they
        # are counted here rather than disappearing without a trace.
        self.whitespace_only_predictions = 0

    def _entities(self, text: str, doc) -> list[Entity]:
        out = []
        for ent in doc.ents:
            raw = Entity.from_text(text, ent.start_char, ent.end_char, ent.label_, "ner")
            canon = canonical_span(text, raw)
            if canon is None:
                self.whitespace_only_predictions += 1
                continue
            out.append(canon)
        return sort_entities(out)

    def detect(self, text: str) -> list[Entity]:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if not text.strip():
            return []
        return self._entities(text, self.nlp(text))

    def detect_many(self, texts: Iterable[str], batch_size: int = 64) -> list[list[Entity]]:
        texts = list(texts)
        for t in texts:
            if not isinstance(t, str):
                raise TypeError(f"text must be str, got {type(t).__name__}")
        return [self._entities(t, doc) for t, doc in zip(texts, self.nlp.pipe(texts, batch_size=batch_size))]


_DEFAULT: NERDetector | None = None


def detect(text: str) -> list[Entity]:
    """Detect entities with the default model (``models/finguard_ner/model-best``)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = NERDetector()
    return _DEFAULT.detect(text)
