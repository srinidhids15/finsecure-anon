"""Smoke tests: the package and every planned module import cleanly."""

import importlib

import pytest

import finguard
from finguard import config

MODULES = [
    "finguard.schema",
    "finguard.config",
    "finguard.canonical",
    "finguard.data.annotations",
    "finguard.data.build_ner_dataset",
    "finguard.data.corrections",
    "finguard.data.loader",
    "finguard.data.prepare",
    "finguard.data.repair",
    "finguard.data.split",
    "finguard.data.statistics",
    "finguard.data.validation",
    "finguard.detection.base",
    "finguard.detection.checksums",
    "finguard.detection.rules",
    "finguard.detection.hybrid",
    "finguard.ner.train",
    "finguard.ner.alignment",
    "finguard.ner.convert",
    "finguard.ner.inference",
    "finguard.ner.tokenizer",
    "finguard.evaluation.freeze",
    "finguard.evaluation.metrics",
    "finguard.evaluation.run",
    "finguard.anonymization",
    "finguard.anonymization.anonymizer",
    "finguard.anonymization.errors",
    "finguard.anonymization.placeholders",
    "finguard.anonymization.result",
    "finguard.anonymization.verify",
    "finguard.pdf.extract",
    "finguard.pdf.redact",
]


def test_version_is_set():
    assert finguard.__version__ == "0.1.0"


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_config_paths_point_inside_project():
    assert (config.PROJECT_ROOT / "pyproject.toml").exists()
    for path in (config.RAW_DATA_DIR, config.PRIVATE_DATA_DIR, config.MODELS_DIR):
        assert config.PROJECT_ROOT in path.parents


def test_raw_text_logging_is_off_by_default():
    assert config.LOG_RAW_TEXT is False
