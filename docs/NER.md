# spaCy NER Model

FinGuard's statistical detector: a spaCy 3.8 pipeline (`tok2vec` + `ner`)
trained from scratch on CPU for the eight PII labels
`name, email, phone, address, ssn, credit_card, company, url`.

## Workflow

```bash
python -m finguard.data.prepare            # Phase 2 (once)
python -m finguard.data.build_ner_dataset  # corrected, template-disjoint partitions
python -m finguard.ner.alignment           # optional: confirm 0 misaligned spans
python -m finguard.ner.convert             # JSONL -> DocBin (.spacy) for train + validation
python -m finguard.ner.train               # train; best model selected on validation
```

| Step | Reads | Writes (git-ignored) |
|---|---|---|
| `ner.convert` | `data/processed/ner/{train,validation}.jsonl` | `data/processed/ner/spacy/{train,validation}.spacy`, `conversion_report.json` |
| `ner.train` | `configs/ner.cfg`, the two `.spacy` corpora, `conversion_report.json`, `split_report.json` | `models/finguard_ner/{model-best,model-last}`, `training_log.jsonl`, `training_manifest.json` |

`internal_test.jsonl` and the official test workbook are never read by
either step. `ner.convert` refuses any partition other than `train` and
`validation`, and `ner.train` refuses corpus paths other than those two files.

## Conversion (`finguard.ner.convert`)

For each JSONL record:

1. Tokenise with `finguard.Tokenizer.v1` (`make_nlp()`).
2. Canonicalise gold spans with `finguard.canonical`: trailing whitespace
   is not part of an entity. Only the credit-card blocks' final `\n` is
   affected: 33,750 train and 5,000 validation spans.
3. `Doc.char_span(start, end, label, alignment_mode="strict")` for each
   span; no `expand`/`contract`.
4. `doc.set_ents(entities, default="outside")`: every other token is `O`.

Any misaligned span, unknown label or overlap stops the conversion. Every
failure is listed (record id, offsets, label, failure pattern), and no file
is written. The report checks that output entities equal input spans.

| Partition | Records → docs | Gold spans → entities | Rejected |
|---|---|---|---|
| train | 35,000 → 35,000 | 327,500 → 327,500 | 0 |
| validation | 5,000 → 5,000 | 46,250 → 46,250 | 0 |

Entities per class:

| Partition | name | email | phone | address | ssn | credit_card | company | url |
|---|---|---|---|---|---|---|---|---|
| train | 62,500 | 35,000 | 35,000 | 38,750 | 41,250 | 33,750 | 46,250 | 35,000 |
| validation | 8,750 | 5,000 | 5,000 | 5,000 | 5,000 | 5,000 | 7,500 | 5,000 |

`spacy debug data` on these corpora: 7 checks passed (8 labels, 0 missing
values, no entity starting or ending with whitespace, no train/dev overlap).

At training time spaCy re-tokenises each text with the *pipeline's*
tokenizer and aligns it to the stored reference docs. That is why
`configs/ner.cfg` names `finguard.Tokenizer.v1` too, and `ner.train`
refuses a config that does not.

## Architecture (`configs/ner.cfg`)

Based on `spacy init config --lang en --pipeline ner --optimize efficiency`;
changes are marked `FinGuard:` in the file.

| Part | Setting | Why |
|---|---|---|
| tokenizer | `finguard.Tokenizer.v1` | same tokens as the converted corpora (0 misaligned spans) |
| tok2vec embed | `MultiHashEmbed.v2`, attrs `NORM, PREFIX, SUFFIX, SHAPE`, no static vectors | PII here is format-heavy: `SHAPE` sees `ddd-dd-dddd`, affixes see `@`, `https`, `Inc` |
| tok2vec encode | `MaxoutWindowEncoder.v2`, width 96, depth 4, window 1 | spaCy's CPU-efficient default; context of ±4 tokens |
| ner | `TransitionBasedParser.v2`, hidden width 64 | handles long multi-token entities (card blocks, addresses) |
| pretrained weights | **none** | nothing to download or license; trains from the data alone |

A transformer or `en_core_web_*` vectors would need an extra download and
GPU time. They are a possible later comparison, not a requirement for this
corpus.

Training settings (all overridable from the command line):

| Setting | Default | CLI flag |
|---|---|---|
| seed | 42 | `--seed` |
| dropout | 0.1 | `--dropout` |
| learn rate (Adam) | 0.001 | `--learn-rate` |
| batch size (words, compounding 100 → 1000) | start 100, stop 1000 | `--batch-start`, `--batch-size` |
| max epochs | 2 | `--max-epochs` |
| max steps | 0 (no limit) | `--max-steps` |
| evaluate on validation every | 500 steps | `--eval-frequency` |
| early stop after no improvement for | 2,000 steps | `--patience` |

Model selection: spaCy evaluates on the validation corpus every
`eval_frequency` steps and saves the best-scoring pipeline (micro entity
F1, exact span and label) as `model-best`; `model-last` is the final state.
Execution is CPU only (`gpu_allocator = null`, `use_gpu=-1`).

## Provenance (`training_manifest.json`)

Written to the output directory and into `model-best/`: FinGuard, spaCy,
Thinc and Python versions; platform; git commit and dirty flag; config
checksum and overrides; architecture; training settings; tokenizer name and
rules; model labels; corpus and source-JSONL checksums with doc/entity
counts; the split report checksum and the train/validation split units;
every logged evaluation (step, epoch, losses, validation P/R/F); the best
evaluation; a per-label re-evaluation of `model-best` on validation;
training time; and warnings raised during training.

## Inference (`finguard.ner.inference`)

```python
from finguard.ner.inference import NERDetector

det = NERDetector()                     # models/finguard_ner/model-best
det.detect("Email asha@example.com")    # [Entity(start=6, end=22, label='email', source='ner', ...)]
det.detect_many(texts)                  # batched
```

`NERDetector` implements `finguard.detection.base.Detector` (like
`RuleDetector`). On load it checks the model's tokenizer is
`finguard.Tokenizer.v1` and its labels are FinGuard labels. `spacy.load`
finds the tokenizer through the `spacy_tokenizers` entry point, so no
explicit import is needed. Predictions are canonicalised like gold spans
(no trailing whitespace); a whitespace-only prediction is counted in
`whitespace_only_predictions` instead of being returned. Empty or
whitespace-only input returns `[]`. Rule and model outputs are not
combined here (Phase 6).

## Training run (2026-10-15, `models/finguard_ner/`)

Command: `python -m finguard.ner.train` (defaults from `configs/ner.cfg`,
seed 42, CPU). Recorded in `training_manifest.json`: spaCy 3.8.16, Thinc 8.3.13,
Python 3.11.17, git commit `a4b7931` **with
uncommitted changes** (this phase's code). Wall time 783 s on one CPU core;
no warnings were raised.

Logged evaluations (validation partition; losses are summed over the
preceding 500 steps):

| Step | Epoch | Loss tok2vec | Loss NER | Val. P | Val. R | Val. F1 |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.00 | 146.32 | 0.00 | 0.00 | 0.00 |
| 500 | 0 | 5207.99 | 7329.40 | 82.75 | 95.47 | 88.66 |
| 1000 | 0 | 1455.70 | 226.87 | 88.78 | 97.56 | 92.96 |
| 1500 | 0 | 106.95 | 69.14 | 98.82 | 96.27 | 97.53 |
| **2000** | 0 | 309.66 | 79.35 | **99.08** | **96.80** | **97.93** |
| 2500 | 0 | 364.29 | 112.36 | 98.93 | 91.79 | 95.23 |
| 3000 | 0 | 129.64 | 29.90 | 89.66 | 99.35 | 94.26 |
| 3500 | 0 | 275.55 | 66.46 | 91.03 | 96.81 | 93.83 |
| 4000 | 0 | 1200.49 | 81.49 | 96.39 | 96.58 | 96.48 |

Training stopped at step 4,000, about half of the first epoch, because
validation F1 had not improved for 2,000 steps (`patience`). `model-best` is
the step-2,000 checkpoint. Re-evaluating it on validation reproduces
P 0.9908 / R 0.9680 / F1 0.9793.

Per label (`model-best`, validation, exact span and label):

| Label | P | R | F1 |
|---|---|---|---|
| name | 0.9638 | 0.9864 | 0.9750 |
| email | 1.0000 | 0.9990 | 0.9995 |
| phone | 1.0000 | 1.0000 | 1.0000 |
| address | 0.9905 | 0.9996 | 0.9950 |
| ssn | 1.0000 | 1.0000 | 1.0000 |
| credit_card | 0.9993 | **0.8128** | 0.8964 |
| company | 0.9948 | 0.9455 | 0.9695 |
| url | 0.9994 | 0.9984 | 0.9989 |

Credit-card errors on validation (5,000 gold blocks): 4,064 exact; 660
not predicted at all; 225 where only the holder line inside the block was
predicted as `name`; 48 with an `address` and/or `name` prediction inside
the block; 3 with a partial `credit_card` span.

These are validation scores. The validation partition was used for
checkpoint selection, so they are optimistic. They cover only two split
units (insurance claim, internal-controls review). No internal-test or
official-test scores have been computed.

## Known limitations

- Validation F1 fluctuates between checkpoints (0.94 to 0.98 after step
  1,500), and training stopped after half an epoch. Parameter averaging
  (`use_averages = true`), a lower learning rate or longer patience may give
  a steadier model. Any such change must again be selected on validation
  only.
- Credit-card recall is the weakest result (0.81): the model sometimes
  labels the holder name inside a block as `name`, against the dataset's
  definition (the holder belongs to the block). The rule detector matches
  these blocks exactly, which is relevant for the hybrid detector (Phase 6).
- The training data comes from 21 template groups of synthetic text.
  Performance on real documents is not measured and will be lower.
- Training is single-threaded on CPU. Results are reproducible with the
  same seed, data and software versions; they were not re-run to measure
  run-to-run variation.
