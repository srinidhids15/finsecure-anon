# Evaluation and Hybrid Detection

## How to reproduce

```bash
python -m finguard.evaluation.run --dataset validation      # development: rules, NER, 3 hybrid variants
python -m finguard.evaluation.freeze --note "..."           # freeze before any held-out run
python -m finguard.evaluation.run --dataset internal_test   # held out; refuses unless the freeze verifies
python -m finguard.evaluation.run --dataset official_test   # held out; refuses unless the freeze verifies
```

Reports (git-ignored, reproducible) go to `reports/evaluation/<dataset>/`:
`metrics.json` (everything, including identifiers), `per_label.csv`,
`per_group.csv`, `errors.csv` (every FP/FN with type and text) and
`report.md`.

## Metric definitions (`finguard.evaluation.metrics`)

* **Canonicalisation:** gold and predictions both pass through
  `finguard.canonical` (trailing whitespace removed from span ends) before
  comparison. Only the credit-card block's final `\n` is affected. Stored
  annotations are unchanged.
* **Exact match:** a true positive needs identical `(start, end, label)`.
  Different offsets or a different label count as an FP plus an FN.
* **Multisets:** a prediction emitted twice is one TP plus one FP.
* **Zero denominators:** precision is undefined (`null`) with no
  predictions, recall with no gold. F1 is `null` only if both are
  undefined; if exactly one is, F1 = 0.
* **Micro:** TP/FP/FN summed over labels and documents (record-weighted).
  **Macro:** unweighted mean over labels with at least one gold or predicted
  entity; undefined P/R count as 0; labels with neither are listed as
  absent. **Group-level:** micro scores per split unit and per template
  group, plus their unweighted mean.
* **Error types:** FP = `boundary` (overlaps gold of the same label),
  `label` (overlaps gold of another label), `spurious`; FN = `boundary`,
  `label`, `missed`.
* Every system is scored by the same `evaluate()` on the same records.

Hand-calculated unit tests: `tests/test_metrics.py`.

## Hybrid policy (`finguard.detection.hybrid`)

Rules and NER run on the same original text; `merge()` combines them
deterministically, using no gold data and no confidence scores:

1. **Deduplicate** identical canonical `(start, end, label)` spans. The
   rule entity's offsets are kept (a card block keeps its original trailing
   newline) and sources are joined, e.g. `rule:credit_card_block|ner`.
2. **Credit-card block precedence:** a block validated by
   `rule:credit_card_block` suppresses every candidate overlapping it
   (holder `name`, bare number, address fragments). Reason
   `inside_credit_card_block`.
3. **Rule priority** for `credit_card, email, phone, ssn, url`: a rule
   entity with one of these labels beats an overlapping NER entity. Reason
   `rule_priority_label`.
4. **Otherwise NER wins** (`ner_preferred`); between rule entities the
   longer span wins (`longer_span`), then the earlier start.

Non-overlapping candidates from both detectors are all kept, and the output
never overlaps. Every suppressed candidate is returned in
`HybridResult.suppressed` with its winner and reason; the rule detector's
own suppressions are passed through in `rule_suppressed`. Tests:
`tests/test_hybrid.py`.

## Policy selection (validation only)

Three variants were declared in code (`run.POLICY_VARIANTS`) before any
validation result was seen, and scored on the same 5,000 validation records:

| Variant | Card precedence | Rule-priority labels | Micro F1 | TP / FP / FN |
|---|---|---|---|---|
| `ner_priority` | no | none | 0.9867 | 45,440 / 415 / 810 |
| `card_block_only` | yes | none | 0.9928 | 45,716 / 130 / 534 |
| `rule_priority_structured` (**selected**) | yes | all five structured | 0.9928 | 45,717 / 129 / 533 |

Decision: card-block precedence is the decisive change. It raises
validation credit-card recall from 0.8128 (NER) to 1.0000 and removes all
234 `name` predictions inside card blocks. Rule priority for structured
labels changes one entity on validation. It was kept because the rule
detector's validation precision is 1.0000 on every structured label (0 FP),
which is the evidence the policy rests on. Nothing was tuned towards a
target score.

## Freeze

After selection the system was frozen with `python -m finguard.evaluation.freeze`
(`configs/evaluation_freeze.json`, committed). It holds SHA-256 hashes of 14
source/config files (detectors, canonicalisation, tokenizer, correction
rule, mojibake repair, metrics, runner, `configs/ner.cfg`), of the model
directory, and of the selected policy:

* `freeze_sha256`: `ff731aeddfc59e2783e226f7ee47f586c1524adb4e2bb5ff731742f03e575392`
* frozen at `2026-10-14T20:40:08+00:00` (UTC)
* model `models/finguard_ner/model-best`: `f483a1d8485a9b6a4fc939f6fcf67c67df03b3c062fe0b45a5a2ad589bd6bf4f`
  (config `366b36dd…`, best step 2000, trained at commit `a4b7931` + uncommitted Phase 4 code)

Held-out runs call `verify_freeze()` first. Each held-out dataset was
evaluated **once**, after the freeze; nothing was changed afterwards.

Data identifiers: validation JSONL `18e4ac01…`, internal test JSONL
`e34761a2…`, split report `589679df…`, official `test.jsonl` `86f59c26…`,
official workbook `abf209b3…` (unchanged before and after).

## Official benchmark: annotation correction

The frozen `dot_dropped_variant` rule (the same code as for the training
workbook, designed in the pre-Phase 4 checkpoint without the test file)
was applied in memory to an evaluation copy. It uses the text and the
workbook's source columns only. It added **7 spans** (5 email, 2 url) to
the 257 original gold spans, with 0 ambiguous cases. Corrected annotations
are the primary benchmark; original annotations are reported as a
diagnostic.

## Validation results (development; used to choose the policy)

**Validation** (gold: corrected, 5000 documents)

| System | Micro P | Micro R | Micro F1 | Macro P | Macro R | Macro F1 | Gold | Pred | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rules | 1.0000 | 0.5088 | 0.6744 | 0.6250 | 0.5882 | 0.6048 | 46250 | 23530 | 23530 | 0 | 22720 |
| ner | 0.9908 | 0.9680 | 0.9793 | 0.9935 | 0.9677 | 0.9793 | 46250 | 45186 | 44771 | 415 | 1479 |
| hybrid:ner_priority | 0.9909 | 0.9825 | 0.9867 | 0.9935 | 0.9844 | 0.9888 | 46250 | 45855 | 45440 | 415 | 810 |
| hybrid:card_block_only | 0.9972 | 0.9885 | 0.9928 | 0.9980 | 0.9913 | 0.9946 | 46250 | 45846 | 45716 | 130 | 534 |
| hybrid:rule_priority_structured | 0.9972 | 0.9885 | 0.9928 | 0.9980 | 0.9914 | 0.9946 | 46250 | 45846 | 45717 | 129 | 533 |

Per-label F1 (P / R):

| System | name | email | phone | address | ssn | credit_card | company | url |
|---|---|---|---|---|---|---|---|---|
| rules | 0.0000 (– / 0.0000) | 0.8784 (1.0000 / 0.7832) | 1.0000 (1.0000 / 1.0000) | 0.0000 (– / 0.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 0.0000 (– / 0.0000) | 0.9599 (1.0000 / 0.9228) |
| ner | 0.9750 (0.9638 / 0.9864) | 0.9995 (1.0000 / 0.9990) | 1.0000 (1.0000 / 1.0000) | 0.9950 (0.9905 / 0.9996) | 1.0000 (1.0000 / 1.0000) | 0.8964 (0.9993 / 0.8128) | 0.9695 (0.9948 / 0.9455) | 0.9989 (0.9994 / 0.9984) |
| hybrid:ner_priority | 0.9750 (0.9638 / 0.9864) | 0.9996 (1.0000 / 0.9992) | 1.0000 (1.0000 / 1.0000) | 0.9950 (0.9905 / 0.9996) | 1.0000 (1.0000 / 1.0000) | 0.9713 (0.9994 / 0.9448) | 0.9695 (0.9948 / 0.9455) | 0.9997 (0.9994 / 1.0000) |
| hybrid:card_block_only | 0.9880 (0.9897 / 0.9864) | 0.9996 (1.0000 / 0.9992) | 1.0000 (1.0000 / 1.0000) | 0.9998 (1.0000 / 0.9996) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 0.9695 (0.9948 / 0.9455) | 0.9997 (0.9994 / 1.0000) |
| hybrid:rule_priority_structured | 0.9880 (0.9897 / 0.9864) | 0.9997 (1.0000 / 0.9994) | 1.0000 (1.0000 / 1.0000) | 0.9998 (1.0000 / 0.9996) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 0.9695 (0.9948 / 0.9455) | 0.9998 (0.9996 / 1.0000) |

Credit-card blocks:

| System | Gold blocks | Exact | Missed | Only other labels inside | Partial card | Block recall | `name` predictions inside blocks |
|---|---|---|---|---|---|---|---|
| rules | 5000 | 5000 | 0 | 0 | 0 | 1.0000 | 0 |
| ner | 5000 | 4064 | 660 | 273 | 3 | 0.8128 | 234 |
| hybrid:ner_priority | 5000 | 4724 | 0 | 273 | 3 | 0.9448 | 234 |
| hybrid:card_block_only | 5000 | 5000 | 0 | 0 | 0 | 1.0000 | 0 |
| hybrid:rule_priority_structured | 5000 | 5000 | 0 | 0 | 0 | 1.0000 | 0 |

Per split unit and template group (micro F1, documents):

| System | Grouping | Results |
|---|---|---|
| rules | unit | `f-insurance_claim` 0.6852 (n=3750); `t36afe448dc` 0.6445 (n=1250) |
| rules | template_group | `t168c7d27f2` 0.6860 (n=2500); `t36afe448dc` 0.6445 (n=1250); `t8ff6b7dd96` 0.6834 (n=1250) |
| ner | unit | `f-insurance_claim` 0.9938 (n=3750); `t36afe448dc` 0.9390 (n=1250) |
| ner | template_group | `t168c7d27f2` 0.9934 (n=2500); `t36afe448dc` 0.9390 (n=1250); `t8ff6b7dd96` 0.9946 (n=1250) |
| hybrid:ner_priority | unit | `f-insurance_claim` 0.9938 (n=3750); `t36afe448dc` 0.9675 (n=1250) |
| hybrid:ner_priority | template_group | `t168c7d27f2` 0.9934 (n=2500); `t36afe448dc` 0.9675 (n=1250); `t8ff6b7dd96` 0.9946 (n=1250) |
| hybrid:card_block_only | unit | `f-insurance_claim` 0.9938 (n=3750); `t36afe448dc` 0.9901 (n=1250) |
| hybrid:card_block_only | template_group | `t168c7d27f2` 0.9934 (n=2500); `t36afe448dc` 0.9901 (n=1250); `t8ff6b7dd96` 0.9946 (n=1250) |
| hybrid:rule_priority_structured | unit | `f-insurance_claim` 0.9938 (n=3750); `t36afe448dc` 0.9901 (n=1250) |
| hybrid:rule_priority_structured | template_group | `t168c7d27f2` 0.9934 (n=2500); `t36afe448dc` 0.9901 (n=1250); `t8ff6b7dd96` 0.9946 (n=1250) |

**Validation, original (uncorrected) annotations — diagnostic** (gold: original, 5000 documents)

| System | Micro P | Micro R | Micro F1 | Macro P | Macro R | Macro F1 | Gold | Pred | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rules | 0.9961 | 0.5279 | 0.6901 | 0.6225 | 0.6250 | 0.6237 | 44395 | 23530 | 23438 | 92 | 20957 |
| ner | 0.9503 | 0.9672 | 0.9586 | 0.9478 | 0.9680 | 0.9546 | 44395 | 45186 | 42938 | 2248 | 1457 |
| hybrid:ner_priority | 0.9510 | 0.9823 | 0.9664 | 0.9479 | 0.9848 | 0.9641 | 44395 | 45855 | 43607 | 2248 | 788 |
| hybrid:card_block_only | 0.9572 | 0.9885 | 0.9726 | 0.9523 | 0.9917 | 0.9699 | 44395 | 45846 | 43883 | 1963 | 512 |
| hybrid:rule_priority_structured | 0.9572 | 0.9885 | 0.9726 | 0.9523 | 0.9917 | 0.9699 | 44395 | 45846 | 43884 | 1962 | 511 |

## Held-out results: internal test (frozen system, single run)

**Internal test** (gold: corrected, 5000 documents)

| System | Micro P | Micro R | Micro F1 | Macro P | Macro R | Macro F1 | Gold | Pred | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rules | 1.0000 | 0.5306 | 0.6934 | 0.6250 | 0.5858 | 0.6032 | 40000 | 21226 | 21226 | 0 | 18774 |
| ner | 0.8930 | 0.8838 | 0.8884 | 0.9285 | 0.8882 | 0.9021 | 40000 | 39591 | 35353 | 4238 | 4647 |
| hybrid:selected | 0.9084 | 0.9191 | 0.9137 | 0.9358 | 0.9353 | 0.9350 | 40000 | 40472 | 36765 | 3707 | 3235 |

Per-label F1 (P / R):

| System | name | email | phone | address | ssn | credit_card | company | url |
|---|---|---|---|---|---|---|---|---|
| rules | 0.0000 (– / 0.0000) | 0.8666 (1.0000 / 0.7645) | 1.0000 (1.0000 / 1.0000) | 0.0000 (– / 0.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 0.0000 (– / 0.0000) | 0.9593 (1.0000 / 0.9218) |
| ner | 0.5092 (0.4648 / 0.5629) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 0.9822 (0.9651 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 0.7678 (0.9991 / 0.6235) | 0.9578 (0.9993 / 0.9195) | 1.0000 (1.0000 / 1.0000) |
| hybrid:selected | 0.5223 (0.4872 / 0.5629) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 0.9578 (0.9993 / 0.9195) | 1.0000 (1.0000 / 1.0000) |

Credit-card blocks:

| System | Gold blocks | Exact | Missed | Only other labels inside | Partial card | Block recall | `name` predictions inside blocks |
|---|---|---|---|---|---|---|---|
| rules | 3750 | 3750 | 0 | 0 | 0 | 1.0000 | 0 |
| ner | 3750 | 2338 | 926 | 484 | 2 | 0.6235 | 348 |
| hybrid:selected | 3750 | 3750 | 0 | 0 | 0 | 1.0000 | 0 |

Per split unit and template group (micro F1, documents):

| System | Grouping | Results |
|---|---|---|
| rules | unit | `f-customer_feedback` 0.7188 (n=3750); `t16a4a8e6b0` 0.5936 (n=1250) |
| rules | template_group | `t16a4a8e6b0` 0.5936 (n=1250); `t74360b80dc` 0.6872 (n=1250); `tcda4a3a527` 0.7359 (n=2500) |
| ner | unit | `f-customer_feedback` 0.8773 (n=3750); `t16a4a8e6b0` 0.9247 (n=1250) |
| ner | template_group | `t16a4a8e6b0` 0.9247 (n=1250); `t74360b80dc` 0.9721 (n=1250); `tcda4a3a527` 0.8238 (n=2500) |
| hybrid:selected | unit | `f-customer_feedback` 0.9104 (n=3750); `t16a4a8e6b0` 0.9247 (n=1250) |
| hybrid:selected | template_group | `t16a4a8e6b0` 0.9247 (n=1250); `t74360b80dc` 0.9721 (n=1250); `tcda4a3a527` 0.8764 (n=2500) |

**Internal test, original annotations — diagnostic** (gold: original, 5000 documents)

| System | Micro P | Micro R | Micro F1 | Macro P | Macro R | Macro F1 | Gold | Pred | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rules | 0.9958 | 0.5516 | 0.7099 | 0.6226 | 0.6250 | 0.6238 | 38322 | 21226 | 21137 | 89 | 17185 |
| ner | 0.8516 | 0.8798 | 0.8654 | 0.8805 | 0.8886 | 0.8758 | 38322 | 39591 | 33714 | 5877 | 4608 |
| hybrid:selected | 0.8679 | 0.9166 | 0.8916 | 0.8875 | 0.9357 | 0.9087 | 38322 | 40472 | 35126 | 5346 | 3196 |

## Held-out results: official benchmark (frozen system, single run)

**Official benchmark, corrected annotations — primary** (gold: corrected, 30 documents)

| System | Micro P | Micro R | Micro F1 | Macro P | Macro R | Macro F1 | Gold | Pred | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rules | 1.0000 | 0.5189 | 0.6833 | 0.6250 | 0.5938 | 0.6081 | 264 | 137 | 137 | 0 | 127 |
| ner | 0.9887 | 0.9924 | 0.9905 | 0.9925 | 0.9949 | 0.9937 | 264 | 265 | 262 | 3 | 2 |
| hybrid:selected | 0.9887 | 0.9924 | 0.9905 | 0.9925 | 0.9949 | 0.9937 | 264 | 265 | 262 | 3 | 2 |

Per-label F1 (P / R):

| System | name | email | phone | address | ssn | credit_card | company | url |
|---|---|---|---|---|---|---|---|---|
| rules | 0.0000 (– / 0.0000) | 0.9020 (1.0000 / 0.8214) | 1.0000 (1.0000 / 1.0000) | 0.0000 (– / 0.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 0.0000 (– / 0.0000) | 0.9630 (1.0000 / 0.9286) |
| ner | 0.9495 (0.9400 / 0.9592) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) |
| hybrid:selected | 0.9495 (0.9400 / 0.9592) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) |

**Official benchmark, original annotations — diagnostic** (gold: original, 30 documents)

| System | Micro P | Micro R | Micro F1 | Macro P | Macro R | Macro F1 | Gold | Pred | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rules | 1.0000 | 0.5331 | 0.6954 | 0.6250 | 0.6250 | 0.6250 | 257 | 137 | 137 | 0 | 120 |
| ner | 0.9623 | 0.9922 | 0.9770 | 0.9613 | 0.9949 | 0.9768 | 257 | 265 | 255 | 10 | 2 |
| hybrid:selected | 0.9623 | 0.9922 | 0.9770 | 0.9613 | 0.9949 | 0.9768 | 257 | 265 | 255 | 10 | 2 |

Per-label F1 (P / R):

| System | name | email | phone | address | ssn | credit_card | company | url |
|---|---|---|---|---|---|---|---|---|
| rules | 0.0000 (– / 0.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 0.0000 (– / 0.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 0.0000 (– / 0.0000) | 1.0000 (1.0000 / 1.0000) |
| ner | 0.9495 (0.9400 / 0.9592) | 0.9020 (0.8214 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 0.9630 (0.9286 / 1.0000) |
| hybrid:selected | 0.9495 (0.9400 / 0.9592) | 0.9020 (0.8214 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 1.0000 (1.0000 / 1.0000) | 0.9630 (0.9286 / 1.0000) |

Credit-card blocks:

| System | Gold blocks | Exact | Missed | Only other labels inside | Partial card | Block recall | `name` predictions inside blocks |
|---|---|---|---|---|---|---|---|
| rules | 27 | 27 | 0 | 0 | 0 | 1.0000 | 0 |
| ner | 27 | 27 | 0 | 0 | 0 | 1.0000 | 0 |
| hybrid:selected | 27 | 27 | 0 | 0 | 0 | 1.0000 | 0 |

## Findings

* **Validation (development).** Hybrid micro F1 0.9928 vs NER 0.9793 and
  rules 0.6744. The gain is almost entirely credit-card blocks.
* **Internal test (held out, unseen templates).** Hybrid micro F1 0.9137,
  NER 0.8884, rules 0.6934. Card-block precedence again fixes every card
  block (NER alone: block recall 0.6235, 348 `name` predictions inside
  blocks). The drop from validation is driven by **`name` (hybrid F1 0.522)**,
  through two systematic errors on templates the model never saw:
  * In "Customer `<name>` from `<company>`" documents the model includes
    the sentence-initial word: it predicts `Customer William Morales` for
    gold `William Morales` (about 2,300 boundary errors).
  * In the auditor's-report template it predicts the title words
    `Independent Auditor` as a name in every document (1,250 spurious FPs).
  These were found after the freeze and were **not** fixed; any fix must be
  developed on validation and evaluated as a new attempt.
* **Official benchmark (30 documents).** Hybrid = NER here (NER found all
  27 card blocks, so nothing was suppressed): micro F1 0.9905 against the
  corrected annotations, 0.9770 against the original ones. The difference
  is the 7 dot-dropped emails/URLs that the original annotations miss.
* **Rules.** Precision 1.0000 on every structured label in all three sets
  (against the corrected annotations).
  URL recall is limited by dot-dropped URLs that are no longer URL-shaped
  (e.g. `https://wwwkeithcom/`). Against the original annotations the rules
  show 92 (validation) and 89 (internal test) URL "false positives". All of
  them are real URLs that the corrections add to gold. Phone FPs: 0 everywhere.

## Limitations

* **Few independent templates.** Validation and internal test each hold
  two split units (three template groups), so per-template variance is
  large: hybrid internal-test F1 per template group ranges from 0.8764 to
  0.9721. These numbers are not estimates of performance on real documents.
* **The official benchmark is small and overlaps training templates.**
  It has 30 documents (264 corrected spans), 16 of which reuse an exact
  training template (Phase 2 key). Its high score is weak evidence of
  generalisation; the internal test is the better held-out estimate.
* **Corrected gold is itself rule-derived.** The corrections come from
  source columns, not human relabelling. Original-annotation scores are
  reported alongside.
* **One trained model, one seed.** Run-to-run variance is not measured.
* **No partial-overlap metric yet.** Only exact-span scores are reported;
  boundary errors are visible through the error types.
