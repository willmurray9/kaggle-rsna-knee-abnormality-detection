# Public report-label source audit

Policy fixed September 16, 2026 **before examining agreement with the 58 observed-label studies**. This is ingestion and provenance review, not a validated extractor.

## Source and limits

The [public dataset](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels), `pilkwang/rsna-knee-llm-labels`, has Kaggle dataset ID 11529209. Its authenticated metadata declares CC0-1.0. Its two listed files, created August 6, 2026, are `api_labeler.py` (12,976 bytes) and `report_labels_v2.csv` (1,004,839 bytes). Only those two named files and the dataset metadata were downloaded to ignored `data/external/public-labels/`. Content hashes identify the exact snapshot in `artifacts/reports/label_audit.json`.

The CSV has 4,406 rows and 37 columns: `StudyInstanceUID`, then a score, `__conf` and `__verdict` for each of the 12 targets. One current training study is absent. Public code defines YES as asserted presence, NO as explicit denial or a normal/intact/preserved structure, and UNK as unmentioned or uncertain. This supports distinguishing negative evidence from missing evidence.

The dataset card explicitly says its scores are rank-usable rather than calibrated probabilities. All 16,129 YES entries have confidence 0.95; all 22,298 NO entries have confidence 0.85; all 14,445 UNK entries have confidence 0.05. These are heuristic verdict weights, not measured reliability. Scores are 0.08 for NO, 0.28 for UNK, and 0.68/0.82/0.94 for YES. The nonzero UNK score and confidence do **not** authorize treating unknown findings as targets.

The included code imports `FINDING_SPEC` and `to_scores` from an unavailable `llm_labeler.py`; exact finding definitions and score derivation are therefore incomplete. Its hosted-model default is `claude-opus-5`, but there is no execution manifest verifying the model/version actually used, complete prompt, report hashes, generation settings or review history. Severity and sentence evidence are absent from the CSV. The code discusses prompt evaluation on annotated studies, so independence of this public extractor from the 58 observed cases is unestablished. A mild YES may also differ from an annotation's clinical threshold.

Public availability and a CC0 declaration do not establish permission to redistribute the original competition reports or to send them to a hosted model. The [competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules) remain the authority. Authenticated rule section 2.6 permits equally accessible public external data/models, subject to its terms. The [pinned host clarification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965), message 3510692 dated August 9, 2026 and reiterated in message 3517694 on August 27, explicitly permits hosted LLM processing of reports for label extraction. Public reuse is supported by those rules and this clarification; the producer's particular service configuration remains unverified. First-party text was retrieved by the authenticated Kaggle API and reviewed in ignored `artifacts/research/public-baseline/audit.md`. No included extractor was executed, and no reports were sent to an external model service.

## Fixed ingestion policy

- Join exact string study IDs; reject missing, blank, padded, duplicate or out-of-training IDs. Partial public coverage is permitted, with uncovered targets left unknown.
- Preserve each observed label unchanged in `target__observed`; it always takes precedence in the final `target` value. Observed labels must be binary or missing.
- Preserve the source score in `target__derived`, original verdict in `target__verdict`, and heuristic confidence in `target__confidence`. Scores and confidences must be finite in [0, 1]; verdicts must be YES, NO or UNK.
- For otherwise missing observed targets, use YES=1 and NO=0 only when confidence is positive. UNK, zero-confidence entries and missing public rows retain a missing target and false `target__mask`. No extraction thresholds or per-target calibration are tuned here. Confidence values are metadata, not default loss weights.
- `target__source` identifies observed supervision or the public dataset/file hash; unknown public verdicts retain source provenance even though their mask is false. The original public columns remain separate even when an observed label overrides them.
- Reuse the saved study/report-group folds, verify their manifest and input hashes, and recompute report hashes against `group_id`. Training a fold excludes **every** study in that held-out fold before selecting usable targets. No report text enters the returned training table.

`build_label_table(train, derived, folds, source=...)` returns one row per training study, saved `group_id`/`fold`, and the columns above. `training_labels(table, heldout_fold=...)` excludes held-out studies and rows with no usable target. For masked loss, select the 12 target columns and matching `__mask` columns; replace missing targets with a finite placeholder **only after preserving masks**, then multiply elementwise loss by masks and normalize by mask count. Never compute loss on NaN and expect multiplication by zero to repair it.

## Reproduction and diagnostics

Run `PYTHONPATH=src .venv/bin/python -m rsnaknee.labels` after metadata, the frozen split and the three named source files exist. It performs local ingestion, writes the joined table to ignored `data/processed/report_labels.csv`, and writes aggregate results plus input/output hashes to ignored `artifacts/reports/label_audit.json`; it does not download data or execute an extractor. `make test` covers observed precedence, ID joins, unknown masking, invalid values, report-group integrity and exclusion of held-out studies.

Those original table/audit artifacts are preserved. Image experiments use the
versioned duplicate-corrected assignments in `data/processed/image-v1/folds.csv`,
the corresponding table `data/processed/image-v1/report_labels.csv`, and
`artifacts/reports/label_audit_image_v1.json`. The saved regeneration script is
`artifacts/reports/generate_label_audit_image_v1.py`. One unlabeled singleton moved
from fold 1 to fold 0 after all nine selected image slices matched another study;
all 58 observed assignments, source labels and masking policy are unchanged.
The revised split SHA-256 is
`23c611d98c910549c5c143b30de436d4217214aae239f15850cf02db2cd6ba21`,
and the revised label-table SHA-256 is
`7d2369e7e936d3335614a71fb32bec5af24737f8d7c62f888fcd1e89075f7f0e`.
Always pair this table with its revised audit and folds when fitting image heads;
the original ingestion command still targets the original artifacts.

Any agreement diagnostic uses only known public verdicts and observed labels and reports its denominator. It measures report-to-label consistency after this policy was fixed; it is **not independent validation**, not image-model accuracy, and not a basis for tuning this public extractor. Image comparisons must keep the same saved folds, report observed-only metrics, and retain the unresolved patient-overlap limitation.

The audited snapshot adds **37,920** known derived targets beyond the 696 observed targets. **14,268** final target cells remain masked. Across the 58 observed studies, **507 of 696** public verdicts are usable for comparison, with **391/507 (77.1%)** binary agreements. This coverage-dependent diagnostic is descriptive only; it did not change the frozen policy. Original usable training counts for held-out folds 0/1/2 were 2,892/2,915/2,901. Under the corrected `image-v1` split they are **2,891/2,916/2,901**; overall coverage and agreement are unchanged.
