# Independent image experiment

This experiment uses the generic `metaresearch/dinov2/PyTorch/small/1` encoder,
separately from the competition-trained [public reference](public-reference.md).
The encoder is frozen and reads neither reports nor labels. The report-derived
targets follow the [audited policy](labels.md). Results and decisions belong in
[the experiment log](experiments.md).

## Completed extraction and validation audit

Private offline feature notebook version 1 completed on Kaggle: 4,410 studies
(4,407 train and 3 example test), 2,307 features per study, in 1,996 seconds
(33.3 minutes). The frozen encoder read no reports or labels. Its outputs and
checkpoint/source hashes are in
`artifacts/kaggle/frozen-features/versions/v1/output/features/`.

All 13,230 selected series were usable. There were no header or sampled-image
decode failures, missing planes, spacing fallbacks or InstanceNumber fallbacks.
The four-study, three-plane preprocessing contact sheet was visually inspected;
the displayed images had no blank tiles or apparent aspect-ratio distortion.
This bounded check does not establish image quality throughout the dataset.

The original-fold audit found one cross-fold pair with all nine sampled images
matching, including dimensions, spacing and slice positions. Their feature vectors
differed by at most 0.00002885; they were numerically near-identical, not bitwise
equal. Their anonymized patient keys differed. Each study belonged to a separate
singleton report group, and neither had observed labels.

The versioned correction in `data/processed/image-v1/folds.csv` moves the one
unlabeled report group from fold 1 to fold 0. All **58 gold study assignments are
unchanged**, and every report group remains together. The original
`data/processed/folds.csv` and its manifest are preserved. The revised split hash
is `23c611d98c910549c5c143b30de436d4217214aae239f15850cf02db2cd6ba21`.
Its sibling manifest records the original split hashes, evidence hashes and move;
`artifacts/reports/revise_image_split_v1.py` preserves the revision procedure.

The revised audit, `artifacts/reports/image_audit_image_v1.json`, has zero sampled
pixel groups crossing folds and no train/test sampled-image matches. The original
audit remains in `artifacts/reports/image_audit.json`. There are 4,410 distinct
patient keys but no issuer or site values, and all deidentification flags are
unknown. There are 59 scanner fingerprints. Patient independence is **not
established**: the confirmed pair shows that different anonymized patient keys
can still identify duplicate image inputs. The audit covers selected samples,
not every image in every series.

The independent image heads and supervision comparisons remain the next fit;
completed extraction alone is not an image-model validation result.

## Fixed recipe

Select one series per sagittal, coronal and axial plane, preferring fluid-sensitive
and fat-suppressed sequences, with a stable series-UID tie break. Order all slice
headers by physical position along a consistent normal. If geometry is missing,
use complete unique InstanceNumber values and record that fallback; inconsistent
orientations fail that series.

Sample three slices across the central 20–80% of each ordered stack. Apply modality
rescaling and MONOCHROME1 inversion, clip each slice to its 1st/99th percentiles,
and resize with physical aspect ratio preserved into a padded 224-pixel square.
The three grayscale slices become the encoder's three input channels, followed by
ImageNet normalization. There are no laterality flips. Concatenate CLS and mean
patch embeddings for each plane, then append three presence flags: 2,307 features.

An isolated failed sampled slice is replaced by the nearest usable sampled slice
with an audit entry. Missing planes have zero features and an absent flag. A study
with no usable plane fails extraction and cannot become a fabricated prediction.

Compare the same per-target logistic heads at C=0.1 under three prespecified
recipes: observed labels only, observed plus derived labels at weight 1.0, and
observed plus derived labels at weight 0.25. Gold labels always take precedence;
unknown targets contribute no training loss. Each fold excludes all held-out
studies before fitting its scaler or heads. Report the unweighted mean of the
three within-fold gold-label macro AUCs, with per-target scores and paired errors.

## Reproduce

Build the private offline feature-extraction notebook locally:

```bash
.venv/bin/python -m rsnaknee.feature_notebook --output artifacts/kaggle/features-new-build
```

The builder emits notebook, metadata and source hashes; it does not upload or run
anything. Kaggle execution uses a T4 and mounts the competition plus the generic
encoder. Images remain on Kaggle. Preserve the pushed version and download its
compact completed outputs; do not use a running or failed feature cache.

Before fitting, inspect `preprocessing_contact_sheet.png` and verify the revised
split against its manifest:

```bash
.venv/bin/python -m rsnaknee.image_audit \
  --features artifacts/kaggle/frozen-features/versions/v1/output/features \
  --folds data/processed/image-v1/folds.csv --train data/raw/train.csv \
  --output artifacts/reports/image_audit_image_v1.json
```

Review patient-key and sampled-image overlap against the frozen split. Hashed
anonymized identifiers do not establish independent patients, and three sampled
slices per selected series cannot exclude every duplicate image. An observed
overlap requires investigation; never silently regenerate a more favorable split.

```bash
.venv/bin/python -m rsnaknee.image_model \
  --features artifacts/kaggle/frozen-features/versions/v1/output/features \
  --labels data/processed/image-v1/report_labels.csv \
  --label-audit artifacts/reports/label_audit_image_v1.json \
  --output artifacts/experiments/frozen-image-supervision-v1
.venv/bin/python -m rsnaknee.image_analysis \
  --experiment artifacts/experiments/frozen-image-supervision-v1 \
  --labels data/processed/image-v1/report_labels.csv \
  --folds data/processed/image-v1/folds.csv \
  --metadata artifacts/experiments/metadata-image-v1/oof.csv \
  --output artifacts/experiments/frozen-image-supervision-v1-analysis
```

Use the label table and audited label provenance attached to the revised split.
Rerun the metadata reference on those same assignments and save it under
`artifacts/experiments/metadata-image-v1/` before comparison. The original
metadata experiment and first-submission evidence remain historical records.
Experiment and analysis directories are immutable. Feature, label, split and OOF
hashes are checked before comparison. The analysis resamples gold studies jointly
within folds, holding models fixed, and discards draws missing either class for
any target in any fold. Its intervals are conditional and omit fitting, split and
unresolved patient dependence; 58 cases give limited evidence for model selection.
The public label generator's independence from these gold cases is also unknown.

Package a selected head for private offline inference:

```bash
.venv/bin/python -m rsnaknee.image_notebook \
  --model artifacts/experiments/frozen-image-supervision-v1/model.json \
  --features-manifest artifacts/kaggle/frozen-features/versions/v1/output/features/manifest.json \
  --output artifacts/kaggle/independent-image
```

The package embeds the same preprocessing, checks checkpoint hashes, reads dynamic
test IDs and uses no training files or reports. Run its example notebook and check
prediction parity with the saved feature-based output before a competition
submission. Record the actual hidden-test result separately from local CV.

The completed first run selected quarter-weight report supervision: local mean
fold AUC **0.697812**, Kaggle public AUC **0.718** (submission 56287107). Offline
example probabilities matched local output within 3.84e-6. A single PCA-32
comparison scored 0.692956 and failed its registered selection rule; the submitted
model was unchanged. See the [submission evidence](submissions.md).

## Reproduce the independent improvement pass

The [improvement plan](independent-improvement-plan.md) fixes the coverage,
cached-attention and matched adaptation recipes; [experiments](experiments.md)
records their results and decisions. All comparisons retain
`data/processed/image-v1/folds.csv`, `report_labels.csv` and
`artifacts/reports/label_audit_image_v1.json`. Use fresh output directories for
reruns; completed artifacts and prior model sources remain immutable.

Build the private offline coverage notebook:

```bash
.venv/bin/python -m rsnaknee.coverage_notebook \
  --output artifacts/kaggle/coverage-features-new-build
```

The completed cache is versioned under
`artifacts/kaggle/coverage-features/versions/v1/output/features`. Download only
compact features, presence flags, audits and provenance. Raw DICOMs and the
roughly 8 GB `pixels_uint8.npy` cache stay private on Kaggle. Inspect the contact
sheet and image audit before fitting. The first coverage comparison uses original
float pixels; both later adaptation arms use the same quantized pixel cache.

The one-off mean-head driver preserves the preregistered quarter-weight
supervision comparison. Its source is also saved as `run_source.py` in each run.
Fit the cached attention candidate through its module CLI:

```bash
.venv/bin/python artifacts/reports/independent-improvement-v1/run_coverage_comparison.py \
  --features artifacts/kaggle/coverage-features/versions/v1/output/features \
  --output artifacts/experiments/coverage-mean-reproduce
.venv/bin/python -m rsnaknee.window_model \
  --features artifacts/kaggle/coverage-features/versions/v1/output/features \
  --labels data/processed/image-v1/report_labels.csv \
  --label-audit artifacts/reports/label_audit_image_v1.json \
  --output artifacts/experiments/coverage-attention-reproduce
```

Build matched frozen-control and limited-adaptation training for Kaggle; the
builder attaches the private coverage kernel output and embeds the audited label
table and saved folds without report text:

```bash
.venv/bin/python -m rsnaknee.finetune_notebook \
  --output artifacts/kaggle/adaptation-training-new-build
```

A disposable throughput probe must meet the recorded budget before training.
Attention and both adaptation arms use six fixed epochs and evaluate only the
final epoch, excluding the entire validation fold from fitting. The adaptation
arms share their sampled training windows and use all windows at inference.

For the September 17 controlled depth experiment, explicitly select the current
two-block control and the six-block candidate. The default pair remains frozen
versus two-block adaptation. Each selected arm runs three folds and a final refit;
the summary records actual trainable block indices and parameter counts.

```bash
.venv/bin/python -m rsnaknee.finetune_notebook \
  --kernel-id willmurray99/rsna-knee-depth-training \
  --arms late_blocks deep_blocks \
  --output artifacts/kaggle/depth-training-new-build
```

Package only the selected candidate after comparison. For a mean or cached
attention head, use `rsnaknee.coverage_notebook` or `rsnaknee.window_notebook`,
respectively, with `--model`, `--features-manifest` and a fresh `--output`. For a
completed adaptation run, choose the reviewed arm explicitly (`frozen`,
`late_blocks`, `deep_blocks` or `soft_targets`) and its actual training kernel. For example,
if the six-block candidate passes the registered comparison:

```bash
.venv/bin/python -m rsnaknee.finetune_notebook \
  --summary artifacts/kaggle/depth-training/versions/v1/output/adaptation/summary.json \
  --arm deep_blocks \
  --training-kernel-id willmurray99/rsna-knee-depth-training \
  --kernel-id willmurray99/rsna-knee-deep-image \
  --output artifacts/kaggle/deep-image-new-build
```

This example selects an arm; it is not a promotion decision. The
inference builder attaches the completed private training output and checks
source, checkpoint and summary hashes. Builders only write local packages.
Current source must match the source embedded in the completed training run.
The previous two-block submission retains its immutable version-1 notebook;
use that saved package or its archived source to reproduce the original run.
Before submission, run the selected package offline on Kaggle and compare all
three example studies against its saved predictions within absolute tolerance
`1e-4`. Verify dynamic hidden-test IDs, probabilities and runtime as well; record
actual submission evidence separately from local validation.

The [preserved-score experiment](soft-target-plan.md) keeps two trainable blocks
in both arms and changes only eligible report-derived BCE targets. Build from
the committed, clean source before launching:

```bash
.venv/bin/python -m rsnaknee.finetune_notebook \
  --kernel-id willmurray99/rsna-knee-soft-target-training \
  --arms late_blocks soft_targets \
  --output artifacts/kaggle/soft-target-training-new-build
```

The builder records Git revision/dirty state and propagates it into the training
summary. Each fit and probe records the target mode and SHA-256 hashes of its
ordered float32 target/weight arrays. Official labels and unknown masks remain
unchanged. A promoted `soft_targets` checkpoint uses its own training kernel
with the existing inference packager; mode and depth provenance are verified
alongside source and weight hashes. These scores are heuristic soft targets,
not calibrated probabilities or newly extracted labels.

Version 1 completed the matched comparison on September 17. The soft-target
candidate scored 0.75994585 versus 0.75991074 for the exactly reproduced binary
control, but improved only one of three folds. It failed the registered promotion
rule and was not packaged or submitted. The existing binary-target two-block
model remains selected; see the [outcome and uncertainty](experiments.md).
