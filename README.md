# RSNA Knee Abnormality Detection

A small, explainable starting point for the [Kaggle competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection), using the same basic structure as our soil-grain project.

**Status:** our independently trained model's best public AUC is **0.780**, up from **0.718** for the frozen image baseline and **0.508** for metadata. Limited DINOv2 adaptation achieved local AUC **0.75991** on the same 58 explicit-label studies; submission **56290319** completed successfully. A separate reproduction of a public competition-trained ensemble scored **0.891**; it has no valid local CV on our folds. The [independent improvement pass](docs/independent-improvement-plan.md) is complete. See [experiments](docs/experiments.md) and [submission evidence](docs/submissions.md). Final deadline: **October 22, 2026, 23:59 UTC**.

The key challenge is supervision: the current training CSV has **4,407 studies**, but only **58 have the 12 condition labels**. All have reports; test studies do not. Missing labels must stay unknown. See [competition details](docs/competition.md) and the [roadmap](docs/roadmap.md).

The [audited public report labels](docs/labels.md) add 37,920 usable condition targets,
giving 4,354 studies some supervision. Unknown verdicts remain masked. The
[independent image workflow](docs/image-model.md) covers frozen DINOv2 features,
duplicate checks, fixed-fold label comparisons and offline inference. Its versioned
split correction moves one unlabeled duplicate while preserving all 58 gold-study
assignments. The [public image reference](docs/public-reference.md) is tracked
separately and is never used for local validation.

The improvement pass compared broader slice coverage, learned aggregation and
matched encoder adaptation without changing labels or folds. The selected model
updates the final two DINOv2 blocks. Our separate report-extraction pilot failed
its quality checks, so none of its annotations entered training.

The September 17 follow-up tested six trainable encoder blocks against the
current two-block model. Local AUC fell from 0.75991 to 0.74828, so it was rejected
and our independent public best remains **0.780**. A [fresh 24-report audit](docs/label-sanity-v2.md)
found threshold mismatches and explicit report/label contradictions; training
labels remain unchanged. Preserving the public labels' graded scores is now a
concrete, untested follow-up hypothesis before full report relabeling.

## Start here

Run from this repository's root:

```bash
uv sync --extra dev --extra imaging --extra training --locked
make download    # Only five metadata CSVs; skips the large MRI dataset
make data        # Validate schemas/IDs/labels and write an aggregate audit
make eda         # Aggregate exploration; create once, then reuse the frozen split
make baselines   # Constant 0.5 probabilities; pipeline check only
make validate SUBMISSION=artifacts/baselines/constant/submission.csv
make test
```

If `uv` is outside your shell's PATH, use `~/.local/bin/uv`. Downloads require Kaggle CLI authentication and competition access. Metadata is already present on the Mac where this repo was created. `make download` uses Kaggle's normal cache behavior; after an announced data update, refresh the five named files explicitly and rerun the audit.

## Layout

```text
configs/data.yaml       Local data and artifact paths
src/rsnaknee/           Data checks, EDA, baseline, offline notebook packaging and CLI
tests/                  Label, leakage, metric, inference and submission checks
notebooks/              Self-contained smoke notebook and packaging instructions
docs/                   Competition brief, roadmap, experiment and submission logs
data/raw/               Downloaded metadata; ignored by Git
artifacts/reports/      Aggregate audit and input hashes; ignored by Git
artifacts/baselines/    Sanity CSV and run manifest; ignored by Git
data/processed/         Frozen study/report-group split and manifest; ignored by Git
artifacts/experiments/  Models, held-out predictions, metrics and hashes; ignored by Git
artifacts/kaggle/       Private notebook builds, run logs and submission evidence; ignored
```

`uv.lock` fixes the local dependency versions. Paths in `configs/data.yaml` are relative to the repository root. The imaging extra installs Pillow and pydicom. The training extra installs PyTorch and Transformers for local cached-feature learning and tests; MRI encoding and encoder adaptation run on Kaggle's GPU.

`make test` uses synthetic fixtures and runs without competition data or Kaggle
credentials. One optional public-reference integration check skips when its
separately downloaded, audited source notebook and manifest are unavailable.

`make metadata-baseline` compares constant 0.5, training-fold prevalence and regularized logistic regression using series counts on the frozen split. It saves a new timestamped run and refuses to overwrite existing experiments. [EDA](docs/eda.md) explains the split and missing-label handling; [experiments](docs/experiments.md) records the fixed recipe, uncertainty and error review. Validation is study/report-grouped, with unresolved patient overlap. The model uses only the 58 observed-label cases and metadata available at inference.

## Kaggle execution

This is a **notebook submission** competition. A local CSV check does not submit it.
The [smoke notebook](notebooks/00-submission-smoke.ipynb) reads the mounted `test.csv` at runtime and produces `/kaggle/working/submission.csv`. It uses constant predictions, no training and no internet. Its private Kaggle version 1 completed successfully and matched local output; it was not submitted to the competition.

The first competition submission uses the learned metadata model in version 2 of [RSNA Knee First Submission](https://www.kaggle.com/code/willmurray99/rsna-knee-first-submission). Its private, offline CPU run matched local predictions exactly. The [packaging instructions](notebooks/README.md) generate a portable notebook from saved model parameters and tested inference functions. Hidden-test IDs are read dynamically, and tests cover 1,300 replacement IDs without training files or reports. Generated notebooks containing weights remain in ignored artifacts.

Keep the full MRI dataset on Kaggle. The completed extraction used its GPU and
downloaded compact features and audits for local model comparisons; raw MRI files
remain on the competition mount.

## How we will work

- One explicit hypothesis and one main change per experiment, with a saved comparison and decision in [the experiment log](docs/experiments.md).
- Commit and push each reviewed logical change on the working branch. Commit the recipe and source before launching new experiments; record outcomes afterward in a separate commit. Earlier runs predate the Git catch-up and retain their original source hashes and dirty-state provenance.
- Freeze validation groups before tuning. All slices/series from one study stay together; use patient grouping if a reliable identifier can be established.
- Record macro and per-target AUC, held-out predictions, seed, split file/hash, preprocessing, code commit, dependencies and weight provenance in each learned run's folder.
- Treat report-derived labels as a separate, auditable source of supervision. Never train a held-out study using its report, pseudo-labels or images.
- Review examples and errors alongside scores. Add complexity only when a controlled comparison supports it.

Results live in the logs, keeping this README focused on setup. No experiment registry, tracking server or model framework is required.
