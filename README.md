# RSNA Knee Abnormality Detection

A small, explainable starting point for the [Kaggle competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection), using the same basic structure as our soil-grain project.

**Status:** the reproduced public ensemble's **0.891** AUC is now our primary submission baseline. Our best independently trained submission is **0.819** from the [all-ten-window diagnostic](docs/all-window-plan.md), ref **56471807**. Three-window training remains the locally selected independent baseline: **0.77478** local AUC versus **0.76965** for ten windows. A [fixed 90/10 rank blend](docs/reference-blend-plan.md) passed exact offline component parity and independent arithmetic verification; version 2 was submitted as **56473633**, with hidden scoring pending. The public ensemble and blend have no valid local CV on our folds. See [experiments](docs/experiments.md) and [submission evidence](docs/submissions.md). Final deadline: **October 22, 2026, 23:59 UTC**.

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
and our independent public best at that point remained **0.780**. A [fresh 24-report audit](docs/label-sanity-v2.md)
found threshold mismatches and explicit report/label contradictions; training
labels remain unchanged. A subsequent matched test of the public labels' graded
scores scored 0.75995 versus 0.75991, improving only one of three folds. It failed
the preregistered promotion rule, so no new competition submission was made.
The selected model still uses binary public targets; no new labels were generated.

The September 18 [three-window comparison](docs/multi-window-plan.md) improved
local AUC from **0.75991 to 0.77478**, with gains in all three folds. It keeps
three slices and 768 features per window while training jointly on three windows
per plane. Labels and model parameter counts are unchanged. Offline inference
matched the saved predictions exactly; submission **56341808** completed with
public AUC **0.801**, our new independent best, **+0.021** over 0.780.

The September 22 [all-ten-window comparison](docs/all-window-plan.md) exactly
reproduced the saved three-window control. Training on all ten windows scored
**0.76965 versus 0.77478**, improving two folds but lowering the mean by **0.00512**.
It failed the promotion rule with labels and folds unchanged. The three-window
model remains selected. Offline inference matched the saved predictions exactly;
diagnostic submission **56471807** completed with public AUC **0.819**, our new
independent public best (**+0.018**). This public gain does not change the frozen
local selection decision; the disagreement highlights the limits of our small
validation set.

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
credentials. Two optional integration checks cover the public reference and
saved-notebook blend packaging; they skip when their separately downloaded,
audited source artifacts are unavailable.
GitHub Actions runs the same tests on pushes to `main` and pull requests using Python 3.12 and the locked dependencies on a CPU runner.

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
