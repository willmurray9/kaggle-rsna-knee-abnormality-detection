# RSNA Knee Abnormality Detection

A small, explainable starting point for the [Kaggle competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection), using the same basic structure as our soil-grain project.

**Status:** metadata downloaded and audited; constant sanity baseline available. No image model trained or competition submission made. Final deadline: **October 22, 2026, 23:59 UTC**.

The key challenge is supervision: the current training CSV has **4,407 studies**, but only **58 have the 12 condition labels**. All have reports; test studies do not. Missing labels must stay unknown. See [competition details](docs/competition.md) and the [roadmap](docs/roadmap.md).

## Start here

Run from this repository's root:

```bash
uv sync --extra dev --locked
make download    # Only five metadata CSVs; skips the large MRI dataset
make data        # Validate schemas/IDs/labels and write an aggregate audit
make baselines   # Constant 0.5 probabilities; pipeline check only
make validate SUBMISSION=artifacts/baselines/constant/submission.csv
make test
```

If `uv` is outside your shell's PATH, use `~/.local/bin/uv`. Downloads require Kaggle CLI authentication and competition access. Metadata is already present on the Mac where this repo was created. `make download` uses Kaggle's normal cache behavior; after an announced data update, refresh the five named files explicitly and rerun the audit.

## Layout

```text
configs/data.yaml       Local data and artifact paths
src/rsnaknee/           Data checks, metric, submission checks, CLI
tests/                  Checks for schema, unknown labels, metric and submission errors
notebooks/              One self-contained Kaggle smoke notebook
docs/                   Competition brief, roadmap, experiment and submission logs
data/raw/               Downloaded metadata; ignored by Git
artifacts/reports/      Aggregate audit and input hashes; ignored by Git
artifacts/baselines/    Sanity CSV and run manifest; ignored by Git
```

`uv.lock` fixes the local dependency versions. Paths in `configs/data.yaml` are relative to the repository root. Heavy imaging/training dependencies will be added with the first image experiment.

## Kaggle execution

This is a **notebook submission** competition. A local CSV check does not submit it.
The [smoke notebook](notebooks/00-submission-smoke.ipynb) reads the mounted `test.csv` at runtime and produces `/kaggle/working/submission.csv`. It uses constant predictions, no training and no internet. Its purpose is to check the execution path before adding an image model. It has been exercised locally against the example metadata; Kaggle execution remains to be checked.

Keep the full MRI dataset on Kaggle initially. Use the Mac for metadata exploration, code and tests; use a small image subset or Kaggle GPU for the first image pipeline.

## How we will work

- One explicit hypothesis and one main change per experiment, with a saved comparison and decision in [the experiment log](docs/experiments.md).
- Freeze validation groups before tuning. All slices/series from one study stay together; use patient grouping if a reliable identifier can be established.
- Record macro and per-target AUC, held-out predictions, seed, split file/hash, preprocessing, code commit, dependencies and weight provenance in each learned run's folder.
- Treat report-derived labels as a separate, auditable source of supervision. Never train a held-out study using its report, pseudo-labels or images.
- Review examples and errors alongside scores. Add complexity only when a controlled comparison supports it.

Results live in the logs, keeping this README focused on setup. No experiment registry, tracking server or model framework is required.
