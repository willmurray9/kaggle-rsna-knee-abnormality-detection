# Notebooks

`00-submission-smoke.ipynb` is a self-contained constant-probability smoke test.
Upload it to a private Kaggle notebook, attach the competition data, and disable
internet. It uses CPU and writes `/kaggle/working/submission.csv` using the current
test IDs. Its 0.5 predictions are intentionally uninformative; a successful run
checks the execution path, not model quality. Creating/running a notebook does
not automatically submit it to the competition.

The notebook can also run locally from this directory or the repository root,
writing to the ignored `artifacts/baselines/notebook_smoke/` directory.
Its cells have been executed locally and as private, offline CPU version 1 of
`willmurray99/rsna-knee-first-submission`. Hosted output matched local output
byte for byte. This smoke version was not submitted to the competition.

## Learned metadata notebook

Version 2 of the same private notebook uses the observed-label metadata model.
It is generated from the tested `build_features`, `predict_model` and
`predict_metadata` functions, with JSON scaler/weights embedded. It needs only
NumPy/pandas and competition test metadata at runtime; it reads no reports or
training data and downloads nothing. Example-test execution matched local
predictions exactly. See [the submission log](../docs/submissions.md) for the
hidden-test result.

Regenerate the first learned notebook from the preserved local model:

```bash
.venv/bin/python -m rsnaknee.notebook \
  --model artifacts/experiments/metadata-v1/model.json \
  --output artifacts/kaggle/metadata/01-metadata-baseline.ipynb
```

To train another identical comparison, run `make eda` then
`make metadata-baseline`; the printed run directory contains its `model.json`.
Use that path with the packager. Experiment directories are never overwritten.
Notebook builds containing weights belong under ignored `artifacts/`.

`tests/test_notebooks.py` executes cells with replacement 1,300-study test
metadata and no training files, checks exact prediction parity, unknown planes,
missing series, stale sample IDs and duplicate-ID rejection in the smoke notebook.

Add EDA notebooks only as useful questions arise. Move reusable image/model
logic into `src/rsnaknee/` and clear notebook outputs before committing.

## Image notebooks

The [independent image workflow](../docs/image-model.md) packages feature extraction
and test-only inference from the same tested preprocessing source. Builds record
source and checkpoint hashes and refuse to overwrite existing directories. Images
stay on Kaggle; compact features support local fixed-fold comparisons.

The [public reference](../docs/public-reference.md) has a separate packager and
retains its original pixel recipe. Its competition-trained weights must not be
used to claim validation performance on our folds.
