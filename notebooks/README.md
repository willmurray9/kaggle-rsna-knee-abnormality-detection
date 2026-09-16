# Notebooks

`00-submission-smoke.ipynb` is a self-contained constant-probability smoke test.
Upload it to a private Kaggle notebook, attach the competition data, and disable
internet. It uses CPU and writes `/kaggle/working/submission.csv` using the current
test IDs. Its 0.5 predictions are intentionally uninformative; a successful run
checks the execution path, not model quality. Creating/running a notebook does
not automatically submit it to the competition.

The notebook can also run locally from this directory or the repository root,
writing to the ignored `artifacts/baselines/notebook_smoke/` directory.
Its cells have been executed locally against the downloaded metadata. Hosted
Kaggle execution and hidden-test runtime have not been verified.

Add EDA notebooks only as useful questions arise. Move reusable image/model
logic into `src/rsnaknee/` and clear notebook outputs before committing.
