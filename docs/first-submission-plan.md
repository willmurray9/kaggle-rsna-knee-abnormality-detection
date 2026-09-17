# First submission plan — September 16, 2026

The user authorized work through an actual first Kaggle submission, including private notebook upload and execution. Use the existing authenticated account and free CPU execution. No paid compute, full image download, external weights, or report upload is needed for this milestone.

1. Recheck official execution requirements and authentication; run the existing audit, tests and constant baseline.
2. Analyze observed-label support, acquisition metadata, report lengths/scripts and duplicate report hashes without publishing report text. Save one fixed three-fold study/report-group split. Patient independence remains unresolved because CSVs contain no patient identifier.
3. Before looking at validation scores, compare constant 0.5, fold-training prevalence and a small regularized logistic model using only series metadata. Fit preprocessing within each training fold, preserve unknown labels, and record held-out predictions and per-target/fold scores. No seed or hyperparameter search.
4. Use the learned model only if its mean validation macro AUC exceeds 0.5; describe its uncertainty and acquisition-protocol shortcut risk. Otherwise submit the constant smoke baseline. This is a first pipeline milestone, not an image-model performance claim.
5. Execute a private, internet-disabled CPU notebook with dynamic hidden-test IDs. Validate its actual output, submit that exact successful notebook version, and wait for Kaggle's scored completion.
6. Record notebook/version, hashes, validation evidence, public result, limitations and the next image-based experiment. Run the full tests and relevant workflows again before finishing.

Keep reusable code in `src/rsnaknee`, unit tests focused on leakage/label/submission mistakes, and generated data/results under ignored `data/` or `artifacts/`. Preserve the self-contained constant smoke notebook.
