# Submission log

Our best independently trained submission is now **0.780** (ref **56290319**), up from **0.718** for the frozen image model and **0.508** for metadata. All four submissions are complete. A separate reproduction of an externally trained image ensemble scored **0.891** (ref **56286555**).

This competition executes a notebook against hidden test data. Local example CSV validation is only a format check. Record actual submissions here when they happen.

| Date UTC | Notebook / version | Code commit | Artifact hashes | Reason to submit | Public AUC | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-09-16 19:21:24 | [RSNA Knee First Submission](https://www.kaggle.com/code/willmurray99/rsna-knee-first-submission), version **2**; ref **56286205** | `3b79c70` + recorded working-tree source hashes | Full hashes below | Prespecified metadata model beat both constant references on frozen CV; offline CPU output verified | **0.508**, COMPLETE | First submission milestone achieved. Preserve as baseline; next investigate image features, without leaderboard-driven tuning. |
| 2026-09-16 19:50:10 | [RSNA Knee Image Reference](https://www.kaggle.com/code/willmurray99/rsna-knee-image-reference), version **1**; ref **56286555** | Working-tree packager; versioned notebook hash below | Versioned build, weights and execution manifests | Audited public image reference; all 20 members × 10 windows passed private offline GPU example execution | **0.891**, COMPLETE | Best submitted public score; external competition-trained ensemble, with no valid CV result on our saved folds. |
| 2026-09-16 20:37:58 | [RSNA Knee Independent Image](https://www.kaggle.com/code/willmurray99/rsna-knee-independent-image), version **1**; ref **56287107** | `3b79c70` + recorded source snapshots/hashes | Versioned build, model and inference manifests | Quarter-weight report supervision selected on corrected frozen folds; PCA-32 rejected; offline parity verified | **0.718**, COMPLETE | Successful hidden-test result, +0.210 over metadata; retain as independent image baseline. |
| 2026-09-17 00:50:24 | [RSNA Knee Adapted Image](https://www.kaggle.com/code/willmurray99/rsna-knee-adapted-image), version **1**; ref **56290319** | `3b79c70` + exact working-tree training/inference source hashes | Versioned training, build, selection and parity manifests | Highest eligible local mean AUC **0.759911**; improves all three folds over original and matched frozen control; exact offline parity | **0.780**, COMPLETE | New independently trained best, **+0.062** over 0.718; no recipe changes from leaderboard feedback. |

Keep training/weight/config hashes with each run; record the generated CSV hash where available. Preserve the local validation result before viewing the leaderboard score.

## Evidence and recipe

- Training run: `artifacts/experiments/metadata-v1/`; 58 observed-label studies, eight fixed acquisition count features, per-target logistic regression with C=0.1. No reports, external weights or images at inference.
- Frozen three-fold study/report-group validation: mean macro AUC **0.598926**, fold SD **0.043192**. Both 0.5 and fold-training prevalence references score 0.500000. Patient independence remains unresolved. Full counts, uncertainty and errors are in [experiments](experiments.md) and [EDA](eda.md).
- Private constant notebook version 1 completed on Kaggle and matched local output; it was not submitted.
- Private learned notebook version 2 completed with CPU and internet disabled. Downloaded source matches the packaged notebook, and all example probabilities match local inference exactly (maximum absolute difference 0).
- Submitted that exact successful version, output filename `submission.csv`. Kaggle API returned ref `56286205` and later `SubmissionStatus.COMPLETE`, public score `0.508`, without an error. The later image-reference submission is recorded separately below; no metadata recipe changes were selected from its public score.
- Local notebook tests exercise 1,300 replacement test IDs with no training/report files, stale sample IDs, missing series and unrecognized planes. The hosted hidden-test run then completed successfully.
- Final verification: `make test` **56 passed**; metadata audit, frozen-split reuse, constant baseline regeneration, hosted-output validation and `git diff --check` all passed. Saved experiment artifacts and modeling/inference source hashes were rechecked successfully.

| Artifact | SHA-256 |
| --- | --- |
| Frozen split | `d1c1f632ed2d87709a984d94365389983b1adbad15704a9aa8e5a98397831910` |
| Final JSON model | `96be94fca140d7a7832b28c75c77e5cfb8d4162ce0d9e7af7ed272b0b7f02dce` |
| Submitted notebook source | `c435d981bdc963153c223f86c558c594cf40b9f370da0bf64c9911fa86a6902e` |
| Local and Kaggle example submission CSV | `9a4c0c6637a3c48e6467f617e6bfe887bca9f0c7ee78ec54349eb2cd4e4bb497` |

The CSV hash identifies the downloadable three-study example output, not the inaccessible hidden-test CSV. Generated notebook, model and account execution records stay in ignored artifacts. `artifacts/kaggle/metadata/push.json`, `run.json` and `submission_record.json` preserve the version, settings, exact submit arguments, timestamps, status and score. The experiment manifest records input hashes, packages, modeling source snapshot and per-source hashes because the implementation was uncommitted at run time.

The modest public score does not support a claim of useful MRI understanding. The validation/public gap is plausible with only 58 labeled studies and acquisition-protocol shortcuts; neither patient independence nor a representative validation population has been established. Keep this submission as an execution and comparison baseline. Next: inspect a bounded DICOM/header sample on Kaggle, check patient and image duplicates, and evaluate a compact image model on the saved split (or explicitly revise the split if stronger grouping evidence requires it).

## Public image reference — version 1

The second submission uses the audited [pilkwang public baseline](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) source (Apache 2.0), public `pilkwang/rsna-knee-weights` (CC0-1.0), and generic `metaresearch/dinov2/PyTorch/small/1`. Its 20 competition-trained members cover four seeds and five source folds. Their training exposure prevents treating this ensemble's predictions as independent validation on our saved folds. See [public reference](public-reference.md) for the unchanged checkpoint preprocessing and fail-closed packaging.

Private, internet-disabled GPU version 1 completed the **three example studies in 83.44 seconds** on a Tesla T4. All **20 members × 10 overlapping windows** finished; restricted checkpoint loads and finite fingerprints passed. Twelve available series slots decoded with zero failed series. The resulting three-row CSV passed exact ID/order, target-column and finite-probability validation. This verifies the example execution, not hidden-test performance or runtime.

The authenticated Kaggle API checked on **September 16, 2026 at 21:49:17 UTC** confirmed ref **56286555 COMPLETE**, public AUC **0.891**, with no error description. It points to submitted script version `350397321`. `scoring_result.json` preserves the check timestamp and returned status/score. This confirms actual hidden-test completion; the 83.44-second measurement above concerns only example data.

| Artifact | SHA-256 |
| --- | --- |
| Reviewed public source notebook | `b32a9155fcc73c519e75e78c08e07d30393efc591c2798a3e40ca92f39832bb8` |
| Submitted private version 1 notebook | `de1130fdac3ef546ef30ed7d1db3826e78c11356267dbc8519196cf2f7f1a55f` |
| Public weights manifest | `496949a3a3e789bc1f4ccff595205c911e471c5b5ef669366a2dd0a58e125844` |
| Downloaded three-study example CSV | `78f0499655c92a0de5486b15c69e546e269e9def647e0d7ea3ec11a16d66ee89` |

Evidence is preserved under `artifacts/kaggle/image-reference/versions/v1/`: the submitted notebook, metadata, build manifest, submission record and `output/run_manifest.json` with every loaded checkpoint hash, window count, package versions, hardware and elapsed time. These CSV and runtime measurements concern only visible example data. Its confirmed public score is **0.891**. The result was obtained without further model changes or leaderboard-based tuning.

## Independent image model — version 1

The generic DINOv2-small encoder was frozen throughout. Twelve regularized heads
were trained using observed labels and masked report-derived labels at weight 0.25,
with usable supervision for 4,354 studies. Mean gold-label three-fold validation
AUC is **0.697812**, compared with 0.651561 for images trained on observed labels
alone and 0.598926 for metadata. Selection used these fixed comparisons; a single
PCA-32 candidate failed its registered AUC requirement and was rejected.

The corrected split groups the demonstrated duplicate pair and preserves all 58
observed-label assignments. Patient independence remains unresolved; the public
label generator's independence from those 58 studies is also unknown. Full
uncertainty and error analysis are in [experiments](experiments.md).

Private offline T4 example inference completed in **8.505 seconds**, with exact
ID/order coverage, verified generic checkpoint hashes and maximum absolute local
prediction difference **3.83794e-6**. This passed the preregistered absolute/relative
1e-4 tolerances. Inference reads only current test metadata and images; it needs
no reports or training labels. Ref **56287107** submitted this exact version.

| Artifact | SHA-256 |
| --- | --- |
| Selected head model | `c1b52ab7e7cf5f67538675e894559aeeafe5361f965faa5f2627803d4080eb3a` |
| Submitted notebook | `877355a1da3e6035e14e3633bfce492b3a5a2185cad1cbae02c5e58524f865c1` |
| Kaggle example CSV | `1ec762d820b40270bd38e3f37a893235584eacca489a21b3ea113a218b9c98a5` |
| Corrected image split | `23c611d98c910549c5c143b30de436d4217214aae239f15850cf02db2cd6ba21` |

Build, inference, parity and submit records live in
`artifacts/kaggle/independent-image/versions/v1/`. Example measurements do not
establish hidden-test completion. Current implementation verification:
**171 tests passed**, including fold exclusion, masking, geometric ordering,
checkpoint/source provenance, replacement IDs and PCA coefficient parity.

Authenticated Kaggle status checked at **20:51 UTC** on September 16 confirmed
ref **56287107 COMPLETE**, public AUC **0.718**, with no error description.
`scoring_result.json` preserves the response fields and check timestamp. This is
the actual competition result, separate from the **0.697812** local CV score.
No further recipe changes or submission decisions were made from this score.

## Independent adapted model — version 1

The selected model starts from generic DINOv2-small and learns its last two
encoder blocks, final LayerNorm and diagnosis-specific attention head. It uses
three selected MRI planes, twelve central slices per plane and all ten adjacent
three-slice windows at inference. Reports are absent from inference. The same
audited quarter-weight supervision provides 4,354 usable training studies; the
58 observed-label cases remain the sole local validation truth. The report pilot
did not change this training table.

The full matched GPU comparison completed all three folds and final refits for
both arms in 1,832.16 seconds. Local mean AUC was **0.759911** for adaptation,
**0.697164** for its matched frozen control and **0.697812** for the original
independent model. Selection was frozen at **00:49:17 UTC**, before this new
submission or score. Conditional paired intervals, target tradeoffs and the
limits of 58 reused validation studies are recorded in [experiments](experiments.md).

Private, offline T4 inference version 1 completed the three visible examples in
**9.343 seconds** and reproduced the saved training-run probabilities exactly
(maximum absolute difference **0**). Model, training-summary, preprocessing and
test metadata hashes passed. The full **224-test** suite includes whole-fold
exclusion, unknown-label masking, actual encoder gradients, safe checkpoint
loading, uint8 preprocessing parity and 1,300 replacement IDs. These checks and
example timing remain distinct from hidden-test scoring.

| Artifact | SHA-256 |
| --- | --- |
| Selected independent full model | `9b4097edc1fa34ba8c2f4f9e4a1cdfbf8f2a135d2e0034762edc2788f8015af0` |
| Training summary | `54e06ca508fba211883160f5457a5401773bbda30977b655bb405a96a841a647` |
| Submitted notebook | `7ddd9155bceb5288c9c8af8d8395caf60ac629aaa3c07973a0f0cc1a8aaf2b8b` |
| Kaggle example CSV | `f478ac815ab921c765bf32195a086a9915c11fb9e23d8dca7aab6cdf40d3d769` |
| Unchanged image split | `23c611d98c910549c5c143b30de436d4217214aae239f15850cf02db2cd6ba21` |

Training checkpoints remain in private
`willmurray99/rsna-knee-adaptation-training`, version 1. Local compact training
evidence lives under `artifacts/kaggle/adaptation-training/versions/v1/output/`;
the submitted package, offline output, parity, request and status history live
under `artifacts/kaggle/adapted-image/versions/v1/`. The example CSV hash identifies
only the visible three rows, not the inaccessible hidden-test predictions.

Kaggle accepted ref **56290319** at **00:50:24 UTC** on September 17. The
authenticated API confirmed **COMPLETE**, public AUC **0.780**, with no error
description, at **01:22:05 UTC**. `scoring_result.json` preserves that response.
This is our new independent best, **+0.062** over 0.718. The interval from
submission to observed completion includes queueing and does not measure hidden
inference alone. No model or selection change followed the leaderboard result.
