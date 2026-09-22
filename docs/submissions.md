# Submission log

The reproduced public ensemble's **0.891** (ref **56286555**) is our primary submission baseline and best scored submission. Our best independently trained submission is **0.819** (ref **56471807**), up from **0.801** for three-window training, **0.780** for one-window adaptation, **0.718** for the frozen image model and **0.508** for metadata. Six submissions are complete. The fixed 90/10 public/independent rank blend passed exact component parity and independent arithmetic checks and was submitted as **56473633**, with hidden scoring pending. Three-window training remains the locally selected independent baseline because the ten-window diagnostic failed its local promotion rule.

This competition executes a notebook against hidden test data. Local example CSV validation is only a format check. Record actual submissions here when they happen.

| Date UTC | Notebook / version | Code commit | Artifact hashes | Reason to submit | Public AUC | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-09-16 19:21:24 | [RSNA Knee First Submission](https://www.kaggle.com/code/willmurray99/rsna-knee-first-submission), version **2**; ref **56286205** | `3b79c70` + recorded working-tree source hashes | Full hashes below | Prespecified metadata model beat both constant references on frozen CV; offline CPU output verified | **0.508**, COMPLETE | First submission milestone achieved. Preserve as baseline; next investigate image features, without leaderboard-driven tuning. |
| 2026-09-16 19:50:10 | [RSNA Knee Image Reference](https://www.kaggle.com/code/willmurray99/rsna-knee-image-reference), version **1**; ref **56286555** | Working-tree packager; versioned notebook hash below | Versioned build, weights and execution manifests | Audited public image reference; all 20 members × 10 windows passed private offline GPU example execution | **0.891**, COMPLETE | Best submitted public score; external competition-trained ensemble, with no valid CV result on our saved folds. |
| 2026-09-16 20:37:58 | [RSNA Knee Independent Image](https://www.kaggle.com/code/willmurray99/rsna-knee-independent-image), version **1**; ref **56287107** | `3b79c70` + recorded source snapshots/hashes | Versioned build, model and inference manifests | Quarter-weight report supervision selected on corrected frozen folds; PCA-32 rejected; offline parity verified | **0.718**, COMPLETE | Successful hidden-test result, +0.210 over metadata; retain as independent image baseline. |
| 2026-09-17 00:50:24 | [RSNA Knee Adapted Image](https://www.kaggle.com/code/willmurray99/rsna-knee-adapted-image), version **1**; ref **56290319** | `3b79c70` + exact working-tree training/inference source hashes | Versioned training, build, selection and parity manifests | Highest eligible local mean AUC **0.759911**; improves all three folds over original and matched frozen control; exact offline parity | **0.780**, COMPLETE | New independently trained best, **+0.062** over 0.718; no recipe changes from leaderboard feedback. |
| 2026-09-18 22:04:43 | [RSNA Knee Multi Window Image](https://www.kaggle.com/code/willmurray99/rsna-knee-multi-window-image), version **1**; ref **56341808** | `2f4daefd1cc0baddf08807867edbd5dd346df2e5` | Full hashes below | Local AUC **0.774776**, all three folds improve over exactly reproduced one-window control; exact offline parity | **0.801**, COMPLETE | Independent best, **+0.021** over 0.780. |
| 2026-09-22 18:38:27 | [RSNA Knee All Window Image](https://www.kaggle.com/code/willmurray99/rsna-knee-all-window-image), version **1**; ref **56471807** | `ec3b6867c8be8b081ba8215054450fe794585c82` | Full hashes below | User-requested diagnostic; local **0.769655** fails promotion versus **0.774776**; exact offline parity | **0.819**, COMPLETE | New independent public best, **+0.018**; retain three-window locally selected baseline and the decision frozen before public feedback. |
| 2026-09-22 20:34:27 | [RSNA Knee Reference Blend](https://www.kaggle.com/code/willmurray99/rsna-knee-reference-blend), version **2**; ref **56473633** | `5a1d9fd77c4d63ab9741c259767db78bdd9cec25` | Full hashes below | Fixed 90/10 targetwise rank blend; both parents reproduce exactly and independent arithmetic matches; no valid local CV | Pending | One preregistered submission; retain 0.891 reference unless the score is strictly higher. |

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

## Independent three-window model — version 1

The September 18 comparison changes training from one to three distinct windows
per MRI plane. Each window still contains three sampled slices and yields 768
encoder features. The same final two DINOv2 blocks and attention head learn
jointly from all selected windows; inference still uses all ten windows per
present plane. Binary supervision, weights and saved folds are unchanged; no new
labels were generated.

Local mean within-fold AUC increased from **0.759911 to 0.774776**, with gains in
all three folds. The rerun control exactly reproduced the saved predictions,
epoch losses and checkpoint hashes. Independent verification agreed with the
preregistered promotion decision, frozen at **22:01:30 UTC** before submission.
The conditional bootstrap interval includes zero; see [experiment outcomes](experiments.md)
for target tradeoffs and the limits of 58 reused gold studies.

Private offline notebook `willmurray99/rsna-knee-multi-window-image`, version 1,
completed the three visible examples in **13.755 seconds**. All probabilities
matched the saved training-run predictions exactly (maximum absolute difference
**0**), passing the fixed absolute/relative 1e-4 tolerances. Selected checkpoint,
training summary, window schedule, preprocessing, test metadata and all twelve
packaged source files were verified. Dynamic-ID tests cover 1,300 replacement
studies. Inference uses test images without reports or labels; these checks and
example timing do not establish hidden-test completion.

| Artifact | SHA-256 |
| --- | --- |
| Selected independent full model | `1abd7c35d59d85d2e08f507c76a8996aca43e2ddd9ea18badf7bf615967b0bd5` |
| Training summary | `05f01fd06bcc2e17b49d7059d16c70d558b6e47a2e0c77872c92a9d4d483fd45` |
| Submitted notebook | `51183a5e235a93869c87c5707735023cf066fcfe7acc891c7fcdc33f8b6289d5` |
| Kaggle example CSV | `38bdb0dd4a938185ebf90cd6de869e645313a1e84797b554871c913201778eaf` |
| Unchanged image split | `23c611d98c910549c5c143b30de436d4217214aae239f15850cf02db2cd6ba21` |

The producing training source is clean, pushed revision
`2f4daefd1cc0baddf08807867edbd5dd346df2e5`. Training checkpoints remain in private
`willmurray99/rsna-knee-multi-window-training`, version 1. Compact training and
independent verification records live under `artifacts/reports/multi-window-v1/`
and `artifacts/kaggle/multi-window-training/versions/v1/output/`. Inference,
parity, request and scoring records live under
`artifacts/kaggle/multi-window-image/versions/v1/`. The example CSV hash identifies
only the visible three rows, not the inaccessible hidden-test predictions.

Kaggle accepted submission **56341808** at **22:04:43 UTC** on September 18.
The authenticated API confirmed **COMPLETE**, public AUC **0.801**, with no error
description, at **22:39:52 UTC**. `scoring_result.json` preserves the response.
This is our new independent best, **+0.021** over 0.780. The approximately
35-minute interval includes queueing and does not measure hidden inference
alone. No model or selection change followed the leaderboard result.

## Independent all-ten-window diagnostic — version 1

The September 22 experiment trains jointly on all ten cached three-slice windows
per MRI plane. Each window still yields 768 features; inference coverage,
architecture, binary supervision and saved folds are unchanged. No new labels
were generated. All eight comparison fits completed in **10,026.11 seconds**
from clean, pushed source `ec3b6867c8be8b081ba8215054450fe794585c82`.

The three-window control exactly reproduced the saved predictions, epoch losses
and recorded checkpoint hashes. Ten-window local mean AUC was **0.769655** versus
**0.774776**, improving two folds but failing the required mean improvement.
Independent verification agreed. The three-window model remains selected;
`selection.json` froze this decision at **18:35:16 UTC**, before public feedback.
As preregistered, this one requested candidate submission is diagnostic despite
failed local promotion. See [experiments](experiments.md) for target tradeoffs,
bootstrap uncertainty and validation limitations.

Private offline notebook `willmurray99/rsna-knee-all-window-image`, version 1,
completed the three visible examples in **16.556 seconds**. Predictions matched
the training-run output exactly (maximum absolute difference **0**), passing the
fixed absolute/relative 1e-4 tolerances. The actual checkpoint, training summary,
window schedule, test metadata and all twelve packaged source files were verified.
IDs and all twelve probabilities per study passed validation. Dynamic-ID tests
cover 1,300 replacement studies. These example checks are separate from hidden
test scoring; inference needs no reports or labels.

| Artifact | SHA-256 |
| --- | --- |
| Diagnostic candidate full model | `d1a796a6945837c7a503271bfb31d028671d41caa340c89349a8972082861172` |
| Training summary | `fb0902ed78f61f31027854e8e35c2264962f04f8c83bc3ee5dcb68bae36ddc9f` |
| Submitted notebook | `3b4ce9dd29b6f5c0bfe8d4e7f5885c979efd1a2d4ba43d7a33a81e9f65150100` |
| Kaggle example CSV | `d705fc6602a8844b39c6ebd599eb33bc48e80a4f855f2bea4327ef7de92dfd88` |
| Frozen selection | `1f4a36b15777280d18f41d8c34e2b9483025c2637cac007660a6b0d830348add` |
| Unchanged image split | `23c611d98c910549c5c143b30de436d4217214aae239f15850cf02db2cd6ba21` |

Training checkpoints remain in private
`willmurray99/rsna-knee-all-window-training`, version 1. Compact comparison,
independent verification and frozen decision records live under
`artifacts/reports/all-window-v1/`. Training evidence is under
`artifacts/kaggle/all-window-training/versions/v1/output/`; inference, exact
parity, request and scoring records are under
`artifacts/kaggle/all-window-image/versions/v1/`. The example CSV hash identifies
the visible three rows only. Source validation passed **336 local tests** and
**335 Linux CI tests**, with one expected optional integration skip in CI.

Kaggle accepted submission **56471807**, requested at **18:38:27 UTC** on
September 22. The authenticated API confirmed **COMPLETE**, public AUC **0.819**,
with no error description, at **19:10:45 UTC**. `scoring_result.json` preserves
the response. This is our new independent public best, **+0.018** over 0.801.
The approximately 32-minute interval includes queueing and does not measure
hidden inference alone.

The public gain and local decrease are both retained in the record. The three-window
model remains the locally selected baseline under the frozen rule; its public
score is still 0.801. The ten-window candidate is now the best independent public
submission, without retroactively changing local promotion or tuning another
recipe from this result.

## Fixed public-reference rank blend — version 2, scoring pending

The [registered experiment](reference-blend-plan.md) combines **90% public
ensemble ranks and 10% independent all-ten-window ranks**, using ascending
average-tie percentile ranks per target over the complete runtime test set. The
parents are public-reference version 1, ref **56286555** (**0.891**), and
independent all-window version 1, ref **56471807** (**0.819**). There is no valid
local CV for this blend because public-checkpoint training exposure is unresolved.
Labels, folds, checkpoints and image recipes remain unchanged.

Version 1 stalled with two public members recorded and was not submitted. The
only repair redirects verbose child output into separate log files. Both parent
notebooks, models, preprocessing and recipe remain identical. Successful replay
supports, but does not prove, an output-capture diagnosis; version 1's termination
remains unconfirmed because the public API exposes no usable session ID for safe
cancellation. See [the execution record](experiments.md) for the preserved evidence.

Private offline T4 version **2** completed the three example studies in
**95.585 seconds** from clean, reviewed source
`5a1d9fd77c4d63ab9741c259767db78bdd9cec25`, after **379 local tests** and passing
Linux CI. Both component outputs matched their scored-parent examples exactly,
and independent rank arithmetic matched the final CSV exactly. All twenty public
members and ten windows, the independent checkpoint/summary/schedule, runtime
test metadata, original runners and source files passed verification.

| Artifact | SHA-256 |
| --- | --- |
| Submitted version 2 notebook | `75f23088a925f8b3e43218fe5ad73a3dedc048b319a5f09f75f75ff0b4be938a` |
| Build manifest | `5762d8db33c32ee5a13c692c6b0a8671e962161ace938c74acef987ecdd6dc10` |
| Kaggle example blend CSV | `d0f674c3008f80bbbf0684e8f98c1c62c35e11535fac30b20fe0c99c25475627` |
| Independent technical verification | `c7316c2ecdd6080b0daee02a62f964e517d2820577117b179e81e361b6606107` |

Kaggle accepted ref **56473633**, requested at **20:34:27 UTC** on September 22,
after the technical submission decision was frozen. Hidden scoring is pending;
the example CSV hash and timing concern only the three visible studies. A public
score strictly above 0.891 will be a provisional new best; equal or lower retains
the pure public ensemble. No weight search or repeat tuning from leaderboard
feedback is part of this experiment. The submitted version's launch, parity,
request and execution evidence is under `artifacts/reports/reference-blend-v2/`
and `artifacts/kaggle/reference-blend/versions/v2/`; version 1 evidence remains
preserved separately.
