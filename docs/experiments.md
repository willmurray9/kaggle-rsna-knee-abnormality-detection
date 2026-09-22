# Experiment log

Keep one row per hypothesis. Detailed outputs live under `artifacts/experiments/<run_id>/`; explain consequential decisions here. Reuse the frozen split and reference across comparisons.

| Date | Run | Hypothesis / change | Local result | Decision |
| --- | --- | --- | --- | --- |
| 2026-09-16 | constant | Verify data, score and output plumbing with 0.5 probabilities | Expected macro AUC 0.500; no learned model or CV | Sanity reference only; no upload |
| 2026-09-16 | metadata-v1 | Acquisition series counts may carry weak disease signal. Compare fixed C=0.1 logistic regression with 0.5 and fold-training prevalence on one frozen three-fold study/report-group split; no reports, images or UIDs as predictors. Hypothesis and selection rule registered before fitting. | Mean fold macro AUC **0.598926**, fold SD 0.043192; both references 0.500000 | Selected by prespecified mean AUC > 0.500 rule. No hyperparameter/seed search or leaderboard-driven changes. |
| 2026-09-16 | public-image-reference | A vetted public DINOv2/report-supervised image ensemble should provide a stronger submission reference. Reproduce its exact pixel recipe with strict output validation and safe checkpoint loading. | Private offline example passed: 20 members × 10 windows; actual submission 56286555 **COMPLETE**, public AUC **0.891** | Best submitted public score. Treat as external reference; public competition-trained weights are not valid local CV. |
| 2026-09-16 | frozen-image-supervision-v1 | Frozen DINOv2-S image features should outperform acquisition counts; additional masked report-derived supervision should improve the same regularized heads. Compare explicit-only, report-derived weight 1.0 and report-derived weight 0.25 on frozen folds, C=0.1, no hyperparameter search. | Mean fold observed-label AUC: explicit-only **0.651561**, full silver **0.691281**, quarter silver **0.697812**; comparison/refit 12.84 seconds | Quarter silver selected by the prespecified highest-mean rule. Its small advantage over full silver is inconclusive; preserve all OOF predictions and inspect probability errors. |
| 2026-09-16 | frozen-image-pca32-v1 | The quarter-weight silver model improves ranking but has worse Brier error than gold-only and 58/696 incorrect probabilities outside 0.05–0.95. Test whether a 32-component unwhitened PCA restriction reduces unstable high-dimensional fitting. Keep encoder, folds, labels, C=0.1 and silver weight 0.25 fixed; fit scaler/PCA only on training rows, seed 20260916, no rank search. | Mean AUC **0.692956**, delta **−0.004856**; two folds improved, mean-fold Brier 0.223942 | Rejected by the registered mean-AUC requirement; retain quarter-weight baseline. This is exploratory reuse of 58 validation cases. |
| 2026-09-16 | coverage-mean-v1 | Twelve central slices and ten neighboring three-slice windows, averaged per plane, may recover information lost by the sparse sample. Keep generic encoder, 224 px, three selected series, labels, silver weight 0.25 and C=0.1 fixed. | Mean AUC **0.712094**, delta **+0.014282**; two folds improved | Passes declared promotion rule; conditional interval includes zero. Offline inference passed; superseded by the adapted candidate before competition submission. |
| 2026-09-16 | coverage-attention-v1 | A small learned diagnosis-specific aggregator of the same frozen window cache may outperform fixed mean pooling and linear heads. Fixed six epochs, 128 hidden units, dropout 0.2, AdamW LR 0.001/weight decay 0.02, batch eight, seed 20260916; silver weight 0.25. | Mean AUC **0.707859**, delta **+0.010047** versus original; **−0.004235** versus coverage mean | Passes original-reference rule but does not improve on mean coverage; retain as a completed comparison. No epoch or hyperparameter search followed. |
| 2026-09-16 | coverage-adaptation-v1 | Compare frozen versus final-two-block DINOv2 adaptation on identical uint8 pixels and deterministic sampled training windows. Same attention head, six epochs, batch eight, AdamW head LR 0.001/backbone LR 0.000008, decay 0.02, silver weight 0.25; all windows at inference. | Frozen control **0.697164**; adapted **0.759911**. Adaptation improves all three folds versus both control and original; complete job 1,832.16 seconds | Adapted final refit selected before leaderboard feedback. Exact offline parity passed; submission **56290319 COMPLETE**, public AUC **0.780**, new independent best. |
| 2026-09-17 | depth-adaptation-v1 | Train final six versus two DINOv2 blocks; keep labels, saved folds, pixels, head, six epochs, optimizer and sampling fixed. Rerun the two-block control and compare against the saved reference. | Two-block control **0.759911**, exactly reproducing saved predictions; six-block candidate **0.748275**, delta **−0.011635**, one of three folds improves; complete job 2,258.52 seconds | Rejected by the preregistered rule. Retain independent public best **0.780**; no new submission or per-target blend. |
| 2026-09-22 | all-window-v1 | Train on all ten versus three windows per plane with unchanged labels, folds, architecture and six-epoch recipe. | Three-window control exactly reproduced **0.774776**; ten windows **0.769655**, delta **−0.005122**, two folds improve; 2.79-hour comparison | Local promotion failed; three-window baseline retained. Requested diagnostic submission **56471807 COMPLETE**, public **0.819**, new independent public best **+0.018**. No leaderboard-driven retuning. |

The independent improvement pass is specified in
[its plan](independent-improvement-plan.md). The separate 120-report rules pilot
failed its release checks; [pilot findings](report-pilot.md) document low coverage
and interpretation errors. Its annotations remain outside model training. The
0.891 external ensemble is not an initializer or valid local comparison model.

For each learned run, save settings/seed, input and split hashes, producing commit and dirty state, dependency versions, preprocessing and weight provenance, held-out predictions, macro/per-target scores, runtime and a brief error review. Prefer a plain `summary.json` plus CSVs to a tracking service. Preserve earlier runs instead of overwriting them.

**Source-control provenance, September 17:** the experiments below were completed
while development changes remained uncommitted on top of scaffold `3b79c70`.
The subsequent Git catch-up groups the implemented work into logical commits;
those commits were created after the runs and are not their original producing
revisions. Reproduce historical runs using their recorded source hashes and
preserved source/build snapshots, rather than the scaffold revision alone.
Original manifests remain unchanged. For future experiments, commit and push
the reviewed source and recipe before launch, record that revision alongside
source hashes, and commit outcomes separately after evaluation.

Before any image-head fitting, the completed image audit found one duplicate pair
across original folds 0 and 1: all nine sampled pixels, dimensions and geometry
matched despite different anonymized patient keys. Neither study had observed
labels and both report groups were singletons. The versioned `image-v1` split
moves one study from fold 1 to fold 0; all 58 observed-label assignments remain
unchanged. Original artifacts are preserved. The new split SHA-256 is
`23c611d98c910549c5c143b30de436d4217214aae239f15850cf02db2cd6ba21`.
All image comparisons use this corrected split, and `metadata-image-v1` refreshes
the unchanged metadata recipe against its provenance. This correction responds to
duplicate evidence, not scores or a new seed search.

The `metadata-image-v1` refresh is verified byte-identical to `metadata-v1` for
`model.json`, `fold_models.json`, `oof.csv` and `submission.csv`. Learned OOF
probabilities differ by exactly zero; mean fold AUC remains **0.5989261703** and
the 1,000-attempt bootstrap is unchanged. Only split/run provenance changed.

Offline image-inference parity used absolute and relative tolerance 1e-4,
fixed before execution. Identical sampled images produced embedding differences
up to 2.9e-5 across GPU batches, so byte equality is not the appropriate check.
The independent quarter-silver version 1 example notebook passed on a Tesla T4
in **8.505 seconds**, with maximum local/Kaggle probability difference
**3.84e-6** across three studies. The subsequent hidden-test submission completed
with public AUC **0.718**; the example runtime remains a separate measurement.

The constant baseline is a regenerable fixture at `artifacts/baselines/constant/`; rerunning it overwrites that fixture. Its `summary.json` records input/config/submission hashes, environment, code revision and the metric sanity check.

## Metadata baseline: first learned submission candidate

Run: `PYTHONPATH=src .venv/bin/python -m rsnaknee.baseline --output artifacts/experiments/metadata-v1`. This directory is preserved; subsequent runs must use a new output directory (`make metadata-baseline` chooses a timestamp). Fitting, validation and 1,000 bootstrap attempts took 51.8 seconds locally. No GPU, images, external data or pretrained weights were used.

Features are total series count, three anatomical-plane counts, two fluid-sensitivity counts and two fat-suppression counts. Categories are fixed; IDs only align rows. The two acquisition flags are redundant in the current snapshot; the recipe was kept unchanged rather than tuning after seeing scores. Each target uses `StandardScaler` and `LogisticRegression(C=0.1, max_iter=1000, random_state=20260916)` with no class weighting. Each fold fits preprocessing and models exclusively on its observed training cases; all 4,349 fully unlabeled studies are excluded from fitting. Labels come only from the observed competition columns.

The seed-42 split and its rationale are in [EDA](eda.md). Split SHA-256: `d1c1f632ed2d87709a984d94365389983b1adbad15704a9aa8e5a98397831910`. Fold macro AUCs were 0.647713, 0.583499 and 0.565566. Both the global 0.5 predictor and each fold's training-prevalence predictor scored exactly 0.5 within every fold. We average within-fold scores; pooling different prevalence constants across folds would create an artificial ranking.

| Target | Mean validation AUC |
| --- | ---: |
| ACL | 0.714206 |
| MCL | 0.534722 |
| Medial Meniscus | 0.426789 |
| Lateral Meniscus | 0.535426 |
| Medial OA | 0.693527 |
| Lateral OA | 0.605502 |
| PF OA | 0.734525 |
| Effusion | 0.441138 |
| Synovitis | 0.654655 |
| Baker's | 0.557217 |
| Contusion | 0.597344 |
| Fracture | 0.692063 |

Joint study resampling within each fold gave an exploratory 95% percentile interval of **+0.0440 to +0.1558** for the mean AUC difference versus the constant reference (point difference +0.0989). Only 694 of 1,000 attempts retained both classes for every target in every fold; 306 were excluded. This conditional interval holds the fitted models fixed and omits fitting, split and unresolved patient-dependence uncertainty. It is not evidence of reliable population generalization. The 58 labeled cases all have distinct report groups, so study and report-group resampling coincide for this experiment.

Error review: Medial Meniscus and Effusion rank below chance in these folds; PF OA and ACL rank highest. We retain all 12 prespecified heads rather than selecting/reversing individual targets after viewing results. Saved `oof.csv` and `largest_errors.csv` support case-level review without raw reports. Metadata cannot distinguish pathology among studies with identical acquisition counts; image inspection and image-based learning are the next substantive step.

There are only 20 distinct metadata profiles among 58 labeled studies; 48 cases share 10 repeated profiles, each with conflicting labels. The most common profile covers 16 labeled cases and contains both classes for all 12 targets. Across 4,407 training studies there are 135 profiles, and the eight-feature matrix has rank four. The 20 largest individual squared-probability errors are positive conditions assigned probabilities 0.036–0.172, especially Lateral OA (6), MCL (4) and Baker's (3). The ten worst study-level predictions have median 6.5 positive findings versus four overall. These aggregate findings motivate image features; they do not justify post-hoc target tuning on this small split.

The final model refits on all 58 explicitly labeled cases. Model SHA-256: `96be94fca140d7a7832b28c75c77e5cfb8d4162ce0d9e7af7ed272b0b7f02dce`. `summary.json` records input/source/artifact hashes, the baseline revision `3b79c70` plus dirty state, package versions, fold results and the selection rule; `baseline_source.py` preserves the exact modeling source. `model.json`, fold models, features, observed labels, held-out predictions and generated submission remain in ignored artifacts.

Rejected for this milestone: replacing missing labels with zero; report inputs at inference; report-derived labels without extraction validation; a full 570 GB image download; GPU training before an image/compute plan; tuning from repeated leaderboard feedback. The metadata result may reflect acquisition or site selection and does not demonstrate MRI understanding or patient-independent performance.

After the recipe and local result were frozen, Kaggle submission 56286205 completed with public AUC **0.508**. The metadata recipe remained unchanged; subsequent image experiments are recorded separately. The gap reinforces the limits above and motivates image-based modeling; it is not a basis for tuning this small validation set or selecting individual heads.

## Frozen image features and report supervision: first comparison

Run: `artifacts/experiments/frozen-image-supervision-v1/`; analysis:
`artifacts/experiments/frozen-image-supervision-v1-analysis/`. Frozen generic
DINOv2-small feature extraction took **1,996.20 seconds** on free Kaggle compute.
The 2,307 features contain CLS and mean-patch embeddings from three fixed MRI
planes plus presence flags. The independently pretrained encoder was not fitted
on competition images or reports. The three supervision recipes share these
features, corrected `image-v1` folds, training-only scaling, and per-target
logistic heads with C=0.1. Comparison and selected-model refit took **12.84 seconds**.

The label policy contributes **37,920** known report-derived target cells in
addition to **696** observed cells; **14,268** unknown cells remain excluded.
Only the 58 observed-label studies supply validation truth. Each training fold
excludes its complete held-out studies, reports and labels before fitting.

| Recipe | Fold 0 AUC | Fold 1 AUC | Fold 2 AUC | Mean AUC | Fold SD | Mean-fold Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Observed only | 0.661358 | 0.664456 | 0.628870 | **0.651561** | 0.019712 | 0.255285 |
| Silver weight 1.0 | 0.737142 | 0.645168 | 0.691533 | **0.691281** | 0.045988 | 0.303533 |
| Silver weight 0.25 | 0.751832 | 0.649249 | 0.692356 | **0.697812** | 0.051509 | 0.272811 |

| Target | Observed only AUC | Full silver AUC | Quarter silver AUC |
| --- | ---: | ---: | ---: |
| ACL | 0.669101 | 0.710688 | 0.684497 |
| MCL | 0.347631 | 0.703840 | 0.658497 |
| Medial Meniscus | 0.500947 | 0.602609 | 0.653157 |
| Lateral Meniscus | 0.790456 | 0.735986 | 0.776816 |
| Medial OA | 0.875446 | 0.873958 | 0.882688 |
| Lateral OA | 0.814281 | 0.656873 | 0.700677 |
| PF OA | 0.627130 | 0.695830 | 0.705025 |
| Effusion | 0.746561 | 0.788360 | 0.827513 |
| Synovitis | 0.527256 | 0.595421 | 0.590067 |
| Baker's | 0.793552 | 0.607143 | 0.602381 |
| Contusion | 0.651770 | 0.739744 | 0.719414 |
| Fracture | 0.474603 | 0.584921 | 0.573016 |

Paired study resampling within the saved folds used 1,000 attempts and seed
20260916. The 58 observed report groups are distinct. **704** attempts retained
both classes for every target in every fold; **296** were excluded. Quarter silver
minus observed-only mean AUC is **+0.04625**, conditional 95% interval
**−0.00783 to +0.09693**; quarter silver minus metadata is **+0.09889**,
interval **+0.02582 to +0.17774**. Full silver minus observed-only is **+0.03972**,
interval **−0.01871 to +0.08777**. These intervals hold predictions fixed and
omit training/selection uncertainty; rare-class exclusions condition the result.
The quarter/full point difference of **0.00653** does not establish superiority.

Error inspection shows better ranking need not mean better probabilities.
Quarter silver has 16 confidently wrong cells (probability below 0.01 on a
positive or above 0.99 on a negative), versus 22 for observed-only and 53 for
full silver, among 696 observed cells. Quarter-silver Brier exceeds observed-only
despite its higher AUC. Mean Synovitis probability is 0.800 against observed
prevalence 0.466; Baker's is 0.474 against 0.207. Compared with observed-only,
quarter silver improves MCL AUC by 0.311 and Medial Meniscus by 0.152 but reduces
Baker's by 0.191 and Lateral OA by 0.114. These are descriptive findings, not
permission to tune individual targets or extraction thresholds. Per-target
errors and probability diagnostics are saved without report text.

Selected quarter-silver model SHA-256:
`c1b52ab7e7cf5f67538675e894559aeeafe5361f965faa5f2627803d4080eb3a`.
The source/image manifests, all recipe OOF predictions, fold models and analysis
input hashes are preserved. The single prespecified PCA-32 restriction was evaluated and rejected, as detailed
below. Extra nonlinear head capacity and encoder training are deferred.

Limits: only 58 gold cases, with two or three positives in some fold/target cells;
selection reuses that validation evidence. The public report-label generator's
independence from these gold studies is unknown, so even group-excluded image
training does not establish fully independent evaluation of the overall labeling
pipeline. Report definitions, selective missingness and annotation thresholds may
differ. The duplicate correction addresses demonstrated overlap only; anonymized
patient IDs and a selected-slice audit do not establish patient independence.

## PCA-32 restriction: rejected

`artifacts/experiments/frozen-image-pca32-v1/` preserves the candidate, OOF
predictions and exact experiment source. Only quarter-weight silver supervision
was tested, using unwhitened randomized PCA with 32 components and seed 20260916
fitted inside each training fold. It retained approximately 58.7–58.9% of training
variance. The PCA and logistic coefficients were combined into the same portable
2,307-input linear form; default non-PCA behavior remained unchanged.

Mean fold AUC was **0.692956**, versus **0.697812** for the saved reference.
Fold differences were **−0.053331, +0.020207, +0.018556**. Two folds improved, but
the required mean-AUC improvement failed, so the candidate was not submitted.
Mean-fold Brier improved from 0.272811 to **0.223942**; that diagnostic improvement
does not replace the competition's ranking metric. The comparison took 1.99
seconds. No component-count, seed or regularization search followed.

The unchanged quarter-silver model passed offline inference and was submitted as
ref **56287107**, private notebook version 1. Its actual hidden-test result is
recorded in [the submission log](submissions.md). The public reference remains
separate; its competition-trained members cannot support honest local ensemble
selection on these folds.

Kaggle subsequently confirmed the unchanged quarter-silver submission
**COMPLETE**, public AUC **0.718** (ref 56287107). The image recipe and PCA rejection
were frozen before this result; the score was not used to tune either model.

The public-reference submission also completed: ref **56286555**, public AUC
**0.891**, checked at 21:49:17 UTC. This is the strongest submitted result from
this pass, compared with our independent model's 0.718 and metadata's 0.508. It is
an externally trained ensemble reproduction, not a locally validated model or
evidence that our 58-case validation supports that score. No blend with this
ensemble was selected from invalid local predictions.


## Neighboring-slice coverage and learned aggregation

Coverage extraction completed all 4,410 studies in 3,576.40 seconds. All 13,230
selected series were usable, with zero decoding, ordering or spacing fallbacks
and no sampled-image matches across saved folds or train/test. The broader check
preserves the existing split; it does not establish patient independence. The
36-tile first-study preview showed no blank images or apparent aspect distortion.
The approximately 8 GB uint8 pixel cache stays private on Kaggle. Downloaded
compact outputs and source hashes were verified before fitting.

| Recipe | Fold 0 | Fold 1 | Fold 2 | Mean AUC |
| --- | ---: | ---: | ---: | ---: |
| Original quarter-weight reference | 0.751832 | 0.649249 | 0.692356 | 0.697812 |
| Coverage mean + same linear heads | 0.766401 | 0.735648 | 0.634233 | **0.712094** |
| Cached-window attention head | 0.739229 | 0.691307 | 0.693043 | **0.707859** |

The coverage mean comparison and refit took 17.51 seconds; cached attention took
14.17 seconds on the Mac. Each used unchanged labels, silver weight 0.25 and the
same saved whole-study folds. The 58 complete gold studies are the only validation
truth. All three fold models and final refits are saved in their respective
`artifacts/experiments/coverage-*-v1` directories.

Coverage versus original has paired AUC delta **+0.01428**, conditional 95%
interval **−0.02738 to +0.05163**. Attention versus original is **+0.01005**,
interval **−0.03088 to +0.05079**; attention versus coverage is **−0.00423**,
interval **−0.04532 to +0.03835**. Each bootstrap retained 704 of 1,000 within-fold
study draws with both classes present. These fixed-prediction intervals exclude
training and selection uncertainty. Coverage passes the predeclared promotion
rule but has substantial fold dispersion, including a 0.05812 loss on fold 2.
Attention has lower fold dispersion, which is not the declared selection metric.
No per-condition mixing, new seed, epoch selection or leaderboard-based change
was introduced after viewing these results.

Exact comparisons, per-target scores, training-ID exclusion checks and input
hashes are saved in
`artifacts/reports/independent-improvement-v1/frozen-comparison/`.
The coverage notebook passed offline inference in 8.83 seconds, with maximum
probability difference 1.84e-6 versus its cached-feature predictions. It was not
submitted because the completed adapted candidate below had stronger local
evidence under the same selection rule.

## Limited encoder adaptation: selected

Private training notebook `willmurray99/rsna-knee-adaptation-training`, version 1,
completed both matched arms in **1,832.16 seconds** after a disposable probe
projected 3,329.59 seconds, below the 27,000-second budget. Each arm completed
three fold fits and one full refit, all six fixed epochs. The final refits use
4,354 studies with known supervision; no pilot annotations enter training.
Both arms use the same saved windows, uint8 pixels and attention architecture.
The adapted arm alone updates the last two generic DINOv2 blocks and final
LayerNorm. No public competition-trained weights are used.

| Recipe | Fold 0 | Fold 1 | Fold 2 | Mean AUC |
| --- | ---: | ---: | ---: | ---: |
| Matched frozen control | 0.723554 | 0.687531 | 0.680408 | 0.697164 |
| Final-two-block adaptation | 0.780411 | 0.732049 | 0.767272 | **0.759911** |

Adaptation versus its matched control has mean AUC delta **+0.06275**, conditional
95% interval **+0.03072 to +0.10034**. Versus the original independent model,
delta is **+0.06210**, interval **+0.01840 to +0.10446**, with all three folds
improved. Versus coverage mean, delta is **+0.04782**, interval **+0.01169 to
+0.08498**, with two folds improved. The same 704 valid bootstrap draws and
58-case limitations apply. These are conditional comparisons, not a guarantee of
leaderboard performance or patient-independent generalization.

The adapted model improves nine target means against the original. ACL, MCL and
Contusion remain below their original values; no per-target replacement or blend
was chosen after seeing these results. The matched frozen control fails the
original-reference mean-AUC rule. Attention fails to improve mean coverage.
All candidates and their unsuccessful comparisons remain preserved.

`artifacts/reports/independent-improvement-v1/full-comparison/` contains exact
scores, paired intervals and decisions. Independent review recalculated all
72 matched-arm fold/target AUCs and verified the exact training IDs, six epochs,
30 compact artifact hashes, 11 source hashes and unchanged split/label hashes.
The eight full checkpoints remain in the private Kaggle training output;
compact predictions, source and provenance are downloaded locally.

The selected full-refit model SHA-256 is
`9b4097edc1fa34ba8c2f4f9e4a1cdfbf8f2a135d2e0034762edc2788f8015af0`.
`selection.json` freezes the decision before the new leaderboard result. Private
offline inference notebook `willmurray99/rsna-knee-adapted-image`, version 1,
completed the three examples in **9.34 seconds**, with **zero probability
difference** versus saved predictions. The complete 224-test suite passed,
including dynamic 1,300-ID inference and leakage checks. Actual competition
submission **56290319** completed with public AUC **0.780**, checked at
**01:22:05 UTC on September 17**. This improves our previous independent public
score by **0.062**. The recipe and choice remained unchanged after submission;
[submission evidence](submissions.md) records this result separately from CV.

## Deeper adaptation and fresh label sanity check — preregistered September 17

User-approved next step: a bounded fresh report-label audit, followed by one
controlled model experiment if no fundamental data integrity problem is found.
Hypothesis: adapting the final six DINOv2-small blocks improves image ranking
over the current final-two-block model with the existing supervision.

Compare `late_blocks` (two trainable encoder blocks) with `deep_blocks` (six),
both including the final LayerNorm and identical attention head. Rerun the
two-block control in the same job. Keep the existing six-epoch schedule,
seed 20260916, batch eight, head/backbone learning rates 0.001/0.000008,
weight decay 0.02, one sampled training window per plane, all ten windows at
inference, 224-pixel images, three selected series, and gold/silver/unknown
weights 1/0.25/0. Fixed final epoch only; no new augmentation, label changes,
per-target blending, seed search, epoch selection or leaderboard tuning.

Preserve the `image-v1` folds and label table exactly. Each arm trains three
whole-fold-excluded models and one fresh full-data refit from generic weights.
A disposable probe must project the complete job below 7.5 hours on free private
Kaggle compute. Existing MRI/pixel caches and weights remain on Kaggle.

Promote six-block adaptation only if its mean within-fold macro AUC is higher
and it improves at least two of three folds versus **both** the rerun two-block
control and the saved two-block reference (mean AUC 0.7599107356, public 0.780).
Recompute metrics from saved OOF predictions and report paired study bootstrap
uncertainty. Freeze the selection before any new competition submission. A
promoted candidate must pass offline inference parity and submission checks;
otherwise retain the current model and record the negative result.

The diagnostic audit samples 24 fresh nongold report groups across language and
linguistic difficulty, excluding gold-linked groups, the prior 120-report pilot
and previously displayed examples. Freeze evidence-based agent interpretations
before joining public labels. Report clear contradictions separately from
severity/definition uncertainty and missing report evidence. This is a targeted
sanity check, not clinical annotation, population accuracy estimation or a new
training label release. No changes to training supervision follow automatically
from individual disagreements. Systematic data alignment, masking or obvious
polarity corruption would block training pending investigation; semantic
disagreements are recorded as limitations of the fixed-label comparison.

All comparisons remain exploratory on 58 repeatedly reused official-label
studies; patient independence and the public extractor's independence from those
labels remain unresolved. Working evidence is kept under ignored
`artifacts/reports/depth-improvement-v1/` and `artifacts/label-sanity-v2/`.

The [fresh audit](label-sanity-v2.md) completed before launch: 24 reports,
137/168 agreement among mutually known targets, 28 threshold mismatches and
three explicit tear-denial contradictions across two reports. Independent
six-report review agreed on 70/72 tri-state decisions. This is targeted
agent interpretation, not clinical accuracy or population error estimation.
No source, join, masking or split corruption was found. The launch decision
permits this one fixed-label comparison with known noisy supervision; it does
not certify label quality or authorize broad model expansion on assumed-clean
labels. The 288 review interpretations remain outside training.

Implementation and final private/offline build passed review; 239 tests passed.
Private notebook `willmurray99/rsna-knee-depth-training` version 1 was launched
at 18:38:44 UTC on September 17 and completed successfully. Prior notebooks
and their embedded source are preserved.

### Depth comparison outcome: rejected

All eight fits completed in **2,258.52 seconds**. Both arms used the exact saved
labels, folds, sampled-window schedule, pixel cache and generic weights; their
runtime library versions match the previous experiment. All six recorded epochs
and exact training-ID sets were verified, including 4,354 studies per final fit.
Saved metadata confirms two versus six trainable blocks plus final LayerNorm.

| Recipe | Fold 0 | Fold 1 | Fold 2 | Mean AUC |
| --- | ---: | ---: | ---: | ---: |
| Saved two-block reference | 0.780411 | 0.732049 | 0.767272 | **0.759911** |
| Rerun two-block control | 0.780411 | 0.732049 | 0.767272 | **0.759911** |
| Six-block candidate | 0.790660 | 0.708115 | 0.746051 | **0.748275** |

The control reproduces all 696 held-out probabilities exactly (maximum absolute
difference zero). Six-block adaptation improves only fold 0 and loses mean AUC
**0.011635**. Its paired conditional 95% bootstrap interval for the difference is
**[−0.046552, +0.022831]**, from 704 accepted draws of 1,000. The interval includes
zero; this is insufficient promotion evidence, not proof that deeper adaptation
is always worse. Training and selection uncertainty and unresolved patient
dependence are excluded, and the 58 validation studies have been reused.

Five of twelve target means improve; seven decline. The largest declines are
PF OA, effusion and Baker cyst; contusion and MCL improve. No per-target mixing,
epoch change or seed search follows this exploratory observation. The six-block
recipe fails against both controls, so the existing **0.780** public model remains
selected. No new inference submission or competition submission was made.

Only 46 compact output files (18.1 MB) were downloaded; all eight checkpoints
remain private on Kaggle. The comparison verifies 30 compact artifacts per run.
Results, source/dependency hashes and the selection record live under
`artifacts/reports/depth-improvement-v1/`; training outputs are in
`artifacts/kaggle/depth-training/versions/v1/output/adaptation/`.

The audit also supplies a cheaper next hypothesis than full report relabeling:
preserve public score gradations instead of hardening every YES verdict to one.
In the targeted audit, 24/28 threshold mismatches have the lowest positive score,
but mild synovitis provides valid positives at that score too. A future matched
soft-target experiment must retain unknown masks and observed-label precedence;
no universal cutoff or calibrated-score claim follows from this small sample.
See the [audit and score diagnostic](label-sanity-v2.md). That follow-up was not
run in this depth experiment, and no training labels were changed.

## Preserved public scores — preregistered September 17

The user approved the next matched supervision experiment and ongoing reviewed
merges to main. [The frozen recipe](soft-target-plan.md) compares the current
binary-public-label two-block model against the same model trained on preserved
public scores for eligible derived cells. Official labels, unknown masks, weights,
folds, images, initialization and schedule remain fixed. This includes softening
public negatives to 0.08 as well as positives to 0.68/0.82/0.94; it introduces no
new labels and makes no calibration claim.

The rerun binary control must reproduce the saved reference within 1e-6 OOF
probability difference and 1e-12 fold-AUC difference. Promotion requires higher
mean AUC and improvement in at least two folds versus both controls, followed by
offline inference checks. No outcome or additional submission exists at
preregistration. The source and recipe will be committed before launch; results
will be recorded separately.

### Preserved-score outcome: rejected

Private offline notebook `willmurray99/rsna-knee-soft-target-training`, version 1,
completed on September 17. The complete comparison took **2,039.62 seconds**. The producing
source was committed and pushed before launch at clean revision
`45b185d5dda81d94cde0810fbb2597b0213bd9ce`; later CI/test-only commits did not
change the launched source. The outcome analyzer and its dependencies were
hashed and frozen before training.

| Recipe | Fold 0 | Fold 1 | Fold 2 | Mean AUC |
| --- | ---: | ---: | ---: | ---: |
| Saved two-block reference | 0.780411 | 0.732049 | 0.767272 | **0.759911** |
| Rerun binary-target control | 0.780411 | 0.732049 | 0.767272 | **0.759911** |
| Preserved-score candidate | 0.791849 | 0.724937 | 0.763052 | **0.759946** |

The binary control reproduces all 696 OOF probabilities exactly, all 24 recorded
epoch losses, and all four recorded checkpoint hashes. Both new arms share exact
training-ID order, supervision weights/counts, saved inputs, window schedule,
initialization, runtime library versions and two-block architecture. The target
fingerprints differ as intended for all four fits. Each final fit uses 4,354
studies. Across all 4,407 rows, the label table still provides 696 official and
37,920 derived target cells, with 14,268 unknown cells masked. No new labels
were generated.

The candidate's mean improvement is only **+0.00003512** and just **one of three
folds improves**, so it fails the registered gate against both controls. The
conditional paired 95% bootstrap interval is **[−0.022141, +0.019756]**, with
704 valid draws from 1,000 attempts at seed 20260916. Seven target means improve
and five decline; no per-target blend or retuning follows these observations.
This near tie does not establish that soft supervision is generally ineffective.
The interval excludes training, split and selection uncertainty, and the 58
gold studies have been repeatedly reused. Patient and public-extractor
independence remain unresolved.

A separate verifier recomputed official truth from the original `train.csv`,
fold/target AUCs, the bootstrap and the promotion rule without the primary metric
helpers. It agreed with the decision and checked training exclusions, target and
weight fingerprints, source provenance and 60 compact artifact hashes across the
new and reference runs. Only 46 output files (18.1 MB) were downloaded; checkpoint
bytes and MRI/pixel data remain on Kaggle. `selection.json` retains the existing
binary-target checkpoint and public AUC **0.780**. No new inference notebook or
competition submission was made. Evidence is under
`artifacts/reports/soft-target-v1/` and
`artifacts/kaggle/soft-target-training/versions/v1/output/adaptation/`.

Prelaunch verification passed 257 local tests and 256 tests plus one expected
optional-integration skip from a clean checkout, along with independent code
review and a generated-notebook smoke build. Added GitHub Actions now runs the
locked, offline test suite on pull requests and main. Its first Linux run exposed
two test portability issues: an oversized `python -c` argument and a one-ULP
cross-batch float32 difference. The fixes execute the same isolated script from a
temporary file and use the existing 1e-7 numerical tolerance while retaining exact
ID/order checks. No training or promotion tolerances changed. Fresh Linux CI at
`58c8f7c` passed **256 tests, one expected skip**.

A later documentation-only commit exposed the same one-ULP assumption in an
existing checkpoint test: reversing 18 inputs changes which studies occupy the
eight-row versus two-row batches. That test now separates exact saved-weight and
same-order prediction roundtrips from the 1e-7-tolerant reordered prediction check.
This is a test-only correction; the model and experimental decision are unchanged.

## Three training windows per plane: preregistered September 18

Following the user's approval, compare the existing one-window `late_blocks`
control with `multi_windows`, using three distinct three-slice windows per plane
jointly during training. Each encoder vector remains 768-dimensional; architecture,
binary labels, saved folds, six epochs and all-ten-window inference remain fixed.
The candidate includes each control window plus two independently hash-selected
positions. No labels are added. The complete recipe, memory/runtime probe,
control-reproduction requirement and promotion rule are fixed in
[the experiment plan](multi-window-plan.md). Commit and push source before launch;
record the outcome separately, whether successful or rejected.

### Three-window local outcome: promotion gate passed

Private offline notebook `willmurray99/rsna-knee-multi-window-training`, version 1,
completed on September 18. The comparison took **3,565.07 seconds** from clean,
pushed source `2f4daefd1cc0baddf08807867edbd5dd346df2e5`. Prelaunch verification
passed 289 local tests, 288 Linux CI tests with one expected optional-integration
skip, independent code review and the generated-notebook smoke build. The
outcome analyzer and independent verifier were frozen before launch.

| Recipe | Fold 0 | Fold 1 | Fold 2 | Mean AUC |
| --- | ---: | ---: | ---: | ---: |
| Saved one-window reference | 0.780411 | 0.732049 | 0.767272 | **0.759911** |
| Rerun one-window control | 0.780411 | 0.732049 | 0.767272 | **0.759911** |
| Three-window candidate | 0.805081 | 0.740177 | 0.779072 | **0.774776** |

The control reproduces all 696 held-out probabilities exactly (maximum absolute
difference zero), along with all 24 epoch losses and four checkpoint hashes.
The candidate improves mean within-fold macro AUC by
**+0.014866** and improves **all three folds** against both controls, passing the
preregistered promotion gate. Fold AUC standard deviation is 0.032664 for the
candidate and 0.025008 for the control. The paired within-fold 95% bootstrap
interval for the difference is **[−0.010256, +0.036274]**, with 704 valid draws
from 1,000 attempts at seed 20260916; it includes zero. This descriptive interval
holds fitted predictions fixed and excludes training, split and selection
uncertainty. The same 58 gold studies have been reused; patient and public-label
extractor independence remain unresolved.

Mean within-fold target AUCs show six improvements and six declines:

| Target | One window | Three windows | Difference |
| --- | ---: | ---: | ---: |
| ACL | 0.562751 | 0.606508 | +0.043757 |
| MCL | 0.608252 | 0.689542 | +0.081291 |
| Medial Meniscus | 0.677315 | 0.673464 | −0.003851 |
| Lateral Meniscus | 0.791257 | 0.778811 | −0.012447 |
| Medial OA | 0.897867 | 0.939236 | +0.041369 |
| Lateral OA | 0.778312 | 0.776353 | −0.001959 |
| PF OA | 0.814491 | 0.829876 | +0.015385 |
| Effusion | 0.911640 | 0.883862 | −0.027778 |
| Synovitis | 0.780825 | 0.755269 | −0.025556 |
| Baker's | 0.891518 | 0.861161 | −0.030357 |
| Contusion | 0.673748 | 0.732601 | +0.058852 |
| Fracture | 0.730952 | 0.770635 | +0.039683 |

Both arms retain identical training-ID sets, target and weight fingerprints,
binary labels, saved folds, generic initialization, optimizer settings and
trainable parameter counts for each corresponding fit. Each final fit uses
4,354 studies with 696 official and 37,920 derived target cells; unknown labels
remain masked. No new labels were generated. The candidate's three windows
cover a mean **7.335 distinct cached slice positions per scheduled plane**, versus
three for the control, while retaining 768-dimensional window vectors and the
same attention head. Broader image exposure and joint attention context change
together, so this experiment cannot attribute the improvement to either alone.
No per-target blending or further tuning follows the target results.

The 64-step full-batch probe recorded peak reserved GPU memory of **1.491 GB** for
the candidate and **0.770 GB** for the control on a **15.636 GB** device. Its
4,956.86-second projection passed the 7.5-hour gate; the inference projection
excluded DICOM preparation. An independent verifier recomputed AUCs, bootstrap
and selection from original official truth and checked fold exclusions,
supervision, source provenance and 61 compact artifact hashes across the new and
reference runs. Only 47 compact output files (18.73 MB) were downloaded; MRI,
pixel-cache and checkpoint bytes remain on Kaggle.

The local decision was frozen at 22:01:30 UTC before leaderboard feedback.
Private offline inference version 1 passed its provenance and submission checks
in 13.75 seconds and reproduced all visible-example probabilities exactly.
Kaggle accepted submission **56341808** at 22:04:43 UTC. At 22:39:52 UTC, the
authenticated API confirmed **COMPLETE**, public AUC **0.801**, with no error.
This improves our independent public best by **0.021** over 0.780. No model or
selection change followed the leaderboard result. See [submission evidence](submissions.md).
Local evidence is under `artifacts/reports/multi-window-v1/`; training outputs
are in `artifacts/kaggle/multi-window-training/versions/v1/output/adaptation/`.

## All ten training windows per plane: preregistered September 22

Compare the current three-window model with all ten cached windows jointly during
training. Labels, folds, architecture, six epochs and all-ten-window inference
stay fixed. The candidate still uses three slices and 768 features per window.
[The fixed recipe](all-window-plan.md) records control reproduction, compute,
local promotion and technical submission gates. The user requested execution
through a new submission; a technically valid candidate that fails local promotion
will be submitted only as a diagnostic, without replacing the selected baseline.
No outcome is known at preregistration. Commit and push source before launch and
record the outcome separately.

### All-ten-window local outcome: promotion failed

Private offline notebook `willmurray99/rsna-knee-all-window-training`, version 1,
completed on September 22. The comparison took **10,026.11 seconds** (2.79 hours)
from clean, pushed source `ec3b6867c8be8b081ba8215054450fe794585c82`.
Prelaunch verification passed **336 local tests**, Linux CI, independent source
review, exact synthetic control parity and a generated-notebook smoke build.
The outcome analyzer was frozen before launch; the supplemental independent
verifier was frozen before outcome analysis, without reading candidate outcomes.

| Recipe | Fold 0 | Fold 1 | Fold 2 | Mean AUC |
| --- | ---: | ---: | ---: | ---: |
| Saved three-window reference | 0.805081 | 0.740177 | 0.779072 | **0.774776** |
| Rerun three-window control | 0.805081 | 0.740177 | 0.779072 | **0.774776** |
| All-ten-window candidate | 0.779243 | 0.742395 | 0.787327 | **0.769655** |

The control reproduces all 696 held-out probabilities exactly (maximum absolute
difference zero), all 24 epoch losses and all four recorded checkpoint hashes.
The candidate improves two of three folds, with fold differences
**−0.025837, +0.002218 and +0.008255**, but lowers mean within-fold macro AUC by
**0.005122**. It therefore fails the registered promotion rule against both saved
and rerun controls. Fold AUC sample standard deviation (`ddof=1`) is 0.023952 for ten windows
and 0.032664 for three windows.

The paired within-fold 95% bootstrap interval for the mean difference is
**[−0.030818, +0.017147]**, with 704 valid draws from 1,000 attempts at seed
20260916. It holds fitted predictions fixed, excludes draws with undefined target
AUC and omits training, split and selection uncertainty. The 58 gold studies have
been repeatedly reused; patient and public-label-extractor independence remain
unresolved. This failed promotion does not establish that more windows are
generally harmful.

Mean within-fold target AUCs show three improvements and nine declines:

| Target | Three windows | Ten windows | Difference |
| --- | ---: | ---: | ---: |
| ACL | 0.606508 | 0.646243 | +0.039735 |
| MCL | 0.689542 | 0.684232 | −0.005310 |
| Medial Meniscus | 0.673464 | 0.649811 | −0.023653 |
| Lateral Meniscus | 0.778811 | 0.752066 | −0.026745 |
| Medial OA | 0.939236 | 0.893204 | −0.046032 |
| Lateral OA | 0.776353 | 0.724893 | −0.051460 |
| PF OA | 0.829876 | 0.828322 | −0.001554 |
| Effusion | 0.883862 | 0.883069 | −0.000794 |
| Synovitis | 0.755269 | 0.729949 | −0.025320 |
| Baker's | 0.861161 | 0.886012 | +0.024851 |
| Contusion | 0.732601 | 0.796947 | +0.064347 |
| Fracture | 0.770635 | 0.761111 | −0.009524 |

Both arms preserve the binary label table, frozen folds, exact training-ID order,
target/weight fingerprints, generic initialization, library versions and trainable
parameter counts for corresponding fits. Each final fit uses all **4,354**
supervised studies with **696 official and 37,920 derived target cells**; unknowns
remain masked. No new labels were generated. Ten windows expose all twelve cached
slice positions jointly, while three sampled windows cover a mean 7.335 positions
per scheduled plane. Image exposure, attention context and removal of window
sampling change together; this comparison does not isolate their effects. No
per-target blending or retuning follows the target results.

Both 64-step probes used batches of eight studies with three present planes.
Peak allocated/reserved CUDA memory was **3.670/4.041 GB** for ten windows and
**1.274/1.491 GB** for three windows on a **15.636 GB** device. The
**12,649.81-second** projection passed the 7.5-hour gate; all six epochs, three
folds and final refits completed without a recipe fallback. The cached-pixel
inference projection excludes DICOM preparation.

The primary analyzer verified 63 compact artifact hashes across the reference
and new runs. A separate verifier recomputed official truth from the original
`train.csv`, fold/target AUCs, bootstrap and promotion, and checked all eight new
fits, complete final-fit membership, fold exclusions and target/weight hashes.
It agreed with the local decision. Only **45 compact output files (11.81 MB)**
were downloaded; MRI, pixel-cache and checkpoint bytes remain on Kaggle.

The selected baseline remains the saved three-window model: local AUC
**0.77478**, public AUC **0.801**. The requested one-time ten-window submission is
**diagnostic**. Selection was frozen at **18:35:16 UTC** before public feedback.
Private offline inference passed checkpoint/source/schedule/metadata validation
and exact visible-example prediction parity in **16.556 seconds**. Submission
**56471807** completed with public AUC **0.819**, confirmed at **19:10:45 UTC**,
our new independent public best (**+0.018** over 0.801). The local promotion
decision remains unchanged. Opposite local and public differences highlight the
limits of 58 reused gold cases; this result does not establish which validation
or distribution factors caused the disagreement. No further candidate or
per-target blend was tuned from the leaderboard score. See
[submission evidence](submissions.md). Local evidence is under
`artifacts/reports/all-window-v1/`, including `comparison/comparison.json` and
`independent_outcome_verification.json`; training outputs are in
`artifacts/kaggle/all-window-training/versions/v1/output/adaptation/`.

## September 22 public-reference blend — preregistered

Shift the primary submission baseline to the public **0.891** ensemble. Test one
fixed **90% public / 10% independent all-window** blend after targetwise average-tie
percentile ranking over the complete current test set. The independent parent
scored **0.819**. Weights are identical across targets; no training, new labels,
coefficient search or preprocessing change. The hypothesis is complementary model
errors, not a claim that the weaker model must help.

There is no valid local CV for the public ensemble or blend on our existing gold
studies. Require both unchanged inference branches to pass provenance and visible
prediction parity, independently verify blend arithmetic, then make one technical
pass-gated submission. A score above 0.891 is a provisional public best; equal or
lower retains the pure reference. Record the outcome without repeated leaderboard
tuning. See [the fixed recipe and execution gates](reference-blend-plan.md).
