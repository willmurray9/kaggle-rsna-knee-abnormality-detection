# All ten training windows per plane

**Hypothesis:** Training jointly on all ten cached windows per plane improves study-level predictions over the current three-window model. Each window still contains three sampled slices and produces 768 features. This changes image exposure, attention context and the removal of random window sampling together; it does not isolate those effects or expand the underlying twelve-slice cache.

The user approved the comparison, execution through a new Kaggle submission, logical local/remote commits and reviewed merges to main. Use existing free Kaggle compute. Generate no new labels.

## Fixed comparison

Compare the unchanged `multi_windows` control with `all_windows`. Both train the final two DINOv2-small blocks, final LayerNorm and existing 30-slot attention head from the same generic initialization, for each of three saved folds and one full-data fit. Keep six epochs, seed 20260916, batch eight, head/backbone learning rates 0.001/0.000008, weight decay 0.02, encoder eval mode, mixed precision, 224-pixel inputs, one selected series per plane and twelve cached slices from the central 20–80% of each series. Inference uses all ten windows in both arms.

Preserve the audited binary `image-v1` table and saved folds: official weight 1, eligible derived weight 0.25, unknown weight 0. Exclude the entire held-out fold before constructing supervision. Score only the 58 official-label studies, distributed 19/19/20 across folds. Reports never enter inference. No augmentation, extra seed, checkpoint selection, label changes or per-target blending.

The control keeps its exact saved deterministic three-window schedule. The candidate's explicit `[6, N, 3, 10]` uint8 schedule contains ascending windows 0–9 for every epoch/study/plane and is saved as `all_window_indices.npy`. Present planes contribute all ten vectors to the same study-level loss and backward pass. Preserve slot identities and mask absent planes. Record schedule hashes, window counts and arm provenance in every fit, probe, training summary and inference build.

## Compute, evaluation and submission

Before launch, check free GPU quota and conflicting jobs. Run the existing disposable 64-step probe for both arms using full batches of eight studies with three present planes, measuring peak allocated/reserved CUDA memory. The candidate retains gradients across all encoder chunks. Stop if memory fails or the projected full comparison exceeds 7.5 hours; do not silently change batch size, detach vectors or split the loss. Keep MRI, pixel cache and checkpoints on Kaggle; download compact results only.

Require the rerun three-window control to reproduce its saved reference (local mean AUC 0.7747764995988035, public 0.801): maximum absolute OOF probability difference at most 1e-6 and fold AUC differences at most 1e-12. Investigate any failure before interpreting the comparison. Promotion requires higher mean within-fold macro AUC and improvement in at least two of three folds against both saved and rerun controls. Report fold and target results and the existing paired within-fold study bootstrap (1,000 attempts, seed 20260916, discard draws with undefined target AUC); its interval is descriptive.

Freeze the local decision before leaderboard feedback. As explicitly requested, submit one technically valid ten-window candidate after the comparison, whether or not it passes the local promotion gate; a failed gate makes this a diagnostic submission, and the three-window model remains the locally selected baseline. Do not tune another candidate from its leaderboard score. Submission requires complete training, valid supervision/fold/source/checkpoint provenance, private offline inference, dynamic test IDs, valid probabilities and example prediction parity at absolute/relative tolerance 1e-4. Invalid runs are never submitted merely to reach a submission count.

The 58 gold studies have been repeatedly reused. Patient independence and the public label extractor's independence remain unresolved. A higher local or public score alone does not remove these limitations.

## Verification and delivery

Test schedule/routing, gradients through all ten windows and encoder chunks, missing planes, unchanged control behavior and fail-closed inference provenance. Independently review the source, run `make test` and a notebook smoke build, and launch from a clean committed/pushed revision. Freeze the outcome analyzer before launch; independently recompute scores and check training exclusions, inputs and artifact hashes. Record the outcome and submission evidence separately from the implementation, then merge with passing CI.

Evidence: ignored `artifacts/reports/all-window-v1/`, with separate versioned training and inference notebook directories under `artifacts/kaggle/`.

## September 22 local outcome: promotion failed

Private offline training version 1 completed from clean, pushed revision
`ec3b6867c8be8b081ba8215054450fe794585c82`. The unchanged three-window control
reproduced all 696 held-out probabilities exactly, all 24 epoch losses and all
four recorded checkpoint hashes. The primary analyzer and separate verifier
agreed on the scores, bootstrap and decision.

| Recipe | Fold 0 | Fold 1 | Fold 2 | Mean AUC |
| --- | ---: | ---: | ---: | ---: |
| Saved and rerun three-window control | 0.805081 | 0.740177 | 0.779072 | **0.774776** |
| All-ten-window candidate | 0.779243 | 0.742395 | 0.787327 | **0.769655** |

Two folds improved, but mean AUC fell by **0.005122**, failing the preregistered
promotion rule against both controls. The paired within-fold 95% bootstrap
interval was **[−0.030818, +0.017147]**, with 704 valid draws from 1,000 attempts.
Three target means improved and nine declined. The result does not establish
that more training windows are generally harmful; the same 58 reused gold studies
and validation limitations still apply. Labels, folds, supervision fingerprints
and model parameter counts were unchanged.

The 64-step probe used batches of eight studies with all three planes in both
arms. Ten-window peak allocated/reserved CUDA memory was **3.670/4.041 GB**,
versus **1.274/1.491 GB** for three windows, on a **15.636 GB** device. The
**12,649.81-second** projection passed the 7.5-hour gate; the comparison actually
took **10,026.11 seconds** (2.79 hours). No recipe fallback was used. The inference
projection excludes DICOM preparation.

The three-window baseline remains locally selected at **0.77478**, with public
AUC **0.801**. Selection was frozen at **18:35:16 UTC**, before submission or
public feedback. The requested one-time ten-window submission is diagnostic.
Private offline inference completed in **16.556 seconds** and reproduced the
three visible examples exactly; checkpoint, source, schedule, metadata and
probability checks passed. Submission **56471807** is awaiting Kaggle scoring.
See [experiments](experiments.md) for target results and
[submission evidence](submissions.md) for execution details.
