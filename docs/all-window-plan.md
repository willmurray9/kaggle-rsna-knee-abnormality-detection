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
