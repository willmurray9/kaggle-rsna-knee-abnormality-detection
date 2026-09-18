# Three training windows per plane

**Hypothesis:** The current attention model learns from too little of each study at once. Compare one versus three windows per plane during training, keeping each window at three slices and each encoder vector at 768 numbers. The treatment increases both image exposure and the evidence available jointly to attention; it cannot separate those effects.

The user has authorized this experiment, a qualifying competition submission and reviewed merges to main. Use existing free Kaggle compute; generate no new labels.

## Fixed comparison

Compare `late_blocks` with the new fixed `multi_windows` arm. Both fine-tune DINOv2-small blocks 10–11, final LayerNorm and the existing 30-slot attention head, starting fresh for each of three folds and one full-data fit. Keep six epochs, seed 20260916, batch eight, head/backbone learning rates 0.001/0.000008, weight decay 0.02, encoder eval mode, mixed precision, 224-pixel inputs, three selected series and twelve cached central slices. Use the unchanged binary `image-v1` labels and saved folds: official weight 1, eligible public-derived weight 0.25, unknown weight 0. Exclude the entire held-out fold before constructing supervision.

The control retains its exact deterministic `[6, N, 3]` window schedule. For each epoch/study/plane, the candidate includes that original window, ranks the other nine window IDs by SHA-256 in a separate versioned namespace containing seed/epoch/study/plane/window, takes the lowest two (ties by window ID), and sorts the resulting triple numerically. Save a separate `[6, N, 3, 3]` uint8 schedule. Sampling cannot use labels or consume training RNG state. Preserve original slot identities; all selected windows contribute to one study-level loss and one backward pass. Missing planes contribute no vectors. Inference remains all ten windows per present plane for both arms.

Keep architecture, supervision, preprocessing, splits, augmentation, epochs and optimizer unchanged. No per-target blending, new seeds, epoch selection or label extraction is part of this comparison. Record per-arm window counts and schedule fingerprints in recipe, fit and probe provenance; require the candidate's provenance during inference while supporting historical summaries.

## Compute and selection gates

Before launch, verify free quota and no conflicting active jobs. Run a disposable 64-step probe using the actual candidate with full batches of eight and three present planes; record peak allocated/reserved CUDA memory and device capacity. Encoder chunks do not release the retained backward graphs. If memory fails or the projected complete comparison exceeds 7.5 hours, defer without silently shrinking batches, detaching features or splitting the loss. Keep MRI files, the approximately 8 GB cache and weights on Kaggle; download compact results only.

First require the rerun control to reproduce the saved two-block reference: maximum absolute OOF probability difference at most 1e-6 and each fold macro-AUC difference at most 1e-12. Investigate any failure before selection. Promote only if the candidate improves mean within-fold macro AUC and at least two of three folds versus both rerun control and saved reference (local mean 0.7599107356, public 0.780). Score only the 58 official-label held-out studies. Report per-target scores and the existing paired study bootstrap within folds: 1,000 attempts, seed 20260916, discarding draws with undefined target AUC. The interval is descriptive, not another selection gate.

Freeze the decision before leaderboard feedback. A promoted candidate must pass private offline inference, source/checkpoint/schedule provenance, dynamic-ID and probability checks, and example prediction parity with absolute and relative tolerance 1e-4. Submit one qualifying notebook and await scoring; otherwise retain the current independent best. Repeated reuse of 58 gold studies and unresolved patient/public-extractor independence limit the evidence.

## Verification and delivery

- Test deterministic unique anchored sampling and unchanged control behavior; invalid indices, absent planes, gradients through extra windows across encoder chunks, and all-window inference.
- Test actual candidate routing through training/probe and fail-closed inference provenance, retaining historical compatibility.
- Freeze an independent outcome analyzer before launch. Review code, run `make test` and a generated-notebook smoke build, then launch from a clean pushed source revision with recorded hashes.
- Independently recompute the outcome from official truth, verify folds/training IDs/supervision and artifact hashes, and record rejected as well as successful results.
- Commit the plan, implementation and outcome in logical steps; merge reviewed work with passing CI.

Evidence: ignored `artifacts/reports/multi-window-v1/`. Use separate versioned training/inference notebook directories so previous runs remain reproducible.

## Local outcome — September 18

Version 1 of the private training notebook completed the comparison in 3,565.07
seconds. The control exactly reproduced the saved probabilities and fold AUCs.
Three windows scored **0.774776** mean within-fold macro AUC versus **0.759911**
for both controls, improving all three folds and passing the registered local
promotion gate. The independent raw-truth verifier agrees with this decision
and confirms fold exclusions, unchanged supervision and compact artifact hashes.
No new labels were generated.

The conditional paired 95% bootstrap interval is **[−0.010256, +0.036274]**, from
704 valid draws of 1,000; six target means improve and six decline. Reuse of 58
gold studies, unresolved patient/extractor independence and the combined change
in image exposure and attention context limit the conclusion. The candidate
passed the memory/runtime probe without recipe changes. See the
[complete local outcome](experiments.md#three-window-local-outcome-promotion-gate-passed)
for fold/target results, source and verification evidence.

The local selection was frozen before leaderboard feedback. Private offline
inference version 1 passed with exact prediction parity; Kaggle accepted
submission **56341808**. Hidden-test scoring completed with public AUC **0.801**,
our new independent best, **+0.021** over 0.780. The model and selection remained
unchanged after scoring; see [submission evidence](submissions.md).
