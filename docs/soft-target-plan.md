# Preserved public scores experiment

> **For agentic workers:** Use test-driven implementation and independent review for the bounded changes below. The user has authorized execution, submission of a qualifying candidate, and merging reviewed work into main.

**Goal:** Test whether the public label table's existing graded scores improve our current two-block image model compared with binary public verdicts.

**Architecture:** Add one fixed `soft_targets` arm to the existing fine-tuning workflow. It shares the two-block architecture with `late_blocks`; only eligible report-derived BCE targets differ. Keep the current defaults and historical artifacts intact.

**Tech stack:** Existing NumPy/pandas, PyTorch, Transformers, pytest and private offline Kaggle GPU workflow; no new dependencies or paid services.

**Basis:** [Fresh label audit](label-sanity-v2.md), especially its preserved-score diagnostic, and the user's approval to continue with the recommended next experiment.

## Frozen recipe and interpretation

Compare `late_blocks` with `soft_targets` in one matched job. Both update the last two DINOv2-small blocks, final LayerNorm and existing attention head. Keep six epochs, seed 20260916, batch eight, head/backbone learning rates 0.001/0.000008, weight decay 0.02, encoder eval mode, mixed precision, 224-pixel images, three selected series, twelve central slices, one deterministic training window per plane and all ten windows at inference. Both arms start fresh from the same generic weights for each of three folds and the final full-data refit.

Keep the existing `image-v1` label table, audit and folds byte-identical. Official observed targets remain strictly 0/1 with weight 1. Eligible public targets retain weight 0.25. Unknowns remain target 0 with weight 0. The control continues to use YES=1 and NO=0. The candidate substitutes the preserved `__derived` score only for active derived cells: observed public tiers are NO=0.08 and YES=0.68/0.82/0.94. UNK=0.28 never becomes supervision. Validate finite [0,1] scores for active derived targets; missing or invalid active scores fail. Remove the entire held-out fold before constructing or validating training supervision. Check binary class support using the existing binary targets before substitution.

These are heuristic soft targets, not calibrated probabilities or a verified severity mapping. The treatment softens both positives and negatives, adds no labels, and does not repair explicit report contradictions. No audit annotations, report extraction, score cutoff, confidence reweighting, augmentation change, epoch/seed search or per-target blending enters this experiment.

A disposable probe must project the complete job below 7.5 hours on existing free Kaggle GPU quota. The approximately 8 GB pixel cache, MRI files and model weights remain on Kaggle. Download compact results and provenance only.

## Selection fixed before training

First require the rerun binary control to reproduce the saved two-block reference: maximum absolute OOF probability difference at most 1e-6 and each fold's macro AUC difference at most 1e-12. If it fails, investigate the comparison before making any promotion decision.

Promote the soft-target candidate only if its mean within-fold macro AUC is higher and at least two of three fold scores improve versus **both** the rerun binary control and saved two-block reference (mean 0.7599107356, public 0.780). Recompute scores from all 58 official-label held-out studies; public-derived labels never serve as validation truth. Report per-target scores and the existing 1,000-attempt paired study bootstrap within folds, seed 20260916, discarding draws with undefined target AUC. Its interval is descriptive and is not an additional selection criterion.

Freeze the selection before any competition submission. A selected checkpoint must pass private offline inference, source/weight provenance, dynamic-ID and probability checks, and example prediction parity (absolute and relative tolerance 1e-4, fixed here). Submit one qualifying candidate; otherwise retain the 0.780 independent best and record the negative result. The 58 reused official-label studies and unresolved patient/extractor independence limit every inference.

## Implementation and verification

- [x] **Supervision and training:** modify only `src/rsnaknee/finetune.py` and its tests. Add the fixed two-block soft arm, keyword-only target mode in `training_partition`, and per-fit target/weight SHA-256 fingerprints. Keep binary defaults unchanged. Test official precedence, unknown gradients, all observed public score tiers, invalid active scores, strict binary official labels, exclusion of held-out scores, and actual candidate routing through both probe and training.
- [x] **Notebook and inference:** modify only `src/rsnaknee/finetune_notebook.py` and its tests. Embed target-mode and Git revision/dirty provenance in the generated job, bind the selected checkpoint to its recipe, and verify both depths and supervision mode during inference. Preserve synthetic-fixture tests and existing notebook defaults. No production API should require access to raw reports at inference.
- [x] **Prelaunch gate:** run targeted failure-first tests, `make test`, a clean-checkout test and a generated-notebook smoke build; obtain independent code review. Commit and push the recipe/source before building the launch notebook. Verify a clean Git state, exact revision/source/input hashes, private offline metadata and remaining free quota. Freeze the outcome-analysis script and dependency hashes before launch.
- [x] **Evaluate and finish:** verify downloaded compact artifact hashes, unchanged data/folds/windows, training IDs, target/weight fingerprints, parameter counts and producing revision. Independently recompute the promotion decision. If promoted, execute and submit the checked inference notebook and wait for scoring. Commit the outcome documentation separately and merge reviewed commits to main.

Expected supervision boundary, illustrated with synthetic values:

```python
# Existing binary labels and weights determine eligibility and class support.
# Only report-derived cells are replaced; official and unknown cells stay exact.
derived = weights[:, column] == 0.25
scores = labels.iloc[training_rows][target + '__derived']
targets[derived, column] = scores.loc[derived].to_numpy(dtype='float32')
# Unknown target/weight stay (0, 0); official targets retain weight 1.
```

Local evidence belongs under ignored `artifacts/reports/soft-target-v1/`; new Kaggle builds use a separate versioned directory and notebook so prior runs remain reproducible.

## Outcome

Completed September 17: soft targets scored 0.75994585 versus 0.75991074,
improving only one fold. Independent verification agreed with the frozen
analysis: retain the binary-target model (public AUC 0.780), with no new
submission. See [the full result](experiments.md) for uncertainty and provenance.
