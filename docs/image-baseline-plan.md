# Image and report supervision plan — September 16, 2026

The user approved the six-stage strategy: reproduce a public image reference, audit report-derived supervision, strengthen validation, train an independent image baseline, compare controlled improvements, and submit the strongest justified candidate. Use private Kaggle notebooks and existing free CPU/GPU resources; no paid compute or external report-processing API. Keep source data and generated models under ignored data/artifacts paths.

## Execution order

1. Audit the public `pilkwang/rsna-knee-baseline-v1` source, attached datasets, model licenses and report-label provenance. Save input/source hashes. Remove silent constant fallbacks and unsafe checkpoint deserialization from the reproduced inference path. Public competition-trained weights are a submission reference, never an independent local CV model.
2. Import the public label table with its YES/NO/UNK semantics, keeping all original labels and provenance. Unknown/uncertain/unmentioned values stay masked even if the source gives them numeric scores. Audit coverage and source confidence; do not tune extraction from held-out explicit-label disagreements.
3. Build tested geometric image ordering, decoding, intensity handling and aspect-preserving resizing. Run header/duplicate checks on Kaggle and retain only hashed identifiers plus aggregate audit outputs. Inspect bounded image samples. Preserve the saved folds unless demonstrated patient/duplicate overlap requires an explicit revision.
4. Extract fixed pretrained image embeddings on Kaggle once, then compare explicit-label-only, explicit plus report-derived labels, and a prespecified lower weight for report-derived labels using the same folds and image features. Missing targets are excluded per target. Public reference checkpoint weights must not enter these independently validated features.
5. Use measured errors/runtime to choose a bounded next improvement (encoder fine-tuning, slice/series coverage or complementary pooling), with the hypothesis recorded before execution. Save all out-of-fold predictions and conditional uncertainty; do not tune from repeated leaderboard scores.
6. Validate private offline inference, record exact notebook versions/artifact hashes, submit the strongest justified candidate, and verify the competition result. Reproduce tests and document limitations and remaining work.

## Storage and compute

Keep the roughly 570 GB of MRI DICOMs on Kaggle's competition mount. Local downloads are limited to named small metadata/source/label files and compact generated outputs. First image recipe: fixed three-plane series selection, a small deterministic slice sample, frozen DINOv2-S features at 224 pixels, and simple regularized heads. Stream/preprocess batches; avoid holding the full image corpus in RAM. Save compact features so fold/model comparisons do not decode the images again. Larger models or paid infrastructure are not authorized by this plan.

## Progress

- Existing first submission remains the 0.508 metadata reference; its files and frozen split are preserved.
- Public source/weights audit and private offline reproduction are complete. Version 1 ran all 20 members × 10 windows in 83.44 seconds on three example studies with strict loading/fingerprints. Submission 56286555 completed with public AUC **0.891**, the best submitted result. This reference has no valid CV result on our folds.
- Public label ingestion is complete: 37,920 added known targets; 14,268 unknown cells masked. Hosted-LLM permission was checked against first-party rules; exact generator provenance and independence from gold labels remain unresolved.
- Frozen generic DINOv2-small feature extraction completed in 1,996.20 seconds. Geometry/decoding and selected-image duplicate checks found one nongold cross-fold duplicate pair. The versioned `image-v1` correction moves one unlabeled study; all 58 gold assignments remain unchanged. Patient independence is still unestablished.
- The metadata baseline was rerun on corrected split provenance, with byte-identical model and OOF predictions and unchanged AUC. The first independent image comparison completed in 12.84 seconds: observed-only/full-silver/quarter-silver mean fold AUC **0.651561/0.691281/0.697812**. Quarter silver is selected by the prespecified mean rule; its advantage over full silver is small and uncertain. Paired bootstrap and error diagnostics are saved.
- A single PCA-32 restriction on the quarter-silver recipe completed and was rejected: AUC 0.692956 versus 0.697812, despite lower Brier error. Scaling/PCA stay inside training folds; the encoder, labels, C=0.1 and folds remain fixed. Error evidence motivates testing lower dimensionality; extra head capacity and encoder training are deferred.
- Independent quarter-silver private offline example inference passed in 8.505 seconds, with maximum local/Kaggle probability difference 3.84e-6 under prespecified 1e-4 tolerances. The selected independent model was submitted as ref 56287107 and completed hidden scoring with public AUC **0.718**. Its code and selection were frozen before that result.
- Work occurs on `codex/image-baselines`; earlier uncommitted first-submission work is preserved.

This pass is complete: source/supervision audit, image inspection and duplicate
correction, three independent supervision comparisons, one rejected PCA
comparison, offline inference parity, and confirmed hidden-test results. Current
verification is 171 passing tests. The larger public reference and independently
trained image heads remain separate; future improvements should address sparse
validation, label-definition/missingness bias and image coverage before broad
model searches.
