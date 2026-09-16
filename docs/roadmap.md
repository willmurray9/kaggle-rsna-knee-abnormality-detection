# Small steps toward a useful baseline

## 1. Working foundation — complete

Audit the actual CSVs, preserve missing labels, verify macro AUC and submission format, and produce a constant 0.5 sanity output. AUC 0.5 is expected for constant predictions; it measures neither image understanding nor generalization. Run the included notebook on Kaggle when ready to verify its hosted execution.

## 2. Establish trustworthy validation

Inspect a small, deliberately selected set of studies and DICOM headers. Confirm decoding, orientation, slice positions, planes and missing-series behavior. Sort slices using physical geometry, not filenames; visually check reconstructed series and preserve aspect ratio. MRI intensities vary by sequence, so document any normalization and inspect its effects.

Check for repeated patients and duplicate studies before splitting. Save a single `data/processed/folds.csv` containing study, group and fold IDs, alongside the seed, input hashes and split reasoning. Use patient groups if trustworthy; otherwise explicitly label validation as study-grouped with unresolved patient overlap. Check both positive and negative counts for each target in every validation fold. If five folds are too sparse, reduce the fold count rather than silently ignoring undefined metrics. Do not search split seeds for better scores.

With only 58 labeled studies, scores will be unstable. Prefer per-target counts and paired held-out errors over small differences in aggregate AUC. Report fold dispersion and, when comparing real candidates, patient/study-level bootstrap uncertainty. Scanner/site-held-out checks can reveal shortcuts when enough metadata and labels support them.

## 3. First image learning loop

Start with the explicit labels and one fixed series-selection rule, a small fixed slice sample and a modest 2D pretrained encoder with simple pooling to one prediction per study. Add PyTorch, DICOM decoders and the required pretrained checkpoint only for this experiment. Choose and record that recipe before looking at its validation scores. Use this limited-data model to test the pipeline, not to claim that 58 cases can establish generalization.

Save held-out probabilities, per-target AUC, runtime, preprocessing, fold identity and checkpoint provenance. Review representative errors and successful cases. Saliency maps, if added, are debugging aids rather than evidence of causal explanations.

## 4. Use reports as auditable training supervision

Review report languages, target definitions, negation and uncertainty. Start with a small reviewed extraction sample before choosing a rules-based or model-assisted extractor. Avoid equating an unmentioned finding with absence without evidence. Store observed labels separately from derived labels, recording the report source, extraction method/version, uncertainty and any manual review.

Validate extraction quality against explicit labels without using held-out reports to tune the extractor. In each image-training fold, exclude held-out studies and their reports/derived labels from all training. Compare the same image model with and without additional report-derived supervision. Report-to-label agreement is a separate measurement from image-model accuracy.

## 5. Iterate only on demonstrated limitations

Change one main factor at a time: better series coverage, an improved label extractor, or limited fine-tuning. Keep the same folds and reference. Add larger 3D models, ensembles or extensive searches only when simpler experiments explain their value. Record runtime as well as score because hidden-test inference must fit the notebook limit.

Before a real submission, execute the notebook offline with attached dependencies/weights, read test IDs dynamically, verify every study receives 12 probabilities, and check scaling beyond the three example studies. Log the notebook version, code commit, artifact hashes, rationale and result. There is no upload command or cloud provisioning in the starter workflow.
