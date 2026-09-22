# Small steps toward a useful baseline

**September 16 progress:** metadata EDA and a provisional frozen study/report-group
split are complete. Constant/prevalence references and one learned series-metadata
baseline have been evaluated. Private offline CPU execution passed; learned
notebook version 2 was submitted. See [EDA](eda.md), [experiments](experiments.md)
and [submission evidence](submissions.md). Private offline frozen-DINOv2 extraction
and image/header inspection are now complete. One verified unlabeled duplicate
pair prompted the explicit `image-v1` split revision, preserving the original
split and all 58 gold assignments. Independent image-head comparisons completed:
observed-only AUC 0.65156, full report-label weight 0.69128, quarter weight 0.69781.
The chosen quarter-weight model completed hidden scoring with public AUC **0.718**.
A single PCA-32 simplification failed its selection rule and was rejected. The
separate public reference completed at **0.891**, the best submitted public score. See [the image workflow](image-model.md).

The next independent pass completed broader slice coverage, cached attention and
a matched encoder-adaptation comparison on the same labels and folds. Mean local
AUCs were 0.71209, 0.70786 and **0.75991** for the adapted candidate, versus
0.69716 for its matched frozen control. The adapted notebook passed exact offline
prediction parity and submission **56290319** completed with public AUC **0.780**,
our new independent best, **+0.062** over 0.718. The
120-report rules pilot was rejected and did not change training labels. See
[the improvement plan](independent-improvement-plan.md) and
[experiment outcomes](experiments.md).

**September 17 follow-up:** a fresh 24-report audit and the fixed-label depth
comparison are complete. The two-block control exactly reproduced its saved
predictions; six-block adaptation scored 0.74828 versus 0.75991 and was rejected.
The independent public best at that point remained **0.780**. The [audit](label-sanity-v2.md)
found threshold mismatches and report contradictions, with most threshold
disagreements concentrated in lower public score tiers. The subsequent matched
test of preserved graded scores scored 0.75995 versus 0.75991 but improved only
one fold and failed promotion. No new submission or training labels resulted;
the selected model and independent public best remain unchanged. Neither deeper
adaptation nor this score-softening recipe demonstrated an improvement. A stronger
reviewed report-extraction pilot remains untested, as do other model changes;
these comparisons do not establish which explains the remaining performance gap.

**September 18 follow-up:** jointly training on three windows per plane improved
local AUC to **0.77478** from the exactly reproduced **0.75991** control. All
three folds improved, passing the fixed promotion rule, although the paired
bootstrap interval includes zero. The model keeps three slices and 768 features
per window, the same architecture and binary labels, and all ten windows per
plane at inference. Offline predictions matched exactly and submission
**56341808** completed with public AUC **0.801**, improving our independent best
by **0.021**. See [the plan and outcome](multi-window-plan.md).

**September 22 follow-up:** training jointly on all ten windows scored **0.76965**
against the exactly reproduced three-window control's **0.77478**. Two folds
improved, but the lower mean failed the fixed promotion rule; the paired 95%
bootstrap interval for the difference was **[−0.030818, +0.017147]**. Labels,
folds and architecture stayed fixed. The 2.79-hour comparison fit the compute
budget. Three-window training remains selected at **0.801** public AUC. A
requested diagnostic submission of the ten-window model passed exact offline
prediction parity; submission **56471807** completed with public AUC **0.819**,
our new independent public best (**+0.018**). The local selection decision remains
unchanged; the disagreement warrants stronger validation rather than tuning to
this leaderboard result. See [the plan and outcome](all-window-plan.md).

## 1. Working foundation — complete

Audit the actual CSVs, preserve missing labels, verify macro AUC and submission format, and produce a constant 0.5 sanity output. AUC 0.5 is expected for constant predictions; it measures neither image understanding nor generalization. The included notebook's hosted execution has now passed.

## 2. Establish trustworthy validation

Completed: 4,410 studies and 13,230 selected series decoded with no header,
sampled-image, spacing or physical-order fallback failures and no missing planes.
The four-study, three-plane preprocessing contact sheet had no blank tiles or
apparent aspect-ratio distortion. The fixed recipe preserves physical aspect ratio
and records per-slice percentile normalization. Broader image-quality review
remains necessary as modeling exposes specific failures.

Preserve the original `data/processed/folds.csv` and first-submission evidence.
Use `data/processed/image-v1/folds.csv` for the image comparison: one unlabeled
singleton report group moved from fold 1 to fold 0 after all nine selected image
samples matched another study. Every gold assignment and report group remains
intact; the revised audit has no sampled-image overlap across folds. Patient
independence remains unresolved because the duplicate had different anonymized
patient keys, issuers/sites were absent, and the image audit was not exhaustive.
Keep this one documented revision fixed across comparisons; do not search splits
for higher scores. Both classes remain present for every target in every fold.

With only 58 labeled studies, scores will be unstable. Prefer per-target counts and paired held-out errors over small differences in aggregate AUC. Report fold dispersion and, when comparing real candidates, patient/study-level bootstrap uncertainty. Scanner/site-held-out checks can reveal shortcuts when enough metadata and labels support them.

## 3. First image learning loop

Completed: frozen generic DINOv2-small features and the prespecified observed-only,
full-weight and quarter-weight supervision comparisons. The improvement pass then
tested twelve central slices, ten neighboring windows and learned aggregation.
Its matched frozen-versus-adapted experiment supports updating the final two
encoder blocks. All runs retain `data/processed/image-v1/report_labels.csv` and
the audited saved folds. Generic feature extraction performed no competition
fitting; supervised adaptation excluded each complete held-out fold. Treat the
58 gold cases as exploratory validation, not established generalization.

Save held-out probabilities, per-target AUC, runtime, preprocessing, fold identity and checkpoint provenance. Review representative errors and successful cases. Saliency maps, if added, are debugging aids rather than evidence of causal explanations.

## 4. Use reports as auditable training supervision

Current training uses the audited public silver table, with gold precedence and
unknowns masked. Our bounded EN/ES rules pilot did not meet its release criteria:
low coverage and negation/scope errors prevented promotion. Its annotations stay
outside training. A stronger multilingual extraction pilot remains a separate,
untested idea; the failed rules prototype does not evaluate that alternative.

Review report languages, target definitions, negation and uncertainty. Start with a small reviewed extraction sample before choosing a rules-based or model-assisted extractor. Avoid equating an unmentioned finding with absence without evidence. Store observed labels separately from derived labels, recording the report source, extraction method/version, uncertainty and any manual review.

Validate extraction quality against explicit labels without using held-out reports to tune the extractor. In each image-training fold, exclude held-out studies and their reports/derived labels from all training. Compare the same image model with and without additional report-derived supervision. Report-to-label agreement is a separate measurement from image-model accuracy.

## 5. Iterate only on demonstrated limitations

Change one main factor at a time: better series coverage, an improved label extractor, or limited fine-tuning. Keep the same folds and reference. Add larger 3D models, ensembles or extensive searches only when simpler experiments explain their value. Record runtime as well as score because hidden-test inference must fit the notebook limit.

Before a real submission, execute the notebook offline with attached dependencies/weights, read test IDs dynamically, verify every study receives 12 probabilities, and check scaling beyond the three example studies. Log the notebook version, code commit, artifact hashes, rationale and result. There is no upload command or cloud provisioning in the starter workflow.
