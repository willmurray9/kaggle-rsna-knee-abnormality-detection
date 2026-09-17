# First metadata exploration and frozen validation

Computed September 16, 2026 from the five downloaded competition CSVs. Reproduce with `PYTHONPATH=src .venv/bin/python -m rsnaknee.eda`. Machine-readable results live in ignored `artifacts/eda/`; study assignments and the source/version manifest live in ignored `data/processed/folds.csv` and `folds_manifest.json`.

The initial exploration below used metadata only and exported no raw report text.
The later completed image/header audit and explicit split revision are recorded
in the image-validation update below; the original split and first-submission
evidence remain preserved.

## Supervision and target balance

There are 4,407 training studies, 24,371 training series and 4,407 nonempty reports. Only 58 studies (1.32%) have observed condition labels; all 12 conditions are observed in those studies. The other 4,349 studies have no observed conditions. There are no partially labeled rows in this snapshot. Missing values remain unknown throughout exploration and validation.

| Target | Observed | Positive | Negative | Observed prevalence |
| --- | ---: | ---: | ---: | ---: |
| ACL | 58 | 24 | 34 | 41.4% |
| MCL | 58 | 9 | 49 | 15.5% |
| Medial Meniscus | 58 | 26 | 32 | 44.8% |
| Lateral Meniscus | 58 | 23 | 35 | 39.7% |
| Medial OA | 58 | 15 | 43 | 25.9% |
| Lateral OA | 58 | 11 | 47 | 19.0% |
| PF OA | 58 | 21 | 37 | 36.2% |
| Effusion | 58 | 35 | 23 | 60.3% |
| Synovitis | 58 | 27 | 31 | 46.6% |
| Baker's | 58 | 12 | 46 | 20.7% |
| Contusion | 58 | 19 | 39 | 32.8% |
| Fracture | 58 | 18 | 40 | 31.0% |

These are prevalences among the 58 observed cases, not estimates established for the unlabeled or hidden-test population. MCL is the scarcest positive condition. A constant 0.5 predictor and a training-only prevalence predictor both have within-fold AUC 0.5; prevalence can change calibration, not discrimination within that fold. Pooling fold-specific prevalence predictions can create an artificial non-0.5 pooled AUC, so compare fold AUCs as well as any pooled score.

The largest observed positive co-occurrences are Effusion/Synovitis (22 studies), Medial Meniscus/Effusion (19), ACL/Effusion (18) and Lateral Meniscus/Effusion (16). No observed study has both MCL/Medial OA or MCL/Baker's, but 58 cases cannot establish mutual exclusion. `positive_cooccurrence.csv` records counts; `pairwise_observed.csv` records their observed-label denominators separately, avoiding the interpretation of unknown values as negatives.

## Series metadata and inference availability

Train and example-test series share the same schema: study ID, series ID, fluid sensitivity, fat suppression and anatomical plane. The test study table has IDs only; reports are absent and cannot be inference features. The three example test studies are too few to estimate the hidden-test distribution.

| Quantity | Training | Example test |
| --- | ---: | ---: |
| Studies | 4,407 | 3 |
| Series | 24,371 | 15 |
| Series per study, median | 5 | 5 |
| Series per study, range | 3–14 | 5–5 |
| Sagittal series | 9,864 | 7 |
| Coronal series | 8,609 | 4 |
| Axial series | 5,898 | 4 |
| Fluid-sensitive series | 14,010 | 9 |

Every study has at least one series in each of the three planes. There are no missing series metadata cells. Mean series count is 5.53 overall, 5.79 for observed-label studies and 5.53 for unlabeled studies. `Fluid_Sensitive` and `Fat_Suppression` are identical on every train and example-test row, so including both adds no information in this snapshot. Series counts and proportions are available during inference and can support a cheap metadata baseline, though protocol information could reflect site or acquisition selection instead of pathology.

Study IDs and series IDs are unique within their respective tables, there are no orphan series or studies without series, and there is no train/test study-ID or series-ID overlap. This does not exclude repeat patients or duplicate images under different identifiers. DICOM filenames do not establish image ordering.

## Report characteristics and duplicate precautions

Reports contain 52–4,743 characters; median 977, interquartile range 587.5–1,459.5. Observed-label studies have median report length 1,205.5 versus 974 for unlabeled studies. This difference and the series-count difference are reasons to examine how the 58 labeled cases were selected before extrapolating validation performance.

There are 220 reports containing Cyrillic characters (including 3 observed-label studies), none containing CJK characters and 2,613 containing non-ASCII characters. These are script clues, not language identification: non-ASCII punctuation or Latin accents alone do not identify a language. Any future report label extraction needs multilingual handling, negation and uncertainty checks, with explicit and derived labels kept separate. Unmentioned findings must not silently become negative labels.

NFKC Unicode normalization, case folding and whitespace collapsing identify **54 duplicate-report groups covering 204 studies**, with the largest group containing 37 studies. One group contains both an observed-label study and unlabeled studies. No group contains conflicting observed labels, and no pair of observed-label studies has the same normalized report.

Identical reports can be templates rather than duplicate scans or patients; grouping them is a conservative leakage precaution. Different reports do not prove patient independence, and this check does not detect near-duplicate reports. Report hashes and IDs stay in ignored local artifacts. Empty reports, if present in future data, get individual study groups instead of being grouped together.

## Original frozen three-fold split

The saved split has one row per training study with columns `StudyInstanceUID`, `group_id`, `fold`. Every image, series, report and future derived label must join this study assignment. All studies with the same normalized report share a fold, including unlabeled studies that might later supply report-derived supervision. When training a fold, exclude **all** studies in its validation fold from model fitting and learned preprocessing, not only the explicitly labeled validation cases.

The method is `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)` on sorted study IDs, grouped by normalized report SHA-256 and stratified by MCL positive, MCL negative and MCL unknown. MCL was chosen because it has the fewest observed positives. Seed 42 was specified before model scoring; one split was generated and checked for support, with no seed search. Three folds preserve more rare-condition examples than five.

| Fold | All studies | Observed-label studies | MCL positives | Smallest positive count, any target | Smallest negative count, any target |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1,469 | 19 | 3 | 3 | 7 |
| 1 | 1,469 | 19 | 3 | 3 | 10 |
| 2 | 1,469 | 20 | 3 | 2 | 6 |

Every target has both positive and negative validation examples in every fold; full counts are in `artifacts/eda/fold_support.csv`. Lateral OA has only two positives in fold 2. Scores from so few cases will be unstable: report per-fold and per-target AUC and inspect errors rather than treating small differences as robust improvements. If comparing learned candidates, uncertainty estimates must resample at the study/report-group level.

This is **study/report-grouped validation with unresolved patient overlap**, not patient-independent validation. There is no patient or site identifier in the CSVs. Establish reliable patient grouping and image-duplicate handling from DICOM evidence before making stronger claims; any necessary split revision must be explicit and comparisons rerun together.

The split CSV SHA-256 is `d1c1f632ed2d87709a984d94365389983b1adbad15704a9aa8e5a98397831910`. The manifest records seed, method, installed scikit-learn version and all input CSV hashes. Rerunning EDA reuses and validates the saved split. Changed metadata, edited assignments, missing studies, cross-fold report groups or a target lacking both classes cause an error rather than silent split replacement.

## Image-validation update and versioned correction

The private offline frozen-DINOv2 extraction completed for all 4,407 training and
3 example-test studies. Every study supplied a usable selected sagittal, coronal
and axial series: 13,230 series with no header/decode failures, missing planes,
spacing fallbacks or geometry-order fallbacks. A contact sheet covering four
studies and all three planes was visually inspected; the displayed images had
no blank tiles or apparent aspect-ratio distortion. See the fixed preprocessing
and output paths in [the image experiment](image-model.md).

Patient hashes were consistent within each study and distinct across all 4,410
studies, but issuer and site tags were absent and deidentification flags were
unknown. There were 59 scanner fingerprints. These fields cannot establish
patient independence or a reliable site-held-out split.

Sampled-image hashing found **one pair of unlabeled studies across folds 0 and 1**
with all nine selected slices matching by pixels and geometry. Their feature
vectors were near-identical (maximum absolute difference 0.00002885), while their
patient hashes and normalized report groups differed. Both report groups were
singletons. No sampled-image match involved a gold study or crossed train/test.
This demonstrates that the anonymized patient identifiers miss at least this
duplicate-image relationship.

The original split is immutable. The explicit `image-v1` revision in
`data/processed/image-v1/folds.csv` coassigns the pair by moving the one unlabeled
report group from fold 1 to fold 0. **All 58 observed-label study assignments and
all gold per-fold target counts remain unchanged.** Revised total fold sizes are
1,470, 1,468 and 1,469. All normalized report groups remain intact.

The revised split SHA-256 is
`23c611d98c910549c5c143b30de436d4217214aae239f15850cf02db2cd6ba21`.
Its `folds_manifest.json` records the original split hashes, duplicate evidence,
unchanged gold assignments and move. The repeatable one-pair revision script is
`artifacts/reports/revise_image_split_v1.py`; the original and revised audits are
`artifacts/reports/image_audit.json` and `image_audit_image_v1.json`.
The revised audit has no sampled pixel groups crossing folds. This is still
study/report-grouped validation with unresolved patient overlap: the selected
image sample is not an exhaustive duplicate audit.

The image workflow uses `data/processed/image-v1/report_labels.csv` with
`artifacts/reports/label_audit_image_v1.json` and the revised folds. Comparisons
must use the metadata reference from
`artifacts/experiments/metadata-image-v1/oof.csv`; do not mix the original and
revised split artifacts. Independent image-head results are not established by
this extraction or audit.

## Consequences for the first submission

Use the constant submission to prove hosted execution and compare a small training-only prevalence or metadata model against it on the saved folds. Neither baseline demonstrates MRI understanding. Reports can supply future image-training supervision after a separate extraction-quality check, but cannot be inputs to hidden-test prediction. Keep the full imaging dataset on Kaggle and inspect a small, selected image/header sample before choosing a compact image baseline.
