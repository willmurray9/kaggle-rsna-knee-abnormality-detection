# Fresh public-label sanity audit

September 17, 2026. **The public table remains noisy weak supervision; this audit does not pass a clean-label gate.** A fresh, report-only review found repeated threshold mismatches and three clear report-to-target contradictions involving meniscus tears in two Croatian reports. It found no ID-join, masking, source-hash or split-integrity corruption. No labels, folds, training code or models were changed.

The findings justify pausing broad model investment that assumes these are accurate competition labels. They do not establish that the entire table is unusable: all 27 reviewer-supported positives agreed with the public table, and a single bounded model comparison can still use the existing table as known noisy supervision, keeping labels and folds fixed. Such an experiment measures the model change under this supervision; it cannot validate or repair the labels. The parent review independently confirmed the explicit-denial findings and approved only the preregistered bounded fixed-label experiment.

## Design and provenance

Twenty-four distinct connected report/known-image-duplicate groups were selected by a fixed SHA-256 ordering, using report-only lexical language and uncertainty/severity/negation strata. Every component touching the 58 observed-label studies, all 120 previous pilot studies, or the ten previously illustrated source rows was excluded. Selection never read public target values. Missing strata fell back to unconstrained selection within the same language proxy. The selection proxies misclassified some Croatian reports; the sample stayed frozen, and manual language correction produced English 3, Spanish 4, Croatian 7, Turkish 4, French 2 and German 4. This is a varied convenience sample, not an estimated population error rate.

Agent interpretations used the [official host target definitions](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343) and the archived pilot protocol: high-grade ligament injury, substantial deep compartment cartilage loss, moderate/large effusion and Baker cyst, definite meniscus tear, synovial inflammation/thickening, impact contusion and acute fracture. The host’s negative treatment of borderline image findings does not turn missing report evidence into a negative. Interpretations and exact evidence spans froze at 18:34:04 UTC, before the public table was opened at comparison. The failed old rules extractor was not used.

All raw reports, study IDs, source-row selections, evidence and comparisons remain in ignored `artifacts/label-sanity-v2/`. `sample_manifest.json`, `annotation_freeze.json` and `summary.json` preserve source/output SHA-256 hashes. The unchanged label-table hash is `7d2369e7e936d3335614a71fb32bec5af24737f8d7c62f888fcd1e89075f7f0e`; the unchanged image-v1 split hash is `23c611d98c910549c5c143b30de436d4217214aae239f15850cf02db2cd6ba21`.

## Results

Reviewer coverage was **175/288 (60.8%)**, versus **218/288 (75.7%)** for public labels. Among 168 cells known to both, **137 agreed (81.5%)**. Agreement is report-to-label consistency, not clinical accuracy. All 31 opposite binary decisions were public-positive/reviewer-negative.

| Target | Present / absent / unknown | Public known | Both-known agreement | Threshold mismatch | Clear contradiction |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACL | 3 / 17 / 4 | 21/24 | 16/20 | 4 | 0 |
| MCL | 1 / 19 / 4 | 21/24 | 17/20 | 3 | 0 |
| Medial Meniscus | 5 / 18 / 1 | 24/24 | 18/23 | 3 | 2 |
| Lateral Meniscus | 5 / 14 / 5 | 21/24 | 18/19 | 0 | 1 |
| Medial OA | 2 / 11 / 11 | 21/24 | 11/13 | 2 | 0 |
| Lateral OA | 1 / 11 / 12 | 19/24 | 10/11 | 1 | 0 |
| PF OA | 1 / 12 / 11 | 21/24 | 7/13 | 6 | 0 |
| Effusion | 4 / 13 / 7 | 23/24 | 9/17 | 8 | 0 |
| Synovitis | 3 / 0 / 21 | 3/24 | 3/3 | 0 | 0 |
| Baker's | 1 / 10 / 13 | 12/24 | 10/11 | 1 | 0 |
| Contusion | 0 / 11 / 13 | 20/24 | 10/10 | 0 | 0 |
| Fracture | 1 / 12 / 11 | 12/24 | 8/8 | 0 | 0 |

| Language | Reports | Reviewer known | Both-known agreement |
| --- | ---: | ---: | ---: |
| Croatian | 7 | 42/84 | 30/41 |
| English | 3 | 20/36 | 16/20 |
| French | 2 | 18/24 | 17/18 |
| German | 4 | 29/48 | 22/27 |
| Spanish | 4 | 32/48 | 26/30 |
| Turkish | 4 | 34/48 | 26/32 |

The 31 opposite decisions comprise:

- **28 severity/definition mismatches:** low-grade ligament injury or degeneration, early/superficial cartilage disease, minimal/small effusions and a small cyst were public-positive despite being below the official threshold. Effusion contributed eight; PF OA six. The public extractor’s complete finding specification is unavailable, so this may reflect a broader abnormality definition rather than isolated missed negation.
- **3 clear report-to-target contradictions:** both menisci in one Croatian report and the medial meniscus in a second were public-positive despite explicit tear denials accompanying degeneration. This repeated degeneration-versus-tear problem is material. It does not prove an MRI annotation is wrong and does not identify whether the extractor ignored negation or intentionally labeled degeneration.

Separately, seven reviewer-known targets were public-unknown (five fracture negatives inferred from normal marrow, one contusion negative and one lateral-cartilage negative). These are possible omitted evidence or scope-policy differences, not established extraction errors. Fifty public-known targets remained reviewer-unknown because of missing/uncertain report evidence, unspecified severity/extent, uncertain tear or trauma attribution, or postoperative ambiguity. Sixty-three were unknown to both. Unknowns were never converted to negatives.

One initial automated discrepancy category was corrected after reading the complete report: a PF finding with a small superficial irregularity and explicit denial of deep defects belongs to severity mismatch. Neither frozen interpretation nor public label changed; initial and reviewed comparison files are both retained.

## Independent review and limits

An independent agent interpreted six reports before seeing these annotations or public values. Tri-state agreement was **70/72 (97.2%)**; all **43/43** mutually known decisions agreed. The two differences were contusion unknown versus absent in reports with reactive/stress-related marrow findings. They expose scope and abstention judgment rather than opposite diagnoses. Neither review was altered after comparison.

The audit validates 24 distinct groups, zero excluded-group overlap, preserved folds/source hashes, all 288 records, 229 primary exact evidence spans and 74 independent exact spans. These are integrity checks, not semantic certification. There is no image review or clinician reference, no established patient independence, and no complete duplicate audit. Positive examples are sparse, including no positive contusion and only one positive MCL injury, fracture and Baker cyst. Language strata are small and unbalanced. Do not extrapolate these percentages or use the sample to tune/relabel the full training set.

## Decision

Keep the current labels immutable for comparability, retain unknown masks and observed-label precedence, and describe them as imperfect report-derived supervision. The audit supplies a concrete reason to improve target-definition alignment before broad training expansion. This pass proceeds with the preregistered bounded fixed-label/fixed-fold model experiment, with these semantic errors recorded. It includes no broad relabeling; the audit neither certifies label quality nor establishes model improvement.

## Preserved public scores: a follow-up hypothesis

A post-audit diagnostic found that 24/28 threshold mismatches have public score
0.68, compared with 3/27 agreed positives. Those three agreed low-score positives
are all synovitis, where mild findings qualify. Four threshold mismatches and
two explicit contradictions have score 0.82. These observations do not support
a universal score cutoff. Only 58/100 public-positive cells were resolvable from
the reports; the sample cannot establish population accuracy or calibration.

Our current ingestion preserves these scores as metadata but trains on binary
YES/NO verdicts. A future matched comparison could test the preserved scores as
heuristic soft targets for known report labels, retaining official labels and
unknown masks. That could be cheaper than full relabeling, but would also soften
negatives (public score 0.08) and would not correct explicit tear contradictions.
The included public API-labeler code explicitly requests three severity levels
for positive findings, but imports its score conversion from an unavailable
module. This supports investigating the preserved score information without
claiming a verified score-to-severity mapping or calibrated probabilities.
No such training change was made in this pass. Exact cell-level diagnostics and
source hashes remain in ignored `artifacts/label-sanity-v2/score_diagnostic.json`.
