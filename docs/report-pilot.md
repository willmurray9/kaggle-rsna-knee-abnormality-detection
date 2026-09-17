# Bounded report-label pilot

September 16, 2026. **Reject this candidate for training or full-dataset extraction.**
The 120-report pilot found important differences between report findings and the
competition thresholds, but the frozen rules also made explicit negation and
anatomical-scope errors. No training labels, saved folds, image weights or Kaggle
submissions were changed. This does not demonstrate improvement over our 0.718
image model.

## Design and provenance

A seeded hash ordering selected one study from each of 120 distinct connected
report/known-image-duplicate groups: 40 development and 80 locked audit reports.
All 58 observed-label studies and their linked components were excluded before
selection. The existing image-v1 folds were preserved. There were 4,198 eligible
components; sampling did not use report contents or public-label values. Existing
duplicate checks are incomplete, so patient independence remains unestablished.

The initial protocol was recorded before reading selected reports. All 40
development reports and initially accepted rule outputs were inspected in the
Codex session, then the protocol and deterministic extractor were hashed before
processing the 80 audit reports. Audit outputs were frozen before joining public
labels. The implemented candidate is a conservative English/Spanish rules pilot,
**not free-form agent annotation of all 120 reports**, human review, clinical gold,
or a validated multilingual extractor. Other languages and unmatched wording are
unknown. No external model API, paid service or report upload was used.

General web search for official definitions incidentally exposed two unrelated
observed-case examples in a discussion search result. These were not used for
sampling or rule fitting; all observed-study groups are excluded from the pilot.

The [official host definitions](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343)
set materially stricter thresholds than mere mention: ACL/MCL tears require high
grade injury; compartment OA requires substantial deep cartilage loss over a
sufficient area; effusion and Baker's cyst require moderate/large size. Meniscus
tears exclude isolated intrasubstance degeneration. Contusion requires traumatic
marrow change; fracture requires an acute break/line. Explicit low-grade findings
can therefore be negative for the competition target, while unspecified severity
remains unknown. Report silence is never automatically negative.

## Frozen audit results

| Target | Present | Absent | Unknown | Agreement with public labels among both-known |
| --- | ---: | ---: | ---: | ---: |
| ACL | 1 | 27 | 52 | 25/28 |
| MCL | 0 | 28 | 52 | 27/28 |
| Medial Meniscus | 21 | 7 | 52 | 28/28 |
| Lateral Meniscus | 8 | 18 | 54 | 23/26 |
| Medial OA | 0 | 4 | 76 | 3/4 |
| Lateral OA | 0 | 9 | 71 | 9/9 |
| PF OA | 0 | 9 | 71 | 2/9 |
| Effusion | 5 | 21 | 54 | 16/26 |
| Synovitis | 10 | 0 | 70 | 10/10 |
| Baker's | 1 | 18 | 61 | 12/19 |
| Contusion | 6 | 8 | 66 | 13/14 |
| Fracture | 0 | 8 | 72 | 8/8 |
| **Total** | **52** | **157** | **751** | **176/209** |

Known coverage is **209/960 (21.8%)**, versus 711/960 for the imported public
source. The candidate recognizes 26 English and 17 Spanish reports; 37 reports
fall outside its language scope. It abstains on 502 cells where the public source
supplies a binary verdict, adding no known cells beyond public coverage. The
40 development reports yielded 116/480 known cells; their agreement is not a
held-out result.

The 84.2% public agreement is a coverage-dependent diagnostic, not accuracy.
Thirty conflicts are candidate-absent/public-present; many concern small fluid
collections or mild disease below the official threshold. Three conflicts are
candidate-present/public-absent and reveal missed negation in lateral-meniscus
sentences. Post-comparison inspection also found an insufficiently scoped
patellofemoral negative and an uncertainty-handling failure. These are observed
failures of this candidate, rather than grounds to treat the public source as
gold. No audit-driven fixes were applied to frozen v1.

## Blinded agent review

Twelve audit reports were independently interpreted against the frozen protocol.
The root agent reviewed 11; a fresh agent reviewed the final case because a
candidate error on that case had inadvertently been disclosed to the root agent.
Both review files were frozen before comparison. This preserves a blinded review
for every case, but provides only one independent interpretation per case, not
multiple clinical readers. No reviewer used public labels as answers.

| Measure | Result |
| --- | ---: |
| Candidate known targets | 16/144 (11.1%) |
| Reviewer known targets | 95/144 (66.0%) |
| Three-state agreement | 59/144 (41.0%) |
| Agreement among both-known targets | 12/14 (85.7%) |
| Candidate unknown, reviewer known | 81 |
| Candidate known, reviewer unknown | 2 |
| Opposite binary decisions | 2 |

Of the 81 abstention disagreements, 56 are outside the prototype's supported
languages and 25 concern unmatched English/Spanish wording. Of the 59 agreements,
47 are shared unknowns. The two opposite binary decisions concern missed meniscal
negation. The two candidate-known/reviewer-unknown decisions concern incomplete
compartment/extent information and conflicting explanations for marrow change.
These differences reinforce the rejection; agreement alone does not establish
which interpretation is clinically correct. The sample is small and selected
without target balancing, and neither images nor clinicians supplied reference
answers. Exact spans and report hashes passed for all 120 reviewer evidence
excerpts. Reviewer annotations remain silver artifacts outside training.

## Decision and next gate

Do not release v1 annotations, run this extractor over all 4,407 reports, or replace
the current public-derived supervision. The immediate reasons are demonstrated
semantic errors, 78.2% abstention, missing positive coverage for several targets,
and highly uneven language support. The blinded review adds evidence of specific
semantic failures. **This rejection applies to the EN/ES rules prototype; the
pilot did not test, and does not reject, a future multilingual LLM extractor.**

A successor should first fix the observed negation, scope and uncertainty errors
on development material and preserve severity explicitly. Freeze it before a
fresh evaluation set; the 80 v1 audit reports are now consumed for development.
Before any broad training use, require no unresolved known semantic defects,
independently reviewed target- and language-stratified evidence with prespecified
precision and coverage criteria, and enough positives and negatives to measure
those criteria. Expert adjudication is appropriate for unresolved clinical
threshold questions; this pilot does not impose a blanket clinician-approval
requirement on every future method. Any claim of model improvement then requires
a matched image experiment on the unchanged folds, excluding held-out studies
and all their reports from training.

## Local artifacts and checks

All report text, IDs, evidence excerpts and source comparisons stay under ignored
`artifacts/report-pilot-v1/`. The directory contains `sample_manifest.json`,
`sample.csv`, `protocol-v1.md`, `audit_freeze.json`, `candidate_freeze.json`,
`extract.py`, per-phase annotation/manifests, `public_comparison.json`,
`summary.json`, the blind reviewer packet, frozen reviewer files,
`blind_combined_annotations.json`, `blind_comparison.json`, and
`blind_summary.json`. `evaluate_blind.py` reproduces the review checks and
comparisons without writing training labels. Each target record preserves its
tri-state decision, reason, exact evidence offsets, literal severity evidence,
uncertainty flag, report hash and method provenance. These are derived silver
artifacts; observed labels retain precedence in the existing training workflow.

`validate_artifacts.py` verified 120 distinct studies/components, the 40/80 split,
gold exclusion, unchanged fold assignments, all 1,440 target records, exact
evidence substrings and report hashes, frozen file hashes, and no labels in the
blind packet. These integrity checks do not establish semantic correctness.
No reusable production code or test fixtures containing reports were introduced.
