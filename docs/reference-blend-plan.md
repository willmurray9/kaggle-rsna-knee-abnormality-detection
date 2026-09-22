# Public-reference rank blend — fixed experiment

Registered September 22, 2026 before building or executing the candidate notebook
and before any candidate leaderboard feedback. The user approved shifting the
main submission baseline to the public **0.891** ensemble and testing one
conservative contribution from our independently trained **0.819** model.

## Hypothesis and recipe

Our independently trained model may make complementary ranking errors despite
its lower standalone AUC. Test exactly one blend, with the same weight for all
twelve conditions:

`prediction = 0.90 * percentile_rank(public) + 0.10 * percentile_rank(independent)`

Rank each target over the complete current test set, using ascending average-tie
ranks divided by its study count. Align both components by exact study IDs before
blending. Never rank separately within prediction batches. The public reference
already emits percentile ranks; reranking preserves its ordering and ties. The
90/10 coefficient was fixed before candidate execution; no coefficient grid,
per-target weighting, model retraining, relabeling or image-recipe change is part
of this experiment. The three visible examples are technical checks only.

The fixed parents are scored public-reference notebook version 1, submission
**56286555** (**0.891**), and independent all-window notebook version 1, submission
**56471807** (**0.819**). Preserve all twenty public members and their ten windows,
and the independent final checkpoint's ten windows per plane. Pin both saved
notebooks, dependencies, twenty public checkpoint hashes, the independent model,
training summary and window schedule. Retain the public source attribution.

## Validation and decision

There is **no valid local CV score for this blend**: the public checkpoints may
have seen our 58 gold studies. Do not score their predictions against those labels
or present parent standalone CV as blend validation. Labels and the independent
training/validation split remain unchanged.

Before the one competition submission, require private offline execution with
all components complete, unchanged checkpoint/source provenance, dynamic test ID
coverage, twelve finite values in [0, 1], and correct rank arithmetic. Each
component must reproduce its saved three-example output within absolute and
relative tolerance 1e-4; the public ranked output must match exactly. Independently
recompute the final blend from those component CSVs. Preserve component CSVs and
manifests, the final CSV, input hashes, source revision and runtime evidence.

Submit once if these technical checks pass. A public score strictly above
**0.891** becomes the provisional best submitted score; equal or lower published
scores retain the pure public ensemble as the preferred submission baseline.
Record this as one limited public-test observation, without claiming statistical
significance or repeating a coefficient search. The independently validated
three-window baseline remains available for future controlled training work.

## Execution and engineering

Run each saved inference notebook in a separate, sequential Python process so
the public pixel cache and GPU allocations are released before the independent
branch. Relocate only the independent notebook's output-directory declaration;
its model and preprocessing source remains identical. Write the final root
`submission.csv` only after both branches and provenance checks succeed. Missing
members, stale outputs, invalid predictions or timeouts stop the run, with no
fallback or partial ensemble.

Use the existing free Kaggle T4, private and internet-disabled, with an eight-hour
shared execution deadline inside the competition's nine-hour limit. The public
cache estimate is about 9.84 GiB at 1,300 studies; its measured-memory gate remains
active. The prior public and independent submissions completed within observed
intervals of approximately 119 and 32 minutes, including queueing. Those intervals
support feasibility but do not measure hidden inference time or guarantee it.
No image/checkpoint download to the Mac or paid compute is needed.

Test ranking/ties, shuffled and replacement IDs, full-test ranking, malformed
outputs, checkpoint and source mismatches, missing public members, branch failure
and notebook packaging. Run `make test`, independently review the code, commit
and push before hosted execution, then record the actual outcome separately and
merge with passing CI. Evidence belongs in ignored
`artifacts/reports/reference-blend-v1/` and versioned
`artifacts/kaggle/reference-blend/` directories.
