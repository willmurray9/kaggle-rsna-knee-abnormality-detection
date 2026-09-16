# Experiment log

Keep one row per hypothesis. Detailed outputs live under `artifacts/experiments/<run_id>/`; explain consequential decisions here. Reuse the frozen split and reference across comparisons.

| Date | Run | Hypothesis / change | Local result | Decision |
| --- | --- | --- | --- | --- |
| 2026-09-16 | constant | Verify data, score and output plumbing with 0.5 probabilities | Expected macro AUC 0.500; no learned model or CV | Sanity reference only; no upload |

For each learned run, save settings/seed, input and split hashes, producing commit and dirty state, dependency versions, preprocessing and weight provenance, held-out predictions, macro/per-target scores, runtime and a brief error review. Prefer a plain `summary.json` plus CSVs to a tracking service. Preserve earlier runs instead of overwriting them.

The constant baseline is a regenerable fixture at `artifacts/baselines/constant/`; rerunning it overwrites that fixture. Its `summary.json` records input/config/submission hashes, environment, code revision and the metric sanity check.
