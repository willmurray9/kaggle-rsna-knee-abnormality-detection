# Project conventions

- Keep this a small, explainable Kaggle project. Match the current style; change only what the task needs. Do not introduce model registries, orchestration frameworks or speculative modules.
- Read README.md, docs/competition.md and docs/roadmap.md before modeling.
- Keep raw data immutable. Data, reports, weights and generated predictions belong in ignored data/ or artifacts/ directories. Never commit credentials or raw radiology reports.
- Missing condition labels mean unknown, never negative. Reports are training supervision; inference must work without them. Keep observed and derived labels distinguishable with provenance.
- Keep every study's images, series, report and derived labels in the same validation partition. Check patient identifiers and duplicates before claiming patient-independent validation. Fit learned preprocessing only on training partitions.
- Preserve one saved split across comparisons. Record the hypothesis before running; record outcomes and rejected ideas in docs/experiments.md. Do not select models from repeated leaderboard feedback alone.
- Keep notebooks small. Reusable modeling logic goes in src/rsnaknee; the self-contained constant smoke notebook is intentionally minimal.
- Test expensive mistakes: label handling, leakage, metric, image ordering/preprocessing and submission IDs/probabilities. Run make test and the relevant workflow after changes.
- Scope downloads to named metadata files by default. Plan image storage and compute before a full download. This scaffold does not authorize uploads, publishing data or spending on cloud compute.
- When looking for a CLI on this Mac, check the shell PATH, /opt/homebrew/bin, /opt/homebrew/sbin, /usr/local/bin, ~/.local/bin and `zsh -lic 'command -v <tool>'` before declaring it unavailable.
