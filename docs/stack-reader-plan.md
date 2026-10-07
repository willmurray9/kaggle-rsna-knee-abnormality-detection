# Public stack plus independent reader — fixed experiment

Registered October 6, 2026, before building, executing or submitting the candidate.
The user approved one attempt to move past our **0.943** primary baseline
(submission 56777807, the [reproduced public frontier](public-frontier-plan.md)).

## Hypothesis and source

An independently trained model with different errors may add ranking information
to the 0.943 stack. The source,
[goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944](https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944)
**version 5**, runs the 0.943 stack unchanged and then adds the author's own 2.5D
ConvNeXt-Tiny reader. The reader uses a small study-level transformer and three
fold checkpoints. Both outputs are rank-blended per target, with **70% stack and
30% reader**. The author reports public AUC **0.944** for this blend, scored on
October 4 by their private run of the same code, checkpoints and weight. The
reader alone reportedly scored **0.929**. We have not verified either score. The
author says the reader was trained on five folds grouped by report text,
using soft targets averaged from four public report-label tables, with the 58
gold-labelled studies excluded from training.

Notebook SHA-256 `941802c45fbc21a763c259d54d35f5fe1600f3a269a39d6773edcbf4185f457e`;
source metadata `ad567d8ff5ea7bf253264fcb2426c9314a3fd9e5435017b87873abf642e6db18`.
Its 23 `%%stack` cells, with that one-line magic removed, are identical in order
to the audited 0.943 notebook. The new code covers:
- a cell magic that can skip the stack
- the reader's preprocessing, model and inference scripts
- the blend cell
- two figure cells

The only new input is `goodpjw2008/rsna-knee-2-5d-convnext-reader` (Apache 2.0).
All of its files were first published on October 5, and version 5 reads its three
checkpoints from it. The author's public example output therefore comes from these
checkpoints; the kernel list shows that run on October 6. Dataset sources attach
their latest version and nothing pins the reader's checkpoint hashes, so reader
parity against that output is required.

## Static audit — completed before launch

Two auditors read every new cell; no code ran locally. The reader reads only
`test.csv`, `test_series.csv` and test DICOMs. It installs pylibjpeg and timm from
attached wheels with `--no-deps` and makes no network calls. Accepted limitations,
recorded rather than changed:

- **Unrestricted checkpoint loading.** The three reader checkpoints are loaded
  with `weights_only=False` and are not hash-pinned, the same risk class as the
  stack's checkpoints, inside the private, internet-disabled container.
- **No reader timeout.** The reader runs after the stack's internal eight-hour
  budget with no timeout of its own. A hang would hit Kaggle's nine-hour kill and
  lose the whole submission.
- **The `%%stack` wrapper swallows stack-cell exceptions.** Most stack failures
  still stop before `submission.csv` is written. One path could publish a stack
  without A5: a non-timeout A5 error. The original 0.943 notebook would have halted
  on that error. Our 0.943 run of the same cells completed on the hidden set.
- **Silent degradation inside the reader.** Per-file and per-series decode errors
  are skipped silently. The blend falls back to the stack's submission if the
  reader fails outright.
- **Some leaderboard selection already in the source.** The author tested 15%,
  30% and 45% reader weights on the public leaderboard (0.944, 0.944 and 0.942).
  The 0.944 therefore includes a small amount of leaderboard selection. We do not
  tune further.

## Recipe

`python -m rsnaknee.frontier --candidate stack-reader` copies the notebook
byte-for-byte into private kernel `willmurray99/rsna-knee-stack-reader`. The kernel
is internet-disabled and uses the source's 2×T4 machine shape and pinned Docker
image. It attaches the competition, 15 public datasets, the two public kernel
outputs and DINOv2-small. No code, checkpoint or weight changes are made; the 30%
reader weight is not tuned.

## Validation and decision

There is no valid local CV for the blend: the stack's weights may have seen our 58
gold studies. The reader reportedly excluded them, but its out-of-fold predictions
are not available on our split, so we make no local claim for it either. On three studies, a
70/30 rank blend can reproduce the stack's ranks exactly, so the final example
CSV cannot test the reader. Before submitting, require:

- every 0.943 stack gate from the previous experiment, with `_public_stack.csv`
  byte-identical to the 0.943 example output (`8f0c5e7b…`);
- the reader's `_own.csv` within absolute tolerance 1e-4 of the author's example
  output, with the same study order;
- the log line `own reader: blended 3 checkpoints at weight 0.3`, no
  `own reader FAILED`, and a valid final `submission.csv`.

Submit once if these pass. A public score strictly above **0.943** makes this the
primary submission baseline; an equal or lower score keeps 56777807. The blend
cell fails open: if the reader errors on the hidden set, the stack's submission is
kept, so a score of exactly 0.943 does not establish that the reader ran. A
0.001 difference is within public-leaderboard rounding and subset noise. The
author reports 6.5–8 hours from submission to score, against a nine-hour limit. A
timeout would use one submission without producing a score. Both
candidates remain eligible for final selection, and no weight search follows.

## Outcome — October 7

All gates passed. The stack output and the reader's `_own.csv` are each
byte-identical to the source's example outputs, as is the hosted notebook.
Submission **56885722** completed with public AUC **0.944**, first observed on
October 7. That is above **0.943**, so this blend is now the primary submission
baseline. The difference is one leaderboard tick. The author reports this same
pipeline at 0.944, 0.944 and 0.943 across runs, so the gain cannot be attributed to
the reader. Both submissions remain final-selection candidates, and no weight
search follows. Evidence and hashes are
in [the submission log](submissions.md).
