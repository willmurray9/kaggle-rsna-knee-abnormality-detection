# Independent image improvement pass — September 16, 2026

**Completed:** the selected adapted model scored **0.780** on Kaggle, up from
our independent baseline's 0.718. All declared comparisons and the report pilot
are recorded in [experiments](experiments.md); [submission evidence](submissions.md)
confirms ref 56290319. The prespecified recipes below are retained unchanged.

Approved by the user after distinguishing our independently trained public AUC
0.718 from the reproduced external ensemble's 0.891. The objective is a measured
improvement to our own model, with a verified offline submission when justified.
The existing six-stage plan remains the foundation. No paid compute or external
report-processing API is part of this pass.

## Fixed comparisons and safeguards

- Keep `data/processed/image-v1/folds.csv`, its report/duplicate groups, the 58
  original labels, and the audited existing label table unchanged. The prior
  model, extraction, notebooks and submission evidence remain immutable.
- The primary reference is quarter-weight report supervision, local mean-fold
  AUC 0.6978123283 and public AUC 0.718. The public ensemble supplies no local
  validation or initialization weights.
- Predeclare the main change before each experiment. Save complete held-out
  probabilities, per-target/fold AUC, paired uncertainty, timings and provenance.
  A candidate needs a higher mean AUC and improvement on at least two of three
  folds before promotion; small improvements remain exploratory on 58 cases.
- Fit preprocessing and models inside each training partition. No held-out
  image, report or derived label may enter task-specific fitting. Generic fixed
  image embeddings may be computed for all studies before splitting.
- Keep DICOMs and GPU training on free Kaggle resources. Benchmark throughput
  before long training; offline hidden inference must retain headroom below nine
  hours. Only bounded compact outputs come back to the Mac.

## Task 1: neighboring-slice coverage

Hypothesis: the sparse three-channel sample discards useful local evidence.
Retain generic frozen DINOv2-small, the same three selected series, 224-pixel
physical-aspect preprocessing, original ordering/intensity handling, linear heads
with C=0.1, silver weight 0.25 and saved folds. Sample 12 ordered slices in the
same 20–80% depth band; encode the ten overlapping three-slice windows and average
their vectors separately within each plane. The feature width stays 2,307.
The window construction and averaging together define this one coverage recipe;
its comparison cannot isolate their individual effects.

Cache the individual window vectors as well as mean vectors, permitting a later
aggregation experiment without re-decoding MRI. Preserve original implementation
files so prior source-hash checks remain reproducible. Test geometry/window
order, missing planes, exact aggregation and extraction/inference parity.

## Task 2: bounded report pilot

Independently sample 120 nongold reports, excluding report/duplicate groups that
touch gold cases. Freeze 40 development and 80 audit reports before reading text
or existing public labels. Record definitions, uncertainty, present/absent/unknown
outcomes, exact evidence and source/method hashes. Freeze the extraction protocol
before the audit portion. Have a separate agent inspect a blind subset.

These are agent-derived silver annotations and a consistency audit, not human
clinical review or new image-based gold truth. Compare with the public table only
after annotations are saved. Scale extraction only if the audit supports the
method; agreement alone does not justify replacing existing labels. Keep report
text and evidence under ignored artifacts, and do not modify training labels as
part of the pilot.

## Task 3: competitive independent training

Use the cached windows to test learned diagnosis-specific aggregation against
fixed averaging with the backbone frozen. Freeze the actual recipe, epochs and
training budget before execution. Then, subject to measured compute feasibility,
test limited backbone fine-tuning using the same image and supervision recipe,
starting from generic weights and excluding each whole held-out fold. Do not
choose epochs from repeated inspection of gold validation.

The first cached-window head recipe is frozen before fitting: 30 window tokens,
LayerNorm then Linear(768,128)/GELU, learned slot embeddings and 12 diagnosis
queries, masked attention and dropout 0.2. Train for six fixed epochs with AdamW,
learning rate 0.001, weight decay 0.02, batch size eight, seed 20260916, CPU single
thread. Gold weights are one, known silver weights 0.25, unknown weights zero;
weighted binary cross entropy is averaged over the batch's 12-target cells.
All data from the validation fold are excluded. Evaluate only the final epoch.
This architecture/optimizer comparison is a complete candidate recipe, not an
isolated test of attention alone.

The optional uint8 MRI cache remains private on Kaggle. Its quantization is not
used to generate the first coverage features; a later frozen-versus-fine-tuned
comparison must run both arms from the same quantized pixels.

### Limited adaptation recipe, frozen before results

Compare a frozen encoder control with the final two DINOv2 blocks and final
LayerNorm trainable. Both start from the same verified generic weights and the
same 128-unit attention head. Both read the exact same uint8 pixels. To bound
compute, train on one neighboring window per plane per study/epoch, with a saved
deterministic window-index table shared by both arms. Keep the selected windows'
identities in the 30-slot attention mask. At inference both arms use all windows.

Both use six final-epoch-only training epochs, effective batch eight, seed
20260916, AdamW head learning rate 0.001, weight decay 0.02, and the unchanged
gold/silver/missing loss weights. Adaptation alone adds backbone learning rate
0.000008. Keep encoder evaluation mode in both arms, with gradients on the
selected final layers; head dropout remains 0.2 during training. No extra image
augmentation, epoch search or study oversampling. All saved folds remain fixed.

A disposable training-only throughput probe must project the complete two-arm,
three-fold and final-refit job below 7.5 hours before execution. Its model/RNG
state cannot seed the real fits. Otherwise defer this job with measured evidence.
This controlled comparison measures adaptation; comparing its frozen control to
the earlier cached head also changes quantization and training-window sampling.

## Task 4: decision and submission

Compare only the declared candidates. Report unsuccessful ideas alongside useful
ones. Promote the strongest justified independent candidate, verify test-only
offline inference on the example and dynamic IDs, submit it, and check the actual
Kaggle result. A leaderboard result does not retroactively choose settings. If no
candidate passes the comparison, retain 0.718 and report the evidence clearly.

Future additional sequences, label replacement/loss balancing and complementary
ensembles depend on these results; they are not an open-ended search in this pass.
