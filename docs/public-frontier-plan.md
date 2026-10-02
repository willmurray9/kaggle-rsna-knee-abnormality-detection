# Public 0.943 frontier reproduction — fixed experiment

Registered October 2, 2026 before building, executing or submitting the candidate.
The user asked for new strategies that beat our public best (**0.891**) within
today's five submissions, stopping once one does.

## Why this candidate

On October 2 the leaderboard top was 0.961, and the strongest *fully public* notebooks
converge on one 0.943 stack: 20 DINOv2-small slot-attention members, A5 attention
pooling, RadImageNet heads, four Raptor views and four CoAtNet readers, rank-blended.
It descends from Mattia Angeli's "Speedy Raptors" v34 and Jiwei Liu's fast 2×T4
inference. Our 0.891 reference (pilkwang) is one of its 20-member components.

The selected source is [yamadan96/rsna-knee-d4-public0946](https://www.kaggle.com/code/yamadan96/rsna-knee-d4-public0946),
**version 2**. Its author states this exact code was submitted on September 28 and
scored **0.943**; the "0946" slug is inherited from a parent whose extra 0.003 came
from a private ConvNeXt that is not attached. Its code is identical to
`skarin/rsna-knee-reproducing-the-0-943-public-stack` v3 and differs from
`jiweiliu/rsna-knee-fast-2xt4-inference` v11 in one line (`_RAD_EXCLUDE` also
excludes Fracture there). Rejected sources: `pjmathematician/rsna-knee-d4-*` and
`aastikrajan15/knee-s75-*` attach unresolvable (private or deleted) datasets;
`mattiaangeli/...-the-original` has moved to v39, after the scored v34.

Notebook SHA-256 `35547f8a28ed09fc68acefa6d23f12fa0bfd6a959f54fdadfdfc3dc3c32768f8`;
source metadata `a69a39714c6e33686aa2b784c75023377689dc1c0d0d9a86376e1167840bca55`.
Its own three-study example output has submission SHA-256
`8f0c5e7b2b561538dae55439a4d2e9173141096da3defd9bc0a4bac351eef37a` and completed
in about 320 seconds on 2×T4.

## Recipe

`python -m rsnaknee.frontier` copies the notebook byte-for-byte. Only the kernel id
and visibility change: private, internet disabled, the source's 2×T4 machine shape
and pinned Docker image, and all its sources — the competition, 14 public datasets,
two public kernel outputs (`sofiaanjenje/rsna-knee-e11-train`, `-e13-train`) and
`metaresearch/dinov2/PyTorch/small/1`. No code, weight, routing or blend change.
Dataset versions resolve to their latest public versions at run time.

Sources and licenses (Kaggle API, October 2): CC0 — the three `dreaddevelopment/raptor-knee-*`,
`mattiaangeli/knee-mri-fold-weights`, the three `mattiaangeli/rsna-knee-coat*`,
`pilkwang/rsna-knee-llm-labels` and `pilkwang/rsna-knee-weights`; CC BY-NC-SA 4.0 —
`marwanmath/resnet-50-radimagenet-marwan` and both `antoinegg1` head sets;
"Other" — `prvsiyan/rsna-knee-v52-radimagenet-heads-20260812`; "Unknown" — the
`mattiaangeli/opencv-python-headless` wheel. All are public and free, meeting rules
section 2.6; section 2.8 exempts incompatibly licensed inputs from winner licensing.

## Static audit — completed before submission

Five parallel auditors read every cell, followed by a completeness critic; no code
ran locally. Three had reported, plus a pattern search of all cells, before the private
example run launched; the DINOv2 and Raptor/CoAt reports arrived during that run and
before any submission. There are no network clients, credential reads, exfiltration or report
or label use at inference. `train.csv` is read only for its row count. The only
subprocesses are local CoAt reader children and `pip install --no-deps --target`
from attached wheels. One embedded payload decodes to plain linear-calibrator
coefficients. We accept these documented limitations, unchanged, for a faithful
reproduction:

- **Unrestricted deserialization.** Third-party checkpoints are unpickled with
  `weights_only=False`: the 20 DINOv2 members, five A5 folds, three Raptor
  checkpoints and SHA-pinned E13 heads. This runs only inside the private,
  internet-disabled Kaggle container.
- **Partial pinning.** Several inputs are not hash-pinned (A5 folds, Raptor
  checkpoints, DINOv2 tails), so a newer dataset version could change predictions
  silently. Example-output parity with the source run is the guard.
- **Fail-open stages.** A failed CoAt reader is dropped and the survivors are
  reweighted, and Raptor/A5/Rad failures become neutral fills. Only the DINOv2
  20/20 gate and the Rad calibrator flag stop the run. Kaggle withholds hidden-run
  logs, so the receipts below verify only the three-study example.
- **Different hidden path.** The hidden run (about 1,300 studies) runs CoAt
  readers sequentially and uses disk-backed pixel caches. It is bounded by an
  internal eight-hour budget, within the nine-hour limit.
- **Two GPUs required.** The run needs exactly two T4s and the pinned Docker
  environment. Hosted metadata confirmed both after the push.

## Validation and decision

There is **no valid local CV**: these competition-trained weights may have seen our
58 gold studies. A static audit records installs, deserialization, fallbacks and
portability risks before launch. Before submitting, the private example run must
complete with every stage receipt present (20 DINOv2 members, A5, RadImageNet,
Raptor, four CoAt readers) and no fallback counters, and its `submission.csv` must
match the source's example output within 1e-6. The receipt's environment, checkpoint
hashes and event kinds must equal the source's. The source records only
`cache_complete`, `dino_shared_path_parity` and `raptor_checkpoint`; any other
kind indicates degradation.

Submit once if those checks pass. A public score strictly above **0.891** makes this
the primary submission baseline and ends today's attempts. Otherwise record the result,
diagnose any failure, and use a remaining submission only on a materially different
candidate. Retain all earlier submissions and the independent three-window baseline.
