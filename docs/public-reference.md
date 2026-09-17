# Public image reference

The private offline notebook reproduces the MRI inference recipe from
[pilkwang/rsna-knee-baseline-v1](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1).
Its public competition-trained weights are an external reference, not a validation
result or initializer for our frozen folds. Independent image learning must start
from the generic pretrained encoder and exclude each held-out group's images,
reports and derived labels from training.

Private [RSNA Knee Image Reference](https://www.kaggle.com/code/willmurray99/rsna-knee-image-reference)
version **1** completed its three-study example run on September 16, 2026 in
**83.44 seconds** on a Tesla T4. All **20 members × 10 windows** completed,
restricted loading and fingerprints passed, and 12 available series slots decoded
with zero failed series. The example CSV passed exact ID/order and probability
validation. The exact notebook was submitted as ref **56286555**; authenticated
Kaggle status at **21:49:17 UTC** confirmed **COMPLETE**, public AUC **0.891**,
with no error description. The hidden result is separate from the example runtime.
Versioned evidence lives in `artifacts/kaggle/image-reference/versions/v1/`;
hashes and submission details are in the [submission log](submissions.md).

Build the reviewed snapshot only:

```bash
uv run python -m rsnaknee.reference \
  --source artifacts/research/public-baseline/rsna-knee-baseline-v1.ipynb \
  --manifest artifacts/research/public-weights/manifest.json \
  --output artifacts/kaggle/image-reference
```

The builder checks exact source and weights-manifest hashes and emits a private,
GPU-enabled, internet-disabled notebook. It attaches only the competition,
`pilkwang/rsna-knee-weights` (CC0) and `metaresearch/dinov2/PyTorch/small/1`.
The public notebook source is licensed Apache 2.0; downloaded source and generated
artifacts remain ignored. Source attribution is
preserved in the generated notebook and build manifest.

The audited package has 20 DINOv2-small members: four seeds across five source
folds, using six recovered series slots, 336 px, 12 slices, a 130 mm crop and
10 overlapping three-slice windows per member. All members must complete;
there is no timed ensemble reduction, training fallback or constant submission.
The generated code keeps the pretrained weights' native preprocessing recipe.

Checkpoint loading is restricted to `weights_only=True`, with no custom
allowlist. An incompatible checkpoint stops the run. The pinned manifest's
finite synthetic fingerprints must match. Local-only backbone loading prohibits
remote model code. IDs, shape, probabilities and study decoding are checked before
writing `submission.csv`; ordering or decode timeout raises an error.
`run_manifest.json` records completed member hashes/window counts, decoded slot
counts, hardware, package versions, elapsed time and output hash, including a
failure status on ordinary exceptions. Abrupt platform termination may prevent
that final evidence file.

At approximately 1,300 hidden studies, the pixel cache requires 9.84 GiB before
worker buffers and model activations. The full ensemble runs approximately 1.56
million encoder images. Memory headroom is checked at runtime; the three-study
visible test run cannot establish hidden-test speed. Review finished logs and
output before submission; compare public scores separately from local CV.

The [official execution requirements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation)
allow public pretrained models and require offline notebooks within nine hours.
[Rules section 2.6](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)
allows qualifying external data/models, subject to reasonable accessibility and
minimal cost. The [pinned August 9 clarification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965)
permits hosted LLM processing of reports for label extraction under those
conditions. No report upload or paid labeling is part of this reproduction.
Full published rules and clarification responses were retrieved with the
authenticated Kaggle API on September 16, 2026 and retained under
`artifacts/research/public-baseline/official-rules.json` and
`host-llm-clarification.json`; the API returns null author names but confirms the
discussion is sticky. The detailed static review is `audit.md` in that folder.
