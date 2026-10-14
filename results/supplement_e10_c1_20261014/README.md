# E10 / C1 experimental supplement — 14 October 2026

This supplement preserves the completed local experiments. Packaging and
verification did not train models, run new model inference, tune thresholds,
or change previously published results.

## E10: a parameter-matched local spatial control

Three seeds (20261001–20261003) use the existing IMD2020 Train/Validation
protocol and Clean-142 decoder. Spatial Mixer has 1,416,100 trainable
parameters; FULL has 1,416,490 (390 fewer in Mixer, a difference of 0.0275%).
The two residual local blocks contain 3×3 depthwise and 1×1 channel convolutions.

| Model | Trainable parameters | Micro pixel F1, mean ± sample SD |
|---|---:|---:|
| FULL | 1,416,490 | 0.757373 ± 0.007043 |
| NO_GRAPH | 555,812 | 0.721672 ± 0.007205 |
| Matched NO_GRAPH | 1,416,110 | 0.695146 ± 0.037559 |
| SPATIAL_MIXER_MATCHED | 1,416,100 | 0.708220 ± 0.007547 |

FULL-minus-Mixer paired pixel F1 differences are 0.046176, 0.049369 and
0.051915; mean **0.049153 ± 0.002875 sample SD**. These are three training
seeds on one Validation split (62 sources / 124 active samples). Validation
also selected checkpoints and detection thresholds. This is not a new blind
test, a source-level confidence interval, or proof isolating kNN topology.

`e10/` contains the original protocols, histories, per-sample detection scores
and pixel counts, lineage, audit report, figures and paper-ready TeX fragment.
`e2_comparators/` preserves the same-seed comparator records, including
NO_PROVENANCE. Original checkpoints are byte-preserved in the Release asset.

## C1: exploratory additional attacks

All three conditions reuse the same 63 previously observed IMD2020 Test
sources, with one protected authentic and one edited sample per source in
each condition. There are 378 prediction records, representing 63 unique
sources; repeated authentic records are not independent new images.

| Condition | Detection AUC | Detection F1 | Micro pixel F1 | Micro pixel IoU |
|---|---:|---:|---:|---:|
| Splicing | 0.997732 | 0.968254 | 0.721975 | 0.564914 |
| Telea inpainting, radius 3 | 0.982615 | 0.898305 | 0.717083 | 0.558947 |
| Copy-Move + JPEG85, 4:4:4 | 0.995465 | 0.960000 | 0.669067 | 0.502705 |

These use the single Clean-142 FULL checkpoint with SHA-256
`1848c54ec1a258ed63f063aab1cac6eca8a519f42f729280932cce465a2fd6ba`,
which is already available in the [v1.1.0 checkpoint release](https://github.com/wunan-snsy/SSPG-Net/releases/tag/v1.1.0).
It is distinct from the E2 three-seed FULL models used in E10.
Detection threshold is the original Validation value 0.6624471545219421;
pixel threshold remains 0.5. Rectangular destination areas follow the frozen
15–30% sampling protocol. JPEG affects the complete edited image, while its
localization GT remains the destination rectangle.

`c1/` preserves original per-sample scores and aggregate metrics. Newly
packaged per-sample pixel-count tables are derived from the saved float32
probabilities and deterministically reconstructed binary GT. Their summed
TP/FP/FN/TN exactly match all three original metric records. The 378 saved
probability arrays are unchanged; the 378 GT arrays are explicitly labelled
as reconstructed, rather than falsely presented as previously saved files.

## Existing inference-efficiency measurement

`inference_efficiency/` contains the already completed RTX 3060 Laptop
6GB measurement: batch 1, FP32, no AMP, 30 warm-up calls, 124 Validation
inputs × 3 repetitions per model, four models (1,488 timing records).
Image I/O, resizing, protection, model loading and warm-up are excluded.
GPU allocated and reserved memory have different meanings; see the original
report and per-model metadata. This upload does not claim a new efficiency
experiment or results on other hardware.

## Downloads and verification

Large binary artifacts are attached to the
[E10/C1 release](https://github.com/wunan-snsy/SSPG-Net/releases/tag/e10-c1-20261014):

- `SSPG-Net-E10-Spatial-Mixer-three-seeds-20261014.zip`: three unchanged
  `best.pth` files, extracted to their historical repository-relative paths.
- `SSPG-Net-C1-dense-predictions-20261014.zip`: all 378 saved probability
  arrays and 378 reconstructed GT arrays. These are model outputs and masks;
  source RGB images are not included.
- `SHA256SUMS`: archive checksums. `ASSET_INDEX.json`, `WEIGHT_INDEX.json`
  and `c1/DENSE_ARRAY_INDEX.json` record archive and per-file identities.

At repository root:

```bash
python scripts/verify_e10_c1_supplement.py
# Optional: download the two ZIPs and verify every archived array/weight:
python scripts/verify_e10_c1_supplement.py --assets-dir /path/to/downloads
```

The verifier uses Python's standard library and requires no dataset, CUDA,
PyTorch, retraining or inference. It checks file bytes, detection scores,
pixel counts, paired seed differences and timing means.

`source_frozen/` stores the exact local experiment entry points. Where the
repository already has a different entry point, that existing file is kept
and the historical source is available here. Full replay requires the
licensed source images, historical manifest identities, decoder and matching
checkpoints. Windows paths and canonical sample keys in frozen records are
historical identities, not paths to silently rewrite: changing keys changes
deterministic payload and attack seeds. The original runners require their
historical local prerequisite layout and refuse to overwrite existing runs.
The summarizer's figure QA additionally uses the recorded local
`nature-figure` audit tooling; offline numeric verification does not need it.

`PROVENANCE_BRIDGE.json` connects source files to byte-identical public copies.
`SHA256SUMS` covers every file inside this supplement except itself. Existing
v1.0.0/v1.1.0 records remain separate and unchanged. No E11–E16 results are
claimed by this release.
