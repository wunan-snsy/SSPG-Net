# E10 and C1 result availability

The completed E10 spatial-mixer control, C1 additional-attack outcomes and
existing E10 inference-efficiency measurements are published in
[`results/supplement_e10_c1_20261014`](../results/supplement_e10_c1_20261014/README.md).

The supplement provides three-seed protocols and histories, original
per-sample predictions, numeric summaries, checkpoint hashes, C1 dense
probability/GT archive indices, vector E10 figures, paper-ready TeX,
and a dataset-free verification script.

Large checkpoints and C1 dense arrays are distributed through the
[E10/C1 Release](https://github.com/wunan-snsy/SSPG-Net/releases/tag/e10-c1-20261014).
The single Clean-142 FULL and decoder needed for C1 are in the existing
[v1.1.0 Release](https://github.com/wunan-snsy/SSPG-Net/releases/tag/v1.1.0).

E10 is a three-seed comparison on the checkpoint-selection Validation split.
C1 reuses already observed Test sources and remains exploratory.
Neither constitutes a new independent blind external evaluation. Seed SD
does not represent a source-level confidence interval. E10 alone cannot
isolate the contribution of feature-similarity graph edges.

```bash
python scripts/verify_e10_c1_supplement.py
```
