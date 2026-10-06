# Data and code availability

## What is included

- Source code for the SSPG-Net reference implementation.
- The leakage-audited relative-path manifest used by the formal experiments.
- Frozen protocol snapshots, training histories, validation predictions, causal audits, LOAO outputs, ablation outputs, and aggregate summaries for v1.0.0.
- E1--E7 extension results and the compact E8 external baseline results, including per-sample scores, pixel counts, and bootstrap intervals.

## What is not included

- **Unpublished manuscript files.** The paper source, bibliography, and paper figure files are kept outside GitHub until publication.
- **IMD2020 image files.** Obtain them from the original dataset provider and comply with the provider's terms. This repository does not assert a right to redistribute them.
- **Large trained checkpoints.** The formal joint, LOAO, ablation, and multiseed checkpoints are excluded from Git. The checkpoint archive is handled separately on Zenodo; add its DOI only after the public record exists.
- **Historical local runs.** Results not aligned with the manuscript's frozen protocol remain excluded to prevent mixing development experiments with reported evidence.

## Expected local layout

```text
data/imd2020/              original dataset contents
checkpoints/               trained .pth files
manifests/                 tracked relative-path CSV manifests
results/csv/               generated run outputs (ignored by Git)
results/release_v1.0.0/    tracked paper-aligned artifacts
results/extension_e1_e7_v1.0.0/  tracked E1--E7 results
results/external_baselines/e8_release/  tracked compact E8 results
```

Manifest paths are interpreted relative to `data/imd2020`. The original v1.0.0 joint, causal, LOAO, and ablation release reports validation results and records `test_accessed=False`. The later E1 and E8 extensions report a 63-source test split. Those test results are labeled separately from the original validation headline.

## Suggested manuscript statement

> Source code, frozen experimental protocols, leakage-audited split manifests, and machine-readable results for the validation analyses and E1--E8 extensions are available in the SSPG-Net GitHub repository. IMD2020 images are not redistributed and must be obtained from the original provider subject to its terms. Trained checkpoints are archived separately. The v1.0.0 joint, causal, LOAO, and ablation results use a predefined validation split; the later E1 and E8 evaluations use the 63-source test split.

Add the Zenodo DOI to this statement only after the public checkpoint record exists.
