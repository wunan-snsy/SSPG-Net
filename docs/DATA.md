# Data and code availability

## What is included

- Source code for the SSPG-Net reference implementation.
- The leakage-audited relative-path manifest used by the formal experiments.
- Frozen protocol snapshots, training histories, validation predictions, causal audits, LOAO outputs, ablation outputs, and aggregate summaries for v1.0.0.
- E1--E7 extension results and the compact E8 external baseline results, including per-sample scores, pixel counts, and bootstrap intervals.
- The v1.1.0 E9 source and attack records, per-sample results, bootstrap summaries, three-seed matched-control records, and Figure 8 provenance under `results/release_v1.1.0/`.

## What is not included

- **Unpublished manuscript files.** The paper source and bibliography are kept outside GitHub. The Figure 8 evidence plate is included in the v1.1.0 supplement for provenance review.
- **IMD2020 image files.** Obtain them from the original dataset provider and comply with the provider's terms. This repository does not assert a right to redistribute them.
- **Large trained checkpoints.** The formal joint, LOAO, ablation, and multiseed checkpoints are excluded from Git. The ten checkpoints needed for E9 and the matched-control comparison have separately prepared ZIP bundles and published SHA-256 indices in `results/release_v1.1.0/`. Their public archive address will be added only after deposit is verified.
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
results/release_v1.1.0/    tracked E9 and matched-control supplement
```

Manifest paths are interpreted relative to `data/imd2020`. The original v1.0.0 joint, causal, LOAO, and ablation release reports validation results and records `test_accessed=False`. The later E1 and E8 extensions report a 63-source test split. Those test results are labeled separately from the original validation headline.

## Suggested manuscript statement

> Source code, frozen experimental protocols, leakage-audited split manifests, and machine-readable results for the validation analyses and E1--E8 extensions are available in the SSPG-Net GitHub repository. IMD2020 images are not redistributed and must be obtained from the original provider subject to its terms. Trained checkpoints are archived separately. The v1.0.0 joint, causal, LOAO, and ablation results use a predefined validation split; the later E1 and E8 evaluations use the 63-source test split.

The v1.1.0 evidence directory adds E9, the three-seed matched control,
and Figure 8 provenance. Its weight indices provide exact hashes but do
not themselves make the checkpoints downloadable. Add the new archive URL
or DOI to the manuscript only after its three ZIP bundles are publicly
deposited and their downloaded hashes are verified.
