# Data and code availability

## What is included

- Source code for the SSPG-Net reference implementation.
- The leakage-audited relative-path manifest used by the formal experiments.
- Frozen protocol snapshots, training histories, validation predictions, causal audits, LOAO outputs, ablation outputs, and aggregate summaries for v1.0.0.

## What is not included

- **Unpublished manuscript files.** The paper source, bibliography, and paper figure files are kept outside GitHub until publication.
- **IMD2020 image files.** Obtain them from the original dataset provider and comply with the provider's terms. This repository does not assert a right to redistribute them.
- **Large trained checkpoints.** The formal joint and ablation checkpoints are approximately 334--344 MB each and are excluded from Git. Before publication, deposit them in a versioned repository such as Zenodo and add the DOI, direct file links, version, and SHA-256 checksums.
- **Historical local runs.** Results not aligned with the manuscript's frozen protocol remain excluded to prevent mixing development experiments with reported evidence.

## Expected local layout

```text
data/imd2020/              original dataset contents
checkpoints/               trained .pth files
manifests/                 tracked relative-path CSV manifests
results/csv/               generated run outputs (ignored by Git)
results/release_v1.0.0/    tracked paper-aligned artifacts
```

Manifest paths are interpreted relative to `data/imd2020`. The formal release reports validation results only and records `test_accessed=False` in the relevant summaries and protocol artifacts.

## Suggested manuscript statement

> The source code, frozen experimental protocols, leakage-audited split manifest, and machine-readable artifacts supporting the reported validation analyses are available in the SSPG-Net GitHub repository (release v1.0.0; URL to be added). The IMD2020 images are not redistributed and must be obtained from the original provider subject to its access and licensing terms. Large trained checkpoints will be archived in a versioned repository, with the persistent identifier and checksums added upon acceptance. The reported formal experiments use a predefined validation split; the final test partition was not accessed.

Replace the parenthetical placeholders only after the public GitHub release and checkpoint archive exist.
