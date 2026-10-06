# Reproducibility guide

## Scope

This repository supports three levels of verification:

1. **Implementation smoke test:** executes a compact forward/backward pass without IMD2020.
2. **Artifact verification:** checks the released CSV/JSON inventory, headline values, validation-only status, and file hashes.
3. **Formal rerun:** reconstructs the frozen joint, causal, LOAO, and ablation experiments and the separately labeled E1--E8 extensions using the required datasets and checkpoints.

## Locally verified environment

The v1.0.0 release was checked on Windows with Python 3.12 and an NVIDIA CUDA environment containing:

| Package | Version |
| --- | --- |
| torch | 2.14.0+cu130 |
| torchvision | 0.29.0+cu130 |
| torch-geometric | 2.8.0.post1 |
| PyYAML | 6.0.3 |
| timm | 1.0.29 |
| opencv-python | 5.0.0.93 |
| scikit-learn | 1.9.1 |
| pandas | 3.0.6 |
| numpy | 2.5.3 |
| Pillow | 12.3.0 |

The broad compatible ranges in `requirements.txt` avoid forcing a CUDA-incompatible PyTorch wheel. For exact archival reproduction, record the final driver, GPU, CUDA runtime, and a complete environment lock alongside the checkpoint deposit.

## Quick checks

```bash
python main.py --config configs/rtx3060_6gb.yaml --mode smoke_test
python scripts/verify_release.py
```

## Formal inputs

- IMD2020 at `data/imd2020`.
- `manifests/imd2020_manifest_e5_clean.csv` for the frozen joint experiments.
- `checkpoints/stego_structured_v33_128bit_best.pth` for the frozen provenance decoder.
- A CUDA-capable environment for the formal protocol.

The formal joint script expects 289 train source records and 62 validation source records, expanded to 578 and 124 active samples, respectively. It selects checkpoints by validation joint loss, selects the detection threshold on validation data using Youden's J statistic, fixes the localization threshold at 0.5, and does not construct or access the final test split.

## Outputs

Fresh runs write checkpoints to `checkpoints/` and CSV/JSON outputs to `results/csv/`. These generated directories are intentionally ignored by Git. Compare new outputs against `results/release_v1.0.0/`, `results/extension_e1_e7_v1.0.0/`, and `results/external_baselines/e8_release/` without overwriting the released evidence. E8 source scripts are in `scripts/e8_*.py`; the E8 release omits dense score maps, source images, and external model weights.

## Known boundaries

- The paper reports a controlled active-forensics protocol derived from one dataset.
- The formal headline values are validation results, not independent final-test results.
- The training-level ablations use different trainable parameter counts and are architecture-level comparisons rather than parameter-matched interventions.
- The later E1 and E8 extensions use a 63-source test split, and E4 reports a CASIA2 cross-dataset evaluation. Their protocols and limitations are recorded with those extension results. Broader post-processing and dataset validation remain future work.
