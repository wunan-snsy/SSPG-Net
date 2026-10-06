# SSPG-Net E1-E7 experiment extension

This update adds the seven experiments used to strengthen the SSPG-Net evaluation. Large checkpoints, datasets, dense prediction masks, and repeated-run model weights are intentionally excluded from Git. Compact metrics, tables, figures, protocols, and scripts are retained for audit and reproduction.

## Results overview

| ID | Experiment | Evaluation scope | Main result |
|---|---|---|---|
| E1 | IMD2020 final test | Frozen checkpoint and validation-selected threshold | ROC-AUC 0.9980, detection F1 0.9756, pixel F1 0.7759 |
| E2 | Multi-seed stability | Three validation runs per model variant | FULL AUC 0.9948 +/- 0.0026, pixel F1 0.7853 +/- 0.0027 |
| E3 | Provenance-only baseline | Validation-only baseline training | AUC 0.9331, detection F1 0.9302, pixel F1 0.5214 |
| E4 | CASIA2 cross-dataset evaluation | 200 CASIA2 authentic sources and 600 controlled attacks | AUC 0.9941, detection F1 0.9344, pixel F1 0.7204 |
| E5 | Benign processing robustness | Nine JPEG, resize, and Gaussian blur conditions | AUC range 0.9935-0.9977 |
| E6 | Embedding strength sensitivity | Frozen-model validation sweep, alpha 0.005-0.020 | alpha 0.010 gives 40.05 dB PSNR and pixel F1 0.7753 |
| E7 | Graph parameter sensitivity | Frozen-model validation sweep, k 4-16 | k=4 gives AUC 0.9969 and pixel F1 0.7858 |

## Evidence boundaries

- E1 is the independent IMD2020 final-test evaluation. The checkpoint and detection threshold were frozen before test access.
- E2, E3, E5, E6, and E7 use validation data and should be described as stability, ablation, robustness, or sensitivity analyses rather than independent test results.
- E4 uses CASIA2 authentic images as source images and applies the controlled active SSPG attack protocol. It measures cross-dataset active-protocol generalization, not performance on the native CASIA2 tampered subset.
- E6 and E7 change inference-time input or topology settings with frozen model weights. They do not replace a full retraining study for every hyperparameter value.
- The preregistered configuration remains alpha=0.010 and k=8. Validation observations were not used to revise E1 test settings.

## Reproduction entry points

| ID | Configuration or script |
|---|---|
| E1 | `scripts/eval_test_final.py` |
| E2 | `scripts/run_multiseed.py`, `scripts/aggregate_multiseed.py` |
| E3 | `scripts/train_provenance_only.py`, `models/provenance_only_model.py` |
| E4 | `scripts/prepare_casia2.py`, `scripts/build_casia_attack_plan.py`, `scripts/eval_casia_cross_dataset.py` |
| E5 | `configs/e5_benign_robustness.json`, `scripts/eval_benign_robustness.py` |
| E6 | `configs/e6_alpha_sensitivity.json`, `scripts/eval_alpha_sensitivity.py` |
| E7 | `configs/e7_graph_sensitivity.json`, `scripts/eval_graph_sensitivity.py` |

The compact release artifacts are stored under `results/extension_e1_e7_v1.0.0/`. Dataset paths and checkpoints remain local and are documented by the existing repository configuration.
