# E8 external same-protocol baseline evaluation

## Repositories and environment

| Method | Official repository | Commit | Checkpoint | SHA-256 |
|---|---|---|---|---|
| TruFor | https://github.com/grip-unina/TruFor | `ae54475df6f41a491d7615100feb19263dec13f7` | `trufor.pth.tar` | `ac1d90e329a72e0d66e8665e123a19e94bfae3209c3ef8a4f9ca3b91578c7844` |
| EditGuard | https://github.com/xuanyuzhang21/EditGuard | `174778fb371640bb31b32fcb9f7bfd440c88766b` | `clean.pth` | `b967b91b94a464ab73f82db63a57ee818e6fc5d99d7ae6aa094a1777cac977d5` |

Python 3.12.10, PyTorch 2.14.0+cu130, CUDA 13.0; NVIDIA RTX 3060 Laptop GPU, batch size 1, float32 inference mode. See `compatibility_notes.md` for preprocessing and adapter details. Third-party source files were not edited.

## Frozen protocol and Validation calibration

The E1 Test preflight reran 63 sources/126 samples and reproduced all four published E1 metrics. `protocol_manifest.json` records hashes of the source manifest, checkpoints, deterministic attack geometry, source images, and masks. TruFor used all 62 Validation sources/124 active samples. Its manipulation-score threshold from Validation-only Youden J was 0.685603023 (J=0.8871). TruFor map threshold is 0.5; EditGuard clean residual threshold is the official 0.2. Neither threshold was chosen from Test.

## Independent Test metrics

| Method | Evaluation input | AUC | Balanced Accuracy | Detection F1 | Sensitivity | Specificity | Pixel F1 | Pixel IoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SSPG-Net | SSPG-protected E1 Test | 0.9980 | 0.9762 | 0.9756 | 0.9524 | 1.0000 | 0.7759 | 0.6338 |
| TruFor | official pretrained model; identical SSPG-protected E1 Test inputs | 0.9642 | 0.8889 | 0.8889 | 0.8889 | 0.8889 | 0.8029 | 0.6707 |

| Method | Protection mechanism | Sources | Pixel F1 | Pixel IoU | Protected-image PSNR |
| --- | --- | --- | --- | --- | --- |
| SSPG-Net | SSPG protection | 63 | 0.7759 | 0.6338 | 40.0403 |
| EditGuard | official clean checkpoint; same sources and attack geometry, method-specific protected images | 63 | 0.8783 | 0.7830 | 36.4362 |

EditGuard detection AUC and F1 are N/A. It uses the same 63 original sources, attack families, attack geometry, and masks, but its own protected images. TruFor receives the identical SSPG-protected E1 active tensors. Pixel metrics use micro TP/FP/FN across all 126 samples, including authentic zero masks. PSNR uses pooled RGB MSE on the 63 original/protected source pairs.

## Paired source-cluster bootstrap

Delta is SSPG-Net minus external baseline. Resampled 63 source IDs with replacement; 2000 replicates; seed 20261006.

| Comparison | Metric | Delta | 95% percentile interval |
|---|---|---:|---:|
| SSPG-Net minus TruFor | auc | +0.0338 | [+0.0123, +0.0602] |
| SSPG-Net minus TruFor | detection_f1 | +0.0867 | [+0.0232, +0.1512] |
| SSPG-Net minus TruFor | pixel_f1 | -0.0270 | [-0.0805, +0.0312] |
| SSPG-Net minus TruFor | pixel_iou | -0.0369 | [-0.1128, +0.0417] |
| SSPG-Net minus EditGuard | pixel_f1 | -0.1024 | [-0.1404, -0.0608] |
| SSPG-Net minus EditGuard | pixel_iou | -0.1492 | [-0.2054, -0.0869] |

## Limits and reproducibility

The EditGuard official test configuration's network scale conflicts with `clean.pth`; the adapter option instantiates the checkpoint's twelve-block architecture, as documented. The two active methods produce different protected images, so Table B is a source/geometry-matched comparison. The official EditGuard testing code normally reads 512x512 images; this evaluation uses the 256x256 E1 source resolution, which its fully convolutional model accepts. The adapter computes residuals from the unclamped recovery output, following official `image_recovery`. A saved official-example sanity check is in `editguard_adapter_audit.json`.

Commands from repository root:

```powershell
python scripts/e8_preflight.py
python scripts/e8_freeze_protocol.py
python scripts/e8_trufor.py
python scripts/e8_editguard.py
python scripts/e8_bootstrap.py
python scripts/e8_versions.py
python scripts/e8_verify.py
python scripts/e8_finalize.py
python scripts/e8_archive.py
```

Official checkpoints must be placed at the paths defined in the adapters. The exact repository commits and checkpoint SHA-256 values above identify the evaluated versions.

Environment setup from the repository root (with the existing CUDA-enabled SSPG Python environment):

    git clone https://github.com/grip-unina/TruFor.git third_party/TruFor
    git -C third_party/TruFor checkout ae54475df6f41a491d7615100feb19263dec13f7
    git clone https://github.com/xuanyuzhang21/EditGuard.git third_party/EditGuard
    git -C third_party/EditGuard checkout 174778fb371640bb31b32fcb9f7bfd440c88766b
    python -m pip install yacs gdown
    python -m pip install basicsr==1.4.2 --no-build-isolation --no-deps
    curl.exe -L -o third_party/TruFor/TruFor_train_test/pretrained_models/TruFor_weights.zip https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip
    Expand-Archive -LiteralPath third_party/TruFor/TruFor_train_test/pretrained_models/TruFor_weights.zip -DestinationPath third_party/TruFor/TruFor_train_test/pretrained_models -Force
    python -m gdown 1w4e1gpdInAv7Lj_NQ7EGgmMuInyfUYgi -O third_party/EditGuard/checkpoints/clean.pth

Verify the official TruFor archive MD5 7bee48f3476c75616c3c5721ab256ff8 and both checkpoint SHA-256 values above before inference. The environment also requires the SSPG dependencies, scikit-learn, torchvision, OpenCV, and PyYAML.

A one-source Test smoke check validated tensor shapes and mask equality before the full run. No threshold or model option was selected from its performance. The 256x256 EditGuard adaptation may differ from its native 512x512 testing behavior.

Shared-evaluator verification: PASS. The saved E1, TruFor, and EditGuard maps and masks were re-evaluated with scripts/e8_metrics.py. After correcting the EditGuard residual calculation to use the unclamped recovery output, its raw pixel counts and paired differences changed slightly; all manuscript values remain unchanged when rounded to four decimal places. The detailed check is in e8_verification.json.
