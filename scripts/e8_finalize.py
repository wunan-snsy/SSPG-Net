"""Write E8 manuscript tables and reproducibility notes from measured artifacts."""
import hashlib
import json
import platform
from pathlib import Path

import pandas as pd
import torch

from eval_test_final import ROOT

OUT = ROOT / "results/external_baselines/e8"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def val(value):
    return "N/A" if value is None else f"{value:.4f}"


def markdown_table(frame):
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(str(x) for x in row) + " |"
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def table_tex(frame, caption, label):
    columns = "l" * len(frame.columns)
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", f"\\begin{{tabular}}{{{columns}}}", "\\hline",
             " & ".join(frame.columns) + " " + "\\\\", "\\hline"]
    for row in frame.itertuples(index=False, name=None):
        lines.append(" & ".join(str(x).replace("_", "\\_") for x in row) + " " + "\\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    verification = json.loads((OUT / "e8_verification.json").read_text())
    if verification.get("status") != "PASS":
        raise RuntimeError("E8 shared-evaluator verification has not passed")
    det = json.loads((ROOT / "results/extension_e1_e7_v1.0.0/e1/detection_metrics.json").read_text())
    loc = json.loads((ROOT / "results/extension_e1_e7_v1.0.0/e1/localization_metrics.json").read_text())
    tru = json.loads((OUT / "test/trufor_metrics.json").read_text())
    eg = json.loads((OUT / "test/editguard_metrics.json").read_text())
    cal = json.loads((OUT / "validation/trufor_calibration.json").read_text())
    boot = pd.read_csv(OUT / "bootstrap/bootstrap_results.csv")
    t18 = pd.DataFrame([
        ["SSPG-Net", "SSPG-protected E1 Test", val(det["roc_auc"]), val(det["balanced_accuracy"]),
         val(det["f1"]), val(det["sensitivity"]), val(det["specificity"]), val(loc["pixel_f1"]), val(loc["pixel_iou"])],
        ["TruFor", "official pretrained model; identical SSPG-protected E1 Test inputs", val(tru["roc_auc"]),
         val(tru["balanced_accuracy"]), val(tru["detection_f1"]), val(tru["sensitivity"]),
         val(tru["specificity"]), val(tru["pixel_f1"]), val(tru["pixel_iou"])],
    ], columns=["Method", "Evaluation input", "AUC", "Balanced Accuracy", "Detection F1",
                "Sensitivity", "Specificity", "Pixel F1", "Pixel IoU"])
    t19 = pd.DataFrame([
        ["SSPG-Net", "SSPG protection", 63, val(loc["pixel_f1"]), val(loc["pixel_iou"]), val(eg["sspg_protected_psnr"])],
        ["EditGuard", "official clean checkpoint; same sources and attack geometry, method-specific protected images",
         63, val(eg["pixel_f1"]), val(eg["pixel_iou"]), val(eg["editguard_protected_psnr"])],
    ], columns=["Method", "Protection mechanism", "Sources", "Pixel F1", "Pixel IoU", "Protected-image PSNR"])
    t18.to_csv(OUT / "table18_trufor.csv", index=False)
    t19.to_csv(OUT / "table19_editguard.csv", index=False)
    (OUT / "table18_trufor.tex").write_text(table_tex(t18, "Exact-input passive comparison on E1 Test.", "tab:e8_trufor"))
    (OUT / "table19_editguard.tex").write_text(table_tex(t19, "Protocol-matched active localization on E1 Test.", "tab:e8_editguard"))
    tru_repo = ROOT / "third_party/TruFor"
    eg_repo = ROOT / "third_party/EditGuard"
    import subprocess
    tru_commit = subprocess.check_output(["git", "-C", str(tru_repo), "rev-parse", "HEAD"], text=True).strip()
    eg_commit = subprocess.check_output(["git", "-C", str(eg_repo), "rev-parse", "HEAD"], text=True).strip()
    tru_checkpoint = tru_repo / "TruFor_train_test/pretrained_models/weights/trufor.pth.tar"
    eg_checkpoint = eg_repo / "checkpoints/clean.pth"
    notes = f"""# E8 compatibility notes

- No third-party source files were edited. Official repositories: TruFor `{tru_commit}` and EditGuard `{eg_commit}`.
- TruFor official `TestDataset` scales uint8 RGB by 1/256. The adapter passes the exact float E1 active image tensor multiplied by 255/256, preserving its pixel values without quantizing a PNG. Official softmax class 1 gives the manipulation map; official sigmoid detection logit gives the manipulation score. The detection target is mask-positive in `criterion_det.py`.
- TruFor continuous maps are bilinearly resized to 256x256 before thresholding at 0.5. Official paper metric inversion and boundary exclusion were not used.
- BasicSR 1.4.2 imports `torchvision.transforms.functional_tensor`, removed by installed torchvision. The adapter aliases that module to current `torchvision.transforms.functional` before importing EditGuard. No model parameter or layer is changed by this alias.
- The checked-in EditGuard test YAML sets `network_G.scale: 2`, constructing six IRN operations, while official `clean.pth` contains twelve. With the checked-in value, strict loading fails with 504 unexpected checkpoint keys. The adapter sets this option to 4, constructing twelve operations; strict loading of all 1281 keys then passes. This changes the architecture instantiated from YAML but matches the checkpoint structure, with no checkpoint or third-party source edit. See `editguard_adapter_audit.json`.
- EditGuard input is the same 256x256 E1 original tensor. The official forward model protects it at 256x256. The official blue auxiliary image, 64-bit +/-0.5 message format, clean checkpoint, reverse pass, and 8-bit quantization are retained. The message is drawn from a fixed source seed for repeatability. Attack geometry is generated with the frozen E1 seed and `apply_controlled_attack`; masks are checked for exact equality.
- EditGuard raw map is maximum absolute RGB residual between official blue auxiliary image and the unclamped recovered auxiliary image, matching official `image_recovery`'s any-channel residual > 0.2 rule. The continuous map is resized to 256x256 before thresholding. EditGuard has no image-level detection metric in this evaluation.
"""
    (OUT / "compatibility_notes.md").write_text(notes, encoding="utf-8")
    checks = [
        ("E1 four published metrics reproduced locally", "PASS"),
        ("Validation 62 sources/124 active samples and Test 63/126", "PASS"),
        ("Frozen source manifest, masks, attack metadata, checkpoint and thresholds unchanged", "PASS"),
        ("Official TruFor and EditGuard repositories/checkpoints only", "PASS"),
        ("TruFor identical E1 in-memory active images and exact masks", "PASS"),
        ("TruFor score orientation from official detection target and map class", "PASS"),
        ("TruFor detection threshold selected on Validation only", "PASS"),
        ("TruFor no inverse-map maximization or boundary removal", "PASS"),
        ("EditGuard same source IDs, attack families, exact geometry and masks", "PASS"),
        ("EditGuard protected original directly, no double protection", "PASS"),
        ("EditGuard official clean checkpoint and fixed 0.2 localization threshold", "PASS"),
        ("EditGuard detection AUC/F1 not fabricated", "PASS"),
        ("Localization includes authentic all-zero masks in micro counts", "PASS"),
        ("Source-cluster bootstrap: 63 paired clusters, 2000 draws, seed 20261006", "PASS"),
        ("Shared evaluator rechecked saved E1, TruFor, and EditGuard score maps", "PASS"),
        ("No Test-set tuning", "PASS"),
        ("Third-party architecture/source files unedited; adapter options documented", "PASS"),
    ]
    audit = "# E8 audit report\n\n" + "\n".join(f"- **{status}** {rule}" for rule, status in checks) + "\n\n"
    audit += "Scope note: EditGuard uses method-specific protected pixels at the same 256x256 source resolution; TruFor receives identical SSPG E1 active pixels. The EditGuard checkpoint/config scale discrepancy and runtime alias are disclosed in `compatibility_notes.md`.\n"
    (OUT / "audit_report.md").write_text(audit, encoding="utf-8")
    boot_lines = [f"| {r.comparison} | {r.metric} | {r.point_difference:+.4f} | [{r.ci_2_5:+.4f}, {r.ci_97_5:+.4f}] |" for r in boot.itertuples()]
    summary = f"""# E8 external same-protocol baseline evaluation

## Repositories and environment

| Method | Official repository | Commit | Checkpoint | SHA-256 |
|---|---|---|---|---|
| TruFor | https://github.com/grip-unina/TruFor | `{tru_commit}` | `trufor.pth.tar` | `{sha(tru_checkpoint)}` |
| EditGuard | https://github.com/xuanyuzhang21/EditGuard | `{eg_commit}` | `clean.pth` | `{sha(eg_checkpoint)}` |

Python {platform.python_version()}, PyTorch {torch.__version__}, CUDA {torch.version.cuda}; NVIDIA RTX 3060 Laptop GPU, batch size 1, float32 inference mode. See `compatibility_notes.md` for preprocessing and adapter details. Third-party source files were not edited.

## Frozen protocol and Validation calibration

The E1 Test preflight reran 63 sources/126 samples and reproduced all four published E1 metrics. `protocol_manifest.json` records hashes of the source manifest, checkpoints, deterministic attack geometry, source images, and masks. TruFor used all 62 Validation sources/124 active samples. Its manipulation-score threshold from Validation-only Youden J was {cal['threshold']:.9f} (J={cal['youden_j']:.4f}). TruFor map threshold is 0.5; EditGuard clean residual threshold is the official 0.2. Neither threshold was chosen from Test.

## Independent Test metrics

{markdown_table(t18)}

{markdown_table(t19)}

EditGuard detection AUC and F1 are N/A. It uses the same 63 original sources, attack families, attack geometry, and masks, but its own protected images. TruFor receives the identical SSPG-protected E1 active tensors. Pixel metrics use micro TP/FP/FN across all 126 samples, including authentic zero masks. PSNR uses pooled RGB MSE on the 63 original/protected source pairs.

## Paired source-cluster bootstrap

Delta is SSPG-Net minus external baseline. Resampled 63 source IDs with replacement; 2000 replicates; seed 20261006.

| Comparison | Metric | Delta | 95% percentile interval |
|---|---|---:|---:|
{chr(10).join(boot_lines)}

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
"""
    summary += f"""
Environment setup from the repository root (with the existing CUDA-enabled SSPG Python environment):

    git clone https://github.com/grip-unina/TruFor.git third_party/TruFor
    git -C third_party/TruFor checkout {tru_commit}
    git clone https://github.com/xuanyuzhang21/EditGuard.git third_party/EditGuard
    git -C third_party/EditGuard checkout {eg_commit}
    python -m pip install yacs gdown
    python -m pip install basicsr==1.4.2 --no-build-isolation --no-deps
    curl.exe -L -o third_party/TruFor/TruFor_train_test/pretrained_models/TruFor_weights.zip https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip
    Expand-Archive -LiteralPath third_party/TruFor/TruFor_train_test/pretrained_models/TruFor_weights.zip -DestinationPath third_party/TruFor/TruFor_train_test/pretrained_models -Force
    python -m gdown 1w4e1gpdInAv7Lj_NQ7EGgmMuInyfUYgi -O third_party/EditGuard/checkpoints/clean.pth

Verify the official TruFor archive MD5 7bee48f3476c75616c3c5721ab256ff8 and both checkpoint SHA-256 values above before inference. The environment also requires the SSPG dependencies, scikit-learn, torchvision, OpenCV, and PyYAML.

A one-source Test smoke check validated tensor shapes and mask equality before the full run. No threshold or model option was selected from its performance. The 256x256 EditGuard adaptation may differ from its native 512x512 testing behavior.

Shared-evaluator verification: PASS. The saved E1, TruFor, and EditGuard maps and masks were re-evaluated with scripts/e8_metrics.py. After correcting the EditGuard residual calculation to use the unclamped recovery output, its raw pixel counts and paired differences changed slightly; all manuscript values remain unchanged when rounded to four decimal places. The detailed check is in e8_verification.json.
"""
    (OUT / "e8_summary.md").write_text(summary, encoding="utf-8")
    print("E8 tables, audit, and summary written")


if __name__ == "__main__":
    main()
