"""Re-evaluate saved E8 maps with one shared E1-compatible evaluator."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from e8_freeze_protocol import records_for
from e8_metrics import detection_metrics, localization_metrics, pixel_counts, sum_pixel_counts
from eval_test_final import CHECKPOINT, MANIFEST, ROOT, STEGO, TEST_SEED
from preprocessing.active_sspg_dataset import ActiveSSPGDataset

OUT = ROOT / "results/external_baselines/e8"


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_numbers(observed, stored, names, tolerance=1e-12):
    for name in names:
        a, b = observed[name], stored[name]
        if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
            if a != b:
                raise RuntimeError(f"{name}: {a} != {b}")
        elif not np.isclose(a, b, rtol=0, atol=tolerance):
            raise RuntimeError(f"{name}: {a} != {b}")


def evaluate_maps(method, split, dataset):
    folder = "validation" if split == "val" else "test"
    frame = pd.read_csv(OUT / folder / f"{method}_predictions.csv")
    if len(frame) != len(dataset):
        raise RuntimeError(f"{method} {split}: incomplete predictions")
    expected_samples = [dataset[i] for i in range(len(dataset))]
    rows = []
    threshold = 0.2 if method == "editguard" else 0.5
    for i, row in frame.iterrows():
        sample = expected_samples[i]
        truth = sample["mask"].numpy().astype(np.uint8)
        if row.sample_key != sample["sample_key"] or int(row.detection_label) != i % 2:
            raise RuntimeError(f"{method} {split}: sample identity/order mismatch at {i}")
        if row.attack != sample["attack"] or row.mask_sha256 != hashlib.sha256(truth.tobytes()).hexdigest():
            raise RuntimeError(f"{method} {split}: attack/mask mismatch at {i}")
        fmap = np.load(OUT / folder / f"{method}_maps" / row.localization_map,
                       allow_pickle=False)
        counts = pixel_counts(fmap, truth[0], threshold, strict=(method == "editguard"))
        same_numbers(counts, {"tp": row.pixel_tp, "fp": row.pixel_fp, "fn": row.pixel_fn,
                              "tn": counts["tn"]}, ("tp", "fp", "fn"))
        rows.append(counts)
    return frame, localization_metrics(sum_pixel_counts(rows), threshold)


def main():
    protocol = json.loads((OUT / "protocol_manifest.json").read_text())
    for path, key in ((MANIFEST, "manifest_sha256"), (CHECKPOINT, "sspgn_checkpoint_sha256"),
                      (STEGO, "stego_checkpoint_sha256")):
        if sha(path) != protocol[key]:
            raise RuntimeError(f"Frozen E1 artifact changed: {path}")
    expected = {
        "trufor": sha(ROOT / "third_party/TruFor/TruFor_train_test/pretrained_models/weights/trufor.pth.tar"),
        "editguard": sha(ROOT / "third_party/EditGuard/checkpoints/clean.pth"),
    }
    versions = json.loads((OUT / "repo_versions.json").read_text()) if (OUT / "repo_versions.json").exists() else None
    if versions:
        for method in expected:
            if expected[method] != versions[method]["checkpoint_sha256"]:
                raise RuntimeError(f"{method} checkpoint changed")
    outputs = {}
    for split, seed, sources in (("val", 20260923, 62), ("test", TEST_SEED, 63)):
        dataset = ActiveSSPGDataset(records_for(split), seed)
        if len(dataset) != 2 * sources:
            raise RuntimeError(f"{split}: source count changed")
        outputs[split] = {}
        for method in ("trufor", "editguard"):
            frame, loc = evaluate_maps(method, split, dataset)
            outputs[split][method] = {"sources": sources, "samples": len(frame),
                                      "pixel_f1": loc["pixel_f1"], "pixel_iou": loc["pixel_iou"],
                                      "tp": loc["tp"], "fp": loc["fp"], "fn": loc["fn"], "tn": loc["tn"]}
            if split == "test":
                stored = json.loads((OUT / "test" / f"{method}_metrics.json").read_text())
                same_numbers({"pixel_f1": loc["pixel_f1"], "pixel_iou": loc["pixel_iou"],
                              "pixel_tp": loc["tp"], "pixel_fp": loc["fp"], "pixel_fn": loc["fn"]},
                             stored, ("pixel_f1", "pixel_iou", "pixel_tp", "pixel_fp", "pixel_fn"))
    val = pd.read_csv(OUT / "validation/trufor_predictions.csv")
    fpr, tpr, thresholds = roc_curve(val.detection_label, val.manipulation_score)
    finite = np.flatnonzero(np.isfinite(thresholds))
    chosen = finite[np.argmax((tpr - fpr)[finite])]
    threshold = float(thresholds[chosen])
    calibration = json.loads((OUT / "validation/trufor_calibration.json").read_text())
    same_numbers({"threshold": threshold}, calibration, ("threshold",))
    if versions:
        same_numbers({"detection_threshold": threshold}, versions["trufor"],
                     ("detection_threshold",))
    test = pd.read_csv(OUT / "test/trufor_predictions.csv")
    det = detection_metrics(test.detection_label, test.manipulation_score, threshold)
    stored = json.loads((OUT / "test/trufor_metrics.json").read_text())
    same_numbers({"roc_auc": det["roc_auc"], "balanced_accuracy": det["balanced_accuracy"],
                  "detection_f1": det["f1"], "sensitivity": det["sensitivity"],
                  "specificity": det["specificity"]}, stored,
                 ("roc_auc", "balanced_accuracy", "detection_f1", "sensitivity", "specificity"))
    sspg = pd.read_csv(OUT / "preflight/test_predictions.csv")
    if not test.sample_key.equals(sspg.sample_key):
        raise RuntimeError("TruFor/SSPG Test source pairing changed")
    sspg_det = detection_metrics(sspg.detection_label, sspg.detection_probability,
                                 protocol["detection_threshold"])
    e1_det = json.loads((ROOT / "results/extension_e1_e7_v1.0.0/e1/detection_metrics.json").read_text())
    same_numbers(sspg_det, e1_det, tuple(sspg_det))
    sspg_dataset = ActiveSSPGDataset(records_for("test"), TEST_SEED)
    sspg_counts = []
    for i, row in sspg.iterrows():
        mask = sspg_dataset[i]["mask"].numpy()[0]
        fmap = np.load(OUT / "preflight/prediction_masks" / row.prediction_mask,
                       allow_pickle=False)
        sspg_counts.append(pixel_counts(fmap, mask, protocol["localization_threshold"]))
    sspg_loc = localization_metrics(sum_pixel_counts(sspg_counts), protocol["localization_threshold"])
    e1_loc = json.loads((ROOT / "results/extension_e1_e7_v1.0.0/e1/localization_metrics.json").read_text())
    same_numbers(sspg_loc, e1_loc, tuple(sspg_loc))
    outputs["test"]["sspg"] = {"detection": sspg_det, "localization": sspg_loc}
    outputs["test"]["trufor"]["detection"] = det
    outputs["status"] = "PASS"
    outputs["evaluator"] = "scripts/e8_metrics.py"
    (OUT / "e8_verification.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print("PASS: all saved E1/E8 maps, masks, scores, counts, and metrics agree")


if __name__ == "__main__":
    main()
