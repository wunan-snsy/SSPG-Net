"""Run official TruFor weights on exact in-memory SSPG E1 active tensors."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve

from e8_metrics import detection_metrics, localization_metrics, pixel_counts, sum_pixel_counts
from e8_freeze_protocol import records_for
from eval_test_final import ROOT, TEST_SEED
from preprocessing.active_sspg_dataset import ActiveSSPGDataset

TRUFOR = ROOT / "third_party/TruFor/TruFor_train_test"
CHECKPOINT = TRUFOR / "pretrained_models/weights/trufor.pth.tar"
OUT = ROOT / "results/external_baselines/e8"


def load_model():
    sys.path.insert(0, str(TRUFOR))
    from lib.config import config, update_config
    from lib.utils import get_model
    from types import SimpleNamespace
    old_cwd = Path.cwd()
    try:
        os.chdir(TRUFOR)
        update_config(config, SimpleNamespace(experiment="trufor_ph3", gpu=[0], opts=["TEST.MODEL_FILE", str(CHECKPOINT)]))
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        model = get_model(config)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    finally:
        os.chdir(old_cwd)
    return model.cuda().eval()


@torch.inference_mode()
def predict(model, image):
    # Official TestDataset converts uint8 RGB to float32 / 256; E1 tensors are /255.
    # Preserve E1 float-valued pixels while applying the official input scale.
    rgb = (image.unsqueeze(0).cuda() * (255.0 / 256.0)).float()
    pred, conf, det, _ = model(rgb, save_np=False)
    if det is None:
        raise RuntimeError("TruFor checkpoint did not return detection score")
    score = float(torch.sigmoid(det).item())
    fmap = F.softmax(pred, dim=1)[:, 1:2]
    fmap = F.interpolate(fmap, size=(256, 256), mode="bilinear", align_corners=False)
    return score, fmap[0, 0].cpu().numpy().astype(np.float32)


def run_split(model, split, limit=None):
    seed = 20260923 if split == "val" else TEST_SEED
    dataset = ActiveSSPGDataset(records_for(split), seed)
    outdir = OUT / ("validation" if split == "val" else "test")
    maps = outdir / "trufor_maps"
    maps.mkdir(parents=True, exist_ok=True)
    rows = []
    per_sample_counts = []
    for i in range(len(dataset) if limit is None else min(limit, len(dataset))):
        sample = dataset[i]
        score, fmap = predict(model, sample["received"])
        truth = sample["mask"].numpy()[0] >= 0.5
        counts = pixel_counts(fmap, truth, 0.5)
        per_sample_counts.append(counts)
        map_name = f"{i:04d}.npy"
        np.save(maps / map_name, fmap)
        rows.append({"sample_index": i, "sample_key": sample["sample_key"],
                     "source_path": sample["source_path"], "attack": sample["attack"],
                     "detection_label": int(sample["detection_label"].item()),
                     "manipulation_score": score, "localization_map": map_name,
                     "mask_sha256": __import__("hashlib").sha256(sample["mask"].numpy().astype(np.uint8).tobytes()).hexdigest(),
                     "pixel_tp": counts["tp"], "pixel_fp": counts["fp"],
                     "pixel_fn": counts["fn"]})
        if (i + 1) % 10 == 0:
            print(f"{split}: {i+1}/{len(dataset)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(outdir / "trufor_predictions.csv", index=False)
    return frame, sum_pixel_counts(per_sample_counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    model = load_model()
    val, _ = run_split(model, "val", args.limit)
    if args.limit is not None:
        return
    fpr, tpr, thresholds = roc_curve(val.detection_label, val.manipulation_score)
    finite = np.where(np.isfinite(thresholds))[0]
    selected = int(finite[np.argmax((tpr - fpr)[finite])])
    threshold = float(thresholds[selected])
    calibration = {"threshold": threshold, "youden_j": float((tpr-fpr)[selected]),
                   "sources": 62, "samples": len(val), "score_orientation": "higher=manipulation",
                   "map_threshold": 0.5, "preprocessing": "Exact E1 float tensor *255/256, official TruFor scale"}
    (OUT / "validation/trufor_calibration.json").write_text(json.dumps(calibration, indent=2))
    test, counts = run_split(model, "test")
    det = detection_metrics(test.detection_label, test.manipulation_score, threshold)
    loc = localization_metrics(counts, 0.5)
    metrics = {"roc_auc": det["roc_auc"], "balanced_accuracy": det["balanced_accuracy"],
               "detection_f1": det["f1"], "sensitivity": det["sensitivity"],
               "specificity": det["specificity"], "pixel_f1": loc["pixel_f1"],
               "pixel_iou": loc["pixel_iou"], "pixel_tp": loc["tp"],
               "pixel_fp": loc["fp"], "pixel_fn": loc["fn"], "pixel_tn": loc["tn"],
               "detection_threshold": threshold, "localization_threshold": 0.5,
               "sources": 63, "samples": len(test)}
    (OUT / "test/trufor_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
