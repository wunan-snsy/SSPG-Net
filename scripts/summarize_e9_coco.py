"""Summarize E9 by source-level bootstrap, preserving the frozen thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = ROOT / "manifests/e9_coco2017"
OUT_ROOT = ROOT / "results/e9_coco2017"
BANDS = ("01_05", "05_10", "10_15", "15_30")
REPLICATES = 2000
SEED = 20261012


def metric(frame: pd.DataFrame, threshold: float) -> dict:
    truth = frame.detection_label.to_numpy(dtype=int)
    score = frame.detection_probability.to_numpy(dtype=float)
    pred = score >= threshold
    det_tp = int(np.sum(pred & (truth == 1)))
    det_fp = int(np.sum(pred & (truth == 0)))
    det_fn = int(np.sum(~pred & (truth == 1)))
    tp = int(frame.pixel_tp.sum())
    fp = int(frame.pixel_fp.sum())
    fn = int(frame.pixel_fn.sum())
    result = {
        "auc": float(roc_auc_score(truth, score)),
        "detection_f1": 2 * det_tp / (2 * det_tp + det_fp + det_fn),
        "detection_recall": det_tp / (det_tp + det_fn),
        "authentic_fpr": det_fp / (det_fp + int(np.sum(~pred & (truth == 0)))),
        "pixel_f1": 2 * tp / (2 * tp + fp + fn),
        "pixel_iou": tp / (tp + fp + fn),
    }
    manipulated = frame[frame.detection_label.eq(1)]
    mtp = int(manipulated.pixel_tp.sum())
    mfp = int(manipulated.pixel_fp.sum())
    mfn = int(manipulated.pixel_fn.sum())
    result["manipulated_pixel_f1"] = 2 * mtp / (2 * mtp + mfp + mfn)
    result["manipulated_pixel_iou"] = mtp / (mtp + mfp + mfn)
    precision = manipulated.boundary_tp_pred.sum() / max(1, manipulated.boundary_pred_count.sum())
    recall = manipulated.boundary_tp_gt.sum() / max(1, manipulated.boundary_gt_count.sum())
    result["boundary_f1_2px"] = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("full", "no_provenance", "no_graph"), default="full")
    args = parser.parse_args()
    protocol = json.loads((MANIFEST_ROOT / "protocol.json").read_text(encoding="utf-8"))
    output_dir = OUT_ROOT / args.variant
    run = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    if run["status"] != "COMPLETE":
        raise RuntimeError("E9 inference has not completed")
    frame = pd.read_csv(output_dir / "predictions.csv")
    if len(frame) != 2600 or frame.source_id.nunique() != 200:
        raise RuntimeError("E9 prediction count mismatch")
    threshold = float(protocol["detection_threshold"])
    ids = frame.source_id.drop_duplicates().to_list()
    rng = np.random.default_rng(SEED)
    resamples = rng.integers(0, len(ids), size=(REPLICATES, len(ids)))
    rows = []
    for band in BANDS:
        band_frame = frame[frame.band.isin(("authentic", band))].copy()
        band_frame["source_order"] = pd.Categorical(band_frame.source_id,
                                                     categories=ids, ordered=True)
        band_frame = band_frame.sort_values(["source_order", "attack"]).reset_index(drop=True)
        if len(band_frame) != 800:
            raise RuntimeError(f"Expected 800 paired samples for {band}")
        if not (band_frame.source_id.to_numpy().reshape(200, 4) == np.asarray(ids)[:, None]).all():
            raise RuntimeError("Band/source pairing mismatch")
        point = metric(band_frame, threshold)
        boot = {key: [] for key in point}
        blocks = np.arange(800).reshape(200, 4)
        for picks in resamples:
            subset = band_frame.iloc[blocks[picks].ravel()]
            sample = metric(subset, threshold)
            for key in point:
                boot[key].append(sample[key])
        for name, value in point.items():
            rows.append({"variant": args.variant, "band": band,
                         "metric": name, "value": value,
                         "ci_low": float(np.percentile(boot[name], 2.5)),
                         "ci_high": float(np.percentile(boot[name], 97.5)),
                         "bootstrap_replicates": REPLICATES,
                         "bootstrap_seed": SEED,
                         "source_clusters": len(ids)})
        print(f"{args.variant} {band}: AUC={point['auc']:.4f}, pixel F1={point['pixel_f1']:.4f}, boundary F1={point['boundary_f1_2px']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(output_dir / "summary_with_95ci.csv", index=False)
    summary = {
        "variant": args.variant, "sources": len(ids), "samples": len(frame),
        "detection_threshold": threshold, "localization_threshold": 0.5,
        "primary_band": "15_30", "bootstrap_seed": SEED,
        "bootstrap_replicates": REPLICATES,
        "metric_definitions": {
            "pixel_f1": "Micro pooled over 200 authentic and 600 manipulated samples per band.",
            "manipulated_pixel_f1": "Micro pooled over 600 manipulated samples per band.",
            "boundary_f1_2px": "Micro boundary precision/recall on manipulated samples, 8-connected one-pixel boundary, 2-pixel tolerance.",
            "detection_f1": "200 authentic plus 600 manipulated samples per band, fixed Validation threshold.",
        },
    }
    (output_dir / "summary_protocol.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
