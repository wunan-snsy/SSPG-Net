"""Paired source bootstrap for frozen SSPG E1 and available E8 baselines."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from e8_metrics import detection_metrics, localization_metrics
from e8_freeze_protocol import records_for
from eval_test_final import ROOT, TEST_SEED
from preprocessing.active_sspg_dataset import ActiveSSPGDataset

OUT = ROOT / "results/external_baselines/e8"


def metrics(frame, indices, score_col, det_threshold, pix_prefix):
    sub = frame.iloc[indices]
    det = detection_metrics(sub.detection_label, sub[score_col], det_threshold)
    tp = int(sub[f"{pix_prefix}_tp"].sum())
    fp = int(sub[f"{pix_prefix}_fp"].sum())
    fn = int(sub[f"{pix_prefix}_fn"].sum())
    loc = localization_metrics({"tp": tp, "fp": fp, "fn": fn,
                                "tn": len(sub)*256*256-tp-fp-fn}, 0.5)
    return {"auc": det["roc_auc"], "detection_f1": det["f1"],
            "pixel_f1": loc["pixel_f1"], "pixel_iou": loc["pixel_iou"]}


def main():
    sspg = pd.read_csv(OUT / "preflight/test_predictions.csv")
    tru = pd.read_csv(OUT / "test/trufor_predictions.csv")
    if len(sspg) != 126 or len(tru) != 126 or not sspg.sample_key.equals(tru.sample_key):
        raise RuntimeError("Paired source/order mismatch")
    dataset = ActiveSSPGDataset(records_for("test"), TEST_SEED)
    tp, fp, fn = [], [], []
    for i, row in sspg.iterrows():
        truth = dataset[i]["mask"].numpy()[0] >= 0.5
        pred = np.load(OUT / "preflight/prediction_masks" / row.prediction_mask) >= 0.5
        tp.append(int(np.sum(pred & truth)))
        fp.append(int(np.sum(pred & ~truth)))
        fn.append(int(np.sum(~pred & truth)))
    frame = tru.copy()
    frame["sspg_score"] = sspg.detection_probability
    frame["sspg_tp"], frame["sspg_fp"], frame["sspg_fn"] = tp, fp, fn
    assert (frame.groupby("sample_key").size() == 2).all()
    source_ids = frame.sample_key.drop_duplicates().tolist()
    assert len(source_ids) == 63
    source_to_indices = {sid: frame.index[frame.sample_key.eq(sid)].to_numpy() for sid in source_ids}
    trum = json.loads((OUT / "test/trufor_metrics.json").read_text())
    sspg_threshold = json.loads((OUT / "protocol_manifest.json").read_text())["detection_threshold"]
    comparisons = []
    all_indices = np.arange(len(frame))
    point_a = metrics(frame, all_indices, "sspg_score", sspg_threshold, "sspg")
    point_b = metrics(frame, all_indices, "manipulation_score", trum["detection_threshold"], "pixel")
    rng = np.random.default_rng(20261006)
    values = {k: [] for k in point_a}
    for _ in range(2000):
        sampled = rng.choice(source_ids, size=63, replace=True)
        indices = np.concatenate([source_to_indices[sid] for sid in sampled])
        a = metrics(frame, indices, "sspg_score", sspg_threshold, "sspg")
        b = metrics(frame, indices, "manipulation_score", trum["detection_threshold"], "pixel")
        for key in values:
            values[key].append(a[key] - b[key])
    for key in values:
        comparisons.append({"comparison": "SSPG-Net minus TruFor", "metric": key,
                            "point_difference": point_a[key] - point_b[key],
                            "ci_2_5": float(np.percentile(values[key], 2.5)),
                            "ci_97_5": float(np.percentile(values[key], 97.5)),
                            "seed": 20261006, "replicates": 2000, "clusters": 63})
    editguard_path = OUT / "test/editguard_predictions.csv"
    if editguard_path.exists():
        edit = pd.read_csv(editguard_path)
        if len(edit) != 126 or not frame.sample_key.equals(edit.sample_key):
            raise RuntimeError("EditGuard source/order mismatch")
        for key in ("tp", "fp", "fn"):
            frame[f"editguard_{key}"] = edit[f"pixel_{key}"]
        def pixel_metric(sub, prefix, metric):
            tp = int(sub[f"{prefix}_tp"].sum())
            fp = int(sub[f"{prefix}_fp"].sum())
            fn = int(sub[f"{prefix}_fn"].sum())
            loc = localization_metrics({"tp": tp, "fp": fp, "fn": fn,
                                        "tn": len(sub)*256*256-tp-fp-fn}, 0.5)
            return loc[metric]
        edit_values = {"pixel_f1": [], "pixel_iou": []}
        rng = np.random.default_rng(20261006)
        for _ in range(2000):
            sampled = rng.choice(source_ids, size=63, replace=True)
            sub = frame.iloc[np.concatenate([source_to_indices[sid] for sid in sampled])]
            for key in edit_values:
                edit_values[key].append(pixel_metric(sub, "sspg", key) - pixel_metric(sub, "editguard", key))
        for key in edit_values:
            point = pixel_metric(frame, "sspg", key) - pixel_metric(frame, "editguard", key)
            comparisons.append({"comparison": "SSPG-Net minus EditGuard", "metric": key,
                                "point_difference": point,
                                "ci_2_5": float(np.percentile(edit_values[key], 2.5)),
                                "ci_97_5": float(np.percentile(edit_values[key], 97.5)),
                                "seed": 20261006, "replicates": 2000, "clusters": 63})
    out = OUT / "bootstrap/bootstrap_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparisons).to_csv(out, index=False)
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == "__main__":
    main()
