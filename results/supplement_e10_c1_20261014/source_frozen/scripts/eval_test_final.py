"""Frozen IMD2020 final-test evaluation for SSPG-Net (E1)."""
from pathlib import Path
import csv, json, sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.sspg_joint_model_v2 import SSPGJointModelV2
from preprocessing.active_sspg_dataset import ActiveSSPGDataset
from training.e53r_training_protocol_v1 import PROTOCOL
from training.train_e53r_joint_v1 import authentic_label, build_loader, resolve_image_path

OUT = ROOT / "results" / "test_evaluation"
MANIFEST = ROOT / "manifests" / "imd2020_manifest_e5_clean.csv"
CHECKPOINT = ROOT / "checkpoints" / "e53r_joint_v1_best.pth"
STEGO = ROOT / "checkpoints" / "stego_structured_v33_128bit_best.pth"
VAL_SUMMARY = ROOT / "results" / "release_v1.0.0" / "joint" / "e53r_joint_v1_summary.csv"
TEST_SEED = 20260924  # fixed before evaluation; follows train=20260922,val=20260923
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_records():
    df = pd.read_csv(MANIFEST)
    part = df[df["split"].astype(str).str.strip().str.lower().eq("test")]
    part = part[part["label"].apply(authentic_label)]
    records = []
    for _, row in part.iterrows():
        path = resolve_image_path(row["image_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({"sample_id": str(row["sample_id"]), "group": str(row["group"]),
                        "split": "test", "label": row["label"], "image_path": str(path)})
    records.sort(key=lambda r: (r["group"], r["sample_id"], r["image_path"]))
    if not records:
        raise RuntimeError("No authentic IMD2020 test sources found")
    return records


@torch.no_grad()
def evaluate(model, loader, det_threshold, loc_threshold=0.5):
    labels, probs, rows = [], [], []
    ltp = lfp = lfn = ltn = 0
    mask_dir = OUT / "prediction_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for i, batch in enumerate(loader):
        received = batch["received"].to(DEVICE).float()
        payload = batch["payload"].to(DEVICE).float()
        truth = batch["mask"].to(DEVICE) >= 0.5
        target = batch["detection_label"].to(DEVICE)
        output = model(received, payload)
        prob = torch.sigmoid(output["detection_logits"])
        loc_prob = torch.sigmoid(model.localization_logits_256(output["localization_grid_logits"]))
        pred = loc_prob >= loc_threshold
        ltp += int((pred & truth).sum()); lfp += int((pred & ~truth).sum())
        lfn += int((~pred & truth).sum()); ltn += int((~pred & ~truth).sum())
        y, p = int(target[0].item()), float(prob[0].item())
        labels.append(y); probs.append(p)
        key = str(batch["sample_key"][0])
        safe = f"{i:04d}_{abs(hash(key)) & 0xffffffff:08x}.npy"
        np.save(mask_dir / safe, loc_prob[0, 0].cpu().numpy().astype(np.float32))
        rows.append({"sample_key": key, "source_path": batch["source_path"][0],
                     "attack": batch["attack"][0], "detection_label": y,
                     "detection_probability": p, "prediction_mask": safe})
    labels, probs = np.asarray(labels), np.asarray(probs)
    pred_det = (probs >= det_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred_det, labels=[0, 1]).ravel()
    f1den, iouden = 2*ltp+lfp+lfn, ltp+lfp+lfn
    metrics = {"roc_auc": float(roc_auc_score(labels, probs)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred_det)),
        "f1": float(f1_score(labels, pred_det, zero_division=0)),
        "sensitivity": float(tp/(tp+fn)) if tp+fn else None,
        "specificity": float(tn/(tn+fp)) if tn+fp else None,
        "threshold": det_threshold, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    loc = {"pixel_f1": float(2*ltp/f1den) if f1den else 1.0,
           "pixel_iou": float(ltp/iouden) if iouden else 1.0,
           "threshold": loc_threshold, "tp": ltp, "fp": lfp, "fn": lfn, "tn": ltn}
    return metrics, loc, rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(VAL_SUMMARY).iloc[0]
    det_threshold = float(val["detection_threshold"])
    records = test_records()
    dataset = ActiveSSPGDataset(records=records, base_seed=TEST_SEED)
    loader = build_loader(dataset, shuffle=False, seed=TEST_SEED)
    model = SSPGJointModelV2(stego_checkpoint=str(STEGO), adapter_bottleneck=PROTOCOL.adapter_bottleneck,
        fusion_hidden_dim=PROTOCOL.fusion_hidden_dim, knn_k=PROTOCOL.knn_k,
        gnn_hidden_dim=PROTOCOL.gnn_hidden_dim, gnn_depth=PROTOCOL.gnn_depth).to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    if ckpt.get("selection_metric") != "validation_joint_loss" or ckpt.get("test_accessed") is not False:
        raise RuntimeError("Checkpoint does not certify frozen validation-only selection")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    det, loc, rows = evaluate(model, loader, det_threshold, float(PROTOCOL.localization_threshold))
    (OUT / "detection_metrics.json").write_text(json.dumps(det, indent=2), encoding="utf-8")
    (OUT / "localization_metrics.json").write_text(json.dumps(loc, indent=2), encoding="utf-8")
    pd.DataFrame([[det[k] for k in ("tn","fp","fn","tp")]], columns=["tn","fp","fn","tp"]).to_csv(OUT/"confusion_matrix.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "test_predictions.csv", index=False)
    summary = f"# E1 Final Test Evaluation\n\n- Frozen checkpoint: `{CHECKPOINT.name}`\n- Validation detection threshold: {det_threshold:.12f}\n- Test seed (pre-evaluation): {TEST_SEED}\n- Sources / active samples: {len(records)} / {len(dataset)}\n- Device: {DEVICE}\n\n## Detection\n\n```json\n{json.dumps(det, indent=2)}\n```\n\n## Localization\n\n```json\n{json.dumps(loc, indent=2)}\n```\n"
    (OUT / "test_summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"detection": det, "localization": loc, "sources": len(records), "samples": len(dataset)}, indent=2))


if __name__ == "__main__":
    main()
