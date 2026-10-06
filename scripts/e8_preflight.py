"""Re-run frozen E1 into an isolated E8 preflight directory."""
import json
from pathlib import Path

import pandas as pd
import torch

import eval_test_final as e1
from models.sspg_joint_model_v2 import SSPGJointModelV2
from preprocessing.active_sspg_dataset import ActiveSSPGDataset
from training.e53r_training_protocol_v1 import PROTOCOL
from training.train_e53r_joint_v1 import build_loader


def main():
    out = e1.ROOT / "results" / "external_baselines" / "e8" / "preflight"
    e1.OUT = out
    out.mkdir(parents=True, exist_ok=True)
    threshold = float(pd.read_csv(e1.VAL_SUMMARY).iloc[0]["detection_threshold"])
    records = e1.test_records()
    dataset = ActiveSSPGDataset(records=records, base_seed=e1.TEST_SEED)
    loader = build_loader(dataset, shuffle=False, seed=e1.TEST_SEED)
    model = SSPGJointModelV2(
        stego_checkpoint=str(e1.STEGO),
        adapter_bottleneck=PROTOCOL.adapter_bottleneck,
        fusion_hidden_dim=PROTOCOL.fusion_hidden_dim,
        knn_k=PROTOCOL.knn_k,
        gnn_hidden_dim=PROTOCOL.gnn_hidden_dim,
        gnn_depth=PROTOCOL.gnn_depth,
    ).to(e1.DEVICE)
    checkpoint = torch.load(e1.CHECKPOINT, map_location=e1.DEVICE, weights_only=False)
    if checkpoint.get("selection_metric") != "validation_joint_loss" or checkpoint.get("test_accessed") is not False:
        raise RuntimeError("Frozen checkpoint certification failed")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    det, loc, rows = e1.evaluate(model, loader, threshold, float(PROTOCOL.localization_threshold))
    observed = {"detection": det, "localization": loc, "sources": len(records), "samples": len(dataset)}
    (out / "metrics.json").write_text(json.dumps(observed, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "test_predictions.csv", index=False)
    expected_det = json.loads((e1.ROOT / "results/extension_e1_e7_v1.0.0/e1/detection_metrics.json").read_text())
    expected_loc = json.loads((e1.ROOT / "results/extension_e1_e7_v1.0.0/e1/localization_metrics.json").read_text())
    checks = {key: round(det[key], 4) == round(expected_det[key], 4) for key in ("roc_auc", "f1")}
    checks.update({key: round(loc[key], 4) == round(expected_loc[key], 4) for key in ("pixel_f1", "pixel_iou")})
    observed["published_four_decimal_checks"] = checks
    (out / "metrics.json").write_text(json.dumps(observed, indent=2), encoding="utf-8")
    print(json.dumps(observed, indent=2))
    if not all(checks.values()):
        raise SystemExit("E1 preflight mismatch; stop before external baselines")


if __name__ == "__main__":
    main()
