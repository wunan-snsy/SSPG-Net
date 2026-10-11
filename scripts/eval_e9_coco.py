"""Frozen E9 inference on the precommitted COCO source and attack manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_dilation, binary_erosion

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.sspg_ablation_model_v2 import SSPGAblationModelV2
from models.sspg_joint_model_v2 import SSPGJointModelV2
from preprocessing.active_sspg_dataset import (
    load_rgb_256, make_payload_deterministic, stable_seed,
)
from training.e53r_training_protocol_v1 import PROTOCOL
from training.train_stego_structured_v33 import embed_payload


MANIFEST_ROOT = ROOT / "manifests/e9_coco2017"
OUT_ROOT = ROOT / "results/e9_coco2017"
DECODER = ROOT / "checkpoints/clean_revision_v2/decoder_budget142/best.pth"
CHECKPOINTS = {
    "full": ROOT / "checkpoints/clean_revision_v2/budget142_full_best.pth",
    "no_provenance": ROOT / "checkpoints/clean_revision_v2/budget142_no_provenance_best.pth",
    "no_graph": ROOT / "checkpoints/clean_revision_v2/budget142_no_graph_best.pth",
}
FIELDS = ["source_id", "band", "attack", "detection_label",
          "detection_probability", "gt_area_fraction", "pixel_tp", "pixel_fp",
          "pixel_fn", "boundary_tp_pred", "boundary_tp_gt",
          "boundary_pred_count", "boundary_gt_count", "latency_ms"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def resolve_source(row: dict) -> Path:
    """Preserve frozen manifest bytes but support portable COCO placement."""
    recorded = Path(row["path"])
    if recorded.is_file():
        return recorded
    fallback = ROOT / "data/coco2017/val2017" / row["name"]
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Missing COCO val2017 image: {row['name']}")


def apply_attack(protected: torch.Tensor, plan: dict) -> tuple[torch.Tensor, np.ndarray]:
    top, left = int(plan["top"]), int(plan["left"])
    height, width = int(plan["height"]), int(plan["width"])
    result = protected.clone()
    truth = np.zeros((256, 256), dtype=bool)
    truth[top:top + height, left:left + width] = True
    region = result[:, top:top + height, left:left + width]
    if plan["attack"] == "erase":
        region.zero_()
    elif plan["attack"] == "copy_move":
        source_top, source_left = int(plan["source_top"]), int(plan["source_left"])
        if not (source_top + height <= top or top + height <= source_top or
                source_left + width <= left or left + width <= source_left):
            raise RuntimeError("Frozen copy-move rectangles overlap")
        region.copy_(protected[:, source_top:source_top + height,
                               source_left:source_left + width])
    elif plan["attack"] == "roll":
        region.copy_(torch.roll(region.clone(),
                                shifts=(max(1, height // 4), max(1, width // 4)),
                                dims=(1, 2)))
    else:
        raise RuntimeError(f"Unknown attack {plan['attack']}")
    return result.clamp(0, 1), truth


def boundary_counts(pred: np.ndarray, truth: np.ndarray) -> tuple[int, int, int, int]:
    if not truth.any():
        return 0, 0, 0, 0
    structure = np.ones((3, 3), dtype=bool)
    pred_boundary = pred & ~binary_erosion(pred, structure=structure, border_value=0)
    gt_boundary = truth & ~binary_erosion(truth, structure=structure, border_value=0)
    near_gt = binary_dilation(gt_boundary, iterations=2)
    near_pred = binary_dilation(pred_boundary, iterations=2)
    return (int((pred_boundary & near_gt).sum()),
            int((gt_boundary & near_pred).sum()),
            int(pred_boundary.sum()), int(gt_boundary.sum()))


def load_model(variant: str, protocol: dict):
    kwargs = dict(stego_checkpoint=str(DECODER),
                  adapter_bottleneck=PROTOCOL.adapter_bottleneck,
                  fusion_hidden_dim=PROTOCOL.fusion_hidden_dim,
                  knn_k=PROTOCOL.knn_k,
                  gnn_hidden_dim=PROTOCOL.gnn_hidden_dim,
                  gnn_depth=PROTOCOL.gnn_depth)
    if variant == "full":
        model = SSPGJointModelV2(**kwargs)
    else:
        model = SSPGAblationModelV2(ablation=variant, **kwargs)
    checkpoint = torch.load(CHECKPOINTS[variant], map_location="cpu", weights_only=False)
    if checkpoint.get("selection_metric") != "validation_joint_loss" or checkpoint.get("test_accessed") is not False:
        raise RuntimeError(f"{variant} checkpoint lacks Validation-only selection certification")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(DEVICE).eval()


def verify(protocol: dict) -> tuple[list[dict], dict[str, list[dict]]]:
    if sha(MANIFEST_ROOT / "source_manifest.csv") != protocol["source_manifest_sha256"]:
        raise RuntimeError("E9 source manifest changed")
    if sha(MANIFEST_ROOT / "attack_plan.csv") != protocol["attack_plan_sha256"]:
        raise RuntimeError("E9 attack plan changed")
    for name, checkpoint in CHECKPOINTS.items():
        if sha(checkpoint) != protocol["checkpoint_sha256"][name]:
            raise RuntimeError(f"Frozen {name} checkpoint changed")
    if sha(DECODER) != protocol["checkpoint_sha256"]["decoder"]:
        raise RuntimeError("Frozen decoder changed")
    sources = read_csv(MANIFEST_ROOT / "source_manifest.csv")
    attacks = defaultdict(list)
    for row in read_csv(MANIFEST_ROOT / "attack_plan.csv"):
        attacks[row["source_id"]].append(row)
    if len(sources) != 200 or len(attacks) != 200 or any(len(v) != 12 for v in attacks.values()):
        raise RuntimeError("E9 sample count mismatch")
    return sources, attacks


@torch.inference_mode()
def infer_one(model, received: torch.Tensor, payload: torch.Tensor,
              truth: np.ndarray, source_id: str, band: str, attack: str) -> dict:
    start = time.perf_counter()
    output = model(received.unsqueeze(0).to(DEVICE), payload.unsqueeze(0).to(DEVICE))
    score = torch.sigmoid(output["detection_logits"]).flatten()[0].item()
    loc_logits = model.localization_logits_256(output["localization_grid_logits"])
    pred = (torch.sigmoid(loc_logits)[0, 0] >= 0.5).cpu().numpy()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) * 1000
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    btp_pred, btp_gt, bpred, bgt = boundary_counts(pred, truth)
    return {"source_id": source_id, "band": band, "attack": attack,
            "detection_label": int(attack != "authentic"),
            "detection_probability": score,
            "gt_area_fraction": float(truth.mean()),
            "pixel_tp": tp, "pixel_fp": fp, "pixel_fn": fn,
            "boundary_tp_pred": btp_pred, "boundary_tp_gt": btp_gt,
            "boundary_pred_count": bpred, "boundary_gt_count": bgt,
            "latency_ms": latency_ms}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(CHECKPOINTS), default="full")
    args = parser.parse_args()
    torch.set_num_threads(4)
    protocol = json.loads((MANIFEST_ROOT / "protocol.json").read_text(encoding="utf-8"))
    sources, attacks = verify(protocol)
    output_dir = OUT_ROOT / args.variant
    if output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite E9 predictions: {output_dir}")
    model = load_model(args.variant, protocol)
    output_dir.mkdir(parents=True)
    (output_dir / "run.json").write_text(json.dumps({
        "variant": args.variant, "device": str(DEVICE),
        "protocol_sha256": sha(MANIFEST_ROOT / "protocol.json"),
        "checkpoint_sha256": protocol["checkpoint_sha256"][args.variant],
        "decoder_sha256": protocol["checkpoint_sha256"]["decoder"],
        "status": "RUNNING"}, indent=2), encoding="utf-8")
    rows_written = 0
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for source in sources:
            source_path = resolve_source(source)
            if sha(source_path) != source["file_sha256"]:
                raise RuntimeError(f"COCO source changed: {source['path']}")
            source_id = source["source_id"]
            original = load_rgb_256(source_path)
            payload = make_payload_deterministic(stable_seed(20261011, source_id))
            protected = embed_payload(original.unsqueeze(0), payload.unsqueeze(0)).squeeze(0)
            writer.writerow(infer_one(model, protected, payload,
                                     np.zeros((256, 256), dtype=bool),
                                     source_id, "authentic", "authentic"))
            rows_written += 1
            for plan in attacks[source_id]:
                received, truth = apply_attack(protected, plan)
                if not (float(plan["area_fraction"]) == float(truth.mean())):
                    raise RuntimeError("Frozen attack area disagrees with mask")
                writer.writerow(infer_one(model, received, payload, truth,
                                         source_id, plan["band"], plan["attack"]))
                rows_written += 1
            stream.flush()
            if int(source_id.split("_")[1]) % 10 == 0:
                print(f"{args.variant}: {source_id}, {rows_written}/2600 samples", flush=True)
    if rows_written != 2600:
        raise RuntimeError(f"Expected 2,600 predictions, got {rows_written}")
    run_path = output_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run.update(status="COMPLETE", prediction_rows=rows_written,
               predictions_sha256=sha(output_dir / "predictions.csv"))
    run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps({"variant": args.variant, "rows": rows_written,
                      "predictions": str(output_dir / "predictions.csv")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
