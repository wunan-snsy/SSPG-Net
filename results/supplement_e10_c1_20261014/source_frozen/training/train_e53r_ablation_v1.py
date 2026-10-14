"""E5.3R-4D formal trained ablations under frozen E5.3R-4B-2 V1."""
from pathlib import Path
import argparse
import gc
import json
import time
import traceback

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

# Reuse the already verified V1 data construction, loss, validation metrics,
# Youden-J threshold logic, early-stopping semantics and atomic checkpoints.
import training.train_e53r_joint_v1 as joint
from preprocessing.active_sspg_dataset import ActiveSSPGDataset
from models.sspg_ablation_model_v2 import SSPGAblationModelV2, VALID_ABLATIONS
from training.e53r_training_protocol_v1 import PROTOCOL, protocol_json

EXPECTED_COUNTS = {"no_provenance": 1341801, "no_graph": 555812, "visual_only": 481123}
EXPERIMENTS = {"no_provenance": "E5.3R-4D-B", "no_graph": "E5.3R-4D-C", "visual_only": "E5.3R-4D-D"}


def paths(variant):
    stem = f"e53r_4d_{variant}"
    return {
        "checkpoint": ROOT / "checkpoints" / f"{stem}_best.pth",
        "history": ROOT / "results" / "csv" / f"{stem}_history.csv",
        "predictions": ROOT / "results" / "csv" / f"{stem}_val_predictions.csv",
        "summary": ROOT / "results" / "csv" / f"{stem}_summary.csv",
        "protocol": ROOT / "results" / "csv" / f"{stem}_protocol.json",
        "failure": ROOT / "results" / "csv" / f"{stem}_failure.json",
    }


def configure_joint_io(output_paths, variant):
    # Existing helpers intentionally use module-level paths; redirect only their
    # outputs for this formal variant, never the frozen FULL files.
    joint.BEST_CHECKPOINT = output_paths["checkpoint"]
    joint.HISTORY_PATH = output_paths["history"]
    joint.VAL_PREDICTIONS_PATH = output_paths["predictions"]
    joint.SUMMARY_PATH = output_paths["summary"]


def write_failure(output_paths, variant, stage, epoch, exc, optimizer_initialized, training_started):
    payload = {"variant": variant, "stage": stage, "epoch": epoch,
               "exception": str(exc), "traceback": traceback.format_exc(),
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "test_accessed": False, "optimizer_initialized": optimizer_initialized,
               "training_started": training_started}
    output_paths["failure"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["failure"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


def preflight(variant):
    if joint.DEVICE.type != "cuda":
        raise RuntimeError("CUDA is required by the frozen formal protocol.")
    if not joint.STEGO_CHECKPOINT.is_file() or not joint.MANIFEST_PATH.is_file():
        raise RuntimeError("Missing frozen V3.3 checkpoint or clean manifest.")
    if not (ROOT / "data" / "imd2020").is_dir():
        raise RuntimeError("Missing IMD2020 image root.")
    if variant not in EXPECTED_COUNTS or variant not in VALID_ABLATIONS:
        raise RuntimeError(f"Unsupported formal variant: {variant}")
    sha = joint.verify_protocol_fingerprint()
    joint.set_global_seed(PROTOCOL.training_seed)
    manifest = pd.read_csv(joint.MANIFEST_PATH)
    # Only train/val source records are ever built; Test is deliberately untouched.
    train_records = joint.build_source_records(manifest, "train")
    val_records = joint.build_source_records(manifest, "val")
    if len(train_records) != 289 or len(val_records) != 62:
        raise RuntimeError("Frozen train/validation source count mismatch.")
    joint.audit_train_val_sources(train_records, val_records)
    train_dataset = ActiveSSPGDataset(records=train_records, base_seed=PROTOCOL.train_seed)
    val_dataset = ActiveSSPGDataset(records=val_records, base_seed=PROTOCOL.val_seed)
    if len(train_dataset) != 578 or len(val_dataset) != 124:
        raise RuntimeError("Frozen active sample count mismatch.")
    return sha, train_records, val_records, train_dataset, val_dataset


def model_preflight(variant):
    model = SSPGAblationModelV2(stego_checkpoint=str(joint.STEGO_CHECKPOINT), ablation=variant,
        adapter_bottleneck=PROTOCOL.adapter_bottleneck, fusion_hidden_dim=PROTOCOL.fusion_hidden_dim,
        knn_k=PROTOCOL.knn_k, gnn_hidden_dim=PROTOCOL.gnn_hidden_dim, gnn_depth=PROTOCOL.gnn_depth).to(joint.DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    decoder_trainable = sum(p.numel() for p in model.frozen_decoder().parameters() if p.requires_grad)
    vit_trainable = sum(p.numel() for p in model.frozen_vit().parameters() if p.requires_grad)
    if trainable != EXPECTED_COUNTS[variant] or decoder_trainable != 0 or vit_trainable != 0:
        raise RuntimeError(f"Architecture preflight failed: trainable={trainable}, decoder={decoder_trainable}, vit={vit_trainable}")
    return model, trainable


def train_variant(variant):
    out = paths(variant)
    existing = [p for p in (out["checkpoint"], out["history"], out["predictions"], out["summary"], out["protocol"]) if p.exists()]
    if existing:
        raise RuntimeError("Refusing to overwrite existing formal output(s): " + ", ".join(map(str, existing)))
    configure_joint_io(out, variant)
    stage, epoch, optimizer_initialized, training_started = "preflight", None, False, False
    model = optimizer = train_loader = val_loader = train_dataset = val_dataset = None
    try:
        sha, train_records, val_records, train_dataset, val_dataset = preflight(variant)
        out["protocol"].parent.mkdir(parents=True, exist_ok=True)
        snapshot = json.loads(protocol_json())
        snapshot.update({"formal_experiment": EXPERIMENTS[variant], "variant": variant,
                         "expected_trainable_parameters": EXPECTED_COUNTS[variant], "test_accessed": False})
        out["protocol"].write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        train_loader = joint.build_loader(train_dataset, True, PROTOCOL.training_seed)
        val_loader = joint.build_loader(val_dataset, False, PROTOCOL.training_seed)
        stage = "model_preflight"
        model, trainable = model_preflight(variant)
        optimizer = joint.build_optimizer(model); optimizer_initialized = True
        history, best_loss, best_epoch, stale = [], float("inf"), None, 0
        started = time.time(); training_started = True; stage = "training"
        for epoch in range(1, PROTOCOL.max_epochs + 1):
            print(f"{variant}: epoch {epoch:02d}/{PROTOCOL.max_epochs:02d}", flush=True)
            train_metrics = joint.train_one_epoch(model, train_loader, optimizer, epoch)
            val_metrics = joint.validate_one_epoch(model, val_loader)
            improved = val_metrics["total_loss"] < best_loss - PROTOCOL.min_delta
            if improved:
                best_loss, best_epoch, stale = float(val_metrics["total_loss"]), epoch, 0
                joint.save_checkpoint_atomic(model, optimizer, epoch, best_loss, sha)
            else: stale += 1
            history.append({"epoch": epoch, **{f"train_{k}": v for k,v in train_metrics.items()},
                            **{f"val_{k}": v for k,v in val_metrics.items()}, "checkpoint_saved": int(improved)})
            joint.save_history(history)
            print(f"  val_joint={val_metrics['total_loss']:.8f}; best={best_loss:.8f}; stale={stale}", flush=True)
            if stale >= PROTOCOL.early_stopping_patience: break
        if not out["checkpoint"].is_file(): raise RuntimeError("Best checkpoint was not created.")
        stage = "best_validation"
        checkpoint = torch.load(out["checkpoint"], map_location=joint.DEVICE, weights_only=False)
        if checkpoint.get("protocol_sha256") != sha or checkpoint.get("selection_metric") != "validation_joint_loss" or checkpoint.get("test_accessed") is not False:
            raise RuntimeError("Best checkpoint integrity failure.")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        summary = joint.evaluate_best_validation(model, val_loader)
        summary.update({"experiment_id": EXPERIMENTS[variant], "variant": variant, "protocol_sha256": sha,
                        "trainable_parameters": trainable, "best_epoch": best_epoch, "best_val_joint_loss": best_loss,
                        "epochs_completed": len(history), "training_seconds": time.time()-started,
                        "train_sources": len(train_records), "train_active_samples": len(train_dataset),
                        "val_sources": len(val_records), "val_active_samples": len(val_dataset), "test_accessed": False})
        joint.save_validation_predictions(summary["prediction_rows"]); joint.save_summary(summary)
        print(json.dumps({k:v for k,v in summary.items() if k != "prediction_rows"}, indent=2), flush=True)
    except Exception as exc:
        write_failure(out, variant, stage, epoch, exc, optimizer_initialized, training_started)
        raise
    finally:
        del model, optimizer, train_loader, val_loader, train_dataset, val_dataset
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def markdown_table(df):
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def write_report(df):
    report=ROOT/"results"/"e53r_4d_ablation_report.md"
    report.write_text("# E5.3R-4D Formal Ablation Study\n\n## Formal Protocol\nFrozen E5.3R-4B-2 V1; validation-only checkpoint selection and Youden-J thresholding; Test was not accessed.\n\n## Architecture Definitions\nFULL: visual + provenance + graph. NO_PROVENANCE: visual + graph. NO_GRAPH: visual + provenance. VISUAL_ONLY: visual only.\n\n## Parameter Counts\n"+markdown_table(df[["variant","trainable_parameters"]])+"\n\n## Training Completion and Validation Results\n"+markdown_table(df)+"\n\n## Comparison with FULL\nDeltas are ablation minus frozen FULL values.\n\n## Detection Ablation Interpretation\nDetection reflects both visual manipulation cues and any incremental provenance contribution; causal audits previously identified a visual shortcut, so these validation results do not establish provenance as the sole cause.\n\n## Localization Ablation Interpretation\nThe relative localization deltas above describe the observed dependence under this frozen protocol; no stronger claim is made beyond these results.\n\n## Limitations\nSingle preregistered training trajectory per variant; validation is not an independent final test.\n\n## Test Status\nTest accessed: False\n", encoding="utf-8")


def aggregate():
    full = {"experiment":"E5.3R-4C-1", "variant":"full", "uses_provenance":True, "uses_graph":True, "trainable_parameters":1416490,
            "best_epoch":8, "best_val_joint_loss":0.467051, "detection_auc":0.996358, "detection_balanced_accuracy":0.983871,
            "detection_f1":0.983607, "detection_sensitivity":0.967742, "detection_specificity":1.0, "detection_tn":62, "detection_fp":0, "detection_fn":2, "detection_tp":60,
            "localization_pixel_f1":0.775291, "localization_pixel_iou":0.633042, "test_accessed":False}
    rows=[full]
    for v in EXPECTED_COUNTS:
        row=pd.read_csv(paths(v)["summary"]).iloc[0].to_dict(); row.update({"experiment":EXPERIMENTS[v], "uses_provenance":v != "no_provenance" and v != "visual_only", "uses_graph":v == "no_provenance"})
        rows.append(row)
    for row in rows:
        row["delta_auc_vs_full"] = row["detection_auc"]-full["detection_auc"]
        row["delta_balanced_accuracy_vs_full"] = row["detection_balanced_accuracy"]-full["detection_balanced_accuracy"]
        row["delta_detection_f1_vs_full"] = row["detection_f1"]-full["detection_f1"]
        row["delta_localization_f1_vs_full"] = row["localization_pixel_f1"]-full["localization_pixel_f1"]
        row["delta_localization_iou_vs_full"] = row["localization_pixel_iou"]-full["localization_pixel_iou"]
    df=pd.DataFrame(rows); target=ROOT/"results"/"csv"; df.to_csv(target/"e53r_4d_ablation_summary.csv", index=False)
    (target/"e53r_4d_ablation_summary.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    write_report(df)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--variant", choices=tuple(EXPECTED_COUNTS), required=True); parser.add_argument("--aggregate", action="store_true")
    args=parser.parse_args(); train_variant(args.variant)
    if args.aggregate: aggregate()
