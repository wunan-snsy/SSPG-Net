"""Verify and summarize the three paired Clean-142 graph-control seeds."""

from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

OUT = ROOT / "results/clean_revision_v2/budget142_param_matched_no_graph_e2"
FULL = ROOT / "results/clean_revision_v2/budget142_e2/full"
OLD = ROOT / "results/clean_revision_v2/budget142_e2/no_graph"
SEEDS = (20261001, 20261002, 20261003)
METRICS = (
    "detection_auc",
    "detection_f1",
    "localization_pixel_f1",
    "localization_pixel_iou",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_sd(values):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1))


def main():
    status = json.loads((OUT / "status.json").read_text(encoding="utf-8"))
    if status["state"] != "COMPLETE" or tuple(status["completed"]) != SEEDS:
        raise RuntimeError("Three seed runs are not complete")
    rows = []
    for seed in SEEDS:
        run = OUT / str(seed)
        lineage = json.loads((run / "lineage.json").read_text(encoding="utf-8"))
        if sha(run / "best.pth") != lineage["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint changed: {seed}")
        if sha(run / "metrics.csv") != lineage["summary_sha256"]:
            raise RuntimeError(f"Metrics changed: {seed}")
        ckpt = torch.load(run / "best.pth", map_location="cpu", weights_only=False)
        matched = pd.read_csv(run / "metrics.csv").iloc[0]
        full = pd.read_csv(FULL / str(seed) / "metrics.csv").iloc[0]
        old = pd.read_csv(OLD / str(seed) / "metrics.csv").iloc[0]
        if (
            ckpt["selection_metric"] != "validation_joint_loss"
            or ckpt["test_accessed"] is not False
            or bool(matched["test_accessed"])
            or int(matched["trainable_parameters"]) != 1_416_110
        ):
            raise RuntimeError(f"Protocol or parameter mismatch: {seed}")
        row = {
            "seed": seed,
            "full_parameters": 1_416_490,
            "matched_parameters": int(matched["trainable_parameters"]),
            "old_no_graph_parameters": int(old["trainable_parameters"]),
            "matched_best_epoch": int(matched["best_epoch"]),
            "matched_epochs_completed": int(matched["epochs_completed"]),
            "matched_training_seconds": float(matched["training_seconds"]),
        }
        for metric in METRICS:
            row[f"full_{metric}"] = float(full[metric])
            row[f"matched_{metric}"] = float(matched[metric])
            row[f"old_no_graph_{metric}"] = float(old[metric])
            row[f"full_minus_matched_{metric}"] = float(full[metric] - matched[metric])
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "paired_seed_results.csv", index=False)
    summary = []
    for metric in METRICS:
        for model in ("full", "matched", "old_no_graph", "full_minus_matched"):
            mean, sd = mean_sd(frame[f"{model}_{metric}"])
            summary.append({"metric": metric, "model_or_difference": model,
                            "mean": mean, "sample_sd": sd, "n_seeds": len(SEEDS)})
    pd.DataFrame(summary).to_csv(OUT / "mean_sd_summary.csv", index=False)

    lines = [
        "# Three-seed, parameter-matched NO_GRAPH comparison",
        "",
        "The three matched-control runs pair by seed with previously completed FULL E2 runs. The preliminary single-seed run is excluded. All checkpoints were chosen by Validation joint loss. Test and E9 were not used for training or selection.",
        "",
        "| Seed | FULL Pixel F1 | Matched NO_GRAPH Pixel F1 | FULL - matched | FULL AUC | Matched AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['full_localization_pixel_f1']:.4f} | "
            f"{row['matched_localization_pixel_f1']:.4f} | "
            f"{row['full_minus_matched_localization_pixel_f1']:+.4f} | "
            f"{row['full_detection_auc']:.4f} | {row['matched_detection_auc']:.4f} |"
        )
    full_mean, full_sd = mean_sd(frame["full_localization_pixel_f1"])
    matched_mean, matched_sd = mean_sd(frame["matched_localization_pixel_f1"])
    diff_mean, diff_sd = mean_sd(frame["full_minus_matched_localization_pixel_f1"])
    old_mean, old_sd = mean_sd(frame["old_no_graph_localization_pixel_f1"])
    lines += [
        "",
        f"FULL Pixel F1: {full_mean:.4f} ± {full_sd:.4f}; matched NO_GRAPH: {matched_mean:.4f} ± {matched_sd:.4f}; paired FULL - matched: {diff_mean:+.4f} ± {diff_sd:.4f} (sample SD across three seeds).",
        f"The smaller existing NO_GRAPH runs scored {old_mean:.4f} ± {old_sd:.4f} Pixel F1, for context.",
        "",
        "The matched control has 1,416,110 trainable parameters versus FULL's 1,416,490 (-0.027%). It applies two per-token residual MLP blocks with no token-to-token exchange. Its results test whether the FULL advantage survives this specific capacity match. They do not establish that graph topology, rather than spatial communication more generally, is uniquely responsible. Three seeds describe training-seed variability on one fixed Validation split; they do not measure cross-dataset generalization. The already inspected E9 set remains exploratory for this newly designed control.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
