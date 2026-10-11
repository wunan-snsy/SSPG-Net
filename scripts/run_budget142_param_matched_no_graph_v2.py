"""One controlled, Validation-only capacity-matched NO_GRAPH run."""

from pathlib import Path
import hashlib
import json
import os
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from models.sspg_param_matched_no_graph_v2 import SSPGParamMatchedNoGraphV2
from scripts.run_clean_stage_a_v2 import gate, sha, write_json

OUT = ROOT / "results/clean_revision_v2/budget142_param_matched_no_graph"
DEC = ROOT / "checkpoints/clean_revision_v2/decoder_budget142/best.pth"
FULL = ROOT / "checkpoints/clean_revision_v2/budget142_full_best.pth"
MODEL = ROOT / "models/sspg_param_matched_no_graph_v2.py"
EXPECTED_PARAMETERS = 1_416_110
FULL_PARAMETERS = 1_416_490
VARIANT = "matched_no_graph"


def paths(_variant):
    if _variant != VARIANT:
        raise ValueError(_variant)
    return {
        "checkpoint": OUT / "best.pth",
        "history": OUT / "training_log.csv",
        "predictions": OUT / "val_predictions.csv",
        "summary": OUT / "metrics.csv",
        "protocol": OUT / "training_protocol.json",
        "failure": OUT / "failure.json",
    }


def main():
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite {OUT}")
    gate()
    torch.set_num_threads(4)
    import training.train_e53r_ablation_v1 as abl

    decoder_state = json.loads(
        (ROOT / "results/clean_revision_v2/decoder_budget142/status.json").read_text(encoding="utf-8")
    )
    if decoder_state["state"] != "COMPLETE" or sha(DEC) != decoder_state["best_checkpoint_sha256"]:
        raise RuntimeError("Frozen decoder changed")
    if not FULL.is_file():
        raise RuntimeError("Missing frozen FULL checkpoint")
    if abs(EXPECTED_PARAMETERS - FULL_PARAMETERS) / FULL_PARAMETERS >= 0.01:
        raise RuntimeError("Parameter match outside 1% tolerance")

    OUT.mkdir(parents=True)
    design = {
        "experiment_id": "CLEAN-BUDGET142-PARAM-MATCHED-NO-GRAPH-V2",
        "comparison": "Frozen Clean-142 FULL vs one newly trained matched NO_GRAPH",
        "architecture": "Existing NO_GRAPH with two per-token residual 256-837-256 MLP blocks; no graph, edges, attention, or cross-token exchange",
        "full_trainable_parameters": FULL_PARAMETERS,
        "control_trainable_parameters": EXPECTED_PARAMETERS,
        "parameter_difference_percent": 100 * (EXPECTED_PARAMETERS - FULL_PARAMETERS) / FULL_PARAMETERS,
        "training": "Same IMD2020 Train/Validation records, decoder, loss, optimizer, seed, 20-epoch maximum, patience=5; select checkpoint by Validation joint loss",
        "evaluation": "Validation only for primary comparison; E9 has already been inspected, so any subsequent E9 comparison is exploratory",
        "test_accessed": False,
        "decoder_sha256": sha(DEC),
        "full_checkpoint_sha256": sha(FULL),
        "model_code_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(OUT / "design.json", design)
    write_json(OUT / "status.json", {"state": "RUNNING", "pid": os.getpid(), "started_at": design["created_at"]})

    abl.joint.STEGO_CHECKPOINT = DEC
    abl.SSPGAblationModelV2 = SSPGParamMatchedNoGraphV2
    abl.VALID_ABLATIONS = (*abl.VALID_ABLATIONS, VARIANT)
    abl.EXPECTED_COUNTS[VARIANT] = EXPECTED_PARAMETERS
    abl.EXPERIMENTS[VARIANT] = design["experiment_id"]
    abl.paths = paths
    try:
        abl.train_variant(VARIANT)
        lineage = {
            "experiment_id": design["experiment_id"],
            "decoder_sha256": sha(DEC),
            "checkpoint_sha256": sha(paths(VARIANT)["checkpoint"]),
            "summary_sha256": sha(paths(VARIANT)["summary"]),
            "test_accessed": False,
        }
        write_json(OUT / "lineage.json", lineage)
        write_json(OUT / "status.json", {"state": "COMPLETE", "pid": os.getpid(), "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    except Exception:
        write_json(OUT / "status.json", {"state": "FAILED", "pid": os.getpid(), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
