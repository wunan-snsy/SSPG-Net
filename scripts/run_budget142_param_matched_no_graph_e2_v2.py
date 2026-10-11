"""Three fixed-seed replications paired to the already completed E2 FULL runs."""

from dataclasses import replace
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_clean_stage_a_v2 import gate, sha, write_json
from scripts.run_budget142_param_matched_no_graph_v2 import (
    DEC,
    MODEL,
    EXPECTED_PARAMETERS,
    SSPGParamMatchedNoGraphV2,
)

OUT = ROOT / "results/clean_revision_v2/budget142_param_matched_no_graph_e2"
FULL_E2 = ROOT / "results/clean_revision_v2/budget142_e2/full"
SEEDS = (20261001, 20261002, 20261003)
VARIANT = "matched_no_graph"
MODEL_SHA256 = "b285db856fff3ae7b5ed9277a2d7d917705a42ca3d667ea94fd1b8b30afcfe48"


def paths(seed):
    d = OUT / str(seed)
    return {
        "checkpoint": d / "best.pth",
        "history": d / "training_log.csv",
        "predictions": d / "val_predictions.csv",
        "summary": d / "metrics.csv",
        "protocol": d / "training_protocol.json",
        "failure": d / "failure.json",
    }


def verify_inputs():
    gate()
    if hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA256:
        raise RuntimeError("Matched model changed since the pilot run")
    state = json.loads(
        (ROOT / "results/clean_revision_v2/decoder_budget142/status.json").read_text(encoding="utf-8")
    )
    if state["state"] != "COMPLETE" or sha(DEC) != state["best_checkpoint_sha256"]:
        raise RuntimeError("Decoder checkpoint changed")
    for seed in SEEDS:
        full = FULL_E2 / str(seed)
        lineage = json.loads((full / "lineage.json").read_text(encoding="utf-8"))
        if sha(full / "best.pth") != lineage["checkpoint_sha256"]:
            raise RuntimeError(f"FULL E2 seed {seed} checkpoint changed")


def run_seed(seed):
    verify_inputs()
    p = paths(seed)
    p["summary"].parent.mkdir(parents=True, exist_ok=False)
    import torch
    import training.e53r_training_protocol_v1 as proto
    import training.train_e53r_joint_v1 as joint
    import training.train_e53r_ablation_v1 as abl

    torch.set_num_threads(4)
    protocol = replace(
        proto.PROTOCOL,
        training_seed=seed,
        experiment_id=f"CLEAN-BUDGET142-E2-PARAM-MATCHED-NO-GRAPH-{seed}-V2",
        protocol_status="CLEAN_BUDGET142_E2_FIXED_REPLICATION",
    )
    proto.PROTOCOL = protocol
    joint.PROTOCOL = protocol
    abl.PROTOCOL = protocol
    digest = hashlib.sha256(proto.protocol_json().encode("utf-8")).hexdigest()
    joint.EXPECTED_PROTOCOL_SHA256 = digest
    joint.STEGO_CHECKPOINT = DEC
    abl.SSPGAblationModelV2 = SSPGParamMatchedNoGraphV2
    abl.VALID_ABLATIONS = (*abl.VALID_ABLATIONS, VARIANT)
    abl.EXPECTED_COUNTS[VARIANT] = EXPECTED_PARAMETERS
    abl.EXPERIMENTS[VARIANT] = protocol.experiment_id
    abl.paths = lambda variant: p if variant == VARIANT else None
    abl.train_variant(VARIANT)
    write_json(p["summary"].parent / "lineage.json", {
        "experiment_id": protocol.experiment_id,
        "training_seed": seed,
        "protocol_sha256": digest,
        "decoder_sha256": sha(DEC),
        "model_code_sha256": MODEL_SHA256,
        "checkpoint_sha256": sha(p["checkpoint"]),
        "summary_sha256": sha(p["summary"]),
        "test_accessed": False,
    })


def pipeline():
    verify_inputs()
    if OUT.exists():
        raise RuntimeError(f"Refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    write_json(OUT / "design.json", {
        "experiment_id": "CLEAN-BUDGET142-E2-PARAM-MATCHED-NO-GRAPH-V2",
        "model_code_sha256": MODEL_SHA256,
        "decoder_sha256": sha(DEC),
        "training_seeds": SEEDS,
        "comparison": "Each matched NO_GRAPH seed pairs with an already completed E2 FULL seed",
        "training": "Unchanged E2 Train/Validation data and protocol; three new matched-control runs only",
        "pilot_excluded": True,
        "test_accessed": False,
        "E9_accessed": False,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    completed = []
    for seed in SEEDS:
        write_json(OUT / "status.json", {
            "state": "RUNNING", "seed": seed, "completed": completed, "pid": os.getpid()
        })
        with (OUT / f"{seed}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, "-u", str(Path(__file__).resolve()), "--seed", str(seed)],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONHASHSEED="42"),
            )
        if result.returncode:
            write_json(OUT / "status.json", {
                "state": "FAILED", "seed": seed, "completed": completed,
                "returncode": result.returncode,
            })
            raise RuntimeError(f"Matched E2 failed at seed {seed}")
        completed.append(seed)
    write_json(OUT / "status.json", {
        "state": "COMPLETE", "completed": completed,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })


def resume_after_first_seed():
    verify_inputs()
    status = json.loads((OUT / "status.json").read_text(encoding="utf-8"))
    if status["state"] != "PAUSED_AFTER_CURRENT" or status["current_seed"] != SEEDS[0]:
        raise RuntimeError("Unexpected paused state")
    if any((OUT / str(seed)).exists() for seed in SEEDS[1:]):
        raise RuntimeError("A remaining seed already has outputs")
    write_json(OUT / "status.json", {
        "state": "WAITING_FOR_FIRST_SEED", "current_seed": SEEDS[0],
        "remaining": SEEDS[1:], "pid": os.getpid(),
    })
    first = OUT / str(SEEDS[0])
    deadline = time.monotonic() + 7200
    while not (first / "lineage.json").is_file():
        if (first / "failure.json").is_file():
            raise RuntimeError("First seed failed")
        if time.monotonic() > deadline:
            raise TimeoutError("First seed did not finish within two hours")
        time.sleep(15)
    lineage = json.loads((first / "lineage.json").read_text(encoding="utf-8"))
    if sha(first / "best.pth") != lineage["checkpoint_sha256"]:
        raise RuntimeError("First seed checkpoint checksum mismatch")
    completed = [SEEDS[0]]
    for seed in SEEDS[1:]:
        write_json(OUT / "status.json", {
            "state": "RUNNING", "seed": seed, "completed": completed, "pid": os.getpid(),
        })
        with (OUT / f"{seed}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, "-u", str(Path(__file__).resolve()), "--seed", str(seed)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONHASHSEED="42"),
            )
        if result.returncode:
            write_json(OUT / "status.json", {
                "state": "FAILED", "seed": seed, "completed": completed,
                "returncode": result.returncode,
            })
            raise RuntimeError(f"Matched E2 failed at seed {seed}")
        completed.append(seed)
    write_json(OUT / "status.json", {
        "state": "COMPLETE", "completed": completed,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.seed is not None and args.resume:
        parser.error("--seed and --resume cannot be combined")
    if args.seed is not None:
        run_seed(args.seed)
    elif args.resume:
        resume_after_first_seed()
    else:
        pipeline()
