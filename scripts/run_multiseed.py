"""Run one E2 model/seed job without overwriting frozen formal outputs."""
import argparse, hashlib, json, sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import training.e53r_training_protocol_v1 as proto
import training.train_e53r_joint_v1 as joint


def output_paths(model, seed):
    d = ROOT / "results" / "multiseed" / model / str(seed)
    d.mkdir(parents=True, exist_ok=True)
    return {"checkpoint": d / "best.pth", "history": d / "training_log.csv",
            "predictions": d / "val_predictions.csv", "summary": d / "metrics.csv",
            "protocol": d / "protocol.json", "failure": d / "failure.json"}


def install_protocol(seed):
    p = replace(proto.PROTOCOL, training_seed=seed,
                experiment_id=f"E2-multiseed-{seed}", protocol_status="E2_FROZEN_REPLICATION")
    proto.PROTOCOL = p
    joint.PROTOCOL = p
    digest = hashlib.sha256(proto.protocol_json().encode("utf-8")).hexdigest()
    joint.EXPECTED_PROTOCOL_SHA256 = digest
    return p, digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("full", "no_graph", "no_provenance"), required=True)
    ap.add_argument("--seed", type=int, choices=(20261001, 20261002, 20261003), required=True)
    args = ap.parse_args()
    out = output_paths(args.model, args.seed)
    if out["summary"].exists():
        print("already complete", out["summary"]); return
    p, digest = install_protocol(args.seed)
    if args.model == "full":
        joint.BEST_CHECKPOINT, joint.HISTORY_PATH = out["checkpoint"], out["history"]
        joint.VAL_PREDICTIONS_PATH, joint.SUMMARY_PATH = out["predictions"], out["summary"]
        joint.PROTOCOL_SNAPSHOT_PATH = out["protocol"]
        joint.main()
    else:
        import training.train_e53r_ablation_v1 as abl
        abl.PROTOCOL = p
        abl.paths = lambda variant: out
        abl.EXPERIMENTS[args.model] = f"E2-{args.model}-{args.seed}"
        abl.train_variant(args.model)
    meta = {"model": args.model, "seed": args.seed, "protocol_sha256": digest,
            "test_accessed": False}
    (out["protocol"].parent / "job.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
