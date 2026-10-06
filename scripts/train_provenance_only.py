"""E3 provenance-only baseline under the frozen train/validation protocol."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import training.train_e53r_ablation_v1 as abl
import training.train_e53r_joint_v1 as joint
from models.provenance_only_model import ProvenanceOnlyModel
from training.e53r_training_protocol_v1 import PROTOCOL

OUT = ROOT / "results" / "provenance_only"
PATHS = {"checkpoint": OUT / "checkpoint" / "best.pth",
         "history": OUT / "training_log.csv", "predictions": OUT / "prediction_maps.csv",
         "summary": OUT / "metrics.csv", "protocol": OUT / "protocol.json",
         "failure": OUT / "failure.json"}


def model_preflight(_variant):
    model = ProvenanceOnlyModel(str(joint.STEGO_CHECKPOINT)).to(joint.DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    decoder_trainable = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    if trainable != 158402 or decoder_trainable != 0:
        raise RuntimeError(f"E3 preflight failed: trainable={trainable}, decoder={decoder_trainable}")
    return model, trainable


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    abl.VALID_ABLATIONS = tuple(abl.VALID_ABLATIONS) + ("provenance_only",)
    abl.EXPECTED_COUNTS["provenance_only"] = 158402
    abl.EXPERIMENTS["provenance_only"] = "E3-PROVENANCE-ONLY"
    abl.paths = lambda _variant: PATHS
    abl.model_preflight = model_preflight
    abl.train_variant("provenance_only")


if __name__ == "__main__": main()
