"""E4 frozen-checkpoint CASIA2 cross-dataset evaluation."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.eval_test_final as ev
from models.sspg_joint_model_v2 import SSPGJointModelV2
from preprocessing.active_sspg_dataset import (load_rgb_256, make_payload_deterministic,
    stable_seed, apply_controlled_attack, ATTACK_GEOMETRY_SEED_OFFSET)
from training.e53r_training_protocol_v1 import PROTOCOL
from training.train_stego_structured_v33 import embed_payload
from training.train_e53r_joint_v1 import build_loader

OUT = ROOT / "results" / "casia_cross_dataset"
SEED = 20261004
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CasiaCrossDataset(Dataset):
    def __init__(self):
        df = pd.read_csv(OUT / "dataset_manifest.csv")
        self.sources = df[df.split.eq("test")].to_dict("records")
        self.items = [(s, a) for s in self.sources for a in ("authentic", "erase", "copy_move", "roll")]
    def __len__(self): return len(self.items)
    def __getitem__(self, index):
        source, attack = self.items[index]
        key = str(source["casia_id"])
        seed = stable_seed(SEED, key)
        original = load_rgb_256(source["source_path"])
        payload = make_payload_deterministic(seed)
        protected = embed_payload(original.unsqueeze(0), payload.unsqueeze(0)).squeeze(0)
        if attack == "authentic":
            received, mask, label = protected, torch.zeros(1,256,256), torch.tensor(0.0)
        else:
            offset = {"erase": 0, "copy_move": 1, "roll": 2}[attack] * 1000003
            rng = np.random.default_rng(seed + ATTACK_GEOMETRY_SEED_OFFSET + offset)
            received, mask = apply_controlled_attack(protected, attack, rng)
            label = torch.tensor(1.0)
        return {"received": received, "payload": payload, "mask": mask,
                "detection_label": label, "attack": attack,
                "source_path": source["source_path"], "sample_key": f"{key}_{attack}"}

def main():
    ev.OUT = OUT
    threshold = float(pd.read_csv(ev.VAL_SUMMARY).iloc[0]["detection_threshold"])
    dataset = CasiaCrossDataset()
    loader = build_loader(dataset, False, SEED)
    model = SSPGJointModelV2(stego_checkpoint=str(ev.STEGO), adapter_bottleneck=PROTOCOL.adapter_bottleneck,
        fusion_hidden_dim=PROTOCOL.fusion_hidden_dim, knn_k=PROTOCOL.knn_k,
        gnn_hidden_dim=PROTOCOL.gnn_hidden_dim, gnn_depth=PROTOCOL.gnn_depth).to(DEVICE)
    ckpt = torch.load(ev.CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    det, loc, rows = ev.evaluate(model, loader, threshold, float(PROTOCOL.localization_threshold))
    payload = {"detection": det, "localization": loc, "sources": 200,
               "authentic_samples": 200, "manipulated_samples": 600,
               "checkpoint": ev.CHECKPOINT.name, "retrained": False, "seed": SEED}
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "cross_dataset_predictions.csv", index=False)
    pd.DataFrame([{"metric": k, "value": v} for k,v in {**det, **{f'loc_{k}':v for k,v in loc.items()}}.items()]).to_csv(OUT / "Table_cross_dataset.csv", index=False)
    print(json.dumps(payload, indent=2))

if __name__ == "__main__": main()
