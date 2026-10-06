"""Freeze deterministic E1 source, attack, and mask identities for E8."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval_test_final import CHECKPOINT, MANIFEST, ROOT, STEGO, TEST_SEED
from preprocessing.active_sspg_dataset import (
    ATTACK_GEOMETRY_SEED_OFFSET,
    ActiveSSPGDataset,
    canonical_sample_key,
    sample_box,
    sample_copy_move_geometry,
    stable_seed,
)
from training.train_e53r_joint_v1 import authentic_label, resolve_image_path


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def records_for(split):
    frame = pd.read_csv(MANIFEST)
    frame = frame[frame["split"].astype(str).str.strip().str.lower().eq(split)]
    frame = frame[frame["label"].apply(authentic_label)]
    records = []
    for _, row in frame.iterrows():
        path = resolve_image_path(row["image_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({"sample_id": str(row["sample_id"]), "group": str(row["group"]),
                        "split": split, "label": row["label"], "image_path": str(path)})
    return sorted(records, key=lambda r: (r["group"], r["sample_id"], r["image_path"]))


def main():
    val_summary = ROOT / "results/release_v1.0.0/joint/e53r_joint_v1_summary.csv"
    result = {"protocol": "SSPG-Net E1 frozen active IMD2020", "manifest_path": str(MANIFEST),
              "manifest_sha256": sha(MANIFEST), "sspgn_checkpoint_path": str(CHECKPOINT),
              "sspgn_checkpoint_sha256": sha(CHECKPOINT), "stego_checkpoint_path": str(STEGO),
              "stego_checkpoint_sha256": sha(STEGO), "validation_summary_sha256": sha(val_summary),
              "detection_threshold": float(pd.read_csv(val_summary).iloc[0]["detection_threshold"]),
              "localization_threshold": 0.5, "splits": {}}
    for split, seed, expected in (("val", 20260923, 62), ("test", TEST_SEED, 63)):
        records = records_for(split)
        if len(records) != expected:
            raise RuntimeError(f"{split}: expected {expected} sources, got {len(records)}")
        dataset = ActiveSSPGDataset(records, seed)
        rows = []
        for i, record in enumerate(records):
            key = canonical_sample_key(record)
            source_seed = stable_seed(seed, key)
            attack = dataset.attack_assignment[i]
            rng = np.random.default_rng(source_seed + ATTACK_GEOMETRY_SEED_OFFSET)
            if attack == "copy_move":
                geometry = sample_copy_move_geometry(rng)
                top, left, height, width = (geometry[k] for k in
                    ("destination_top", "destination_left", "box_h", "box_w"))
            else:
                top, left, height, width = sample_box(rng)
                geometry = {"top": top, "left": left, "box_h": height, "box_w": width}
                if attack == "roll":
                    geometry.update(shift_y=max(1, height // 4), shift_x=max(1, width // 4))
            sample = dataset[2 * i + 1]
            mask = sample["mask"].numpy().astype(np.uint8)
            expected_mask = np.zeros((1, 256, 256), dtype=np.uint8)
            expected_mask[:, top:top + height, left:left + width] = 1
            if not np.array_equal(mask, expected_mask) or sample["attack"] != attack:
                raise RuntimeError(f"Attack metadata/mask mismatch: {key}")
            rows.append({"sample_key": key, "source_sha256": sha(record["image_path"]),
                         "source_seed": source_seed, "attack": attack, "geometry": geometry,
                         "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest()})
        result["splits"][split] = {"seed": seed, "sources": len(records),
            "active_samples": len(dataset), "source_attack_mask_records": rows,
            "split_manifest_sha256": canonical_hash(records),
            "attack_metadata_sha256": canonical_hash([{"sample_key": r["sample_key"],
                "attack": r["attack"], "geometry": r["geometry"]} for r in rows]),
            "masks_sha256": canonical_hash([r["mask_sha256"] for r in rows])}
    out = ROOT / "results/external_baselines/e8/protocol_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
