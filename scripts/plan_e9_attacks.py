"""Freeze E9 attack geometries before reading any E9 model prediction."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.active_sspg_dataset import (
    enumerate_non_overlapping_sources,
    sample_box,
    stable_seed,
)


OUT = ROOT / "manifests/e9_coco2017"
SOURCE_MANIFEST = OUT / "source_manifest.csv"
ATTACK_PLAN = OUT / "attack_plan.csv"
PROTOCOL = OUT / "protocol.json"
BANDS = (("01_05", 0.01, 0.05), ("05_10", 0.05, 0.10),
         ("10_15", 0.10, 0.15), ("15_30", 0.15, 0.30))
ATTACKS = ("erase", "copy_move", "roll")
SEED = 20261011
IMAGE_SIZE = 256


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometry(rng: np.random.Generator, attack: str, low: float, high: float) -> dict:
    for _ in range(1000):
        top, left, height, width = sample_box(rng, min_area=low, max_area=high)
        fraction = height * width / (IMAGE_SIZE * IMAGE_SIZE)
        # The original sampler rounds dimensions; enforce realized area bins.
        if not low <= fraction < high:
            continue
        result = {"top": top, "left": left, "height": height,
                  "width": width, "area_fraction": fraction,
                  "source_top": "", "source_left": ""}
        if attack == "copy_move":
            legal = enumerate_non_overlapping_sources(top, left, height, width)
            if not legal:
                continue
            source_top, source_left = legal[int(rng.integers(0, len(legal)))]
            result.update(source_top=source_top, source_left=source_left)
        return result
    raise RuntimeError(f"Could not generate {attack} in area band {low}-{high}")


def main() -> None:
    if ATTACK_PLAN.exists() or PROTOCOL.exists():
        raise RuntimeError("Refusing to replace frozen E9 attack plan/protocol")
    with SOURCE_MANIFEST.open("r", encoding="utf-8", newline="") as stream:
        sources = list(csv.DictReader(stream))
    if len(sources) != 200:
        raise RuntimeError(f"Expected 200 frozen sources, got {len(sources)}")
    rows = []
    for source in sources:
        source_id = source["source_id"]
        for band, low, high in BANDS:
            for attack in ATTACKS:
                seed = stable_seed(SEED, f"{source_id}|{band}|{attack}")
                shape = geometry(np.random.default_rng(seed), attack, low, high)
                rows.append({"source_id": source_id, "band": band,
                             "attack": attack, "seed": seed, **shape})
    fields = ["source_id", "band", "attack", "seed", "top", "left",
              "height", "width", "area_fraction", "source_top", "source_left"]
    with ATTACK_PLAN.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if len(rows) != 2400:
        raise RuntimeError(f"Expected 2,400 attacks, got {len(rows)}")
    checkpoint_root = ROOT / "checkpoints/clean_revision_v2"
    checkpoints = {
        "decoder": checkpoint_root / "decoder_budget142/best.pth",
        "full": checkpoint_root / "budget142_full_best.pth",
        "no_provenance": checkpoint_root / "budget142_no_provenance_best.pth",
        "no_graph": checkpoint_root / "budget142_no_graph_best.pth",
    }
    threshold = 0.6624471545219421
    protocol = {
        "id": "E9-COCO-VAL2017-V1",
        "status": "FROZEN_BEFORE_E9_INFERENCE",
        "source_manifest_sha256": sha(SOURCE_MANIFEST),
        "attack_plan_sha256": sha(ATTACK_PLAN),
        "sources": len(sources),
        "protected_authentic_per_source": 1,
        "manipulated_per_source": 12,
        "total_samples": 2600,
        "bands": [{"id": band, "min_inclusive": low, "max_exclusive": high}
                  for band, low, high in BANDS],
        "attacks": ATTACKS,
        "image_size": IMAGE_SIZE,
        "payload_bits": 128,
        "embedding_alpha": 0.010,
        "knn_k": 8,
        "detection_threshold": threshold,
        "localization_threshold": 0.5,
        "payload_seed": SEED,
        "geometry_seed": SEED,
        "checkpoint_sha256": {name: sha(path) for name, path in checkpoints.items()},
        "primary_comparison": "15_30 band: 200 protected authentic and 600 manipulated, same 1:3 class ratio as CASIA2 E4",
        "small_region_analysis": "Each band: 200 sources x 3 attacks; report manipulated-only localization and detection recall, with shared authentic FPR separately.",
        "bootstrap": "2,000 source-level paired resamples; seed 20261012; fixed thresholds",
        "no_graph_interpretation": "Structure ablation only; parameters not matched.",
        "source_distribution_limit": "COCO images are authentic sources for controlled attacks, not native forgery annotations.",
    }
    PROTOCOL.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"attack_rows": len(rows),
                      "attack_plan_sha256": protocol["attack_plan_sha256"],
                      "protocol": str(PROTOCOL)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
