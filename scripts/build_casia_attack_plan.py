"""Freeze E4 attack assignments: each of 200 held-out sources gets all 3 attacks."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "casia_cross_dataset"
manifest = pd.read_csv(OUT / "dataset_manifest.csv")
test = manifest[manifest["split"].eq("test")].copy()
if len(test) != 200:
    raise RuntimeError(f"Expected 200 held-out CASIA sources, got {len(test)}")
rows = []
for _, source in test.iterrows():
    for attack in ("erase", "copy_move", "roll"):
        rows.append({"sample_id": f"{source.casia_id}_{attack}", "casia_id": source.casia_id,
                     "source_path": source.source_path, "attack": attack,
                     "mask_required": True, "training_used": False})
plan = pd.DataFrame(rows)
plan.to_csv(OUT / "attack_statistics.csv", index=False)
print(plan.groupby("attack").size().to_dict(), "total", len(plan))
