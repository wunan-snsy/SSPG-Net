"""Validate CASIA2 authentic images and freeze the E4 1000-image manifest."""
from pathlib import Path
import hashlib
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "CASIA2" / "authenti" / "Au"
OUT = ROOT / "results" / "casia_cross_dataset"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

files = sorted(p.resolve() for p in SOURCE.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
if len(files) < 1000:
    raise RuntimeError(f"Need at least 1000 authentic images, found {len(files)}")

rows, invalid = [], []
for path in files:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            width, height = im.size
            mode = im.mode
    except Exception as exc:
        invalid.append({"path": str(path), "error": str(exc)})
        continue
    digest = hashlib.sha256(str(path.relative_to(SOURCE)).encode("utf-8")).hexdigest()
    rows.append({"source_path": str(path), "relative_path": str(path.relative_to(SOURCE)),
                 "width": width, "height": height, "mode": mode, "selection_hash": digest})

valid = pd.DataFrame(rows).sort_values(["selection_hash", "relative_path"]).head(1000).reset_index(drop=True)
valid.insert(0, "casia_id", [f"CASIA2_{i:04d}" for i in range(1, 1001)])
valid["split"] = ["train"] * 800 + ["test"] * 200
OUT.mkdir(parents=True, exist_ok=True)
valid.to_csv(OUT / "dataset_manifest.csv", index=False)
pd.DataFrame(invalid, columns=["path", "error"]).to_csv(OUT / "invalid_images.csv", index=False)
print({"discovered": len(files), "valid": len(rows), "invalid": len(invalid),
       "selected": len(valid), "train": 800, "test": 200,
       "manifest": str(OUT / 'dataset_manifest.csv')})
