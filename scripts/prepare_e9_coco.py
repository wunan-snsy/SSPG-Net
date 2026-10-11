"""Audit COCO val2017 sources and freeze an outcome-blind E9 source manifest.

This script reads image content only. It does not load models or predictions.
Existing frozen manifests and experiment outputs are never modified.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn


ROOT = Path(__file__).resolve().parents[1]
COCO = ROOT / "data/coco2017/val2017"
CASIA_MANIFEST = ROOT / "results/casia_cross_dataset/dataset_manifest.csv"
IMD_LINEAGE = ROOT / "reports/clean_revision_v2/audit/image_lineage.csv"
OUT = ROOT / "manifests/e9_coco2017"
SELECTION_SALT = "E9-COCO-VAL2017-200-SOURCES-V1"
N_SOURCES = 200
MAX_NEAR_DISTANCE = 6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_record(path: Path, dataset: str) -> dict:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        gray = rgb.convert("L")
        small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
        coeff = dctn(small, type=2, norm="ortho")[:8, :8].reshape(-1)
        ph = coeff > np.median(coeff[1:])
        ph[0] = False
        dh = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
    return {
        "dataset": dataset,
        "path": str(path.resolve()),
        "name": path.name,
        "width": width,
        "height": height,
        "file_sha256": sha256_file(path),
        "phash": np.packbits(ph).tobytes().hex(),
        "dhash": np.packbits(dh[:, 1:] > dh[:, :-1]).tobytes().hex(),
    }


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def min_distance(a: dict, b: dict) -> tuple[int, int]:
    return (
        (int(a["phash"], 16) ^ int(b["phash"], 16)).bit_count(),
        (int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count(),
    )


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Refusing to replace frozen E9 manifest: {OUT}")
    paths = sorted(COCO.glob("*.jpg"))
    if len(paths) != 5000:
        raise RuntimeError(f"Expected 5,000 COCO val2017 JPGs, found {len(paths)}")
    if not CASIA_MANIFEST.is_file() or not IMD_LINEAGE.is_file():
        raise RuntimeError("Missing existing CASIA2 or IMD2020 source audit")

    imd = [
        {"dataset": "IMD2020", "path": row["image_path"],
         "file_sha256": row["file_sha256"], "phash": row["phash"],
         "dhash": row["dhash"]}
        for row in read_csv(IMD_LINEAGE)
    ]
    casia = [image_record(Path(row["source_path"]), "CASIA2")
             for row in read_csv(CASIA_MANIFEST)]
    prior = imd + casia
    prior_exact = {row["file_sha256"] for row in prior}
    coco = []
    invalid = []
    for index, path in enumerate(paths, 1):
        try:
            coco.append(image_record(path, "COCO-val2017"))
        except Exception as exc:
            invalid.append({"name": path.name, "error": str(exc)})
        if index % 500 == 0:
            print(f"Audited COCO {index}/{len(paths)}", flush=True)

    order = sorted(coco, key=lambda row: (
        hashlib.sha256(f"{SELECTION_SALT}|{row['name']}".encode()).hexdigest(),
        row["name"],
    ))
    selected = []
    rejected = []
    for candidate in order:
        exact = candidate["file_sha256"] in prior_exact
        near_matches = []
        if not exact:
            for other in prior + selected:
                ph, dh = min_distance(candidate, other)
                if ph <= MAX_NEAR_DISTANCE or dh <= MAX_NEAR_DISTANCE:
                    near_matches.append({"dataset": other["dataset"],
                                         "path": other["path"],
                                         "phash_distance": ph, "dhash_distance": dh})
        if exact or near_matches:
            rejected.append({"name": candidate["name"], "exact": exact,
                             "near_matches": near_matches})
        else:
            selected.append(candidate)
            if len(selected) == N_SOURCES:
                break
    if len(selected) != N_SOURCES or invalid:
        raise RuntimeError(f"Audit incomplete: selected={len(selected)}, invalid={len(invalid)}")

    OUT.mkdir(parents=True)
    fields = ["source_id", "dataset", "path", "name", "width", "height",
              "file_sha256", "phash", "dhash"]
    manifest = [dict(source_id=f"COCO_{i:04d}", **row)
                for i, row in enumerate(selected, 1)]
    write_csv(OUT / "source_manifest.csv", manifest, fields)
    manifest_sha = sha256_file(OUT / "source_manifest.csv")
    audit = {
        "experiment": "E9 Independent Source-Image and Small-Region Evaluation",
        "source_dataset": "COCO val2017",
        "available_jpgs": len(paths),
        "validated_jpgs": len(coco),
        "invalid_jpgs": invalid,
        "selected_sources": len(manifest),
        "selection_salt": SELECTION_SALT,
        "selection_rule": "Sort SHA256(salt|filename); take first 200 with no exact file or conservative pHash/dHash <=6 match against prior IMD2020/CASIA2 images and already selected COCO images.",
        "imd_compared": len(imd),
        "casia_compared": len(casia),
        "near_hash_distance_cutoff": MAX_NEAR_DISTANCE,
        "rejected_before_selection_complete": rejected,
        "source_manifest_sha256": manifest_sha,
        "audit_limit": "Hash checks screen exact and visual near duplicates; they do not prove absence of crops or semantic duplicates.",
        "model_outputs_consulted": False,
    }
    (OUT / "source_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"selected": len(manifest), "rejected": len(rejected),
                      "manifest_sha256": manifest_sha}, indent=2), flush=True)


if __name__ == "__main__":
    main()
