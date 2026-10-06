from pathlib import Path
from PIL import Image
import csv
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "data" / "imd2020"
OUT = PROJECT_ROOT / "manifests" / "imd2020_manifest.csv"

SEED = 2026
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

OUT.parent.mkdir(parents=True, exist_ok=True)

groups = []
records = []
problems = []

# ---------------------------------------------------------
# 1. Find every directory containing an *_orig image
# ---------------------------------------------------------

for folder in sorted([p for p in ROOT.rglob("*") if p.is_dir()]):

    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    origs = [
        p for p in files
        if p.stem.lower().endswith("_orig")
    ]

    if not origs:
        continue

    if len(origs) != 1:
        problems.append(
            (str(folder.relative_to(ROOT)),
             f"orig_count={len(origs)}")
        )
        continue

    orig = origs[0]

    # Every *_mask image should have a corresponding manipulated image
    masks = [
        p for p in files
        if p.stem.lower().endswith("_mask")
    ]

    fake_pairs = []

    for mask in masks:

        # c8swtoq_0_mask.png -> c8swtoq_0
        fake_stem = mask.stem[:-5]

        fake_candidates = [
            p for p in files
            if p.stem == fake_stem
            and not p.stem.lower().endswith("_mask")
        ]

        if len(fake_candidates) != 1:
            problems.append(
                (
                    str(mask.relative_to(ROOT)),
                    f"fake_candidates={len(fake_candidates)}"
                )
            )
            continue

        fake = fake_candidates[0]

        try:
            with Image.open(fake) as im:
                fake_size = im.size

            with Image.open(mask) as im:
                mask_size = im.size

            with Image.open(orig) as im:
                orig_size = im.size

        except Exception as exc:
            problems.append(
                (str(fake.relative_to(ROOT)),
                 f"read_error={exc}")
            )
            continue

        # Localization requires fake and mask to align exactly.
        if fake_size != mask_size:
            problems.append(
                (
                    str(fake.relative_to(ROOT)),
                    f"fake_size={fake_size}, mask_size={mask_size}"
                )
            )
            continue

        fake_pairs.append(
            {
                "orig": orig,
                "fake": fake,
                "mask": mask,
                "orig_size": orig_size,
                "fake_size": fake_size,
            }
        )

    if fake_pairs:
        groups.append(
            {
                "group": str(folder.relative_to(ROOT)),
                "pairs": fake_pairs,
            }
        )

# ---------------------------------------------------------
# 2. Group-level split
# IMPORTANT:
# all manipulations derived from one original stay in one split.
# ---------------------------------------------------------

rng = random.Random(SEED)
rng.shuffle(groups)

n = len(groups)

n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

train_groups = groups[:n_train]
val_groups = groups[n_train:n_train + n_val]
test_groups = groups[n_train + n_val:]

split_groups = {
    "train": train_groups,
    "val": val_groups,
    "test": test_groups,
}

# ---------------------------------------------------------
# 3. Build records
#
# Each group contributes:
#   one authentic record
#   N manipulated records
#
# Authentic localization mask is represented as NONE.
# ---------------------------------------------------------

sample_id = 0

for split, group_list in split_groups.items():

    for group_info in group_list:

        group_name = group_info["group"]
        pairs = group_info["pairs"]

        # Authentic image: add once only
        orig = pairs[0]["orig"]

        with Image.open(orig) as im:
            ow, oh = im.size

        records.append(
            {
                "sample_id": sample_id,
                "group": group_name,
                "split": split,
                "label": 0,
                "image_path": str(orig.relative_to(ROOT)),
                "mask_path": "",
                "width": ow,
                "height": oh,
            }
        )

        sample_id += 1

        # Manipulated images
        for pair in pairs:

            fake = pair["fake"]
            mask = pair["mask"]

            fw, fh = pair["fake_size"]

            records.append(
                {
                    "sample_id": sample_id,
                    "group": group_name,
                    "split": split,
                    "label": 1,
                    "image_path": str(fake.relative_to(ROOT)),
                    "mask_path": str(mask.relative_to(ROOT)),
                    "width": fw,
                    "height": fh,
                }
            )

            sample_id += 1

# ---------------------------------------------------------
# 4. Write manifest
# ---------------------------------------------------------

fieldnames = [
    "sample_id",
    "group",
    "split",
    "label",
    "image_path",
    "mask_path",
    "width",
    "height",
]

with OUT.open("w", newline="", encoding="utf-8-sig") as f:

    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(records)

# ---------------------------------------------------------
# 5. Statistics
# ---------------------------------------------------------

print("=" * 72)
print("IMD2020 MANIFEST BUILDER")
print("=" * 72)

print(f"Dataset root       : {ROOT}")
print(f"Valid source groups: {len(groups)}")
print(f"Total records      : {len(records)}")
print(f"Problems           : {len(problems)}")

print()

for split in ["train", "val", "test"]:

    subset = [r for r in records if r["split"] == split]

    authentic = sum(r["label"] == 0 for r in subset)
    manipulated = sum(r["label"] == 1 for r in subset)

    group_count = len(split_groups[split])

    print(
        f"{split.upper():5s} | "
        f"groups={group_count:4d} | "
        f"authentic={authentic:4d} | "
        f"manipulated={manipulated:4d} | "
        f"total={len(subset):4d}"
    )

print()
print("Manifest:", OUT)

# ---------------------------------------------------------
# 6. Leakage check
# ---------------------------------------------------------

train_names = {g["group"] for g in train_groups}
val_names = {g["group"] for g in val_groups}
test_names = {g["group"] for g in test_groups}

leakage = (
    (train_names & val_names)
    | (train_names & test_names)
    | (val_names & test_names)
)

print(f"Group leakage      : {len(leakage)}")

# ---------------------------------------------------------
# 7. Show problems
# ---------------------------------------------------------

if problems:

    print("\nFirst problems:")

    for p in problems[:20]:
        print(" ", p)

# ---------------------------------------------------------
# 8. Final status
# ---------------------------------------------------------

if records and len(leakage) == 0:

    print("\nMANIFEST BUILD: PASS")

else:

    print("\nMANIFEST BUILD: REVIEW REQUIRED")
