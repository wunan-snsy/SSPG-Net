from pathlib import Path
from collections import Counter
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "data" / "imd2020"

orig_files = sorted(ROOT.rglob("*_orig.*"))
fake_files = sorted(ROOT.rglob("*_fake.*"))

# 排除 *_fake_mask.png
fake_files = [p for p in fake_files if "_fake_mask" not in p.stem]

mask_files = sorted(ROOT.rglob("*_fake_mask.*"))

print("=" * 70)
print("IMD2020 DATASET INSPECTION")
print("=" * 70)

print(f"Root:  {ROOT}")
print(f"Orig:  {len(orig_files)}")
print(f"Fake:  {len(fake_files)}")
print(f"Mask:  {len(mask_files)}")

ext_counter = Counter(
    p.suffix.lower()
    for p in ROOT.rglob("*")
    if p.is_file()
)

print("\nFile extensions:")
for ext, count in sorted(ext_counter.items()):
    print(f"  {ext}: {count}")

pairs = []
missing = []

for fake in fake_files:
    stem = fake.stem

    if not stem.endswith("_fake"):
        continue

    base = stem[:-5]

    orig_candidates = list(fake.parent.glob(base + "_orig.*"))
    mask_candidates = list(fake.parent.glob(base + "_fake_mask.*"))

    if len(orig_candidates) == 1 and len(mask_candidates) == 1:
        pairs.append(
            (
                orig_candidates[0],
                fake,
                mask_candidates[0],
            )
        )
    else:
        missing.append(
            (
                fake,
                len(orig_candidates),
                len(mask_candidates),
            )
        )

print("\n" + "=" * 70)
print("PAIR CHECK")
print("=" * 70)

print(f"Complete triplets: {len(pairs)}")
print(f"Incomplete fake samples: {len(missing)}")

if missing:
    print("\nFirst incomplete samples:")
    for item in missing[:20]:
        print(item)

print("\n" + "=" * 70)
print("IMAGE SIZE CHECK")
print("=" * 70)

size_mismatch = []
read_errors = []

for i, (orig, fake, mask) in enumerate(pairs, start=1):
    try:
        with Image.open(orig) as im:
            orig_size = im.size

        with Image.open(fake) as im:
            fake_size = im.size

        with Image.open(mask) as im:
            mask_size = im.size

        if not (orig_size == fake_size == mask_size):
            size_mismatch.append(
                (
                    orig,
                    orig_size,
                    fake_size,
                    mask_size,
                )
            )

    except Exception as exc:
        read_errors.append((fake, str(exc)))

    if i % 250 == 0:
        print(f"Checked {i}/{len(pairs)} triplets...")

print("\n" + "=" * 70)
print("FINAL REPORT")
print("=" * 70)

print(f"Complete triplets : {len(pairs)}")
print(f"Size mismatches   : {len(size_mismatch)}")
print(f"Read errors       : {len(read_errors)}")

if size_mismatch:
    print("\nFirst size mismatches:")
    for item in size_mismatch[:20]:
        print(item)

if read_errors:
    print("\nFirst read errors:")
    for item in read_errors[:20]:
        print(item)

print("\nExample valid samples:")

for orig, fake, mask in pairs[:5]:
    print("-" * 70)
    print("ORIG:", orig.relative_to(ROOT))
    print("FAKE:", fake.relative_to(ROOT))
    print("MASK:", mask.relative_to(ROOT))

if (
    len(pairs) > 0
    and len(missing) == 0
    and len(size_mismatch) == 0
    and len(read_errors) == 0
):
    print("\nIMD2020 CHECK: PASS")
else:
    print("\nIMD2020 CHECK: REVIEW REQUIRED")
