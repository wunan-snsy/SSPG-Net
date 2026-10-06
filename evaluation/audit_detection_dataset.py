from pathlib import Path
import hashlib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST = (
    PROJECT_ROOT
    / "manifests"
    / "imd2020_manifest.csv"
)

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "imd2020"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "results"
    / "csv"
    / "e5_detection_dataset_audit.csv"
)


def separator():
    print("=" * 78)


def subsection():
    print("-" * 78)


def normalize_path(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    return DATA_ROOT / Path(value)


def file_md5(path, chunk_size=1024 * 1024):
    h = hashlib.md5()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def main():

    separator()
    print("E5.0 SSPG-NET FORGERY DETECTION DATASET AUDIT")
    separator()

    print("Manifest :", MANIFEST)
    print("Data root:", DATA_ROOT)
    print()

    df = pd.read_csv(MANIFEST)

    required_columns = [
        "sample_id",
        "group",
        "split",
        "label",
        "image_path",
        "mask_path",
        "width",
        "height",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # ========================================================
    # 1. Basic manifest statistics
    # ========================================================

    subsection()
    print("1. BASIC MANIFEST STATISTICS")
    subsection()

    print("Total samples :", len(df))
    print("Unique IDs    :", df["sample_id"].nunique())
    print("Unique groups :", df["group"].nunique())
    print()

    print("Split counts:")
    print(df["split"].value_counts().sort_index())
    print()

    print("Label counts:")
    print(df["label"].value_counts().sort_index())
    print()

    # ========================================================
    # 2. Split x label
    # ========================================================

    subsection()
    print("2. SPLIT / LABEL DISTRIBUTION")
    subsection()

    table = pd.crosstab(
        df["split"],
        df["label"],
        margins=True,
    )

    print(table)
    print()

    for split in sorted(df["split"].unique()):

        part = df[df["split"] == split]

        n = len(part)
        authentic = int((part["label"] == 0).sum())
        manipulated = int((part["label"] == 1).sum())

        print(
            f"{split:>8s}: "
            f"N={n:4d} | "
            f"authentic={authentic:4d} "
            f"({authentic / n * 100:6.2f}%) | "
            f"manipulated={manipulated:4d} "
            f"({manipulated / n * 100:6.2f}%)"
        )

    print()

    # ========================================================
    # 3. Label validity
    # ========================================================

    subsection()
    print("3. LABEL VALIDITY")
    subsection()

    invalid_labels = df[
        ~df["label"].isin([0, 1])
    ]

    print(
        "Invalid labels:",
        len(invalid_labels),
    )

    # ========================================================
    # 4. Duplicate sample IDs / paths
    # ========================================================

    subsection()
    print("4. DUPLICATE IDENTIFIERS / PATHS")
    subsection()

    duplicated_ids = df[
        df["sample_id"].duplicated(
            keep=False
        )
    ]

    duplicated_paths = df[
        df["image_path"].duplicated(
            keep=False
        )
    ]

    print(
        "Duplicated sample_id rows:",
        len(duplicated_ids),
    )

    print(
        "Duplicated image_path rows:",
        len(duplicated_paths),
    )

    # ========================================================
    # 5. Group-level split leakage
    # ========================================================

    subsection()
    print("5. GROUP-LEVEL SPLIT LEAKAGE")
    subsection()

    group_split_counts = (
        df.groupby("group")["split"]
        .nunique()
    )

    leaking_groups = (
        group_split_counts[
            group_split_counts > 1
        ]
    )

    print(
        "Groups appearing in >1 split:",
        len(leaking_groups),
    )

    if len(leaking_groups) > 0:

        print()

        print(
            df[
                df["group"].isin(
                    leaking_groups.index
                )
            ][
                [
                    "group",
                    "split",
                    "sample_id",
                    "label",
                    "image_path",
                ]
            ]
            .sort_values(
                ["group", "split"]
            )
            .to_string(index=False)
        )

    # ========================================================
    # 6. Group composition
    # ========================================================

    subsection()
    print("6. GROUP COMPOSITION")
    subsection()

    group_sizes = (
        df.groupby("group")
        .size()
    )

    print(
        "Group size min    :",
        int(group_sizes.min()),
    )

    print(
        "Group size median :",
        float(group_sizes.median()),
    )

    print(
        "Group size mean   :",
        float(group_sizes.mean()),
    )

    print(
        "Group size max    :",
        int(group_sizes.max()),
    )

    print()

    groups_without_authentic = 0
    groups_without_manipulated = 0

    for group, part in df.groupby("group"):

        labels = set(
            part["label"].tolist()
        )

        if 0 not in labels:
            groups_without_authentic += 1

        if 1 not in labels:
            groups_without_manipulated += 1

    print(
        "Groups without authentic   :",
        groups_without_authentic,
    )

    print(
        "Groups without manipulated :",
        groups_without_manipulated,
    )

    # ========================================================
    # 7. File existence
    # ========================================================

    subsection()
    print("7. FILE EXISTENCE")
    subsection()

    missing_images = []
    missing_masks = []

    for _, row in df.iterrows():

        image_file = normalize_path(
            row["image_path"]
        )

        if (
            image_file is None
            or not image_file.exists()
        ):
            missing_images.append(
                row["sample_id"]
            )

        if row["label"] == 1:

            mask_file = normalize_path(
                row["mask_path"]
            )

            if (
                mask_file is None
                or not mask_file.exists()
            ):
                missing_masks.append(
                    row["sample_id"]
                )

    print(
        "Missing image files:",
        len(missing_images),
    )

    print(
        "Missing manipulated masks:",
        len(missing_masks),
    )

    # ========================================================
    # 8. Mask consistency
    # ========================================================

    subsection()
    print("8. LABEL / MASK CONSISTENCY")
    subsection()

    authentic = df[
        df["label"] == 0
    ]

    manipulated = df[
        df["label"] == 1
    ]

    authentic_with_mask = (
        authentic["mask_path"]
        .notna()
        .sum()
    )

    manipulated_without_mask = (
        manipulated["mask_path"]
        .isna()
        .sum()
    )

    print(
        "Authentic samples with mask:",
        int(authentic_with_mask),
    )

    print(
        "Manipulated samples without mask:",
        int(manipulated_without_mask),
    )

    # ========================================================
    # 9. Image dimension audit
    # ========================================================

    subsection()
    print("9. IMAGE DIMENSIONS")
    subsection()

    print(
        "Width  min / median / max:",
        int(df["width"].min()),
        "/",
        float(df["width"].median()),
        "/",
        int(df["width"].max()),
    )

    print(
        "Height min / median / max:",
        int(df["height"].min()),
        "/",
        float(df["height"].median()),
        "/",
        int(df["height"].max()),
    )

    invalid_dimensions = df[
        (df["width"] <= 0)
        | (df["height"] <= 0)
    ]

    print(
        "Invalid dimensions:",
        len(invalid_dimensions),
    )

    # ========================================================
    # 10. Exact binary duplicate check
    # ========================================================

    subsection()
    print("10. EXACT IMAGE-BINARY DUPLICATE CHECK")
    subsection()

    hashes = []
    hash_errors = 0

    for index, row in df.iterrows():

        path = normalize_path(
            row["image_path"]
        )

        if path is None or not path.exists():

            hashes.append(None)
            hash_errors += 1

            continue

        try:

            hashes.append(
                file_md5(path)
            )

        except Exception:

            hashes.append(None)
            hash_errors += 1

    df["image_md5"] = hashes

    valid_hash_df = df[
        df["image_md5"].notna()
    ]

    duplicate_hash_rows = (
        valid_hash_df[
            valid_hash_df["image_md5"]
            .duplicated(
                keep=False
            )
        ]
    )

    duplicate_hash_groups = (
        duplicate_hash_rows[
            "image_md5"
        ]
        .nunique()
    )

    print(
        "Hash read errors:",
        hash_errors,
    )

    print(
        "Duplicate binary hash groups:",
        duplicate_hash_groups,
    )

    print(
        "Rows involved in binary duplicates:",
        len(duplicate_hash_rows),
    )

    # Check whether identical binary images cross splits.

    cross_split_hashes = 0

    if len(duplicate_hash_rows) > 0:

        hash_split_counts = (
            duplicate_hash_rows
            .groupby("image_md5")["split"]
            .nunique()
        )

        cross_split_hashes = int(
            (
                hash_split_counts > 1
            ).sum()
        )

    print(
        "Identical image hashes crossing splits:",
        cross_split_hashes,
    )

    # ========================================================
    # 11. Save audit table
    # ========================================================

    subsection()
    print("11. SAVE AUDIT TABLE")
    subsection()

    audit_df = df.copy()

    audit_df[
        "image_exists"
    ] = audit_df[
        "image_path"
    ].apply(
        lambda x:
        normalize_path(x) is not None
        and normalize_path(x).exists()
    )

    audit_df[
        "mask_exists"
    ] = audit_df.apply(
        lambda row:
        (
            True
            if row["label"] == 0
            else (
                normalize_path(
                    row["mask_path"]
                )
                is not None
                and normalize_path(
                    row["mask_path"]
                ).exists()
            )
        ),
        axis=1,
    )

    audit_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        "Saved:",
        OUTPUT_CSV,
    )

    # ========================================================
    # 12. Final audit verdict
    # ========================================================

    subsection()
    print("12. E5.0 AUDIT VERDICT")
    subsection()

    critical_failures = []

    if len(invalid_labels) > 0:
        critical_failures.append(
            "invalid labels"
        )

    if len(duplicated_ids) > 0:
        critical_failures.append(
            "duplicated sample IDs"
        )

    if len(duplicated_paths) > 0:
        critical_failures.append(
            "duplicated image paths"
        )

    if len(leaking_groups) > 0:
        critical_failures.append(
            "group-level split leakage"
        )

    if len(missing_images) > 0:
        critical_failures.append(
            "missing images"
        )

    if len(missing_masks) > 0:
        critical_failures.append(
            "missing manipulated masks"
        )

    if authentic_with_mask > 0:
        critical_failures.append(
            "authentic samples unexpectedly have masks"
        )

    if manipulated_without_mask > 0:
        critical_failures.append(
            "manipulated samples lack masks"
        )

    if len(invalid_dimensions) > 0:
        critical_failures.append(
            "invalid image dimensions"
        )

    if cross_split_hashes > 0:
        critical_failures.append(
            "exact image duplicates cross splits"
        )

    if len(critical_failures) == 0:

        print(
            "PASS: no critical dataset-integrity "
            "failure detected."
        )

    else:

        print("FAIL:")

        for failure in critical_failures:
            print(" -", failure)

    separator()


if __name__ == "__main__":
    main()
