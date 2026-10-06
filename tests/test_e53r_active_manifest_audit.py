from pathlib import Path
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from preprocessing.active_sspg_dataset import (
    ATTACKS,
    MIN_AREA,
    MAX_AREA,
    ATTACK_GEOMETRY_SEED_OFFSET,
    canonical_sample_key,
    stable_seed,
    make_payload_deterministic,
    load_rgb_256,
    build_balanced_attack_assignment,
    apply_controlled_attack,
)


# ============================================================
# E5.3R-4A.1-R3.1
# FINAL REAL TRAIN/VAL MANIFEST AUDIT
#
# IMPORTANT:
#
# - Uses REAL IMD2020 Train/Val authentic source images.
# - Does NOT load Test images.
# - Uses the SAME canonical source identity as training.
# - Uses the SAME payload seed as training.
# - Uses the SAME R3 attack assignment as training.
# - Uses the SAME R2 attack geometry as training.
#
# No detector/localizer is trained.
# ============================================================


MANIFEST = (
    ROOT
    / "manifests"
    / "imd2020_manifest_e5_clean.csv"
)

IMAGE_ROOT = (
    ROOT
    / "data"
    / "imd2020"
)

TRAIN_SEED = 20260922
VAL_SEED = 20260923

EXPECTED_TRAIN_AUTHENTIC = 289
EXPECTED_VAL_AUTHENTIC = 62

EXPECTED_TRAIN_ACTIVE = 578
EXPECTED_VAL_ACTIVE = 124

AREA_TOLERANCE_MIN = 0.145
AREA_TOLERANCE_MAX = 0.305


def resolve_image_path(
    relative_path,
):
    """
    Resolve manifest Windows-style relative path.
    """

    normalized = str(
        relative_path
    ).replace(
        "\\",
        "/",
    )

    return (
        IMAGE_ROOT
        / Path(normalized)
    )


def dataframe_to_records(
    dataframe,
):
    """
    Convert manifest rows into the exact record structure
    expected by ActiveSSPGDataset.

    image_path is converted to an absolute path.

    group/sample_id are preserved so canonical_sample_key()
    is identical between audit and training.
    """

    records = []

    for _, row in dataframe.iterrows():

        image_path = resolve_image_path(
            row["image_path"]
        )

        record = {
            "sample_id":
                row["sample_id"],

            "group":
                row["group"],

            "image_path":
                str(image_path),
        }

        records.append(
            record
        )

    return records


def audit_split(
    split_name,
    dataframe,
    base_seed,
    expected_sources,
    expected_counts,
):
    print()
    print("=" * 78)
    print(
        f"{split_name.upper()} "
        "AUTHENTIC SOURCE AUDIT"
    )
    print("=" * 78)

    source_df = dataframe[
        (
            dataframe["split"]
            == split_name
        )
        &
        (
            dataframe["label"]
            == 0
        )
    ].copy()

    source_df = (
        source_df
        .sort_values(
            by=[
                "group",
                "sample_id",
                "image_path",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    n_sources = len(
        source_df
    )

    if n_sources != expected_sources:
        raise RuntimeError(
            f"{split_name}: expected "
            f"{expected_sources} authentic sources, "
            f"got {n_sources}."
        )

    records = dataframe_to_records(
        source_df
    )

    print(
        "Authentic sources :",
        n_sources,
    )

    print(
        "Active samples     :",
        n_sources * 2,
    )

    # --------------------------------------------------------
    # Path existence
    # --------------------------------------------------------

    source_paths = []

    for record in records:

        path = Path(
            record[
                "image_path"
            ]
        )

        if not path.is_file():
            raise RuntimeError(
                f"Missing source image: {path}"
            )

        source_paths.append(
            str(
                path.resolve()
            ).lower()
        )

    if len(
        set(source_paths)
    ) != n_sources:
        raise RuntimeError(
            f"{split_name}: duplicate source paths."
        )

    # --------------------------------------------------------
    # Canonical sample keys
    # --------------------------------------------------------

    sample_keys = [
        canonical_sample_key(
            record
        )
        for record in records
    ]

    if len(
        set(sample_keys)
    ) != n_sources:
        raise RuntimeError(
            f"{split_name}: duplicate canonical "
            "sample keys."
        )

    # --------------------------------------------------------
    # R3 attack assignment
    # --------------------------------------------------------

    attack_assignment = (
        build_balanced_attack_assignment(
            n_sources,
            base_seed,
        )
    )

    attack_counts = Counter(
        attack_assignment
    )

    actual_counts = {
        attack:
            attack_counts[
                attack
            ]
        for attack in ATTACKS
    }

    print()
    print("-" * 78)
    print(
        "ATTACK DISTRIBUTION"
    )
    print("-" * 78)

    for attack in ATTACKS:

        count = attack_counts[
            attack
        ]

        print(
            f"{attack:10s}: "
            f"{count:4d} "
            f"({count / n_sources * 100:6.2f}%)"
        )

    if actual_counts != expected_counts:
        raise RuntimeError(
            f"{split_name}: unexpected R3 "
            f"attack counts: {actual_counts}"
        )

    values = list(
        actual_counts.values()
    )

    if (
        max(values)
        - min(values)
        > 1
    ):
        raise RuntimeError(
            f"{split_name}: R3 attack imbalance."
        )

    # --------------------------------------------------------
    # Payload + geometry audit
    # --------------------------------------------------------

    payload_signatures = []

    mask_areas = []

    actual_change_ratios = []

    deterministic_failures = 0

    outside_change_failures = 0

    mask_binary_failures = 0

    area_failures = 0

    for source_index, record in enumerate(
        records
    ):
        sample_key = (
            canonical_sample_key(
                record
            )
        )

        seed = stable_seed(
            base_seed,
            sample_key,
        )

        # ----------------------------------------------------
        # Payload determinism
        # ----------------------------------------------------

        payload_1 = (
            make_payload_deterministic(
                seed
            )
        )

        payload_2 = (
            make_payload_deterministic(
                seed
            )
        )

        if not torch.equal(
            payload_1,
            payload_2,
        ):
            deterministic_failures += 1

        payload_signature = (
            "".join(
                str(
                    int(x)
                )
                for x in payload_1.tolist()
            )
        )

        payload_signatures.append(
            payload_signature
        )

        # ----------------------------------------------------
        # Real source image
        #
        # For this geometric audit we intentionally do NOT
        # embed V3.3 yet. That is checked by the subsequent
        # full Dataset integration gate.
        # ----------------------------------------------------

        original = load_rgb_256(
            record[
                "image_path"
            ]
        )

        attack = attack_assignment[
            source_index
        ]

        rng_1 = np.random.default_rng(
            seed
            + ATTACK_GEOMETRY_SEED_OFFSET
        )

        rng_2 = np.random.default_rng(
            seed
            + ATTACK_GEOMETRY_SEED_OFFSET
        )

        manipulated_1, mask_1 = (
            apply_controlled_attack(
                original,
                attack,
                rng_1,
            )
        )

        manipulated_2, mask_2 = (
            apply_controlled_attack(
                original,
                attack,
                rng_2,
            )
        )

        # ----------------------------------------------------
        # Geometry determinism
        # ----------------------------------------------------

        if not torch.equal(
            mask_1,
            mask_2,
        ):
            deterministic_failures += 1

        if not torch.equal(
            manipulated_1,
            manipulated_2,
        ):
            deterministic_failures += 1

        # ----------------------------------------------------
        # Mask shape
        # ----------------------------------------------------

        if tuple(
            mask_1.shape
        ) != (
            1,
            256,
            256,
        ):
            raise RuntimeError(
                f"{split_name}: invalid mask shape."
            )

        # ----------------------------------------------------
        # Binary mask
        # ----------------------------------------------------

        unique_mask = torch.unique(
            mask_1
        )

        valid_binary = all(
            float(x) in (
                0.0,
                1.0,
            )
            for x in unique_mask
        )

        if not valid_binary:
            mask_binary_failures += 1

        # ----------------------------------------------------
        # Area
        # ----------------------------------------------------

        mask_area = float(
            mask_1.mean().item()
        )

        mask_areas.append(
            mask_area
        )

        if not (
            AREA_TOLERANCE_MIN
            <= mask_area
            <= AREA_TOLERANCE_MAX
        ):
            area_failures += 1

        # ----------------------------------------------------
        # Verify nothing changes outside GT.
        # ----------------------------------------------------

        difference = (
            manipulated_1
            - original
        ).abs()

        changed_pixel = (
            difference.max(
                dim=0
            ).values
            > 1e-7
        )

        mask_bool = (
            mask_1[
                0
            ]
            > 0.5
        )

        outside_changed = (
            changed_pixel
            &
            (~mask_bool)
        )

        if bool(
            outside_changed.any()
        ):
            outside_change_failures += 1

        # ----------------------------------------------------
        # Diagnostic only:
        # actual changed-pixel ratio inside GT.
        #
        # No post-hoc pass threshold is introduced here.
        # ----------------------------------------------------

        inside_count = int(
            mask_bool.sum().item()
        )

        if inside_count <= 0:
            raise RuntimeError(
                f"{split_name}: empty manipulation mask."
            )

        actual_inside_change = float(
            (
                changed_pixel
                &
                mask_bool
            ).sum().item()
            / inside_count
        )

        actual_change_ratios.append(
            actual_inside_change
        )

    # --------------------------------------------------------
    # Payload statistics
    # --------------------------------------------------------

    unique_payloads = len(
        set(
            payload_signatures
        )
    )

    repeated_payloads = (
        n_sources
        - unique_payloads
    )

    print()
    print("-" * 78)
    print(
        "MASK AREA DISTRIBUTION"
    )
    print("-" * 78)

    areas = np.asarray(
        mask_areas,
        dtype=np.float64,
    )

    print(
        "Minimum :",
        f"{areas.min():.6f}",
    )

    print(
        "Median  :",
        f"{np.median(areas):.6f}",
    )

    print(
        "Mean    :",
        f"{areas.mean():.6f}",
    )

    print(
        "Maximum :",
        f"{areas.max():.6f}",
    )

    changes = np.asarray(
        actual_change_ratios,
        dtype=np.float64,
    )

    print()
    print("-" * 78)
    print(
        "ACTUAL PIXEL CHANGE INSIDE GT"
    )
    print("-" * 78)

    print(
        "Minimum :",
        f"{changes.min():.6f}",
    )

    print(
        "Median  :",
        f"{np.median(changes):.6f}",
    )

    print(
        "Mean    :",
        f"{changes.mean():.6f}",
    )

    print(
        "Maximum :",
        f"{changes.max():.6f}",
    )

    print()
    print("-" * 78)
    print(
        "PAYLOAD AUDIT"
    )
    print("-" * 78)

    print(
        "Unique payloads   :",
        unique_payloads,
    )

    print(
        "Repeated payloads :",
        repeated_payloads,
    )

    print()
    print("-" * 78)
    print(
        "FAILURE COUNTS"
    )
    print("-" * 78)

    print(
        "Determinism failures     :",
        deterministic_failures,
    )

    print(
        "Mask binary failures     :",
        mask_binary_failures,
    )

    print(
        "Area failures            :",
        area_failures,
    )

    print(
        "Outside-change failures  :",
        outside_change_failures,
    )

    if deterministic_failures != 0:
        raise RuntimeError(
            f"{split_name}: determinism failure."
        )

    if mask_binary_failures != 0:
        raise RuntimeError(
            f"{split_name}: non-binary masks."
        )

    if area_failures != 0:
        raise RuntimeError(
            f"{split_name}: manipulation area failure."
        )

    if outside_change_failures != 0:
        raise RuntimeError(
            f"{split_name}: pixels changed outside GT."
        )

    if repeated_payloads != 0:
        raise RuntimeError(
            f"{split_name}: repeated payloads detected."
        )

    print()
    print(
        f"{split_name.upper()} SPLIT AUDIT: PASS"
    )

    return {
        "paths":
            set(source_paths),

        "groups":
            set(
                source_df[
                    "group"
                ].astype(
                    str
                ).tolist()
            ),

        "sample_keys":
            set(sample_keys),

        "n_sources":
            n_sources,

        "n_active":
            n_sources * 2,

        "attack_counts":
            actual_counts,
    }


def main():
    print("=" * 78)
    print(
        "E5.3R-4A.1-R3.1 FINAL REAL "
        "TRAIN/VAL MANIFEST AUDIT"
    )
    print("=" * 78)

    print(
        "Manifest   :",
        MANIFEST,
    )

    print(
        "Image root :",
        IMAGE_ROOT,
    )

    print(
        "Train seed :",
        TRAIN_SEED,
    )

    print(
        "Val seed   :",
        VAL_SEED,
    )

    if not MANIFEST.is_file():
        raise RuntimeError(
            f"Manifest not found: {MANIFEST}"
        )

    if not IMAGE_ROOT.is_dir():
        raise RuntimeError(
            f"Image root not found: {IMAGE_ROOT}"
        )

    df = pd.read_csv(
        MANIFEST
    )

    print()
    print(
        "Manifest rows :",
        len(df),
    )

    print(
        "Columns       :",
        list(df.columns),
    )

    split_counts = (
        df[
            "split"
        ]
        .value_counts()
        .to_dict()
    )

    print(
        "Split metadata:",
        split_counts,
    )

    expected_split_counts = {
        "train": 1750,
        "val": 374,
        "test": 298,
    }

    if split_counts != expected_split_counts:
        raise RuntimeError(
            "Unexpected clean-manifest split counts."
        )

    # --------------------------------------------------------
    # Development dataframe contains Train/Val ONLY.
    #
    # We read Test metadata from the CSV because split counts
    # are audited, but NO Test image path is opened.
    # --------------------------------------------------------

    development_df = df[
        df[
            "split"
        ].isin(
            [
                "train",
                "val",
            ]
        )
    ].copy()

    if bool(
        (
            development_df[
                "split"
            ]
            == "test"
        ).any()
    ):
        raise RuntimeError(
            "Test row entered development dataframe."
        )

    print()
    print(
        "TEST IMAGE ACCESS GATE: PASS"
    )

    print(
        "No Test image row is present "
        "in the development dataframe."
    )

    print()
    print("-" * 78)
    print(
        "PREREGISTERED SOURCE COUNTS"
    )
    print("-" * 78)

    print(
        "Train authentic sources :",
        EXPECTED_TRAIN_AUTHENTIC,
    )

    print(
        "Val authentic sources   :",
        EXPECTED_VAL_AUTHENTIC,
    )

    print(
        "Train active samples    :",
        EXPECTED_TRAIN_ACTIVE,
    )

    print(
        "Val active samples      :",
        EXPECTED_VAL_ACTIVE,
    )

    train_result = audit_split(
        split_name="train",
        dataframe=development_df,
        base_seed=TRAIN_SEED,
        expected_sources=
            EXPECTED_TRAIN_AUTHENTIC,
        expected_counts={
            "erase": 97,
            "copy_move": 96,
            "roll": 96,
        },
    )

    val_result = audit_split(
        split_name="val",
        dataframe=development_df,
        base_seed=VAL_SEED,
        expected_sources=
            EXPECTED_VAL_AUTHENTIC,
        expected_counts={
            "erase": 21,
            "copy_move": 21,
            "roll": 20,
        },
    )

    # --------------------------------------------------------
    # Train / Val leakage audit
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "TRAIN / VAL LEAKAGE AUDIT"
    )
    print("=" * 78)

    path_overlap = (
        train_result[
            "paths"
        ]
        &
        val_result[
            "paths"
        ]
    )

    group_overlap = (
        train_result[
            "groups"
        ]
        &
        val_result[
            "groups"
        ]
    )

    key_overlap = (
        train_result[
            "sample_keys"
        ]
        &
        val_result[
            "sample_keys"
        ]
    )

    print(
        "Source-path overlap :",
        len(path_overlap),
    )

    print(
        "Group overlap       :",
        len(group_overlap),
    )

    print(
        "Sample-key overlap  :",
        len(key_overlap),
    )

    if path_overlap:
        raise RuntimeError(
            "Train/Val source-path leakage."
        )

    if group_overlap:
        raise RuntimeError(
            "Train/Val group leakage."
        )

    if key_overlap:
        raise RuntimeError(
            "Train/Val canonical-key leakage."
        )

    if (
        train_result[
            "n_active"
        ]
        != EXPECTED_TRAIN_ACTIVE
    ):
        raise RuntimeError(
            "Unexpected Train active count."
        )

    if (
        val_result[
            "n_active"
        ]
        != EXPECTED_VAL_ACTIVE
    ):
        raise RuntimeError(
            "Unexpected Val active count."
        )

    print()
    print(
        "Train/Val leakage gate : PASS"
    )

    print()
    print("=" * 78)
    print(
        "E5.3R-4A.1-R3.1 REAL "
        "MANIFEST AUDIT: PASS"
    )
    print("=" * 78)

    print()
    print(
        "Train authentic sources :",
        train_result[
            "n_sources"
        ],
    )

    print(
        "Train active samples    :",
        train_result[
            "n_active"
        ],
    )

    print(
        "Val authentic sources   :",
        val_result[
            "n_sources"
        ],
    )

    print(
        "Val active samples      :",
        val_result[
            "n_active"
        ],
    )

    print()
    print(
        "Frozen Train attacks:"
    )

    print(
        "  erase=97, "
        "copy_move=96, "
        "roll=96"
    )

    print(
        "Frozen Val attacks:"
    )

    print(
        "  erase=21, "
        "copy_move=21, "
        "roll=20"
    )

    print()
    print(
        "No detector/localization "
        "head was trained."
    )

    print(
        "No Test image was loaded."
    )


if __name__ == "__main__":
    main()
