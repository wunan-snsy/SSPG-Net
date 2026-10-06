from pathlib import Path
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from preprocessing.active_sspg_dataset import (
    ActiveSSPGDataset,
    canonical_sample_key,
    stable_seed,
    make_payload_deterministic,
)


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

EXPECTED_TRAIN_SOURCES = 289
EXPECTED_VAL_SOURCES = 62

EXPECTED_TRAIN_ACTIVE = 578
EXPECTED_VAL_ACTIVE = 124

EXPECTED_TRAIN_ATTACKS = {
    "erase": 97,
    "copy_move": 96,
    "roll": 96,
}

EXPECTED_VAL_ATTACKS = {
    "erase": 21,
    "copy_move": 21,
    "roll": 20,
}


def resolve_image_path(relative_path):
    normalized = str(
        relative_path
    ).replace(
        "\\",
        "/",
    )

    return str(
        IMAGE_ROOT
        / Path(normalized)
    )


def build_records(df, split_name):
    source_df = df[
        (df["split"] == split_name)
        &
        (df["label"] == 0)
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
        .reset_index(drop=True)
    )

    records = []

    for _, row in source_df.iterrows():
        records.append(
            {
                "sample_id":
                    row["sample_id"],

                "group":
                    row["group"],

                "image_path":
                    resolve_image_path(
                        row["image_path"]
                    ),
            }
        )

    return records


def tensor_equal(a, b):
    return torch.equal(
        a.cpu(),
        b.cpu(),
    )


def audit_dataset(
    name,
    records,
    base_seed,
    expected_sources,
    expected_active,
    expected_attacks,
):
    print()
    print("=" * 78)
    print(
        f"{name} FULL DATASET INTEGRATION"
    )
    print("=" * 78)

    if len(records) != expected_sources:
        raise RuntimeError(
            f"{name}: source count mismatch."
        )

    dataset = ActiveSSPGDataset(
        records=records,
        base_seed=base_seed,
    )

    print(
        "Authentic sources :",
        len(records),
    )

    print(
        "Dataset samples   :",
        len(dataset),
    )

    if len(dataset) != expected_active:
        raise RuntimeError(
            f"{name}: active sample count mismatch."
        )

    # --------------------------------------------------------
    # Dataset attack assignment itself.
    # --------------------------------------------------------

    counts = Counter(
        dataset.attack_assignment
    )

    actual_attacks = {
        attack:
            counts[attack]
        for attack in (
            "erase",
            "copy_move",
            "roll",
        )
    }

    print()
    print(
        "Attack assignment :",
        actual_attacks,
    )

    if actual_attacks != expected_attacks:
        raise RuntimeError(
            f"{name}: dataset attack assignment mismatch."
        )

    # --------------------------------------------------------
    # Full dataset traversal.
    #
    # This is the important part:
    # __getitem__ really executes V3.3 embed_payload().
    # --------------------------------------------------------

    authentic_count = 0
    manipulated_count = 0

    payload_signatures = set()

    mask_areas = []

    finite_failures = 0
    shape_failures = 0
    label_failures = 0
    pair_payload_failures = 0
    pair_key_failures = 0
    authentic_mask_failures = 0
    manipulated_mask_failures = 0

    for source_index in range(
        len(records)
    ):
        authentic = dataset[
            2 * source_index
        ]

        manipulated = dataset[
            2 * source_index + 1
        ]

        # ----------------------------------------------------
        # Shapes
        # ----------------------------------------------------

        for sample in (
            authentic,
            manipulated,
        ):
            if tuple(
                sample["received"].shape
            ) != (
                3,
                256,
                256,
            ):
                shape_failures += 1

            if tuple(
                sample["payload"].shape
            ) != (
                128,
            ):
                shape_failures += 1

            if tuple(
                sample["mask"].shape
            ) != (
                1,
                256,
                256,
            ):
                shape_failures += 1

            if not torch.isfinite(
                sample["received"]
            ).all():
                finite_failures += 1

            if not torch.isfinite(
                sample["payload"]
            ).all():
                finite_failures += 1

            if not torch.isfinite(
                sample["mask"]
            ).all():
                finite_failures += 1

        # ----------------------------------------------------
        # Pair identity
        # ----------------------------------------------------

        if (
            authentic["sample_key"]
            != manipulated["sample_key"]
        ):
            pair_key_failures += 1

        if not tensor_equal(
            authentic["payload"],
            manipulated["payload"],
        ):
            pair_payload_failures += 1

        # ----------------------------------------------------
        # Canonical key / seed / payload consistency.
        # ----------------------------------------------------

        expected_key = (
            canonical_sample_key(
                records[source_index]
            )
        )

        if authentic["sample_key"] != expected_key:
            pair_key_failures += 1

        expected_seed = stable_seed(
            base_seed,
            expected_key,
        )

        if (
            authentic["source_seed"]
            != expected_seed
        ):
            pair_key_failures += 1

        expected_payload = (
            make_payload_deterministic(
                expected_seed
            )
        )

        if not tensor_equal(
            authentic["payload"],
            expected_payload,
        ):
            pair_payload_failures += 1

        # ----------------------------------------------------
        # Authentic sample.
        # ----------------------------------------------------

        if float(
            authentic[
                "detection_label"
            ].item()
        ) != 0.0:
            label_failures += 1

        if (
            authentic["attack"]
            != "authentic"
        ):
            label_failures += 1

        if float(
            authentic["mask"].sum().item()
        ) != 0.0:
            authentic_mask_failures += 1

        authentic_count += 1

        # ----------------------------------------------------
        # Manipulated sample.
        # ----------------------------------------------------

        if float(
            manipulated[
                "detection_label"
            ].item()
        ) != 1.0:
            label_failures += 1

        expected_attack = (
            dataset.attack_assignment[
                source_index
            ]
        )

        if (
            manipulated["attack"]
            != expected_attack
        ):
            label_failures += 1

        area = float(
            manipulated[
                "mask"
            ].mean().item()
        )

        mask_areas.append(
            area
        )

        if not (
            0.145
            <= area
            <= 0.305
        ):
            manipulated_mask_failures += 1

        manipulated_count += 1

        payload_signature = "".join(
            str(int(x))
            for x in authentic[
                "payload"
            ].tolist()
        )

        payload_signatures.add(
            payload_signature
        )

    # --------------------------------------------------------
    # Exact detection balance.
    # --------------------------------------------------------

    print()
    print("-" * 78)
    print(
        "FULL PIPELINE COUNTS"
    )
    print("-" * 78)

    print(
        "Authentic samples   :",
        authentic_count,
    )

    print(
        "Manipulated samples :",
        manipulated_count,
    )

    if authentic_count != expected_sources:
        raise RuntimeError(
            f"{name}: authentic count failure."
        )

    if manipulated_count != expected_sources:
        raise RuntimeError(
            f"{name}: manipulated count failure."
        )

    # --------------------------------------------------------
    # Payload uniqueness.
    # --------------------------------------------------------

    print()
    print(
        "Unique source payloads :",
        len(payload_signatures),
    )

    if (
        len(payload_signatures)
        != expected_sources
    ):
        raise RuntimeError(
            f"{name}: repeated payload detected."
        )

    # --------------------------------------------------------
    # Mask area.
    # --------------------------------------------------------

    areas = np.asarray(
        mask_areas,
        dtype=np.float64,
    )

    print()
    print("-" * 78)
    print(
        "MANIPULATED MASK AREA"
    )
    print("-" * 78)

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

    # --------------------------------------------------------
    # Failure counts.
    # --------------------------------------------------------

    print()
    print("-" * 78)
    print(
        "FAILURE COUNTS"
    )
    print("-" * 78)

    failures = {
        "shape":
            shape_failures,

        "finite":
            finite_failures,

        "label":
            label_failures,

        "pair payload":
            pair_payload_failures,

        "pair key/seed":
            pair_key_failures,

        "authentic mask":
            authentic_mask_failures,

        "manipulated mask":
            manipulated_mask_failures,
    }

    for key, value in failures.items():
        print(
            f"{key:20s}: {value}"
        )

    if any(
        value != 0
        for value in failures.values()
    ):
        raise RuntimeError(
            f"{name}: full dataset integration failure."
        )

    # --------------------------------------------------------
    # Deterministic repeat check.
    #
    # Use fixed positions from beginning/middle/end.
    # --------------------------------------------------------

    repeat_indices = sorted(
        set(
            [
                0,
                1,
                len(dataset) // 2,
                len(dataset) // 2 + 1,
                len(dataset) - 2,
                len(dataset) - 1,
            ]
        )
    )

    repeat_failures = 0

    for index in repeat_indices:
        sample_a = dataset[
            index
        ]

        sample_b = dataset[
            index
        ]

        if not tensor_equal(
            sample_a["received"],
            sample_b["received"],
        ):
            repeat_failures += 1

        if not tensor_equal(
            sample_a["payload"],
            sample_b["payload"],
        ):
            repeat_failures += 1

        if not tensor_equal(
            sample_a["mask"],
            sample_b["mask"],
        ):
            repeat_failures += 1

        if (
            sample_a["attack"]
            != sample_b["attack"]
        ):
            repeat_failures += 1

    print()
    print(
        "Deterministic repeat failures :",
        repeat_failures,
    )

    if repeat_failures != 0:
        raise RuntimeError(
            f"{name}: dataset is not deterministic."
        )

    print()
    print(
        f"{name} FULL DATASET "
        "INTEGRATION: PASS"
    )

    return {
        "sample_keys":
            {
                canonical_sample_key(r)
                for r in records
            },

        "groups":
            {
                str(
                    r["group"]
                )
                for r in records
            },
    }


def main():
    print("=" * 78)
    print(
        "E5.3R-4A.2 FULL ACTIVE DATASET "
        "INTEGRATION GATE"
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
            "Clean manifest not found."
        )

    df = pd.read_csv(
        MANIFEST
    )

    # --------------------------------------------------------
    # Explicit development-only dataframe.
    # --------------------------------------------------------

    development_df = df[
        df["split"].isin(
            [
                "train",
                "val",
            ]
        )
    ].copy()

    if bool(
        (
            development_df["split"]
            == "test"
        ).any()
    ):
        raise RuntimeError(
            "Test row entered development data."
        )

    print()
    print(
        "TEST IMAGE ACCESS GATE: PASS"
    )

    print(
        "Only Train/Val records will be "
        "instantiated."
    )

    train_records = build_records(
        development_df,
        "train",
    )

    val_records = build_records(
        development_df,
        "val",
    )

    train_result = audit_dataset(
        name="TRAIN",
        records=train_records,
        base_seed=TRAIN_SEED,
        expected_sources=
            EXPECTED_TRAIN_SOURCES,
        expected_active=
            EXPECTED_TRAIN_ACTIVE,
        expected_attacks=
            EXPECTED_TRAIN_ATTACKS,
    )

    val_result = audit_dataset(
        name="VAL",
        records=val_records,
        base_seed=VAL_SEED,
        expected_sources=
            EXPECTED_VAL_SOURCES,
        expected_active=
            EXPECTED_VAL_ACTIVE,
        expected_attacks=
            EXPECTED_VAL_ATTACKS,
    )

    # --------------------------------------------------------
    # Final leakage check.
    # --------------------------------------------------------

    key_overlap = (
        train_result[
            "sample_keys"
        ]
        &
        val_result[
            "sample_keys"
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

    print()
    print("=" * 78)
    print(
        "FINAL TRAIN / VAL LEAKAGE CHECK"
    )
    print("=" * 78)

    print(
        "Canonical-key overlap :",
        len(key_overlap),
    )

    print(
        "Group overlap         :",
        len(group_overlap),
    )

    if key_overlap:
        raise RuntimeError(
            "Canonical-key leakage."
        )

    if group_overlap:
        raise RuntimeError(
            "Group leakage."
        )

    print()
    print("=" * 78)
    print(
        "E5.3R-4A.2 FULL ACTIVE DATASET "
        "INTEGRATION GATE: PASS"
    )
    print("=" * 78)

    print()
    print(
        "The real training data pipeline has now "
        "executed:"
    )

    print(
        "authentic image -> canonical key -> payload "
        "-> V3.3 embedding -> controlled attack -> "
        "received/mask/label"
    )

    print()
    print(
        "No detector/localization head was trained."
    )

    print(
        "No Test image was loaded."
    )


if __name__ == "__main__":
    main()
