from pathlib import Path
import sys
from collections import Counter

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from preprocessing.active_sspg_dataset import (
    ActiveSSPGDataset,
    canonical_sample_key,
)

from training.e53r_training_protocol_v1 import (
    PROTOCOL,
)

from training.train_e53r_joint_v1 import (
    MANIFEST_PATH,
    build_source_records,
)


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


def main():

    print("=" * 78)
    print(
        "E5.3R-4C-1 FORMAL TRAINING "
        "PREFLIGHT"
    )
    print("=" * 78)

    print("Manifest :", MANIFEST_PATH)
    print("Train seed:", PROTOCOL.train_seed)
    print("Val seed  :", PROTOCOL.val_seed)

    if not MANIFEST_PATH.is_file():
        raise RuntimeError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    # ========================================================
    # Manifest
    # ========================================================

    df = pd.read_csv(
        MANIFEST_PATH
    )

    print()
    print("-" * 78)
    print("MANIFEST")
    print("-" * 78)

    print("Rows :", len(df))

    split_counts = (
        df["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts()
        .to_dict()
    )

    print(
        "Split counts :",
        split_counts,
    )

    if len(df) != 2422:
        raise RuntimeError(
            "Clean manifest must contain "
            "2422 rows."
        )

    # ========================================================
    # Explicit development-only dataframe
    # ========================================================

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
            ==
            "test"
        ).any()
    ):
        raise RuntimeError(
            "Test row entered development dataframe."
        )

    print()
    print("TEST ROW IN DEVELOPMENT DATA : NO")

    # ========================================================
    # Build source records using the SAME functions that
    # formal training will use.
    # ========================================================

    train_records = build_source_records(
        development_df,
        "train",
    )

    val_records = build_source_records(
        development_df,
        "val",
    )

    print()
    print("-" * 78)
    print("SOURCE RECORDS")
    print("-" * 78)

    print(
        "Train authentic sources :",
        len(train_records),
    )

    print(
        "Val authentic sources   :",
        len(val_records),
    )

    if (
        len(train_records)
        !=
        EXPECTED_TRAIN_SOURCES
    ):
        raise RuntimeError(
            "Train source count mismatch."
        )

    if (
        len(val_records)
        !=
        EXPECTED_VAL_SOURCES
    ):
        raise RuntimeError(
            "Val source count mismatch."
        )

    # ========================================================
    # Every resolved source path must exist.
    #
    # This checks paths only.
    # It does NOT open/decode the images.
    # ========================================================

    missing_train = [
        record["image_path"]
        for record in train_records
        if not Path(
            record["image_path"]
        ).is_file()
    ]

    missing_val = [
        record["image_path"]
        for record in val_records
        if not Path(
            record["image_path"]
        ).is_file()
    ]

    print()
    print(
        "Missing Train paths :",
        len(missing_train),
    )

    print(
        "Missing Val paths   :",
        len(missing_val),
    )

    if missing_train:
        raise RuntimeError(
            "Missing Train source path."
        )

    if missing_val:
        raise RuntimeError(
            "Missing Val source path."
        )

    # ========================================================
    # Identity / leakage
    # ========================================================

    train_keys = {
        canonical_sample_key(record)
        for record in train_records
    }

    val_keys = {
        canonical_sample_key(record)
        for record in val_records
    }

    train_groups = {
        str(record["group"])
        for record in train_records
    }

    val_groups = {
        str(record["group"])
        for record in val_records
    }

    train_paths = {
        str(
            Path(
                record["image_path"]
            ).resolve()
        )
        for record in train_records
    }

    val_paths = {
        str(
            Path(
                record["image_path"]
            ).resolve()
        )
        for record in val_records
    }

    key_overlap = (
        train_keys
        &
        val_keys
    )

    group_overlap = (
        train_groups
        &
        val_groups
    )

    path_overlap = (
        train_paths
        &
        val_paths
    )

    print()
    print("-" * 78)
    print("TRAIN / VAL LEAKAGE")
    print("-" * 78)

    print(
        "Canonical-key overlap :",
        len(key_overlap),
    )

    print(
        "Group overlap         :",
        len(group_overlap),
    )

    print(
        "Path overlap          :",
        len(path_overlap),
    )

    if key_overlap:
        raise RuntimeError(
            "Canonical-key leakage."
        )

    if group_overlap:
        raise RuntimeError(
            "Group leakage."
        )

    if path_overlap:
        raise RuntimeError(
            "Path leakage."
        )

    # ========================================================
    # Active datasets
    #
    # Constructor only.
    # __getitem__ is NOT called.
    # Therefore images are NOT loaded.
    # ========================================================

    train_dataset = ActiveSSPGDataset(
        records=train_records,
        base_seed=PROTOCOL.train_seed,
    )

    val_dataset = ActiveSSPGDataset(
        records=val_records,
        base_seed=PROTOCOL.val_seed,
    )

    print()
    print("-" * 78)
    print("ACTIVE DATASET")
    print("-" * 78)

    print(
        "Train active samples :",
        len(train_dataset),
    )

    print(
        "Val active samples   :",
        len(val_dataset),
    )

    if (
        len(train_dataset)
        !=
        EXPECTED_TRAIN_ACTIVE
    ):
        raise RuntimeError(
            "Train active count mismatch."
        )

    if (
        len(val_dataset)
        !=
        EXPECTED_VAL_ACTIVE
    ):
        raise RuntimeError(
            "Val active count mismatch."
        )

    # ========================================================
    # Attack assignment
    # ========================================================

    train_attack_counts = dict(
        Counter(
            train_dataset.attack_assignment
        )
    )

    val_attack_counts = dict(
        Counter(
            val_dataset.attack_assignment
        )
    )

    print()
    print(
        "Train attacks :",
        train_attack_counts,
    )

    print(
        "Val attacks   :",
        val_attack_counts,
    )

    if (
        train_attack_counts
        !=
        EXPECTED_TRAIN_ATTACKS
    ):
        raise RuntimeError(
            "Train attack assignment mismatch."
        )

    if (
        val_attack_counts
        !=
        EXPECTED_VAL_ATTACKS
    ):
        raise RuntimeError(
            "Val attack assignment mismatch."
        )

    # ========================================================
    # Canonical identity sanity
    # ========================================================

    if len(train_keys) != 289:
        raise RuntimeError(
            "Train canonical keys are not unique."
        )

    if len(val_keys) != 62:
        raise RuntimeError(
            "Val canonical keys are not unique."
        )

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 78)

    print(
        "E5.3R-4C-1 FORMAL TRAINING "
        "PREFLIGHT: PASS"
    )

    print("=" * 78)

    print()
    print(
        "Train sources       : 289"
    )

    print(
        "Train active samples: 578"
    )

    print(
        "Val sources         : 62"
    )

    print(
        "Val active samples  : 124"
    )

    print()

    print(
        "No dataset __getitem__ was called."
    )

    print(
        "No image was decoded."
    )

    print(
        "No model was instantiated."
    )

    print(
        "No optimizer was created."
    )

    print(
        "No optimizer step was executed."
    )

    print(
        "No validation performance "
        "metric was observed."
    )

    print(
        "No Test image was accessed."
    )


if __name__ == "__main__":
    main()