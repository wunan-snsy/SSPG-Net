from pathlib import Path
from collections import Counter

import pandas as pd

from preprocessing.active_sspg_dataset import (
    ActiveSSPGDataset,
)


# ============================================================
# E5.3R-4C-4-1
# FORMAL LOAO SPLIT GATE
#
# PURPOSE
# ------------------------------------------------------------
# Construct the FINAL leave-one-attack-out development protocol
# from the already frozen ActiveSSPGDataset assignment.
#
# For each fold:
#
#   Train:
#       ONLY source pairs assigned to the two seen attacks
#
#   Val-Calibration:
#       ONLY Val source pairs assigned to the two seen attacks
#       Used later for:
#           - early stopping
#           - checkpoint selection
#           - detection threshold calibration
#
#   Held-Out Evaluation:
#       ONLY Val source pairs assigned to the held-out attack
#       NEVER used for:
#           - gradient updates
#           - early stopping
#           - checkpoint selection
#           - threshold calibration
#
# NO MODEL
# NO TRAINING
# NO OPTIMIZER
# NO CHECKPOINT
# NO TEST
# ============================================================


ROOT = Path(
    __file__
).resolve().parents[1]

MANIFEST_PATH = (
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

ATTACKS = (
    "erase",
    "copy_move",
    "roll",
)

EXPECTED_TRAIN_SOURCES = 289
EXPECTED_VAL_SOURCES = 62

EXPECTED_ORIGINAL_TRAIN_ATTACKS = {
    "erase": 97,
    "copy_move": 96,
    "roll": 96,
}

EXPECTED_ORIGINAL_VAL_ATTACKS = {
    "erase": 21,
    "copy_move": 21,
    "roll": 20,
}

EXPECTED_FOLDS = {
    "erase": {
        "train_sources": 192,
        "calibration_sources": 41,
        "heldout_sources": 21,
    },
    "copy_move": {
        "train_sources": 193,
        "calibration_sources": 41,
        "heldout_sources": 21,
    },
    "roll": {
        "train_sources": 193,
        "calibration_sources": 42,
        "heldout_sources": 20,
    },
}


def normalize_split(
    value,
):
    return (
        str(value)
        .strip()
        .lower()
    )


def authentic_label(
    value,
):
    """
    Match the frozen E5.3R training convention robustly.
    """

    if pd.isna(value):
        return False

    if isinstance(
        value,
        (int, float),
    ):
        return int(value) == 0

    text = (
        str(value)
        .strip()
        .lower()
    )

    return text in {
        "0",
        "authentic",
        "real",
        "clean",
        "negative",
    }


def resolve_image_path(
    value,
):
    normalized = (
        str(value)
        .replace(
            "\\",
            "/",
        )
    )

    path = Path(
        normalized
    )

    if path.is_absolute():
        return str(
            path
        )

    return str(
        IMAGE_ROOT
        / path
    )


def build_source_records(
    dataframe,
    split_name,
):
    split_name = normalize_split(
        split_name
    )

    if split_name == "test":
        raise RuntimeError(
            "TEST ACCESS PROHIBITED."
        )

    if split_name not in {
        "train",
        "val",
    }:
        raise RuntimeError(
            f"Unsupported split: {split_name}"
        )

    split_series = (
        dataframe[
            "split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    split_df = dataframe[
        split_series
        ==
        split_name
    ].copy()

    source_df = split_df[
        split_df[
            "label"
        ].apply(
            authentic_label
        )
    ].copy()

    # --------------------------------------------------------
    # IMPORTANT:
    # Must reproduce frozen canonical source ordering.
    # --------------------------------------------------------

    source_df = (
        source_df
        .sort_values(
            by=[
                "group",
                "sample_id",
                "image_path",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    records = []

    for _, row in source_df.iterrows():
        records.append(
            {
                "sample_id":
                    str(
                        row[
                            "sample_id"
                        ]
                    ),

                "group":
                    str(
                        row[
                            "group"
                        ]
                    ),

                "image_path":
                    resolve_image_path(
                        row[
                            "image_path"
                        ]
                    ),
            }
        )

    return records


def source_identity(
    record,
):
    return (
        str(
            record[
                "group"
            ]
        ),
        str(
            record[
                "sample_id"
            ]
        ),
        str(
            record[
                "image_path"
            ]
        ),
    )


def get_original_attack_assignment(
    records,
    base_seed,
):
    """
    Recover the attack assigned by the frozen
    ActiveSSPGDataset.

    We inspect only each manipulated member:
        index = 2 * source_index + 1

    Train/Val images may be read.
    Test is never constructed.
    """

    dataset = ActiveSSPGDataset(
        records=records,
        base_seed=base_seed,
    )

    if len(dataset) != (
        2
        * len(records)
    ):
        raise RuntimeError(
            "Unexpected ActiveSSPGDataset length."
        )

    assignments = []

    for source_index in range(
        len(records)
    ):
        manipulated = dataset[
            2
            * source_index
            + 1
        ]

        attack = (
            str(
                manipulated[
                    "attack"
                ]
            )
            .strip()
            .lower()
        )

        if attack not in ATTACKS:
            raise RuntimeError(
                "Unexpected frozen attack: "
                f"{attack}"
            )

        if int(
            manipulated[
                "detection_label"
            ].item()
        ) != 1:
            raise RuntimeError(
                "Frozen manipulated label != 1."
            )

        if float(
            manipulated[
                "mask"
            ].sum().item()
        ) <= 0.0:
            raise RuntimeError(
                "Frozen manipulated mask is empty."
            )

        assignments.append(
            attack
        )

    return assignments


def partition_records(
    records,
    assignments,
    held_out_attack,
):
    seen_records = []
    heldout_records = []

    for record, attack in zip(
        records,
        assignments,
    ):
        if attack == held_out_attack:
            heldout_records.append(
                record
            )
        else:
            seen_records.append(
                record
            )

    return (
        seen_records,
        heldout_records,
    )


def attack_counts_for_records(
    records,
    assignments,
    selected_records,
):
    attack_by_identity = {
        source_identity(record):
            attack
        for record, attack in zip(
            records,
            assignments,
        )
    }

    counter = Counter()

    for record in selected_records:
        identity = source_identity(
            record
        )

        counter[
            attack_by_identity[
                identity
            ]
        ] += 1

    return counter


def assert_disjoint(
    name_a,
    records_a,
    name_b,
    records_b,
):
    identities_a = {
        source_identity(
            record
        )
        for record in records_a
    }

    identities_b = {
        source_identity(
            record
        )
        for record in records_b
    }

    overlap = (
        identities_a
        &
        identities_b
    )

    if overlap:
        raise RuntimeError(
            f"Source leakage between "
            f"{name_a} and {name_b}: "
            f"{len(overlap)}"
        )


def main():
    print()
    print("=" * 78)
    print(
        "E5.3R-4C-4-1 "
        "FORMAL LOAO SPLIT GATE"
    )
    print("=" * 78)

    print()
    print(
        "No model."
    )
    print(
        "No training."
    )
    print(
        "No optimizer."
    )
    print(
        "No checkpoint."
    )
    print(
        "Test is prohibited."
    )

    if not MANIFEST_PATH.exists():
        raise RuntimeError(
            f"Manifest not found: "
            f"{MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    required = {
        "sample_id",
        "group",
        "split",
        "label",
        "image_path",
    }

    missing = (
        required
        -
        set(
            manifest.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Manifest missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    # ========================================================
    # HARD TEST EXCLUSION
    # ========================================================

    development = manifest[
        manifest[
            "split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "train",
                "val",
            ]
        )
    ].copy()

    if (
        development[
            "split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(
            "test"
        )
        .any()
    ):
        raise RuntimeError(
            "Test leakage into development dataframe."
        )

    train_records = (
        build_source_records(
            development,
            "train",
        )
    )

    val_records = (
        build_source_records(
            development,
            "val",
        )
    )

    if len(
        train_records
    ) != EXPECTED_TRAIN_SOURCES:
        raise RuntimeError(
            "Train source count mismatch: "
            f"{len(train_records)}"
        )

    if len(
        val_records
    ) != EXPECTED_VAL_SOURCES:
        raise RuntimeError(
            "Val source count mismatch: "
            f"{len(val_records)}"
        )

    assert_disjoint(
        "Train",
        train_records,
        "Val",
        val_records,
    )

    print()
    print(
        "Recovering frozen Train attack assignment..."
    )

    train_assignment = (
        get_original_attack_assignment(
            train_records,
            TRAIN_SEED,
        )
    )

    print(
        "Recovering frozen Val attack assignment..."
    )

    val_assignment = (
        get_original_attack_assignment(
            val_records,
            VAL_SEED,
        )
    )

    train_original_counts = Counter(
        train_assignment
    )

    val_original_counts = Counter(
        val_assignment
    )

    print()
    print("-" * 78)
    print(
        "FROZEN ORIGINAL ASSIGNMENT"
    )
    print("-" * 78)

    print(
        "Train:",
        dict(
            train_original_counts
        ),
    )

    print(
        "Val  :",
        dict(
            val_original_counts
        ),
    )

    if dict(
        train_original_counts
    ) != EXPECTED_ORIGINAL_TRAIN_ATTACKS:
        raise RuntimeError(
            "Frozen Train attack assignment "
            "does not match E5.3R-4A."
        )

    if dict(
        val_original_counts
    ) != EXPECTED_ORIGINAL_VAL_ATTACKS:
        raise RuntimeError(
            "Frozen Val attack assignment "
            "does not match E5.3R-4A."
        )

    # ========================================================
    # FORMAL LOAO FOLDS
    # ========================================================

    for held_out_attack in ATTACKS:
        expected = EXPECTED_FOLDS[
            held_out_attack
        ]

        (
            fold_train_records,
            train_heldout_records,
        ) = partition_records(
            train_records,
            train_assignment,
            held_out_attack,
        )

        (
            calibration_records,
            heldout_eval_records,
        ) = partition_records(
            val_records,
            val_assignment,
            held_out_attack,
        )

        # ----------------------------------------------------
        # Train must exclude held-out sources entirely.
        # ----------------------------------------------------

        if len(
            fold_train_records
        ) != expected[
            "train_sources"
        ]:
            raise RuntimeError(
                f"{held_out_attack}: "
                "Train source count mismatch."
            )

        if len(
            calibration_records
        ) != expected[
            "calibration_sources"
        ]:
            raise RuntimeError(
                f"{held_out_attack}: "
                "Calibration source count mismatch."
            )

        if len(
            heldout_eval_records
        ) != expected[
            "heldout_sources"
        ]:
            raise RuntimeError(
                f"{held_out_attack}: "
                "Held-out source count mismatch."
            )

        # ----------------------------------------------------
        # Internal source separation.
        # ----------------------------------------------------

        assert_disjoint(
            "Fold Train",
            fold_train_records,
            "Calibration",
            calibration_records,
        )

        assert_disjoint(
            "Fold Train",
            fold_train_records,
            "Held-Out Evaluation",
            heldout_eval_records,
        )

        assert_disjoint(
            "Calibration",
            calibration_records,
            "Held-Out Evaluation",
            heldout_eval_records,
        )

        train_counts = (
            attack_counts_for_records(
                train_records,
                train_assignment,
                fold_train_records,
            )
        )

        calibration_counts = (
            attack_counts_for_records(
                val_records,
                val_assignment,
                calibration_records,
            )
        )

        heldout_counts = (
            attack_counts_for_records(
                val_records,
                val_assignment,
                heldout_eval_records,
            )
        )

        # ----------------------------------------------------
        # Core attack-isolation assertions.
        # ----------------------------------------------------

        if train_counts[
            held_out_attack
        ] != 0:
            raise RuntimeError(
                f"{held_out_attack}: "
                "held-out attack leaked into Train."
            )

        if calibration_counts[
            held_out_attack
        ] != 0:
            raise RuntimeError(
                f"{held_out_attack}: "
                "held-out attack leaked into Calibration."
            )

        if heldout_counts[
            held_out_attack
        ] != len(
            heldout_eval_records
        ):
            raise RuntimeError(
                f"{held_out_attack}: "
                "Held-Out Evaluation contains "
                "non-held-out attack."
            )

        for attack in ATTACKS:
            if (
                attack != held_out_attack
                and heldout_counts[
                    attack
                ] != 0
            ):
                raise RuntimeError(
                    f"{held_out_attack}: "
                    "seen attack leaked into "
                    "Held-Out Evaluation."
                )

        seen_attacks = [
            attack
            for attack in ATTACKS
            if attack != held_out_attack
        ]

        for attack in seen_attacks:
            if train_counts[
                attack
            ] <= 0:
                raise RuntimeError(
                    f"{held_out_attack}: "
                    f"Train missing {attack}."
                )

            if calibration_counts[
                attack
            ] <= 0:
                raise RuntimeError(
                    f"{held_out_attack}: "
                    f"Calibration missing {attack}."
                )

        # ----------------------------------------------------
        # Print frozen fold.
        # ----------------------------------------------------

        print()
        print("=" * 78)
        print(
            "HELD-OUT ATTACK:",
            held_out_attack.upper(),
        )
        print("=" * 78)

        print(
            "Seen attacks          :",
            tuple(
                seen_attacks
            ),
        )

        print()
        print(
            "TRAIN"
        )
        print(
            "Sources               :",
            len(
                fold_train_records
            ),
        )
        print(
            "Active samples        :",
            2
            * len(
                fold_train_records
            ),
        )
        print(
            "Authentic             :",
            len(
                fold_train_records
            ),
        )
        print(
            "Manipulated           :",
            len(
                fold_train_records
            ),
        )
        print(
            "Attack counts         :",
            dict(
                train_counts
            ),
        )

        print()
        print(
            "VAL-CALIBRATION"
        )
        print(
            "Sources               :",
            len(
                calibration_records
            ),
        )
        print(
            "Active samples        :",
            2
            * len(
                calibration_records
            ),
        )
        print(
            "Authentic             :",
            len(
                calibration_records
            ),
        )
        print(
            "Manipulated           :",
            len(
                calibration_records
            ),
        )
        print(
            "Attack counts         :",
            dict(
                calibration_counts
            ),
        )

        print()
        print(
            "HELD-OUT EVALUATION"
        )
        print(
            "Sources               :",
            len(
                heldout_eval_records
            ),
        )
        print(
            "Active samples        :",
            2
            * len(
                heldout_eval_records
            ),
        )
        print(
            "Authentic             :",
            len(
                heldout_eval_records
            ),
        )
        print(
            "Manipulated           :",
            len(
                heldout_eval_records
            ),
        )
        print(
            "Attack counts         :",
            dict(
                heldout_counts
            ),
        )

        print()
        print(
            "Held-out in Train      : 0"
        )
        print(
            "Held-out in Calibration: 0"
        )
        print(
            "Seen in Held-Out Eval  : 0"
        )
        print(
            "Source overlap         : 0"
        )
        print(
            "Test accessed          : False"
        )
        print(
            "STATUS                 : PASS"
        )

    print()
    print("=" * 78)
    print(
        "E5.3R-4C-4-1 "
        "FORMAL LOAO SPLIT GATE: PASS"
    )
    print("=" * 78)

    print()
    print(
        "Formal rule frozen:"
    )
    print(
        "  Train = seen attacks only."
    )
    print(
        "  Val-Calibration = seen attacks only."
    )
    print(
        "  Held-Out Evaluation = unseen attack only."
    )
    print(
        "  Held-Out Evaluation cannot select "
        "checkpoint or threshold."
    )
    print(
        "  Test remains untouched."
    )


if __name__ == "__main__":
    main()
