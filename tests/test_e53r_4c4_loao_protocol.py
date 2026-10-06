from pathlib import Path
from collections import Counter

import pandas as pd
import torch

from preprocessing.active_sspg_loao_dataset import (
    ALL_ATTACKS,
    LOAOActiveSSPGDataset,
    seen_attacks_for,
)


# ============================================================
# E5.3R-4C-4-0
# LEAVE-ONE-ATTACK-OUT PROTOCOL SAFETY GATE
#
# NO TRAINING
# NO OPTIMIZER
# NO MODEL
# NO CHECKPOINT
# NO TEST ACCESS
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

EXPECTED_TRAIN_SOURCES = 289
EXPECTED_VAL_SOURCES = 62

EXPECTED_TRAIN_ACTIVE = 578
EXPECTED_VAL_ACTIVE = 124


def resolve_image_path(
    value,
):
    normalized = str(
        value
    ).replace(
        "\\",
        "/",
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


def build_records(
    dataframe,
    split,
):
    subset = dataframe[
        (
            dataframe[
                "split"
            ]
            .astype(str)
            .str.lower()
            == split
        )
        &
        (
            pd.to_numeric(
                dataframe[
                    "label"
                ],
                errors="raise",
            )
            == 0
        )
    ].copy()

    subset = subset.sort_values(
        by=[
            "group",
            "sample_id",
            "image_path",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    records = []

    for _, row in subset.iterrows():
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


def count_dataset(
    dataset,
):
    counts = Counter()

    labels = Counter()

    sample_keys = []

    for index in range(
        len(dataset)
    ):
        sample = dataset[
            index
        ]

        attack = str(
            sample[
                "attack"
            ]
        )

        label = int(
            sample[
                "detection_label"
            ].item()
        )

        counts[
            attack
        ] += 1

        labels[
            label
        ] += 1

        sample_keys.append(
            str(
                sample[
                    "sample_key"
                ]
            )
        )

        if label == 0:
            if attack != "authentic":
                raise RuntimeError(
                    "Label-0 sample is not authentic."
                )

            if float(
                sample[
                    "mask"
                ].sum().item()
            ) != 0.0:
                raise RuntimeError(
                    "Authentic mask is not zero."
                )

        elif label == 1:
            if attack == "authentic":
                raise RuntimeError(
                    "Label-1 sample marked authentic."
                )

            if float(
                sample[
                    "mask"
                ].sum().item()
            ) <= 0.0:
                raise RuntimeError(
                    "Manipulated mask is empty."
                )

        else:
            raise RuntimeError(
                f"Unexpected label: {label}"
            )

    return (
        counts,
        labels,
        sample_keys,
    )


def verify_pairing(
    dataset,
):
    for source_index in range(
        len(
            dataset.records
        )
    ):
        authentic = dataset[
            2
            * source_index
        ]

        manipulated = dataset[
            2
            * source_index
            + 1
        ]

        if str(
            authentic[
                "sample_key"
            ]
        ) != str(
            manipulated[
                "sample_key"
            ]
        ):
            raise RuntimeError(
                "Pair sample_key mismatch."
            )

        if not torch.equal(
            authentic[
                "payload"
            ],
            manipulated[
                "payload"
            ],
        ):
            raise RuntimeError(
                "Pair payload mismatch."
            )

        if int(
            authentic[
                "source_seed"
            ]
        ) != int(
            manipulated[
                "source_seed"
            ]
        ):
            raise RuntimeError(
                "Pair source seed mismatch."
            )


def main():
    print()
    print("=" * 76)
    print(
        "E5.3R-4C-4-0 "
        "LOAO PROTOCOL SAFETY GATE"
    )
    print("=" * 76)

    print()
    print(
        "No training."
    )

    print(
        "No model."
    )

    print(
        "No checkpoint."
    )

    print(
        "Test is excluded before dataset construction."
    )

    if not MANIFEST_PATH.exists():
        raise RuntimeError(
            f"Manifest missing: {MANIFEST_PATH}"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    required = {
        "split",
        "label",
        "group",
        "sample_id",
        "image_path",
    }

    missing = (
        required
        - set(
            manifest.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # Explicit Test exclusion.
    # --------------------------------------------------------

    development = manifest[
        manifest[
            "split"
        ]
        .astype(str)
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
        build_records(
            development,
            "train",
        )
    )

    val_records = (
        build_records(
            development,
            "val",
        )
    )

    if (
        len(
            train_records
        )
        != EXPECTED_TRAIN_SOURCES
    ):
        raise RuntimeError(
            "Unexpected Train source count: "
            f"{len(train_records)}"
        )

    if (
        len(
            val_records
        )
        != EXPECTED_VAL_SOURCES
    ):
        raise RuntimeError(
            "Unexpected Val source count: "
            f"{len(val_records)}"
        )

    train_keys = {
        (
            r[
                "group"
            ],
            r[
                "sample_id"
            ],
            r[
                "image_path"
            ],
        )
        for r in train_records
    }

    val_keys = {
        (
            r[
                "group"
            ],
            r[
                "sample_id"
            ],
            r[
                "image_path"
            ],
        )
        for r in val_records
    }

    if (
        train_keys
        & val_keys
    ):
        raise RuntimeError(
            "Train/Val source leakage detected."
        )

    overall_pass = True

    for held_out in ALL_ATTACKS:
        print()
        print("=" * 76)
        print(
            f"LOAO HELD-OUT = "
            f"{held_out.upper()}"
        )
        print("=" * 76)

        seen = seen_attacks_for(
            held_out
        )

        print(
            "Train allowed attacks : "
            f"{seen}"
        )

        print(
            "Validation attack     : "
            f"{held_out}"
        )

        train_dataset = (
            LOAOActiveSSPGDataset(
                records=train_records,
                base_seed=TRAIN_SEED,
                held_out_attack=held_out,
                mode="train",
            )
        )

        val_dataset = (
            LOAOActiveSSPGDataset(
                records=val_records,
                base_seed=VAL_SEED,
                held_out_attack=held_out,
                mode="val",
            )
        )

        if (
            len(
                train_dataset
            )
            != EXPECTED_TRAIN_ACTIVE
        ):
            raise RuntimeError(
                "Unexpected Train active count."
            )

        if (
            len(
                val_dataset
            )
            != EXPECTED_VAL_ACTIVE
        ):
            raise RuntimeError(
                "Unexpected Val active count."
            )

        (
            train_counts,
            train_labels,
            _,
        ) = count_dataset(
            train_dataset
        )

        (
            val_counts,
            val_labels,
            _,
        ) = count_dataset(
            val_dataset
        )

        verify_pairing(
            train_dataset
        )

        verify_pairing(
            val_dataset
        )

        print()
        print(
            f"Train sources       : "
            f"{len(train_records)}"
        )

        print(
            f"Train active        : "
            f"{len(train_dataset)}"
        )

        print(
            f"Train authentic     : "
            f"{train_counts['authentic']}"
        )

        for attack in ALL_ATTACKS:
            print(
                f"Train {attack:13s}: "
                f"{train_counts[attack]}"
            )

        print()
        print(
            f"Val sources         : "
            f"{len(val_records)}"
        )

        print(
            f"Val active          : "
            f"{len(val_dataset)}"
        )

        print(
            f"Val authentic       : "
            f"{val_counts['authentic']}"
        )

        for attack in ALL_ATTACKS:
            print(
                f"Val {attack:15s}: "
                f"{val_counts[attack]}"
            )

        print()
        print(
            "Train labels 0/1    : "
            f"{train_labels[0]}/"
            f"{train_labels[1]}"
        )

        print(
            "Val labels 0/1      : "
            f"{val_labels[0]}/"
            f"{val_labels[1]}"
        )

        # ----------------------------------------------------
        # Core LOAO assertions.
        # ----------------------------------------------------

        if (
            train_counts[
                held_out
            ]
            != 0
        ):
            raise RuntimeError(
                "HELD-OUT ATTACK LEAKED INTO TRAIN."
            )

        for attack in seen:
            if (
                train_counts[
                    attack
                ]
                <= 0
            ):
                raise RuntimeError(
                    "Seen Train attack absent."
                )

        seen_counts = [
            train_counts[
                attack
            ]
            for attack in seen
        ]

        if abs(
            seen_counts[0]
            - seen_counts[1]
        ) > 1:
            raise RuntimeError(
                "Train seen attacks are not balanced."
            )

        if (
            val_counts[
                held_out
            ]
            != EXPECTED_VAL_SOURCES
        ):
            raise RuntimeError(
                "Validation held-out count incorrect."
            )

        for attack in ALL_ATTACKS:
            if (
                attack
                != held_out
                and val_counts[
                    attack
                ]
                != 0
            ):
                raise RuntimeError(
                    "Seen attack leaked into held-out Validation."
                )

        if (
            train_labels[0]
            != EXPECTED_TRAIN_SOURCES
            or train_labels[1]
            != EXPECTED_TRAIN_SOURCES
        ):
            raise RuntimeError(
                "Train detection balance failed."
            )

        if (
            val_labels[0]
            != EXPECTED_VAL_SOURCES
            or val_labels[1]
            != EXPECTED_VAL_SOURCES
        ):
            raise RuntimeError(
                "Validation detection balance failed."
            )

        print()
        print(
            "Held-out attack in Train : 0"
        )

        print(
            "Test accessed            : False"
        )

        print(
            "Pairing integrity        : PASS"
        )

        print(
            "Detection balance        : PASS"
        )

        print(
            "Attack isolation         : PASS"
        )

        print(
            "STATUS                   : PASS"
        )

    print()
    print("=" * 76)
    print(
        "E5.3R-4C-4-0 "
        "LOAO PROTOCOL SAFETY GATE: PASS"
    )
    print("=" * 76)

    print()
    print(
        "All three held-out attacks passed."
    )

    print(
        "No Test split was constructed."
    )

    print(
        "No model training was performed."
    )


if __name__ == "__main__":
    main()
