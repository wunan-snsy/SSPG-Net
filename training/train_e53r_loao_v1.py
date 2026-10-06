# ============================================================
# E5.3R-4C-4-2
# FORMAL LEAVE-ONE-ATTACK-OUT TRAINING
#
# Protocol:
#   Train              = seen attacks only
#   Val-Calibration    = seen attacks only
#   Held-Out Evaluation= unseen attack only
#
# Held-out evaluation is NEVER used for:
#   - gradient updates
#   - checkpoint selection
#   - early stopping
#   - detection threshold calibration
#
# Test is NEVER constructed or accessed.
#
# This script intentionally reuses the already validated
# E5.3R-4C-1 training engine wherever possible.
# ============================================================

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Subset

from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

import training.train_e53r_joint_v1 as base

from preprocessing.active_sspg_dataset import (
    ActiveSSPGDataset,
)

from models.sspg_joint_model_v2 import (
    SSPGJointModelV2,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    __file__
).resolve().parents[1]

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "csv"
)

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

STEGO_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "stego_structured_v33_128bit_best.pth"
)


# ============================================================
# FROZEN PROTOCOL
# ============================================================

EXPERIMENT_ID = "E5.3R-4C-4-2-V1"

TRAIN_SEED = 20260922
VAL_SEED = 20260923
TRAINING_SEED = 20260925

ATTACKS = (
    "erase",
    "copy_move",
    "roll",
)

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


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_attack(
    attack,
):
    attack = (
        str(attack)
        .strip()
        .lower()
    )

    if attack not in ATTACKS:
        raise RuntimeError(
            f"Unsupported held-out attack: {attack}"
        )

    return attack


def set_deterministic_seed(
    seed,
):
    seed = int(seed)

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def protocol_dictionary(
    held_out_attack,
):
    held_out_attack = normalize_attack(
        held_out_attack
    )

    seen_attacks = [
        attack
        for attack in ATTACKS
        if attack != held_out_attack
    ]

    return {
        "experiment_id":
            EXPERIMENT_ID,

        "held_out_attack":
            held_out_attack,

        "seen_attacks":
            seen_attacks,

        "train_seed":
            TRAIN_SEED,

        "val_seed":
            VAL_SEED,

        "training_seed":
            TRAINING_SEED,

        "max_epochs":
            int(
                base.PROTOCOL.max_epochs
            ),

        "early_stopping_patience":
            int(
                base.PROTOCOL.early_stopping_patience
            ),

        "min_delta":
            float(
                base.PROTOCOL.min_delta
            ),

        "batch_size":
            int(
                base.PROTOCOL.batch_size
            ),

        "gradient_accumulation_steps":
            int(
                base.PROTOCOL.gradient_accumulation_steps
            ),

        "learning_rate":
            float(
                base.PROTOCOL.learning_rate
            ),

        "weight_decay":
            float(
                base.PROTOCOL.weight_decay
            ),

        "adapter_bottleneck":
            int(
                base.PROTOCOL.adapter_bottleneck
            ),

        "fusion_hidden_dim":
            int(
                base.PROTOCOL.fusion_hidden_dim
            ),

        "knn_k":
            int(
                base.PROTOCOL.knn_k
            ),

        "gnn_hidden_dim":
            int(
                base.PROTOCOL.gnn_hidden_dim
            ),

        "gnn_depth":
            int(
                base.PROTOCOL.gnn_depth
            ),

        "localization_threshold":
            float(
                base.PROTOCOL.localization_threshold
            ),

        "checkpoint_selection":
            "val_calibration_joint_loss",

        "checkpoint_selection_mode":
            "min",

        "detection_threshold_selection":
            "val_calibration_youden_j",

        "heldout_used_for_training":
            False,

        "heldout_used_for_checkpoint":
            False,

        "heldout_used_for_threshold":
            False,

        "test_accessed":
            False,
    }


def protocol_sha256(
    held_out_attack,
):
    payload = json.dumps(
        protocol_dictionary(
            held_out_attack
        ),
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


# ============================================================
# SOURCE RECORDS
# ============================================================

def load_source_records():
    if not MANIFEST_PATH.is_file():
        raise RuntimeError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    dataframe = pd.read_csv(
        MANIFEST_PATH
    )

    # --------------------------------------------------------
    # HARD TEST EXCLUSION BEFORE record construction.
    # --------------------------------------------------------

    development = dataframe[
        dataframe[
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
            "TEST leakage into development dataframe."
        )

    train_records = (
        base.build_source_records(
            development,
            "train",
        )
    )

    val_records = (
        base.build_source_records(
            development,
            "val",
        )
    )

    if len(
        train_records
    ) != 289:
        raise RuntimeError(
            "Frozen Train source count must be 289."
        )

    if len(
        val_records
    ) != 62:
        raise RuntimeError(
            "Frozen Val source count must be 62."
        )

    base.audit_train_val_sources(
        train_records,
        val_records,
    )

    return (
        train_records,
        val_records,
    )


# ============================================================
# FROZEN ACTIVE DATASETS
# ============================================================

def build_full_datasets(
    train_records,
    val_records,
):
    train_dataset = ActiveSSPGDataset(
        records=train_records,
        base_seed=TRAIN_SEED,
    )

    val_dataset = ActiveSSPGDataset(
        records=val_records,
        base_seed=VAL_SEED,
    )

    if len(
        train_dataset
    ) != 578:
        raise RuntimeError(
            "Frozen Train active count must be 578."
        )

    if len(
        val_dataset
    ) != 124:
        raise RuntimeError(
            "Frozen Val active count must be 124."
        )

    return (
        train_dataset,
        val_dataset,
    )


# ============================================================
# RECOVER FROZEN ATTACK ASSIGNMENT
# ============================================================

def recover_attack_assignment(
    dataset,
    n_sources,
):
    assignment = []

    for source_index in range(
        int(n_sources)
    ):
        manipulated_index = (
            2
            * source_index
            +
            1
        )

        sample = dataset[
            manipulated_index
        ]

        attack = normalize_attack(
            sample[
                "attack"
            ]
        )

        label = int(
            sample[
                "detection_label"
            ].item()
        )

        if label != 1:
            raise RuntimeError(
                "Manipulated member has label != 1."
            )

        if float(
            sample[
                "mask"
            ].sum().item()
        ) <= 0.0:
            raise RuntimeError(
                "Manipulated member has empty mask."
            )

        assignment.append(
            attack
        )

    return assignment


# ============================================================
# SUBSET CONSTRUCTION
# ============================================================

def source_pair_indices(
    source_indices,
):
    indices = []

    for source_index in source_indices:
        indices.append(
            2
            * int(source_index)
        )

        indices.append(
            2
            * int(source_index)
            +
            1
        )

    return indices


def build_formal_subsets(
    train_dataset,
    val_dataset,
    train_assignment,
    val_assignment,
    held_out_attack,
):
    held_out_attack = normalize_attack(
        held_out_attack
    )

    train_source_indices = [
        index
        for index, attack
        in enumerate(
            train_assignment
        )
        if attack != held_out_attack
    ]

    calibration_source_indices = [
        index
        for index, attack
        in enumerate(
            val_assignment
        )
        if attack != held_out_attack
    ]

    heldout_source_indices = [
        index
        for index, attack
        in enumerate(
            val_assignment
        )
        if attack == held_out_attack
    ]

    expected = EXPECTED_FOLDS[
        held_out_attack
    ]

    if len(
        train_source_indices
    ) != expected[
        "train_sources"
    ]:
        raise RuntimeError(
            "LOAO Train source count mismatch."
        )

    if len(
        calibration_source_indices
    ) != expected[
        "calibration_sources"
    ]:
        raise RuntimeError(
            "LOAO calibration source count mismatch."
        )

    if len(
        heldout_source_indices
    ) != expected[
        "heldout_sources"
    ]:
        raise RuntimeError(
            "LOAO held-out source count mismatch."
        )

    calibration_set = set(
        calibration_source_indices
    )

    heldout_set = set(
        heldout_source_indices
    )

    if calibration_set & heldout_set:
        raise RuntimeError(
            "Calibration/Held-out source overlap."
        )

    if (
        calibration_set
        |
        heldout_set
    ) != set(
        range(
            len(
                val_assignment
            )
        )
    ):
        raise RuntimeError(
            "Val partition does not cover all sources."
        )

    train_subset = Subset(
        train_dataset,
        source_pair_indices(
            train_source_indices
        ),
    )

    calibration_subset = Subset(
        val_dataset,
        source_pair_indices(
            calibration_source_indices
        ),
    )

    heldout_subset = Subset(
        val_dataset,
        source_pair_indices(
            heldout_source_indices
        ),
    )

    return {
        "train":
            train_subset,

        "calibration":
            calibration_subset,

        "heldout":
            heldout_subset,

        "train_source_indices":
            train_source_indices,

        "calibration_source_indices":
            calibration_source_indices,

        "heldout_source_indices":
            heldout_source_indices,
    }


# ============================================================
# SUBSET AUDIT
# ============================================================

def audit_subset(
    subset,
    allowed_manipulated_attacks,
):
    allowed_manipulated_attacks = set(
        allowed_manipulated_attacks
    )

    labels = Counter()
    attacks = Counter()

    for sample in subset:
        label = int(
            sample[
                "detection_label"
            ].item()
        )

        attack = (
            str(
                sample[
                    "attack"
                ]
            )
            .strip()
            .lower()
        )

        labels[
            label
        ] += 1

        attacks[
            attack
        ] += 1

        if label == 0:
            if attack != "authentic":
                raise RuntimeError(
                    "Authentic member attack mismatch."
                )

        elif label == 1:
            if attack not in allowed_manipulated_attacks:
                raise RuntimeError(
                    f"Unexpected manipulated attack: {attack}"
                )

        else:
            raise RuntimeError(
                f"Unexpected detection label: {label}"
            )

    if labels[
        0
    ] != labels[
        1
    ]:
        raise RuntimeError(
            "Detection class balance failed."
        )

    return (
        labels,
        attacks,
    )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint_atomic(
    path,
    model,
    optimizer,
    epoch,
    calibration_joint_loss,
    held_out_attack,
    fingerprint,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        ".tmp"
    )

    checkpoint = {
        "experiment_id":
            EXPERIMENT_ID,

        "held_out_attack":
            held_out_attack,

        "protocol_sha256":
            fingerprint,

        "selection_metric":
            "val_calibration_joint_loss",

        "selection_mode":
            "min",

        "selection_data":
            "seen_attacks_only",

        "heldout_used_for_training":
            False,

        "heldout_used_for_checkpoint":
            False,

        "heldout_used_for_threshold":
            False,

        "test_accessed":
            False,

        "epoch":
            int(epoch),

        "calibration_joint_loss":
            float(
                calibration_joint_loss
            ),

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),
    }

    torch.save(
        checkpoint,
        temporary,
    )

    temporary.replace(
        path
    )


# ============================================================
# FIXED-THRESHOLD HELD-OUT EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_fixed_threshold(
    model,
    loader,
    detection_threshold,
):
    model.eval()

    labels = []
    probabilities = []
    rows = []

    localization_tp = 0
    localization_fp = 0
    localization_fn = 0
    localization_tn = 0

    localization_threshold = float(
        base.PROTOCOL.localization_threshold
    )

    for batch in loader:
        (
            received,
            payload,
            mask,
            detection_target,
        ) = base.move_batch_to_device(
            batch
        )

        output = model(
            received,
            payload,
        )

        detection_probability = torch.sigmoid(
            output[
                "detection_logits"
            ]
        )

        localization_logits_256 = (
            model.localization_logits_256(
                output[
                    "localization_grid_logits"
                ]
            )
        )

        localization_probability = torch.sigmoid(
            localization_logits_256
        )

        localization_prediction = (
            localization_probability
            >=
            localization_threshold
        )

        localization_truth = (
            mask
            >=
            0.5
        )

        localization_tp += int(
            (
                localization_prediction
                &
                localization_truth
            ).sum().item()
        )

        localization_fp += int(
            (
                localization_prediction
                &
                ~localization_truth
            ).sum().item()
        )

        localization_fn += int(
            (
                ~localization_prediction
                &
                localization_truth
            ).sum().item()
        )

        localization_tn += int(
            (
                ~localization_prediction
                &
                ~localization_truth
            ).sum().item()
        )

        label_value = int(
            detection_target[
                0
            ].item()
        )

        probability_value = float(
            detection_probability[
                0
            ].item()
        )

        labels.append(
            label_value
        )

        probabilities.append(
            probability_value
        )

        sample_pred = (
            localization_prediction[
                0
            ]
        )

        sample_truth = (
            localization_truth[
                0
            ]
        )

        sample_tp = int(
            (
                sample_pred
                &
                sample_truth
            ).sum().item()
        )

        sample_fp = int(
            (
                sample_pred
                &
                ~sample_truth
            ).sum().item()
        )

        sample_fn = int(
            (
                ~sample_pred
                &
                sample_truth
            ).sum().item()
        )

        f1_denominator = (
            2
            * sample_tp
            +
            sample_fp
            +
            sample_fn
        )

        iou_denominator = (
            sample_tp
            +
            sample_fp
            +
            sample_fn
        )

        sample_f1 = (
            1.0
            if f1_denominator == 0
            else (
                2.0
                * sample_tp
                /
                f1_denominator
            )
        )

        sample_iou = (
            1.0
            if iou_denominator == 0
            else (
                sample_tp
                /
                iou_denominator
            )
        )

        rows.append(
            {
                "sample_key":
                    batch[
                        "sample_key"
                    ][0],

                "source_path":
                    batch[
                        "source_path"
                    ][0],

                "attack":
                    batch[
                        "attack"
                    ][0],

                "detection_label":
                    label_value,

                "detection_probability":
                    probability_value,

                "localization_f1":
                    sample_f1,

                "localization_iou":
                    sample_iou,
            }
        )

    labels_array = np.asarray(
        labels,
        dtype=np.int64,
    )

    probabilities_array = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if len(
        np.unique(
            labels_array
        )
    ) != 2:
        raise RuntimeError(
            "Held-out evaluation must contain both classes."
        )

    detection_auc = float(
        roc_auc_score(
            labels_array,
            probabilities_array,
        )
    )

    detection_prediction = (
        probabilities_array
        >=
        float(
            detection_threshold
        )
    ).astype(
        np.int64
    )

    balanced_accuracy = float(
        balanced_accuracy_score(
            labels_array,
            detection_prediction,
        )
    )

    detection_f1 = float(
        f1_score(
            labels_array,
            detection_prediction,
            zero_division=0,
        )
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            labels_array,
            detection_prediction,
            labels=[
                0,
                1,
            ],
        )
        .ravel()
        .tolist()
    )

    sensitivity = (
        float(
            tp
            /
            (
                tp
                +
                fn
            )
        )
        if (
            tp
            +
            fn
        )
        >
        0
        else float("nan")
    )

    specificity = (
        float(
            tn
            /
            (
                tn
                +
                fp
            )
        )
        if (
            tn
            +
            fp
        )
        >
        0
        else float("nan")
    )

    pixel_f1_denominator = (
        2
        * localization_tp
        +
        localization_fp
        +
        localization_fn
    )

    pixel_iou_denominator = (
        localization_tp
        +
        localization_fp
        +
        localization_fn
    )

    localization_pixel_f1 = (
        2.0
        * localization_tp
        /
        pixel_f1_denominator
        if pixel_f1_denominator
        >
        0
        else 1.0
    )

    localization_pixel_iou = (
        localization_tp
        /
        pixel_iou_denominator
        if pixel_iou_denominator
        >
        0
        else 1.0
    )

    return {
        "detection_auc":
            detection_auc,

        "detection_threshold":
            float(
                detection_threshold
            ),

        "detection_balanced_accuracy":
            balanced_accuracy,

        "detection_f1":
            detection_f1,

        "detection_sensitivity":
            sensitivity,

        "detection_specificity":
            specificity,

        "detection_tn":
            int(tn),

        "detection_fp":
            int(fp),

        "detection_fn":
            int(fn),

        "detection_tp":
            int(tp),

        "localization_threshold":
            localization_threshold,

        "localization_pixel_f1":
            float(
                localization_pixel_f1
            ),

        "localization_pixel_iou":
            float(
                localization_pixel_iou
            ),

        "localization_tp":
            int(
                localization_tp
            ),

        "localization_fp":
            int(
                localization_fp
            ),

        "localization_fn":
            int(
                localization_fn
            ),

        "localization_tn":
            int(
                localization_tn
            ),

        "prediction_rows":
            rows,
    }


# ============================================================
# OUTPUT HELPERS
# ============================================================

def output_paths(
    held_out_attack,
):
    stem = (
        f"e53r_4c4_loao_"
        f"{held_out_attack}"
    )

    return {
        "checkpoint":
            CHECKPOINT_DIR
            /
            f"{stem}_best.pth",

        "history":
            RESULT_DIR
            /
            f"{stem}_history.csv",

        "calibration_predictions":
            RESULT_DIR
            /
            f"{stem}_calibration_predictions.csv",

        "heldout_predictions":
            RESULT_DIR
            /
            f"{stem}_heldout_predictions.csv",

        "summary":
            RESULT_DIR
            /
            f"{stem}_summary.csv",

        "protocol":
            RESULT_DIR
            /
            f"{stem}_protocol.json",
    }


def save_dataframe(
    rows,
    path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        rows
    ).to_csv(
        path,
        index=False,
    )


def save_protocol(
    held_out_attack,
    fingerprint,
    path,
):
    content = protocol_dictionary(
        held_out_attack
    )

    content[
        "protocol_sha256"
    ] = fingerprint

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            content,
            handle,
            indent=2,
            sort_keys=True,
        )


# ============================================================
# PREFLIGHT
# ============================================================

def run_preflight(
    held_out_attack,
    train_dataset,
    val_dataset,
    train_assignment,
    val_assignment,
    subsets,
):
    seen_attacks = tuple(
        attack
        for attack in ATTACKS
        if attack != held_out_attack
    )

    print()
    print("=" * 78)
    print(
        "E5.3R-4C-4-2 "
        "FORMAL LOAO TRAINING PREFLIGHT"
    )
    print("=" * 78)

    print(
        "Held-out attack       :",
        held_out_attack.upper(),
    )

    print(
        "Seen attacks           :",
        seen_attacks,
    )

    print()
    print(
        "Frozen Train assignment:",
        dict(
            Counter(
                train_assignment
            )
        ),
    )

    print(
        "Frozen Val assignment  :",
        dict(
            Counter(
                val_assignment
            )
        ),
    )

    if dict(
        Counter(
            train_assignment
        )
    ) != EXPECTED_TRAIN_ATTACKS:
        raise RuntimeError(
            "Frozen Train attack assignment mismatch."
        )

    if dict(
        Counter(
            val_assignment
        )
    ) != EXPECTED_VAL_ATTACKS:
        raise RuntimeError(
            "Frozen Val attack assignment mismatch."
        )

    train_labels, train_attacks = audit_subset(
        subsets[
            "train"
        ],
        seen_attacks,
    )

    calibration_labels, calibration_attacks = audit_subset(
        subsets[
            "calibration"
        ],
        seen_attacks,
    )

    heldout_labels, heldout_attacks = audit_subset(
        subsets[
            "heldout"
        ],
        (
            held_out_attack,
        ),
    )

    expected = EXPECTED_FOLDS[
        held_out_attack
    ]

    if len(
        subsets[
            "train"
        ]
    ) != (
        2
        * expected[
            "train_sources"
        ]
    ):
        raise RuntimeError(
            "Train active count mismatch."
        )

    if len(
        subsets[
            "calibration"
        ]
    ) != (
        2
        * expected[
            "calibration_sources"
        ]
    ):
        raise RuntimeError(
            "Calibration active count mismatch."
        )

    if len(
        subsets[
            "heldout"
        ]
    ) != (
        2
        * expected[
            "heldout_sources"
        ]
    ):
        raise RuntimeError(
            "Held-out active count mismatch."
        )

    print()
    print("-" * 78)
    print(
        "TRAIN"
    )
    print("-" * 78)

    print(
        "Sources                :",
        expected[
            "train_sources"
        ],
    )

    print(
        "Active samples         :",
        len(
            subsets[
                "train"
            ]
        ),
    )

    print(
        "Labels                 :",
        dict(
            train_labels
        ),
    )

    print(
        "Attacks                :",
        dict(
            train_attacks
        ),
    )

    print()
    print("-" * 78)
    print(
        "VAL-CALIBRATION"
    )
    print("-" * 78)

    print(
        "Sources                :",
        expected[
            "calibration_sources"
        ],
    )

    print(
        "Active samples         :",
        len(
            subsets[
                "calibration"
            ]
        ),
    )

    print(
        "Labels                 :",
        dict(
            calibration_labels
        ),
    )

    print(
        "Attacks                :",
        dict(
            calibration_attacks
        ),
    )

    print()
    print("-" * 78)
    print(
        "HELD-OUT EVALUATION"
    )
    print("-" * 78)

    print(
        "Sources                :",
        expected[
            "heldout_sources"
        ],
    )

    print(
        "Active samples         :",
        len(
            subsets[
                "heldout"
            ]
        ),
    )

    print(
        "Labels                 :",
        dict(
            heldout_labels
        ),
    )

    print(
        "Attacks                :",
        dict(
            heldout_attacks
        ),
    )

    print()
    print(
        "Held-out used for training   : False"
    )

    print(
        "Held-out used for checkpoint : False"
    )

    print(
        "Held-out used for threshold  : False"
    )

    print(
        "Test accessed                : False"
    )

    print()
    print("=" * 78)
    print(
        "PREFLIGHT: PASS"
    )
    print("=" * 78)


# ============================================================
# MODEL
# ============================================================

def build_model():
    model = SSPGJointModelV2(
        stego_checkpoint=str(
            STEGO_CHECKPOINT
        ),

        adapter_bottleneck=(
            base.PROTOCOL.adapter_bottleneck
        ),

        fusion_hidden_dim=(
            base.PROTOCOL.fusion_hidden_dim
        ),

        knn_k=(
            base.PROTOCOL.knn_k
        ),

        gnn_hidden_dim=(
            base.PROTOCOL.gnn_hidden_dim
        ),

        gnn_depth=(
            base.PROTOCOL.gnn_depth
        ),
    ).to(
        base.DEVICE
    )

    decoder_trainable = sum(
        parameter.numel()
        for parameter
        in (
            model
            .frontend
            .frontend
            .decoder
            .parameters()
        )
        if parameter.requires_grad
    )

    vit_trainable = sum(
        parameter.numel()
        for parameter
        in (
            model
            .frontend
            .frontend
            .visual
            .vit
            .parameters()
        )
        if parameter.requires_grad
    )

    if decoder_trainable != 0:
        raise RuntimeError(
            "Frozen decoder became trainable."
        )

    if vit_trainable != 0:
        raise RuntimeError(
            "Frozen ViT became trainable."
        )

    return model


# ============================================================
# FORMAL TRAINING
# ============================================================

def run_training(
    held_out_attack,
    subsets,
):
    paths = output_paths(
        held_out_attack
    )

    fingerprint = protocol_sha256(
        held_out_attack
    )

    save_protocol(
        held_out_attack,
        fingerprint,
        paths[
            "protocol"
        ],
    )

    # --------------------------------------------------------
    # Important:
    # Held-out loader exists, but is NOT touched inside epoch
    # loop and NOT passed to checkpoint selection.
    # --------------------------------------------------------

    train_loader = base.build_loader(
        subsets[
            "train"
        ],
        shuffle=True,
        seed=TRAINING_SEED,
    )

    calibration_loader = base.build_loader(
        subsets[
            "calibration"
        ],
        shuffle=False,
        seed=TRAINING_SEED,
    )

    heldout_loader = base.build_loader(
        subsets[
            "heldout"
        ],
        shuffle=False,
        seed=TRAINING_SEED,
    )

    set_deterministic_seed(
        TRAINING_SEED
    )

    print()
    print("=" * 78)
    print(
        "MODEL INITIALIZATION"
    )
    print("=" * 78)

    model = build_model()

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Device               :",
        base.DEVICE,
    )

    print(
        "Total parameters     :",
        f"{total_parameters:,}",
    )

    print(
        "Trainable parameters :",
        f"{trainable_parameters:,}",
    )

    optimizer = base.build_optimizer(
        model
    )

    print(
        "Optimizer            :",
        optimizer.__class__.__name__,
    )

    print(
        "Learning rate        :",
        optimizer.param_groups[
            0
        ][
            "lr"
        ],
    )

    print(
        "Weight decay         :",
        optimizer.param_groups[
            0
        ][
            "weight_decay"
        ],
    )

    print(
        "Gradient accumulation:",
        base.PROTOCOL.gradient_accumulation_steps,
    )

    print()
    print("=" * 78)
    print(
        "FORMAL LOAO TRAINING START"
    )
    print("=" * 78)

    history = []

    best_calibration_loss = float(
        "inf"
    )

    best_epoch = None

    epochs_without_improvement = 0

    training_start = time.time()

    for epoch in range(
        1,
        base.PROTOCOL.max_epochs
        +
        1,
    ):
        print()
        print(
            f"Epoch "
            f"{epoch:02d}"
            f"/"
            f"{base.PROTOCOL.max_epochs:02d}"
        )

        print(
            "-" * 78
        )

        train_metrics = base.train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch,
        )

        # ----------------------------------------------------
        # ONLY seen-attack calibration is evaluated here.
        # Held-out loader is NOT touched.
        # ----------------------------------------------------

        calibration_metrics = (
            base.validate_one_epoch(
                model,
                calibration_loader,
            )
        )

        print()
        print(
            "Train Joint Loss       :",
            f"{train_metrics['total_loss']:.8f}",
        )

        print(
            "Calibration Joint Loss :",
            f"{calibration_metrics['total_loss']:.8f}",
        )

        print(
            "Train Det Loss         :",
            f"{train_metrics['detection_loss']:.8f}",
        )

        print(
            "Calibration Det Loss   :",
            f"{calibration_metrics['detection_loss']:.8f}",
        )

        print(
            "Train Loc Loss         :",
            f"{train_metrics['localization_loss']:.8f}",
        )

        print(
            "Calibration Loc Loss   :",
            f"{calibration_metrics['localization_loss']:.8f}",
        )

        print(
            "Optimizer steps        :",
            train_metrics[
                "optimizer_steps"
            ],
        )

        improved = (
            calibration_metrics[
                "total_loss"
            ]
            <
            (
                best_calibration_loss
                -
                base.PROTOCOL.min_delta
            )
        )

        if improved:
            best_calibration_loss = float(
                calibration_metrics[
                    "total_loss"
                ]
            )

            best_epoch = int(
                epoch
            )

            epochs_without_improvement = 0

            save_checkpoint_atomic(
                path=paths[
                    "checkpoint"
                ],
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                calibration_joint_loss=(
                    best_calibration_loss
                ),
                held_out_attack=(
                    held_out_attack
                ),
                fingerprint=(
                    fingerprint
                ),
            )

            print(
                "Checkpoint              : SAVED"
            )

        else:
            epochs_without_improvement += 1

            print(
                "Checkpoint              : not saved"
            )

        history.append(
            {
                "epoch":
                    epoch,

                "train_joint_loss":
                    train_metrics[
                        "total_loss"
                    ],

                "train_detection_loss":
                    train_metrics[
                        "detection_loss"
                    ],

                "train_localization_loss":
                    train_metrics[
                        "localization_loss"
                    ],

                "train_localization_bce":
                    train_metrics[
                        "localization_bce"
                    ],

                "train_localization_dice":
                    train_metrics[
                        "localization_dice"
                    ],

                "calibration_joint_loss":
                    calibration_metrics[
                        "total_loss"
                    ],

                "calibration_detection_loss":
                    calibration_metrics[
                        "detection_loss"
                    ],

                "calibration_localization_loss":
                    calibration_metrics[
                        "localization_loss"
                    ],

                "calibration_localization_bce":
                    calibration_metrics[
                        "localization_bce"
                    ],

                "calibration_localization_dice":
                    calibration_metrics[
                        "localization_dice"
                    ],

                "optimizer_steps":
                    train_metrics[
                        "optimizer_steps"
                    ],

                "epoch_seconds":
                    train_metrics[
                        "seconds"
                    ],

                "checkpoint_saved":
                    int(
                        improved
                    ),
            }
        )

        save_dataframe(
            history,
            paths[
                "history"
            ],
        )

        print(
            "Best epoch              :",
            best_epoch,
        )

        print(
            "Best Calibration Loss   :",
            f"{best_calibration_loss:.8f}",
        )

        print(
            "No-improve count        :",
            epochs_without_improvement,
            "/",
            base.PROTOCOL.early_stopping_patience,
        )

        if (
            epochs_without_improvement
            >=
            base.PROTOCOL.early_stopping_patience
        ):
            print()
            print(
                "Early stopping triggered."
            )

            break

    total_training_seconds = (
        time.time()
        -
        training_start
    )

    if not paths[
        "checkpoint"
    ].is_file():
        raise RuntimeError(
            "Best LOAO checkpoint was not created."
        )

    # ========================================================
    # RELOAD FROZEN BEST CHECKPOINT
    # ========================================================

    print()
    print("=" * 78)
    print(
        "FORMAL TRAINING COMPLETE"
    )
    print("=" * 78)

    print(
        "Best epoch              :",
        best_epoch,
    )

    print(
        "Best Calibration Loss   :",
        f"{best_calibration_loss:.8f}",
    )

    print(
        "Training seconds        :",
        f"{total_training_seconds:.2f}",
    )

    checkpoint = torch.load(
        paths[
            "checkpoint"
        ],
        map_location=base.DEVICE,
        weights_only=False,
    )

    if checkpoint.get(
        "protocol_sha256"
    ) != fingerprint:
        raise RuntimeError(
            "LOAO checkpoint protocol fingerprint mismatch."
        )

    if checkpoint.get(
        "held_out_attack"
    ) != held_out_attack:
        raise RuntimeError(
            "LOAO checkpoint held-out attack mismatch."
        )

    if checkpoint.get(
        "selection_metric"
    ) != "val_calibration_joint_loss":
        raise RuntimeError(
            "LOAO checkpoint selection metric mismatch."
        )

    if checkpoint.get(
        "heldout_used_for_checkpoint"
    ) is not False:
        raise RuntimeError(
            "Checkpoint does not certify held-out isolation."
        )

    if checkpoint.get(
        "test_accessed"
    ) is not False:
        raise RuntimeError(
            "Checkpoint does not certify Test isolation."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    # ========================================================
    # CALIBRATION
    #
    # This is the ONLY place detection threshold is selected.
    # ========================================================

    print()
    print("=" * 78)
    print(
        "BEST-CHECKPOINT "
        "VAL-CALIBRATION EVALUATION"
    )
    print("=" * 78)

    calibration_summary = (
        base.evaluate_best_validation(
            model,
            calibration_loader,
        )
    )

    frozen_detection_threshold = float(
        calibration_summary[
            "detection_threshold"
        ]
    )

    save_dataframe(
        calibration_summary[
            "prediction_rows"
        ],
        paths[
            "calibration_predictions"
        ],
    )

    print(
        "Calibration ROC-AUC      :",
        f"{calibration_summary['detection_auc']:.8f}",
    )

    print(
        "Frozen detection threshold:",
        f"{frozen_detection_threshold:.12f}",
    )

    print(
        "Calibration Balanced Acc :",
        f"{calibration_summary['detection_balanced_accuracy']:.8f}",
    )

    print(
        "Calibration Detection F1 :",
        f"{calibration_summary['detection_f1']:.8f}",
    )

    print(
        "Calibration Loc Pixel F1 :",
        f"{calibration_summary['localization_pixel_f1']:.8f}",
    )

    print(
        "Calibration Loc Pixel IoU:",
        f"{calibration_summary['localization_pixel_iou']:.8f}",
    )

    # ========================================================
    # HELD-OUT EVALUATION
    #
    # IMPORTANT:
    #   No checkpoint selection.
    #   No early stopping.
    #   No threshold optimization.
    #
    # Detection threshold is already frozen above.
    # ========================================================

    print()
    print("=" * 78)
    print(
        "HELD-OUT ATTACK EVALUATION"
    )
    print("=" * 78)

    print(
        "Held-out attack          :",
        held_out_attack.upper(),
    )

    print(
        "Detection threshold      :",
        f"{frozen_detection_threshold:.12f}",
        "(FROZEN FROM CALIBRATION)",
    )

    heldout_summary = evaluate_fixed_threshold(
        model=model,
        loader=heldout_loader,
        detection_threshold=(
            frozen_detection_threshold
        ),
    )

    save_dataframe(
        heldout_summary[
            "prediction_rows"
        ],
        paths[
            "heldout_predictions"
        ],
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    final_summary = {
        "experiment_id":
            EXPERIMENT_ID,

        "held_out_attack":
            held_out_attack,

        "protocol_sha256":
            fingerprint,

        "best_epoch":
            best_epoch,

        "best_calibration_joint_loss":
            best_calibration_loss,

        "epochs_completed":
            len(
                history
            ),

        "training_seconds":
            total_training_seconds,

        "train_sources":
            len(
                subsets[
                    "train_source_indices"
                ]
            ),

        "train_active_samples":
            len(
                subsets[
                    "train"
                ]
            ),

        "calibration_sources":
            len(
                subsets[
                    "calibration_source_indices"
                ]
            ),

        "calibration_active_samples":
            len(
                subsets[
                    "calibration"
                ]
            ),

        "heldout_sources":
            len(
                subsets[
                    "heldout_source_indices"
                ]
            ),

        "heldout_active_samples":
            len(
                subsets[
                    "heldout"
                ]
            ),

        "calibration_detection_auc":
            calibration_summary[
                "detection_auc"
            ],

        "calibration_detection_threshold":
            frozen_detection_threshold,

        "calibration_detection_balanced_accuracy":
            calibration_summary[
                "detection_balanced_accuracy"
            ],

        "calibration_detection_f1":
            calibration_summary[
                "detection_f1"
            ],

        "calibration_localization_pixel_f1":
            calibration_summary[
                "localization_pixel_f1"
            ],

        "calibration_localization_pixel_iou":
            calibration_summary[
                "localization_pixel_iou"
            ],

        "heldout_detection_auc":
            heldout_summary[
                "detection_auc"
            ],

        "heldout_detection_threshold":
            heldout_summary[
                "detection_threshold"
            ],

        "heldout_detection_balanced_accuracy":
            heldout_summary[
                "detection_balanced_accuracy"
            ],

        "heldout_detection_f1":
            heldout_summary[
                "detection_f1"
            ],

        "heldout_detection_sensitivity":
            heldout_summary[
                "detection_sensitivity"
            ],

        "heldout_detection_specificity":
            heldout_summary[
                "detection_specificity"
            ],

        "heldout_detection_tn":
            heldout_summary[
                "detection_tn"
            ],

        "heldout_detection_fp":
            heldout_summary[
                "detection_fp"
            ],

        "heldout_detection_fn":
            heldout_summary[
                "detection_fn"
            ],

        "heldout_detection_tp":
            heldout_summary[
                "detection_tp"
            ],

        "heldout_localization_threshold":
            heldout_summary[
                "localization_threshold"
            ],

        "heldout_localization_pixel_f1":
            heldout_summary[
                "localization_pixel_f1"
            ],

        "heldout_localization_pixel_iou":
            heldout_summary[
                "localization_pixel_iou"
            ],

        "heldout_localization_tp":
            heldout_summary[
                "localization_tp"
            ],

        "heldout_localization_fp":
            heldout_summary[
                "localization_fp"
            ],

        "heldout_localization_fn":
            heldout_summary[
                "localization_fn"
            ],

        "heldout_localization_tn":
            heldout_summary[
                "localization_tn"
            ],

        "heldout_used_for_training":
            False,

        "heldout_used_for_checkpoint":
            False,

        "heldout_used_for_threshold":
            False,

        "test_accessed":
            False,
    }

    save_dataframe(
        [
            final_summary
        ],
        paths[
            "summary"
        ],
    )

    print()
    print("-" * 78)

    print(
        "Held-Out Detection ROC-AUC :",
        f"{heldout_summary['detection_auc']:.8f}",
    )

    print(
        "Held-Out Balanced Acc      :",
        f"{heldout_summary['detection_balanced_accuracy']:.8f}",
    )

    print(
        "Held-Out Detection F1      :",
        f"{heldout_summary['detection_f1']:.8f}",
    )

    print(
        "Held-Out Sensitivity       :",
        f"{heldout_summary['detection_sensitivity']:.8f}",
    )

    print(
        "Held-Out Specificity       :",
        f"{heldout_summary['detection_specificity']:.8f}",
    )

    print(
        "Held-Out confusion         :",
        "TN=",
        heldout_summary[
            "detection_tn"
        ],
        "FP=",
        heldout_summary[
            "detection_fp"
        ],
        "FN=",
        heldout_summary[
            "detection_fn"
        ],
        "TP=",
        heldout_summary[
            "detection_tp"
        ],
    )

    print()
    print(
        "Held-Out Localization F1  :",
        f"{heldout_summary['localization_pixel_f1']:.8f}",
    )

    print(
        "Held-Out Localization IoU :",
        f"{heldout_summary['localization_pixel_iou']:.8f}",
    )

    print()
    print(
        "Held-out used for training   : False"
    )

    print(
        "Held-out used for checkpoint : False"
    )

    print(
        "Held-out used for threshold  : False"
    )

    print(
        "Test accessed                : False"
    )

    print()
    print("=" * 78)
    print(
        "E5.3R-4C-4-2 FORMAL LOAO COMPLETE"
    )
    print("=" * 78)

    print()
    print(
        "Checkpoint:",
        paths[
            "checkpoint"
        ],
    )

    print(
        "History:",
        paths[
            "history"
        ],
    )

    print(
        "Calibration predictions:",
        paths[
            "calibration_predictions"
        ],
    )

    print(
        "Held-out predictions:",
        paths[
            "heldout_predictions"
        ],
    )

    print(
        "Summary:",
        paths[
            "summary"
        ],
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "E5.3R-4C-4 Formal "
            "Leave-One-Attack-Out training"
        )
    )

    parser.add_argument(
        "--held-out",
        required=True,
        choices=ATTACKS,
        help=(
            "Attack family excluded from Train and "
            "Val-Calibration."
        ),
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Audit formal LOAO split only. "
            "No model, optimizer, checkpoint, or training."
        ),
    )

    args = parser.parse_args()

    held_out_attack = normalize_attack(
        args.held_out
    )

    print()
    print("=" * 78)
    print(
        EXPERIMENT_ID
    )
    print("=" * 78)

    print(
        "Held-out attack:",
        held_out_attack.upper(),
    )

    print(
        "Mode:",
        (
            "PREFLIGHT"
            if args.preflight
            else
            "FORMAL TRAINING"
        ),
    )

    print(
        "Test accessed: False"
    )

    # --------------------------------------------------------
    # Build ONLY Train / Val source records.
    # --------------------------------------------------------

    (
        train_records,
        val_records,
    ) = load_source_records()

    (
        train_dataset,
        val_dataset,
    ) = build_full_datasets(
        train_records,
        val_records,
    )

    print()
    print(
        "Recovering frozen Train attack assignment..."
    )

    train_assignment = (
        recover_attack_assignment(
            train_dataset,
            len(
                train_records
            ),
        )
    )

    print(
        "Recovering frozen Val attack assignment..."
    )

    val_assignment = (
        recover_attack_assignment(
            val_dataset,
            len(
                val_records
            ),
        )
    )

    subsets = build_formal_subsets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_assignment=train_assignment,
        val_assignment=val_assignment,
        held_out_attack=held_out_attack,
    )

    # --------------------------------------------------------
    # Always run the safety audit before any possible model
    # initialization.
    # --------------------------------------------------------

    run_preflight(
        held_out_attack=held_out_attack,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_assignment=train_assignment,
        val_assignment=val_assignment,
        subsets=subsets,
    )

    if args.preflight:
        print()
        print(
            "Preflight-only mode complete."
        )

        print(
            "No model initialized."
        )

        print(
            "No optimizer initialized."
        )

        print(
            "No checkpoint created."
        )

        print(
            "No training performed."
        )

        print(
            "Test accessed: False"
        )

        return

    # --------------------------------------------------------
    # Formal training begins ONLY after preflight PASS.
    # --------------------------------------------------------

    run_training(
        held_out_attack=held_out_attack,
        subsets=subsets,
    )


if __name__ == "__main__":
    main()
