from pathlib import Path
import sys
import csv
import json
import hashlib
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# Project root
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


# ============================================================
# Project imports
# ============================================================

from preprocessing.active_sspg_dataset import (
    ActiveSSPGDataset,
    canonical_sample_key,
)

from models.sspg_joint_model_v2 import (
    SSPGJointModelV2,
)

from training.e53r_training_protocol_v1 import (
    PROTOCOL,
    protocol_json,
    mask_to_patch_target,
    compute_joint_loss,
    build_optimizer,
)


# ============================================================
# Paths
# ============================================================

MANIFEST_PATH = (
    ROOT
    / "manifests"
    / "imd2020_manifest_e5_clean.csv"
)

STEGO_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "stego_structured_v33_128bit_best.pth"
)

BEST_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "e53r_joint_v1_best.pth"
)

HISTORY_PATH = (
    ROOT
    / "results"
    / "csv"
    / "e53r_joint_v1_history.csv"
)

VAL_PREDICTIONS_PATH = (
    ROOT
    / "results"
    / "csv"
    / "e53r_joint_v1_val_predictions.csv"
)

SUMMARY_PATH = (
    ROOT
    / "results"
    / "csv"
    / "e53r_joint_v1_summary.csv"
)

PROTOCOL_SNAPSHOT_PATH = (
    ROOT
    / "results"
    / "csv"
    / "e53r_joint_v1_protocol.json"
)


# ============================================================
# Frozen protocol fingerprint
# ============================================================

EXPECTED_PROTOCOL_SHA256 = (
    "2a478f83513378b7e34e8f3994ccb988"
    "2a1be6c69340003f383a49e6e2833c09"
)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

def set_global_seed(seed):

    seed = int(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    # Reproducibility is preferred over maximum speed.
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# Protocol fingerprint
# ============================================================

def verify_protocol_fingerprint():

    serialized = protocol_json()

    actual = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()

    print(
        "Protocol SHA256 :",
        actual,
    )

    if actual != EXPECTED_PROTOCOL_SHA256:

        raise RuntimeError(
            "\n"
            "TRAINING ABORTED.\n"
            "Formal protocol fingerprint changed.\n"
            "\n"
            f"Expected: {EXPECTED_PROTOCOL_SHA256}\n"
            f"Actual  : {actual}\n"
            "\n"
            "Do not train under E5.3R-4C-1 "
            "with a modified protocol."
        )

    return actual


# ============================================================
# Manifest utilities
# ============================================================

def resolve_image_path(value):

    path = Path(
        str(value)
    )

    if path.is_absolute():
        return path

    return (
        ROOT
        / "data"
        / "imd2020"
        / path
    ).resolve()


def authentic_label(value):
    """
    Conservative parser for the clean manifest.

    Returns True only for recognized authentic labels.
    """

    if pd.isna(value):
        return False

    text = str(value).strip().lower()

    authentic_values = {
        "0",
        "0.0",
        "authentic",
        "real",
        "original",
        "pristine",
        "clean",
        "negative",
    }

    return (
        text
        in authentic_values
    )


def build_source_records(
    dataframe,
    split_name,
):

    split_name = (
        str(split_name)
        .strip()
        .lower()
    )

    # --------------------------------------------------------
    # Hard Test prohibition
    # --------------------------------------------------------

    if split_name == "test":

        raise RuntimeError(
            "TEST ACCESS PROHIBITED during "
            "E5.3R-4C-1."
        )

    if split_name not in {
        "train",
        "val",
    }:

        raise RuntimeError(
            f"Unsupported split: {split_name}"
        )

    required_columns = {
        "sample_id",
        "group",
        "split",
        "label",
        "image_path",
    }

    missing = (
        required_columns
        -
        set(
            dataframe.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Manifest missing columns: "
            + ", ".join(
                sorted(missing)
            )
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

    if len(split_df) == 0:

        raise RuntimeError(
            f"No rows for split={split_name}"
        )

    # --------------------------------------------------------
    # Use authentic source images ONLY.
    #
    # ActiveSSPGDataset itself creates:
    #
    # source -> protected authentic
    # source -> protected manipulated
    # --------------------------------------------------------

    source_df = split_df[
        split_df[
            "label"
        ].apply(
            authentic_label
        )
    ].copy()

    if len(source_df) == 0:

        raise RuntimeError(
            f"No authentic sources found "
            f"for split={split_name}."
        )

    records = []

    for _, row in (
        source_df.iterrows()
    ):

        image_path = resolve_image_path(
            row[
                "image_path"
            ]
        )

        if not image_path.is_file():

            raise RuntimeError(
                "Source image not found: "
                f"{image_path}"
            )

        record = {
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

            "split":
                split_name,

            "label":
                row[
                    "label"
                ],

            "image_path":
                str(
                    image_path
                ),
        }

        # Preserve explicit sample_key if a future manifest
        # contains one. Current clean manifest does not
        # require it.
        if (
            "sample_key"
            in source_df.columns
            and not pd.isna(
                row.get(
                    "sample_key"
                )
            )
        ):

            record[
                "sample_key"
            ] = str(
                row[
                    "sample_key"
                ]
            )

        records.append(
            record
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # This ordering MUST match frozen E5.3R-4A.
    #
    # R3 balanced attack assignment is index-based.
    # --------------------------------------------------------

    records.sort(
        key=lambda record: (
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
    )

    return records


# ============================================================
# Leakage audit
# ============================================================

def audit_train_val_sources(
    train_records,
    val_records,
):

    train_keys = {
        canonical_sample_key(
            record
        )
        for record
        in train_records
    }

    val_keys = {
        canonical_sample_key(
            record
        )
        for record
        in val_records
    }

    key_overlap = (
        train_keys
        &
        val_keys
    )

    train_groups = {
        str(
            record[
                "group"
            ]
        )
        for record
        in train_records
    }

    val_groups = {
        str(
            record[
                "group"
            ]
        )
        for record
        in val_records
    }

    group_overlap = (
        train_groups
        &
        val_groups
    )

    train_paths = {
        str(
            Path(
                record[
                    "image_path"
                ]
            ).resolve()
        )
        for record
        in train_records
    }

    val_paths = {
        str(
            Path(
                record[
                    "image_path"
                ]
            ).resolve()
        )
        for record
        in val_records
    }

    path_overlap = (
        train_paths
        &
        val_paths
    )

    print(
        "Canonical-key overlap :",
        len(
            key_overlap
        ),
    )

    print(
        "Group overlap         :",
        len(
            group_overlap
        ),
    )

    print(
        "Path overlap          :",
        len(
            path_overlap
        ),
    )

    if key_overlap:

        raise RuntimeError(
            "Train/Val canonical-key leakage."
        )

    if group_overlap:

        raise RuntimeError(
            "Train/Val group leakage."
        )

    if path_overlap:

        raise RuntimeError(
            "Train/Val path leakage."
        )


# ============================================================
# DataLoader
# ============================================================

def build_loader(
    dataset,
    shuffle,
    seed,
):

    generator = torch.Generator()

    generator.manual_seed(
        int(seed)
    )

    return DataLoader(
        dataset,
        batch_size=(
            PROTOCOL.batch_size
        ),
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(
            DEVICE.type
            ==
            "cuda"
        ),
        drop_last=False,
        generator=generator,
    )


# ============================================================
# Batch transfer
# ============================================================

def move_batch_to_device(
    batch,
):

    received = (
        batch[
            "received"
        ]
        .to(
            DEVICE,
            non_blocking=True,
        )
        .float()
    )

    payload = (
        batch[
            "payload"
        ]
        .to(
            DEVICE,
            non_blocking=True,
        )
        .float()
    )

    mask = (
        batch[
            "mask"
        ]
        .to(
            DEVICE,
            non_blocking=True,
        )
        .float()
    )

    detection_target = (
        batch[
            "detection_label"
        ]
        .to(
            DEVICE,
            non_blocking=True,
        )
        .float()
    )

    return (
        received,
        payload,
        mask,
        detection_target,
    )


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    epoch,
):

    model.train()

    total_sum = 0.0
    detection_sum = 0.0
    localization_sum = 0.0
    localization_bce_sum = 0.0
    localization_dice_sum = 0.0

    sample_count = 0
    optimizer_steps = 0

    num_batches = len(
        loader
    )

    accumulation = int(
        PROTOCOL.gradient_accumulation_steps
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    epoch_start = time.time()

    for batch_index, batch in enumerate(
        loader
    ):

        (
            received,
            payload,
            mask,
            detection_target,
        ) = move_batch_to_device(
            batch
        )

        batch_size = int(
            received.shape[
                0
            ]
        )

        # ----------------------------------------------------
        # Determine actual accumulation-window size.
        #
        # This handles the final partial window correctly.
        # ----------------------------------------------------

        window_position = (
            batch_index
            %
            accumulation
        )

        window_start = (
            batch_index
            -
            window_position
        )

        remaining_from_window_start = (
            num_batches
            -
            window_start
        )

        actual_window_size = min(
            accumulation,
            remaining_from_window_start,
        )

        output = model(
            received,
            payload,
        )

        localization_target = (
            mask_to_patch_target(
                mask
            )
        )

        losses = compute_joint_loss(
            detection_logits=(
                output[
                    "detection_logits"
                ]
            ),
            detection_target=(
                detection_target
            ),
            localization_logits=(
                output[
                    "localization_grid_logits"
                ]
            ),
            localization_target=(
                localization_target
            ),
        )

        raw_total_loss = (
            losses[
                "total_loss"
            ]
        )

        if not torch.isfinite(
            raw_total_loss
        ):

            raise RuntimeError(
                "Non-finite training loss "
                f"at epoch {epoch}, "
                f"batch {batch_index}."
            )

        # ----------------------------------------------------
        # Gradient accumulation.
        #
        # Divide by the actual current window size.
        # ----------------------------------------------------

        backward_loss = (
            raw_total_loss
            /
            float(
                actual_window_size
            )
        )

        backward_loss.backward()

        # ----------------------------------------------------
        # Statistics use RAW loss, not divided loss.
        # ----------------------------------------------------

        total_sum += (
            float(
                losses[
                    "total_loss"
                ].detach().item()
            )
            *
            batch_size
        )

        detection_sum += (
            float(
                losses[
                    "detection_loss"
                ].detach().item()
            )
            *
            batch_size
        )

        localization_sum += (
            float(
                losses[
                    "localization_loss"
                ].detach().item()
            )
            *
            batch_size
        )

        localization_bce_sum += (
            float(
                losses[
                    "localization_bce"
                ].detach().item()
            )
            *
            batch_size
        )

        localization_dice_sum += (
            float(
                losses[
                    "localization_dice"
                ].detach().item()
            )
            *
            batch_size
        )

        sample_count += (
            batch_size
        )

        end_of_window = (
            (
                window_position
                ==
                actual_window_size
                -
                1
            )
            or
            (
                batch_index
                ==
                num_batches
                -
                1
            )
        )

        if end_of_window:

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            optimizer_steps += 1

        # ----------------------------------------------------
        # Progress only.
        # No metric-based decisions.
        # ----------------------------------------------------

        if (
            (
                batch_index
                +
                1
            )
            %
            50
            ==
            0
            or
            batch_index
            ==
            num_batches
            -
            1
        ):

            print(
                f"  Train "
                f"{batch_index + 1:4d}"
                f"/{num_batches:4d} "
                f"loss="
                f"{total_sum / sample_count:.6f}",
                flush=True,
            )

    elapsed = (
        time.time()
        -
        epoch_start
    )

    if sample_count != len(
        loader.dataset
    ):

        raise RuntimeError(
            "Training sample-count mismatch."
        )

    expected_optimizer_steps = (
        num_batches
        +
        accumulation
        -
        1
    ) // accumulation

    if (
        optimizer_steps
        !=
        expected_optimizer_steps
    ):

        raise RuntimeError(
            "Optimizer-step count mismatch: "
            f"expected "
            f"{expected_optimizer_steps}, "
            f"got {optimizer_steps}."
        )

    return {
        "total_loss":
            total_sum
            /
            sample_count,

        "detection_loss":
            detection_sum
            /
            sample_count,

        "localization_loss":
            localization_sum
            /
            sample_count,

        "localization_bce":
            localization_bce_sum
            /
            sample_count,

        "localization_dice":
            localization_dice_sum
            /
            sample_count,

        "optimizer_steps":
            optimizer_steps,

        "seconds":
            elapsed,
    }


# ============================================================
# Validation one epoch
# ============================================================

@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
):

    model.eval()

    total_sum = 0.0
    detection_sum = 0.0
    localization_sum = 0.0
    localization_bce_sum = 0.0
    localization_dice_sum = 0.0

    sample_count = 0

    for batch in loader:

        (
            received,
            payload,
            mask,
            detection_target,
        ) = move_batch_to_device(
            batch
        )

        batch_size = int(
            received.shape[
                0
            ]
        )

        output = model(
            received,
            payload,
        )

        localization_target = (
            mask_to_patch_target(
                mask
            )
        )

        losses = compute_joint_loss(
            detection_logits=(
                output[
                    "detection_logits"
                ]
            ),
            detection_target=(
                detection_target
            ),
            localization_logits=(
                output[
                    "localization_grid_logits"
                ]
            ),
            localization_target=(
                localization_target
            ),
        )

        if not torch.isfinite(
            losses[
                "total_loss"
            ]
        ):

            raise RuntimeError(
                "Non-finite validation loss."
            )

        total_sum += (
            float(
                losses[
                    "total_loss"
                ].item()
            )
            *
            batch_size
        )

        detection_sum += (
            float(
                losses[
                    "detection_loss"
                ].item()
            )
            *
            batch_size
        )

        localization_sum += (
            float(
                losses[
                    "localization_loss"
                ].item()
            )
            *
            batch_size
        )

        localization_bce_sum += (
            float(
                losses[
                    "localization_bce"
                ].item()
            )
            *
            batch_size
        )

        localization_dice_sum += (
            float(
                losses[
                    "localization_dice"
                ].item()
            )
            *
            batch_size
        )

        sample_count += (
            batch_size
        )

    if sample_count != len(
        loader.dataset
    ):

        raise RuntimeError(
            "Validation sample-count mismatch."
        )

    return {
        "total_loss":
            total_sum
            /
            sample_count,

        "detection_loss":
            detection_sum
            /
            sample_count,

        "localization_loss":
            localization_sum
            /
            sample_count,

        "localization_bce":
            localization_bce_sum
            /
            sample_count,

        "localization_dice":
            localization_dice_sum
            /
            sample_count,
    }


# ============================================================
# Atomic checkpoint save
# ============================================================

def save_checkpoint_atomic(
    model,
    optimizer,
    epoch,
    val_joint_loss,
    protocol_sha256,
):

    BEST_CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        BEST_CHECKPOINT
        .with_suffix(
            ".tmp"
        )
    )

    checkpoint = {
        "experiment_id":
            "E5.3R-4C-1-V1",

        "protocol_experiment_id":
            PROTOCOL.experiment_id,

        "protocol_sha256":
            protocol_sha256,

        "selection_metric":
            "validation_joint_loss",

        "selection_mode":
            "min",

        "epoch":
            int(
                epoch
            ),

        "validation_joint_loss":
            float(
                val_joint_loss
            ),

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "test_accessed":
            False,
    }

    torch.save(
        checkpoint,
        temporary_path,
    )

    temporary_path.replace(
        BEST_CHECKPOINT
    )


# ============================================================
# CSV history
# ============================================================

def save_history(
    history,
):

    HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(
        history
    )

    dataframe.to_csv(
        HISTORY_PATH,
        index=False,
    )


# ============================================================
# Validation final metrics
# ============================================================

@torch.no_grad()
def evaluate_best_validation(
    model,
    loader,
):

    model.eval()

    labels = []
    probabilities = []

    prediction_rows = []

    localization_tp = 0
    localization_fp = 0
    localization_fn = 0
    localization_tn = 0

    threshold_loc = float(
        PROTOCOL.localization_threshold
    )

    for batch in loader:

        (
            received,
            payload,
            mask,
            detection_target,
        ) = move_batch_to_device(
            batch
        )

        output = model(
            received,
            payload,
        )

        detection_probability = (
            torch.sigmoid(
                output[
                    "detection_logits"
                ]
            )
        )

        localization_logits_256 = (
            model.localization_logits_256(
                output[
                    "localization_grid_logits"
                ]
            )
        )

        localization_probability = (
            torch.sigmoid(
                localization_logits_256
            )
        )

        localization_prediction = (
            localization_probability
            >=
            threshold_loc
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

        label_value = float(
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
            int(
                label_value
            )
        )

        probabilities.append(
            probability_value
        )

        # ----------------------------------------------------
        # Per-sample localization F1 / IoU.
        #
        # For an authentic sample:
        #   if GT=0 and prediction=0 everywhere,
        #   F1 and IoU are recorded as 1.0.
        #
        # This is for descriptive per-sample CSV only.
        # Formal aggregate pixel metrics below use global
        # TP/FP/FN counts.
        # ----------------------------------------------------

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
            *
            sample_tp
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

        if f1_denominator == 0:
            sample_f1 = 1.0
        else:
            sample_f1 = (
                2.0
                *
                sample_tp
                /
                f1_denominator
            )

        if iou_denominator == 0:
            sample_iou = 1.0
        else:
            sample_iou = (
                sample_tp
                /
                iou_denominator
            )

        prediction_rows.append(
            {
                "sample_key":
                    batch[
                        "sample_key"
                    ][
                        0
                    ],

                "source_path":
                    batch[
                        "source_path"
                    ][
                        0
                    ],

                "attack":
                    batch[
                        "attack"
                    ][
                        0
                    ],

                "detection_label":
                    int(
                        label_value
                    ),

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

    # --------------------------------------------------------
    # Detection ROC-AUC
    # --------------------------------------------------------

    detection_auc = float(
        roc_auc_score(
            labels_array,
            probabilities_array,
        )
    )

    # --------------------------------------------------------
    # Validation-only Youden J threshold
    #
    # This is exactly the pre-registered rule.
    # --------------------------------------------------------

    fpr, tpr, thresholds = (
        roc_curve(
            labels_array,
            probabilities_array,
        )
    )

    youden_j = (
        tpr
        -
        fpr
    )

    finite_indices = np.where(
        np.isfinite(
            thresholds
        )
    )[0]

    if len(
        finite_indices
    ) == 0:

        raise RuntimeError(
            "No finite validation ROC threshold."
        )

    best_local_index = int(
        np.argmax(
            youden_j[
                finite_indices
            ]
        )
    )

    best_index = int(
        finite_indices[
            best_local_index
        ]
    )

    detection_threshold = float(
        thresholds[
            best_index
        ]
    )

    detection_prediction = (
        probabilities_array
        >=
        detection_threshold
    ).astype(
        np.int64
    )

    detection_balanced_accuracy = float(
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

    # --------------------------------------------------------
    # Aggregate pixel localization metrics
    # --------------------------------------------------------

    pixel_f1_denominator = (
        2
        *
        localization_tp
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
        *
        localization_tp
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
            detection_threshold,

        "detection_balanced_accuracy":
            detection_balanced_accuracy,

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
            threshold_loc,

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
            prediction_rows,
    }


# ============================================================
# Save validation predictions
# ============================================================

def save_validation_predictions(
    rows,
):

    VAL_PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        rows
    ).to_csv(
        VAL_PREDICTIONS_PATH,
        index=False,
    )


# ============================================================
# Save summary
# ============================================================

def save_summary(
    summary,
):

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    flat_summary = {
        key: value
        for key, value
        in summary.items()
        if key
        !=
        "prediction_rows"
    }

    pd.DataFrame(
        [
            flat_summary
        ]
    ).to_csv(
        SUMMARY_PATH,
        index=False,
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 78)

    print(
        "E5.3R-4C-1 FORMAL "
        "TRAIN / VALIDATION V1"
    )

    print("=" * 78)

    print(
        "Device           :",
        DEVICE,
    )

    print(
        "Manifest         :",
        MANIFEST_PATH,
    )

    print(
        "Stego checkpoint :",
        STEGO_CHECKPOINT,
    )

    print()

    print(
        "TEST ACCESS POLICY:"
    )

    print(
        "  Test images are prohibited."
    )

    print(
        "  Test thresholds are prohibited."
    )

    print(
        "  Test checkpoint selection is prohibited."
    )

    # ========================================================
    # File checks
    # ========================================================

    if not MANIFEST_PATH.is_file():

        raise RuntimeError(
            "Clean manifest not found: "
            f"{MANIFEST_PATH}"
        )

    if not STEGO_CHECKPOINT.is_file():

        raise RuntimeError(
            "V3.3 checkpoint not found: "
            f"{STEGO_CHECKPOINT}"
        )

    # ========================================================
    # Frozen protocol
    # ========================================================

    print()

    print("-" * 78)

    print(
        "FROZEN PROTOCOL"
    )

    print("-" * 78)

    protocol_sha256 = (
        verify_protocol_fingerprint()
    )

    PROTOCOL_SNAPSHOT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        PROTOCOL_SNAPSHOT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            protocol_json()
        )

        file.write(
            "\n"
        )

    # ========================================================
    # Seed
    # ========================================================

    set_global_seed(
        PROTOCOL.training_seed
    )

    # ========================================================
    # Manifest
    # ========================================================

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    print()

    print("-" * 78)

    print(
        "MANIFEST / ACTIVE DATA"
    )

    print("-" * 78)

    print(
        "Manifest rows :",
        len(
            manifest
        ),
    )

    # --------------------------------------------------------
    # We inspect split names only.
    #
    # We DO NOT construct or load Test dataset.
    # --------------------------------------------------------

    normalized_splits = (
        manifest[
            "split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    print(
        "Manifest split counts:"
    )

    print(
        normalized_splits
        .value_counts()
        .to_dict()
    )

    train_records = (
        build_source_records(
            manifest,
            "train",
        )
    )

    val_records = (
        build_source_records(
            manifest,
            "val",
        )
    )

    # --------------------------------------------------------
    # Frozen 4A counts
    # --------------------------------------------------------

    print(
        "Train authentic sources :",
        len(
            train_records
        ),
    )

    print(
        "Val authentic sources   :",
        len(
            val_records
        ),
    )

    if len(
        train_records
    ) != 289:

        raise RuntimeError(
            "Frozen Train source count "
            "must be 289."
        )

    if len(
        val_records
    ) != 62:

        raise RuntimeError(
            "Frozen Val source count "
            "must be 62."
        )

    audit_train_val_sources(
        train_records,
        val_records,
    )

    # ========================================================
    # Active datasets
    # ========================================================

    train_dataset = (
        ActiveSSPGDataset(
            records=train_records,
            base_seed=(
                PROTOCOL.train_seed
            ),
        )
    )

    val_dataset = (
        ActiveSSPGDataset(
            records=val_records,
            base_seed=(
                PROTOCOL.val_seed
            ),
        )
    )

    print(
        "Train active samples    :",
        len(
            train_dataset
        ),
    )

    print(
        "Val active samples      :",
        len(
            val_dataset
        ),
    )

    if len(
        train_dataset
    ) != 578:

        raise RuntimeError(
            "Frozen Train active count "
            "must be 578."
        )

    if len(
        val_dataset
    ) != 124:

        raise RuntimeError(
            "Frozen Val active count "
            "must be 124."
        )

    # ========================================================
    # DataLoaders
    # ========================================================

    train_loader = build_loader(
        train_dataset,
        shuffle=True,
        seed=(
            PROTOCOL.training_seed
        ),
    )

    val_loader = build_loader(
        val_dataset,
        shuffle=False,
        seed=(
            PROTOCOL.training_seed
        ),
    )

    # ========================================================
    # Model
    # ========================================================

    print()

    print("-" * 78)

    print(
        "MODEL"
    )

    print("-" * 78)

    model = SSPGJointModelV2(
        stego_checkpoint=str(
            STEGO_CHECKPOINT
        ),

        adapter_bottleneck=(
            PROTOCOL.adapter_bottleneck
        ),

        fusion_hidden_dim=(
            PROTOCOL.fusion_hidden_dim
        ),

        knn_k=(
            PROTOCOL.knn_k
        ),

        gnn_hidden_dim=(
            PROTOCOL.gnn_hidden_dim
        ),

        gnn_depth=(
            PROTOCOL.gnn_depth
        ),
    ).to(
        DEVICE
    )

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
        "Total parameters     :",
        f"{total_parameters:,}",
    )

    print(
        "Trainable parameters :",
        f"{trainable_parameters:,}",
    )

    # Frozen decoder / ViT invariant.

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

    print(
        "Decoder trainable    :",
        decoder_trainable,
    )

    print(
        "ViT trainable        :",
        vit_trainable,
    )

    if decoder_trainable != 0:

        raise RuntimeError(
            "Frozen decoder became trainable."
        )

    if vit_trainable != 0:

        raise RuntimeError(
            "Frozen ViT became trainable."
        )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = build_optimizer(
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
        PROTOCOL.gradient_accumulation_steps,
    )

    # ========================================================
    # Formal training
    # ========================================================

    print()

    print("=" * 78)

    print(
        "FORMAL TRAINING START"
    )

    print("=" * 78)

    history = []

    best_val_loss = float(
        "inf"
    )

    best_epoch = None

    epochs_without_improvement = 0

    training_start = time.time()

    for epoch in range(
        1,
        PROTOCOL.max_epochs
        +
        1,
    ):

        print()

        print(
            f"Epoch "
            f"{epoch:02d}"
            f"/"
            f"{PROTOCOL.max_epochs:02d}"
        )

        print("-" * 78)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch,
        )

        val_metrics = validate_one_epoch(
            model,
            val_loader,
        )

        print()

        print(
            "Train Joint Loss :",
            f"{train_metrics['total_loss']:.8f}",
        )

        print(
            "Val Joint Loss   :",
            f"{val_metrics['total_loss']:.8f}",
        )

        print(
            "Train Det Loss   :",
            f"{train_metrics['detection_loss']:.8f}",
        )

        print(
            "Val Det Loss     :",
            f"{val_metrics['detection_loss']:.8f}",
        )

        print(
            "Train Loc Loss   :",
            f"{train_metrics['localization_loss']:.8f}",
        )

        print(
            "Val Loc Loss     :",
            f"{val_metrics['localization_loss']:.8f}",
        )

        print(
            "Optimizer steps  :",
            train_metrics[
                "optimizer_steps"
            ],
        )

        print(
            "Epoch seconds    :",
            f"{train_metrics['seconds']:.2f}",
        )

        # ----------------------------------------------------
        # Checkpoint decision uses ONLY validation joint loss.
        # ----------------------------------------------------

        improved = (
            val_metrics[
                "total_loss"
            ]
            <
            (
                best_val_loss
                -
                PROTOCOL.min_delta
            )
        )

        if improved:

            best_val_loss = float(
                val_metrics[
                    "total_loss"
                ]
            )

            best_epoch = int(
                epoch
            )

            epochs_without_improvement = 0

            save_checkpoint_atomic(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_joint_loss=(
                    best_val_loss
                ),
                protocol_sha256=(
                    protocol_sha256
                ),
            )

            print(
                "Checkpoint       : SAVED "
                "(best Val Joint Loss)"
            )

        else:

            epochs_without_improvement += 1

            print(
                "Checkpoint       : not saved"
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

                "val_joint_loss":
                    val_metrics[
                        "total_loss"
                    ],

                "val_detection_loss":
                    val_metrics[
                        "detection_loss"
                    ],

                "val_localization_loss":
                    val_metrics[
                        "localization_loss"
                    ],

                "val_localization_bce":
                    val_metrics[
                        "localization_bce"
                    ],

                "val_localization_dice":
                    val_metrics[
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

        save_history(
            history
        )

        print(
            "Best epoch       :",
            best_epoch,
        )

        print(
            "Best Val Loss    :",
            f"{best_val_loss:.8f}",
        )

        print(
            "No-improve count :",
            epochs_without_improvement,
            "/",
            PROTOCOL.early_stopping_patience,
        )

        # ----------------------------------------------------
        # Frozen early stopping rule.
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >=
            PROTOCOL.early_stopping_patience
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

    # ========================================================
    # Best checkpoint integrity
    # ========================================================

    if not BEST_CHECKPOINT.is_file():

        raise RuntimeError(
            "Best checkpoint was not created."
        )

    print()

    print("=" * 78)

    print(
        "FORMAL TRAINING COMPLETE"
    )

    print("=" * 78)

    print(
        "Best epoch          :",
        best_epoch,
    )

    print(
        "Best Val Joint Loss :",
        f"{best_val_loss:.8f}",
    )

    print(
        "Training seconds    :",
        f"{total_training_seconds:.2f}",
    )

    # ========================================================
    # Reload ONLY the checkpoint selected by Val Joint Loss
    # ========================================================

    print()

    print(
        "Reloading frozen best checkpoint..."
    )

    checkpoint = torch.load(
        BEST_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    if (
        checkpoint.get(
            "protocol_sha256"
        )
        !=
        EXPECTED_PROTOCOL_SHA256
    ):

        raise RuntimeError(
            "Best checkpoint protocol "
            "fingerprint mismatch."
        )

    if (
        checkpoint.get(
            "selection_metric"
        )
        !=
        "validation_joint_loss"
    ):

        raise RuntimeError(
            "Checkpoint selection metric "
            "mismatch."
        )

    if checkpoint.get(
        "test_accessed"
    ) is not False:

        raise RuntimeError(
            "Checkpoint does not certify "
            "Test isolation."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    # ========================================================
    # Final Validation-only evaluation
    #
    # This happens only AFTER checkpoint selection.
    #
    # It cannot influence checkpoint choice.
    # ========================================================

    print()

    print("=" * 78)

    print(
        "BEST-CHECKPOINT "
        "VALIDATION-ONLY EVALUATION"
    )

    print("=" * 78)

    summary = evaluate_best_validation(
        model,
        val_loader,
    )

    summary[
        "experiment_id"
    ] = "E5.3R-4C-1-V1"

    summary[
        "protocol_sha256"
    ] = protocol_sha256

    summary[
        "best_epoch"
    ] = best_epoch

    summary[
        "best_val_joint_loss"
    ] = best_val_loss

    summary[
        "epochs_completed"
    ] = len(
        history
    )

    summary[
        "training_seconds"
    ] = total_training_seconds

    summary[
        "train_sources"
    ] = len(
        train_records
    )

    summary[
        "train_active_samples"
    ] = len(
        train_dataset
    )

    summary[
        "val_sources"
    ] = len(
        val_records
    )

    summary[
        "val_active_samples"
    ] = len(
        val_dataset
    )

    summary[
        "test_accessed"
    ] = False

    save_validation_predictions(
        summary[
            "prediction_rows"
        ]
    )

    save_summary(
        summary
    )

    print()

    print(
        "Detection ROC-AUC       :",
        f"{summary['detection_auc']:.8f}",
    )

    print(
        "Detection threshold     :",
        f"{summary['detection_threshold']:.12f}",
    )

    print(
        "Detection Balanced Acc  :",
        f"{summary['detection_balanced_accuracy']:.8f}",
    )

    print(
        "Detection F1            :",
        f"{summary['detection_f1']:.8f}",
    )

    print(
        "Detection Sensitivity   :",
        f"{summary['detection_sensitivity']:.8f}",
    )

    print(
        "Detection Specificity   :",
        f"{summary['detection_specificity']:.8f}",
    )

    print(
        "Detection confusion     :",
        "TN=",
        summary[
            "detection_tn"
        ],
        "FP=",
        summary[
            "detection_fp"
        ],
        "FN=",
        summary[
            "detection_fn"
        ],
        "TP=",
        summary[
            "detection_tp"
        ],
    )

    print()

    print(
        "Localization threshold  :",
        summary[
            "localization_threshold"
        ],
    )

    print(
        "Localization Pixel F1   :",
        f"{summary['localization_pixel_f1']:.8f}",
    )

    print(
        "Localization Pixel IoU  :",
        f"{summary['localization_pixel_iou']:.8f}",
    )

    print()

    print(
        "History:"
    )

    print(
        " ",
        HISTORY_PATH,
    )

    print(
        "Validation predictions:"
    )

    print(
        " ",
        VAL_PREDICTIONS_PATH,
    )

    print(
        "Summary:"
    )

    print(
        " ",
        SUMMARY_PATH,
    )

    print(
        "Best checkpoint:"
    )

    print(
        " ",
        BEST_CHECKPOINT,
    )

    print()

    print("=" * 78)

    print(
        "E5.3R-4C-1 FORMAL "
        "TRAIN / VALIDATION V1: COMPLETE"
    )

    print("=" * 78)

    print()

    print(
        "TEST ACCESSED : NO"
    )

    print(
        "Checkpoint selected ONLY by "
        "Validation Joint Loss."
    )

    print(
        "Detection threshold selected ONLY "
        "from Validation by Youden J."
    )

    print(
        "Localization threshold remained "
        "fixed at 0.5."
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "These are Validation results, "
        "NOT final Test results."
    )


if __name__ == "__main__":

    main()
