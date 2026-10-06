from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)

from preprocessing.active_sspg_dataset import (
    ActiveSSPGDataset,
)

from models.sspg_joint_model_v2 import (
    SSPGJointModelV2,
)


# ============================================================
# E5.3R-4C-2
# Validation Causal & Shortcut Audit
#
# PURPOSE
# ------------------------------------------------------------
# Audit the already-trained E5.3R-4C-1-V1 model.
#
# This script:
#   - DOES NOT train
#   - DOES NOT call optimizer.step()
#   - DOES NOT modify checkpoint
#   - DOES NOT access Test
#   - DOES NOT tune model parameters
#
# Conditions:
#
#   NORMAL
#       correct payload + normal provenance evidence
#
#   WRONG_PAYLOAD
#       same received image
#       payload deterministically flipped: 1 - payload
#
#   ZERO_EVIDENCE
#       same received image
#       visual path retained
#       provenance patch evidence forced to zero
#       fusion / graph / GNN / heads recomputed
#
# Additional audits:
#   - attack-wise performance
#   - paired authentic/manipulated score separation
#
# IMPORTANT
# ------------------------------------------------------------
# No post-hoc PASS threshold is defined for causal degradation.
# Results are descriptive and must be preserved whether
# favorable or unfavorable.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

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

CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "e53r_joint_v1_best.pth"
)

STEGO_CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "stego_structured_v33_128bit_best.pth"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "csv"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "e53r_4c2_validation_causal_predictions.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "e53r_4c2_validation_causal_summary.csv"
)

ATTACK_PATH = (
    OUTPUT_DIR
    / "e53r_4c2_validation_attackwise.csv"
)

PAIRED_PATH = (
    OUTPUT_DIR
    / "e53r_4c2_validation_paired.csv"
)

AUDIT_JSON_PATH = (
    OUTPUT_DIR
    / "e53r_4c2_validation_causal_audit.json"
)


VAL_SEED = 20260923
AUDIT_SEED = 20260926

EXPECTED_VAL_SOURCES = 62
EXPECTED_VAL_ACTIVE = 124

LOCALIZATION_THRESHOLD = 0.5

CONDITIONS = (
    "NORMAL",
    "WRONG_PAYLOAD",
    "ZERO_EVIDENCE",
)


# ============================================================
# REPRODUCIBILITY
# ============================================================


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# PATH RESOLUTION
# ============================================================


def resolve_image_path(value):
    """
    Match the already-audited E5.3R-4A path convention.
    """

    normalized = str(value).replace(
        "\\",
        "/",
    )

    path = Path(normalized)

    if path.is_absolute():
        return str(path)

    return str(
        IMAGE_ROOT / path
    )


# ============================================================
# VALIDATION SOURCE RECORDS
# ============================================================


def build_validation_records():
    if not MANIFEST_PATH.exists():
        raise RuntimeError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    df = pd.read_csv(
        MANIFEST_PATH
    )

    required_columns = {
        "split",
        "label",
        "group",
        "sample_id",
        "image_path",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Manifest missing columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Explicit development-only filter.
    #
    # Test rows are excluded BEFORE records are constructed.
    # --------------------------------------------------------

    development_df = df[
        df["split"].isin(
            ["train", "val"]
        )
    ].copy()

    if (
        development_df["split"]
        .astype(str)
        .str.lower()
        .eq("test")
        .any()
    ):
        raise RuntimeError(
            "Test row entered development dataframe."
        )

    val_df = development_df[
        (
            development_df["split"]
            .astype(str)
            .str.lower()
            == "val"
        )
        &
        (
            pd.to_numeric(
                development_df["label"],
                errors="raise",
            )
            == 0
        )
    ].copy()

    val_df = val_df.sort_values(
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

    for _, row in val_df.iterrows():
        image_path = resolve_image_path(
            row["image_path"]
        )

        record = {
            "sample_id":
                str(row["sample_id"]),

            "group":
                str(row["group"]),

            "image_path":
                image_path,
        }

        records.append(
            record
        )

    if len(records) != EXPECTED_VAL_SOURCES:
        raise RuntimeError(
            "Unexpected validation source count: "
            f"{len(records)} != "
            f"{EXPECTED_VAL_SOURCES}"
        )

    # Path existence only.
    # This is validation-only.
    for record in records:
        path = Path(
            record["image_path"]
        )

        if not path.exists():
            raise RuntimeError(
                f"Validation source missing: {path}"
            )

    return records


# ============================================================
# CHECKPOINT LOADING
# ============================================================


def extract_state_dict(checkpoint):
    """
    Support the common checkpoint containers while refusing
    to silently guess beyond explicit known possibilities.
    """

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            "Checkpoint is not a dictionary."
        )

    for key in (
        "model_state_dict",
        "state_dict",
        "model",
    ):
        value = checkpoint.get(
            key
        )

        if isinstance(
            value,
            dict,
        ):
            return value

    # A raw state_dict has tensor values directly.
    if checkpoint:
        if all(
            torch.is_tensor(v)
            for v in checkpoint.values()
        ):
            return checkpoint

    raise RuntimeError(
        "Could not locate model state_dict "
        "inside checkpoint."
    )


def build_model(device):
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(
            f"Joint checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

    if not STEGO_CHECKPOINT_PATH.exists():
        raise RuntimeError(
            f"Stego checkpoint not found: "
            f"{STEGO_CHECKPOINT_PATH}"
        )

    model = SSPGJointModelV2(
        stego_checkpoint=
            STEGO_CHECKPOINT_PATH,

        adapter_bottleneck=96,
        fusion_hidden_dim=96,
        knn_k=8,
        gnn_hidden_dim=256,
        gnn_depth=2,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(
        device
    )

    model.eval()

    # Explicit frozen-module invariant.
    if model.frontend.frontend.decoder.training:
        raise RuntimeError(
            "Decoder unexpectedly in train mode."
        )

    if model.frontend.frontend.visual.vit.training:
        raise RuntimeError(
            "ViT unexpectedly in train mode."
        )

    return model, checkpoint


# ============================================================
# ZERO-EVIDENCE FORWARD
# ============================================================


@torch.no_grad()
def forward_zero_evidence(
    model,
    received,
):
    """
    Preserve received image and visual information while
    removing provenance evidence.

    IMPORTANT:
        We do NOT reuse normal fused_tokens.

        Instead:
            received
              -> visual backbone
              -> visual_tokens

            zero patch evidence
              -> fusion
              -> graph
              -> GNN
              -> heads

    Thus provenance evidence is removed both from:
        1. token fusion
        2. graph node/edge evidence features
    """

    graph_frontend = (
        model.frontend
    )

    frontend = (
        graph_frontend.frontend
    )

    # --------------------------------------------------------
    # Visual path only.
    # --------------------------------------------------------

    visual_input = F.interpolate(
        received,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )

    visual_tokens = (
        frontend.visual(
            visual_input
        )
    )

    batch_size = (
        received.shape[0]
    )

    # --------------------------------------------------------
    # Remove provenance evidence.
    # Shape must match [B,196,1].
    # --------------------------------------------------------

    zero_patch_evidence = torch.zeros(
        batch_size,
        196,
        1,
        dtype=visual_tokens.dtype,
        device=visual_tokens.device,
    )

    # --------------------------------------------------------
    # Recompute fusion from zero evidence.
    # --------------------------------------------------------

    fused_tokens = (
        frontend.fusion(
            visual_tokens,
            zero_patch_evidence,
        )
    )

    # --------------------------------------------------------
    # Recompute graph using zero evidence.
    # --------------------------------------------------------

    graph = (
        graph_frontend.graph_builder(
            fused_tokens,
            zero_patch_evidence,
        )
    )

    # --------------------------------------------------------
    # Recompute GNN.
    # --------------------------------------------------------

    graph_tokens = (
        graph_frontend.gnn(
            graph["node_features"],
            graph["edge_index"],
            graph["edge_attr"],
        )
    )

    # --------------------------------------------------------
    # Same frozen trained heads.
    # --------------------------------------------------------

    head_output = (
        model.heads(
            graph_tokens
        )
    )

    return {
        "graph_tokens":
            graph_tokens,

        "detection_logits":
            head_output[
                "detection_logits"
            ],

        "localization_node_logits":
            head_output[
                "localization_node_logits"
            ],

        "localization_grid_logits":
            head_output[
                "localization_grid_logits"
            ],
    }


# ============================================================
# METRICS
# ============================================================


def choose_youden_threshold(
    labels,
    probabilities,
):
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    fpr, tpr, thresholds = (
        roc_curve(
            labels,
            probabilities,
        )
    )

    finite = np.isfinite(
        thresholds
    )

    if not finite.any():
        raise RuntimeError(
            "No finite ROC threshold."
        )

    j = (
        tpr
        - fpr
    )

    j = np.where(
        finite,
        j,
        -np.inf,
    )

    best_index = int(
        np.argmax(j)
    )

    return float(
        thresholds[
            best_index
        ]
    )


def detection_metrics(
    labels,
    probabilities,
    threshold,
):
    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    predictions = (
        probabilities
        >= threshold
    ).astype(
        np.int64
    )

    auc = roc_auc_score(
        labels,
        probabilities,
    )

    balanced_acc = (
        balanced_accuracy_score(
            labels,
            predictions,
        )
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    specificity = (
        float(tn)
        / float(tn + fp)
        if (tn + fp) > 0
        else float("nan")
    )

    sensitivity = (
        float(tp)
        / float(tp + fn)
        if (tp + fn) > 0
        else float("nan")
    )

    return {
        "roc_auc":
            float(auc),

        "balanced_accuracy":
            float(balanced_acc),

        "f1":
            float(f1),

        "sensitivity":
            float(sensitivity),

        "specificity":
            float(specificity),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),
    }


def localization_global_metrics(
    masks,
    probability_maps,
    threshold=LOCALIZATION_THRESHOLD,
):
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for mask, probability in zip(
        masks,
        probability_maps,
    ):
        target = (
            np.asarray(mask)
            >= 0.5
        )

        prediction = (
            np.asarray(probability)
            >= threshold
        )

        tp += int(
            np.logical_and(
                prediction,
                target,
            ).sum()
        )

        fp += int(
            np.logical_and(
                prediction,
                np.logical_not(target),
            ).sum()
        )

        fn += int(
            np.logical_and(
                np.logical_not(prediction),
                target,
            ).sum()
        )

        tn += int(
            np.logical_and(
                np.logical_not(prediction),
                np.logical_not(target),
            ).sum()
        )

    f1_denominator = (
        2 * tp
        + fp
        + fn
    )

    iou_denominator = (
        tp
        + fp
        + fn
    )

    pixel_f1 = (
        (2.0 * tp)
        / f1_denominator
        if f1_denominator > 0
        else 0.0
    )

    pixel_iou = (
        float(tp)
        / iou_denominator
        if iou_denominator > 0
        else 0.0
    )

    return {
        "pixel_f1":
            float(pixel_f1),

        "pixel_iou":
            float(pixel_iou),

        "loc_tp":
            int(tp),

        "loc_fp":
            int(fp),

        "loc_fn":
            int(fn),

        "loc_tn":
            int(tn),
    }


# ============================================================
# MODEL OUTPUT HELPERS
# ============================================================


def extract_detection_logits(
    output,
):
    logits = output[
        "detection_logits"
    ]

    return logits.reshape(
        -1
    )


def extract_localization_probability(
    model,
    output,
):
    grid_logits = output[
        "localization_grid_logits"
    ]

    logits_256 = (
        model.localization_logits_256(
            grid_logits
        )
    )

    probability = torch.sigmoid(
        logits_256
    )

    return probability


# ============================================================
# VALIDATION INFERENCE
# ============================================================


@torch.no_grad()
def run_validation(
    model,
    dataset,
    device,
):
    rows = []

    localization_storage = {
        condition: {
            "masks": [],
            "probabilities": [],
        }
        for condition in CONDITIONS
    }

    print()
    print("=" * 76)
    print("VALIDATION INFERENCE")
    print("=" * 76)

    for index in range(
        len(dataset)
    ):
        sample = dataset[
            index
        ]

        received = (
            sample["received"]
            .unsqueeze(0)
            .to(device)
        )

        payload = (
            sample["payload"]
            .unsqueeze(0)
            .to(device)
        )

        mask = (
            sample["mask"]
            .unsqueeze(0)
            .to(device)
        )

        detection_label = int(
            float(
                sample[
                    "detection_label"
                ].item()
            )
        )

        # ----------------------------------------------------
        # NORMAL
        # ----------------------------------------------------

        normal_output = model(
            received,
            payload,
        )

        # ----------------------------------------------------
        # WRONG PAYLOAD
        #
        # Deterministic bit complement.
        #
        # Same received image.
        # Same decoder.
        # Only payload interpretation changes.
        # ----------------------------------------------------

        wrong_payload = (
            1.0
            - payload
        )

        wrong_output = model(
            received,
            wrong_payload,
        )

        # ----------------------------------------------------
        # ZERO EVIDENCE
        #
        # Same received image.
        # Same visual path.
        # Provenance evidence removed.
        # ----------------------------------------------------

        zero_output = (
            forward_zero_evidence(
                model,
                received,
            )
        )

        outputs = {
            "NORMAL":
                normal_output,

            "WRONG_PAYLOAD":
                wrong_output,

            "ZERO_EVIDENCE":
                zero_output,
        }

        mask_np = (
            mask[
                0,
                0,
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        row = {
            "dataset_index":
                int(index),

            "source_index":
                int(
                    sample[
                        "source_index"
                    ]
                ),

            "sample_key":
                str(
                    sample[
                        "sample_key"
                    ]
                ),

            "source_path":
                str(
                    sample[
                        "source_path"
                    ]
                ),

            "source_seed":
                int(
                    sample[
                        "source_seed"
                    ]
                ),

            "attack":
                str(
                    sample[
                        "attack"
                    ]
                ),

            "detection_label":
                detection_label,

            "mask_area_fraction":
                float(
                    mask_np.mean()
                ),
        }

        for condition in CONDITIONS:
            output = outputs[
                condition
            ]

            detection_logits = (
                extract_detection_logits(
                    output
                )
            )

            detection_probability = (
                torch.sigmoid(
                    detection_logits
                )
            )

            localization_probability = (
                extract_localization_probability(
                    model,
                    output,
                )
            )

            det_logit_value = float(
                detection_logits[
                    0
                ]
                .detach()
                .cpu()
                .item()
            )

            det_probability_value = float(
                detection_probability[
                    0
                ]
                .detach()
                .cpu()
                .item()
            )

            loc_probability_np = (
                localization_probability[
                    0,
                    0,
                ]
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            row[
                f"{condition}_det_logit"
            ] = det_logit_value

            row[
                f"{condition}_det_probability"
            ] = det_probability_value

            row[
                f"{condition}_loc_mean_probability"
            ] = float(
                loc_probability_np.mean()
            )

            localization_storage[
                condition
            ]["masks"].append(
                mask_np
            )

            localization_storage[
                condition
            ]["probabilities"].append(
                loc_probability_np
            )

        rows.append(
            row
        )

        if (
            (index + 1) % 10 == 0
            or (index + 1) == len(dataset)
        ):
            print(
                f"Processed "
                f"{index + 1:3d}/"
                f"{len(dataset)}"
            )

    return (
        pd.DataFrame(
            rows
        ),
        localization_storage,
    )


# ============================================================
# OVERALL CONDITION SUMMARY
# ============================================================


def build_condition_summary(
    predictions_df,
    localization_storage,
):
    labels = (
        predictions_df[
            "detection_label"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Detection threshold is selected ONCE from NORMAL
    # validation predictions.
    #
    # The exact same threshold is then frozen and applied
    # to WRONG_PAYLOAD and ZERO_EVIDENCE.
    #
    # We do NOT optimize a new threshold for each ablation.
    # --------------------------------------------------------

    normal_probabilities = (
        predictions_df[
            "NORMAL_det_probability"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    frozen_detection_threshold = (
        choose_youden_threshold(
            labels,
            normal_probabilities,
        )
    )

    summary_rows = []

    for condition in CONDITIONS:
        probabilities = (
            predictions_df[
                f"{condition}_det_probability"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        det = detection_metrics(
            labels,
            probabilities,
            frozen_detection_threshold,
        )

        loc = (
            localization_global_metrics(
                localization_storage[
                    condition
                ]["masks"],

                localization_storage[
                    condition
                ]["probabilities"],

                threshold=
                    LOCALIZATION_THRESHOLD,
            )
        )

        row = {
            "condition":
                condition,

            "n_samples":
                int(
                    len(
                        predictions_df
                    )
                ),

            "detection_threshold":
                float(
                    frozen_detection_threshold
                ),

            "localization_threshold":
                float(
                    LOCALIZATION_THRESHOLD
                ),

            **det,
            **loc,
        }

        summary_rows.append(
            row
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    # --------------------------------------------------------
    # Descriptive degradation relative to NORMAL.
    # No post-hoc PASS threshold.
    # --------------------------------------------------------

    normal_row = (
        summary_df[
            summary_df[
                "condition"
            ]
            == "NORMAL"
        ]
        .iloc[0]
    )

    for metric in (
        "roc_auc",
        "balanced_accuracy",
        "f1",
        "pixel_f1",
        "pixel_iou",
    ):
        normal_value = float(
            normal_row[
                metric
            ]
        )

        summary_df[
            f"delta_{metric}_vs_normal"
        ] = (
            summary_df[
                metric
            ]
            - normal_value
        )

    return (
        summary_df,
        frozen_detection_threshold,
    )


# ============================================================
# ATTACK-WISE AUDIT
# ============================================================


def build_attackwise_summary(
    predictions_df,
    frozen_detection_threshold,
):
    rows = []

    authentic_df = (
        predictions_df[
            predictions_df[
                "detection_label"
            ]
            == 0
        ]
    )

    manipulated_df = (
        predictions_df[
            predictions_df[
                "detection_label"
            ]
            == 1
        ]
    )

    for attack in (
        "erase",
        "copy_move",
        "roll",
    ):
        attack_manipulated = (
            manipulated_df[
                manipulated_df[
                    "attack"
                ]
                == attack
            ]
        )

        if attack_manipulated.empty:
            raise RuntimeError(
                f"No manipulated samples for "
                f"attack={attack}"
            )

        # ----------------------------------------------------
        # Detection attack-wise subset:
        #
        # all authentic validation samples
        # +
        # manipulated samples of this attack
        #
        # This avoids comparing only one class.
        # ----------------------------------------------------

        subset = pd.concat(
            [
                authentic_df,
                attack_manipulated,
            ],
            axis=0,
            ignore_index=True,
        )

        labels = (
            subset[
                "detection_label"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        for condition in CONDITIONS:
            probabilities = (
                subset[
                    f"{condition}_det_probability"
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            det = detection_metrics(
                labels,
                probabilities,
                frozen_detection_threshold,
            )

            # ------------------------------------------------
            # Localization attack-wise:
            # manipulated samples only.
            #
            # Per-sample pixel metrics are calculated from
            # predictions saved during inference below.
            # Since full maps are not stored in the CSV,
            # attack-wise localization is handled later from
            # per-sample scalar overlap fields.
            # ------------------------------------------------

            rows.append(
                {
                    "attack":
                        attack,

                    "condition":
                        condition,

                    "n_authentic":
                        int(
                            len(
                                authentic_df
                            )
                        ),

                    "n_manipulated":
                        int(
                            len(
                                attack_manipulated
                            )
                        ),

                    **det,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PER-SAMPLE LOCALIZATION OVERLAP
# ============================================================


def add_per_sample_localization_metrics(
    predictions_df,
    localization_storage,
):
    df = predictions_df.copy()

    for condition in CONDITIONS:
        sample_f1 = []
        sample_iou = []

        masks = (
            localization_storage[
                condition
            ]["masks"]
        )

        probabilities = (
            localization_storage[
                condition
            ]["probabilities"]
        )

        for mask, probability in zip(
            masks,
            probabilities,
        ):
            target = (
                np.asarray(mask)
                >= 0.5
            )

            prediction = (
                np.asarray(
                    probability
                )
                >= LOCALIZATION_THRESHOLD
            )

            tp = int(
                np.logical_and(
                    prediction,
                    target,
                ).sum()
            )

            fp = int(
                np.logical_and(
                    prediction,
                    np.logical_not(
                        target
                    ),
                ).sum()
            )

            fn = int(
                np.logical_and(
                    np.logical_not(
                        prediction
                    ),
                    target,
                ).sum()
            )

            f1_den = (
                2 * tp
                + fp
                + fn
            )

            iou_den = (
                tp
                + fp
                + fn
            )

            if f1_den > 0:
                f1_value = (
                    2.0 * tp
                    / f1_den
                )
            else:
                # Both target and prediction empty.
                f1_value = 1.0

            if iou_den > 0:
                iou_value = (
                    float(tp)
                    / iou_den
                )
            else:
                iou_value = 1.0

            sample_f1.append(
                float(
                    f1_value
                )
            )

            sample_iou.append(
                float(
                    iou_value
                )
            )

        df[
            f"{condition}_loc_sample_f1"
        ] = sample_f1

        df[
            f"{condition}_loc_sample_iou"
        ] = sample_iou

    return df


def add_attackwise_localization(
    attack_df,
    predictions_df,
):
    output = attack_df.copy()

    output[
        "manipulated_loc_mean_f1"
    ] = np.nan

    output[
        "manipulated_loc_mean_iou"
    ] = np.nan

    output[
        "manipulated_loc_median_f1"
    ] = np.nan

    output[
        "manipulated_loc_median_iou"
    ] = np.nan

    for row_index in output.index:
        attack = output.loc[
            row_index,
            "attack",
        ]

        condition = output.loc[
            row_index,
            "condition",
        ]

        subset = predictions_df[
            (
                predictions_df[
                    "detection_label"
                ]
                == 1
            )
            &
            (
                predictions_df[
                    "attack"
                ]
                == attack
            )
        ]

        f1_values = (
            subset[
                f"{condition}_loc_sample_f1"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        iou_values = (
            subset[
                f"{condition}_loc_sample_iou"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        output.loc[
            row_index,
            "manipulated_loc_mean_f1",
        ] = float(
            np.mean(
                f1_values
            )
        )

        output.loc[
            row_index,
            "manipulated_loc_mean_iou",
        ] = float(
            np.mean(
                iou_values
            )
        )

        output.loc[
            row_index,
            "manipulated_loc_median_f1",
        ] = float(
            np.median(
                f1_values
            )
        )

        output.loc[
            row_index,
            "manipulated_loc_median_iou",
        ] = float(
            np.median(
                iou_values
            )
        )

    return output


# ============================================================
# PAIRED AUTHENTIC / MANIPULATED AUDIT
# ============================================================


def build_paired_audit(
    predictions_df,
):
    rows = []

    grouped = predictions_df.groupby(
        "source_index",
        sort=True,
    )

    for source_index, group in grouped:
        if len(group) != 2:
            raise RuntimeError(
                "Expected exactly two active samples "
                f"for source_index={source_index}, "
                f"got {len(group)}."
            )

        authentic = group[
            group[
                "detection_label"
            ]
            == 0
        ]

        manipulated = group[
            group[
                "detection_label"
            ]
            == 1
        ]

        if (
            len(authentic) != 1
            or len(manipulated) != 1
        ):
            raise RuntimeError(
                "Invalid authentic/manipulated pair "
                f"for source_index={source_index}."
            )

        authentic = authentic.iloc[
            0
        ]

        manipulated = manipulated.iloc[
            0
        ]

        row = {
            "source_index":
                int(
                    source_index
                ),

            "sample_key":
                str(
                    authentic[
                        "sample_key"
                    ]
                ),

            "attack":
                str(
                    manipulated[
                        "attack"
                    ]
                ),

            "mask_area_fraction":
                float(
                    manipulated[
                        "mask_area_fraction"
                    ]
                ),
        }

        for condition in CONDITIONS:
            authentic_score = float(
                authentic[
                    f"{condition}_det_probability"
                ]
            )

            manipulated_score = float(
                manipulated[
                    f"{condition}_det_probability"
                ]
            )

            row[
                f"{condition}_authentic_score"
            ] = authentic_score

            row[
                f"{condition}_manipulated_score"
            ] = manipulated_score

            row[
                f"{condition}_paired_delta"
            ] = (
                manipulated_score
                - authentic_score
            )

        rows.append(
            row
        )

    paired_df = pd.DataFrame(
        rows
    )

    if len(paired_df) != EXPECTED_VAL_SOURCES:
        raise RuntimeError(
            "Unexpected paired source count: "
            f"{len(paired_df)}"
        )

    return paired_df


# ============================================================
# REPORTING
# ============================================================


def print_condition_summary(
    summary_df,
):
    print()
    print("=" * 76)
    print("OVERALL VALIDATION CAUSAL AUDIT")
    print("=" * 76)

    for _, row in summary_df.iterrows():
        print()
        print(
            f"[{row['condition']}]"
        )

        print(
            "Detection ROC-AUC       : "
            f"{row['roc_auc']:.6f}"
        )

        print(
            "Detection Balanced Acc  : "
            f"{row['balanced_accuracy']:.6f}"
        )

        print(
            "Detection F1            : "
            f"{row['f1']:.6f}"
        )

        print(
            "Sensitivity             : "
            f"{row['sensitivity']:.6f}"
        )

        print(
            "Specificity             : "
            f"{row['specificity']:.6f}"
        )

        print(
            "Detection TN/FP/FN/TP    : "
            f"{int(row['tn'])}/"
            f"{int(row['fp'])}/"
            f"{int(row['fn'])}/"
            f"{int(row['tp'])}"
        )

        print(
            "Localization Pixel F1    : "
            f"{row['pixel_f1']:.6f}"
        )

        print(
            "Localization Pixel IoU   : "
            f"{row['pixel_iou']:.6f}"
        )

        print(
            "Delta AUC vs NORMAL      : "
            f"{row['delta_roc_auc_vs_normal']:+.6f}"
        )

        print(
            "Delta Loc F1 vs NORMAL   : "
            f"{row['delta_pixel_f1_vs_normal']:+.6f}"
        )


def print_attackwise(
    attack_df,
):
    print()
    print("=" * 76)
    print("ATTACK-WISE VALIDATION AUDIT")
    print("=" * 76)

    for attack in (
        "erase",
        "copy_move",
        "roll",
    ):
        print()
        print(
            f"ATTACK = {attack.upper()}"
        )

        subset = attack_df[
            attack_df[
                "attack"
            ]
            == attack
        ]

        for _, row in subset.iterrows():
            print(
                f"  {row['condition']:14s} "
                f"AUC={row['roc_auc']:.6f}  "
                f"BalAcc="
                f"{row['balanced_accuracy']:.6f}  "
                f"LocMeanF1="
                f"{row['manipulated_loc_mean_f1']:.6f}"
            )


def print_paired_summary(
    paired_df,
):
    print()
    print("=" * 76)
    print("PAIRED AUTHENTIC / MANIPULATED AUDIT")
    print("=" * 76)

    for condition in CONDITIONS:
        values = (
            paired_df[
                f"{condition}_paired_delta"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        positive_fraction = float(
            np.mean(
                values > 0.0
            )
        )

        print()
        print(
            f"[{condition}]"
        )

        print(
            "Median manipulated-authentic "
            "score delta : "
            f"{np.median(values):+.6f}"
        )

        print(
            "Mean manipulated-authentic "
            "score delta   : "
            f"{np.mean(values):+.6f}"
        )

        print(
            "Fraction delta > 0                    : "
            f"{positive_fraction * 100.0:.2f}%"
        )


# ============================================================
# MAIN
# ============================================================


def main():
    print()
    print("=" * 76)
    print(
        "E5.3R-4C-2 VALIDATION "
        "CAUSAL & SHORTCUT AUDIT"
    )
    print("=" * 76)

    print()
    print(
        "This audit does NOT train the model."
    )

    print(
        "This audit does NOT access Test."
    )

    print(
        "This audit does NOT modify the checkpoint."
    )

    print(
        "No post-hoc causal PASS threshold is used."
    )

    set_deterministic(
        AUDIT_SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        f"Device             : {device}"
    )

    print(
        f"Manifest           : {MANIFEST_PATH}"
    )

    print(
        f"Joint checkpoint   : {CHECKPOINT_PATH}"
    )

    print(
        f"Stego checkpoint   : {STEGO_CHECKPOINT_PATH}"
    )

    # --------------------------------------------------------
    # Validation-only source records.
    # --------------------------------------------------------

    records = (
        build_validation_records()
    )

    print()
    print(
        "Validation sources : "
        f"{len(records)}"
    )

    dataset = ActiveSSPGDataset(
        records=records,
        base_seed=VAL_SEED,
    )

    print(
        "Validation active  : "
        f"{len(dataset)}"
    )

    if len(dataset) != EXPECTED_VAL_ACTIVE:
        raise RuntimeError(
            "Unexpected active validation count: "
            f"{len(dataset)} != "
            f"{EXPECTED_VAL_ACTIVE}"
        )

    # --------------------------------------------------------
    # Model.
    # --------------------------------------------------------

    print()
    print(
        "Loading frozen best checkpoint..."
    )

    model, checkpoint = (
        build_model(
            device
        )
    )

    print(
        "Checkpoint loaded strictly."
    )

    # --------------------------------------------------------
    # Inference.
    # --------------------------------------------------------

    (
        predictions_df,
        localization_storage,
    ) = run_validation(
        model,
        dataset,
        device,
    )

    # --------------------------------------------------------
    # Per-sample localization.
    # --------------------------------------------------------

    predictions_df = (
        add_per_sample_localization_metrics(
            predictions_df,
            localization_storage,
        )
    )

    # --------------------------------------------------------
    # Overall summary.
    # --------------------------------------------------------

    (
        summary_df,
        frozen_detection_threshold,
    ) = build_condition_summary(
        predictions_df,
        localization_storage,
    )

    # --------------------------------------------------------
    # Attack-wise.
    # --------------------------------------------------------

    attack_df = (
        build_attackwise_summary(
            predictions_df,
            frozen_detection_threshold,
        )
    )

    attack_df = (
        add_attackwise_localization(
            attack_df,
            predictions_df,
        )
    )

    # --------------------------------------------------------
    # Paired audit.
    # --------------------------------------------------------

    paired_df = (
        build_paired_audit(
            predictions_df
        )
    )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    attack_df.to_csv(
        ATTACK_PATH,
        index=False,
    )

    paired_df.to_csv(
        PAIRED_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Compact machine-readable audit record.
    # --------------------------------------------------------

    normal = (
        summary_df[
            summary_df[
                "condition"
            ]
            == "NORMAL"
        ]
        .iloc[0]
    )

    wrong = (
        summary_df[
            summary_df[
                "condition"
            ]
            == "WRONG_PAYLOAD"
        ]
        .iloc[0]
    )

    zero = (
        summary_df[
            summary_df[
                "condition"
            ]
            == "ZERO_EVIDENCE"
        ]
        .iloc[0]
    )

    audit_record = {
        "experiment":
            "E5.3R-4C-2",

        "scope":
            "validation_only",

        "training_performed":
            False,

        "optimizer_step_performed":
            False,

        "checkpoint_modified":
            False,

        "test_accessed":
            False,

        "posthoc_causal_pass_threshold":
            False,

        "validation_sources":
            int(
                len(records)
            ),

        "validation_active_samples":
            int(
                len(dataset)
            ),

        "normal_selected_detection_threshold":
            float(
                frozen_detection_threshold
            ),

        "localization_threshold":
            float(
                LOCALIZATION_THRESHOLD
            ),

        "normal_detection_auc":
            float(
                normal[
                    "roc_auc"
                ]
            ),

        "wrong_payload_detection_auc":
            float(
                wrong[
                    "roc_auc"
                ]
            ),

        "zero_evidence_detection_auc":
            float(
                zero[
                    "roc_auc"
                ]
            ),

        "wrong_payload_auc_delta":
            float(
                wrong[
                    "roc_auc"
                ]
                - normal[
                    "roc_auc"
                ]
            ),

        "zero_evidence_auc_delta":
            float(
                zero[
                    "roc_auc"
                ]
                - normal[
                    "roc_auc"
                ]
            ),

        "normal_localization_f1":
            float(
                normal[
                    "pixel_f1"
                ]
            ),

        "wrong_payload_localization_f1":
            float(
                wrong[
                    "pixel_f1"
                ]
            ),

        "zero_evidence_localization_f1":
            float(
                zero[
                    "pixel_f1"
                ]
            ),

        "wrong_payload_loc_f1_delta":
            float(
                wrong[
                    "pixel_f1"
                ]
                - normal[
                    "pixel_f1"
                ]
            ),

        "zero_evidence_loc_f1_delta":
            float(
                zero[
                    "pixel_f1"
                ]
                - normal[
                    "pixel_f1"
                ]
            ),
    }

    with open(
        AUDIT_JSON_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            audit_record,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Reporting.
    # --------------------------------------------------------

    print_condition_summary(
        summary_df
    )

    print_attackwise(
        attack_df
    )

    print_paired_summary(
        paired_df
    )

    print()
    print("=" * 76)
    print("OUTPUT FILES")
    print("=" * 76)

    print(
        PREDICTIONS_PATH
    )

    print(
        SUMMARY_PATH
    )

    print(
        ATTACK_PATH
    )

    print(
        PAIRED_PATH
    )

    print(
        AUDIT_JSON_PATH
    )

    print()
    print("=" * 76)
    print(
        "E5.3R-4C-2 VALIDATION "
        "CAUSAL & SHORTCUT AUDIT: COMPLETE"
    )
    print("=" * 76)

    print()
    print(
        "Scientific boundary:"
    )

    print(
        "These are validation-only causal diagnostics."
    )

    print(
        "They are NOT Test results."
    )

    print(
        "They are NOT final paper performance claims."
    )

    print(
        "No causal PASS/FAIL threshold was selected "
        "after observing the results."
    )


if __name__ == "__main__":
    main()
