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
)

from preprocessing.active_sspg_dataset import (
    ActiveSSPGDataset,
)

from models.sspg_joint_model_v2 import (
    SSPGJointModelV2,
)


# ============================================================
# E5.3R-4C-3
# REPRESENTATION ABLATION AUDIT
#
# PURPOSE
# ------------------------------------------------------------
# Diagnose how much the already-trained model depends on:
#
#   1. visual representation
#   2. provenance evidence
#
# This is NOT a new training experiment.
#
# This script:
#   - does NOT train
#   - does NOT call optimizer.step()
#   - does NOT modify any checkpoint
#   - does NOT access Test
#   - does NOT tune thresholds
#
# CONDITIONS
# ------------------------------------------------------------
#
# FULL
#     visual tokens + correct provenance evidence
#
# VISUAL_ONLY
#     visual tokens + zero provenance evidence
#
# WRONG_PAYLOAD
#     visual tokens + provenance evidence computed using
#     deterministically complemented payload
#
# EVIDENCE_ONLY_ABLATION
#     zero visual tokens + correct provenance evidence
#
# IMPORTANT
# ------------------------------------------------------------
# EVIDENCE_ONLY_ABLATION is a representation ablation.
#
# It must NOT be described as a formally trained
# "provenance-only model", because the downstream network
# was not trained under zero-visual-token inputs.
#
# PRIMARY DIAGNOSTICS
# ------------------------------------------------------------
#
# Detection provenance contribution:
#
#     FULL detection logit
#       - VISUAL_ONLY detection logit
#
# Localization provenance contribution:
#
#     FULL localization probability
#       - VISUAL_ONLY localization probability
#
# For manipulated samples:
#
#     mean contribution inside GT mask
#       vs
#     mean contribution outside GT mask
#
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

JOINT_CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "e53r_joint_v1_best.pth"
)

STEGO_CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "stego_structured_v33_128bit_best.pth"
)

C4C2_PREDICTIONS_PATH = (
    ROOT
    / "results"
    / "csv"
    / "e53r_4c2_validation_causal_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "csv"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "e53r_4c3_representation_predictions.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "e53r_4c3_representation_summary.csv"
)

CONTRIBUTION_PATH = (
    OUTPUT_DIR
    / "e53r_4c3_provenance_contribution.csv"
)

ATTACKWISE_PATH = (
    OUTPUT_DIR
    / "e53r_4c3_attackwise_contribution.csv"
)

AUDIT_JSON_PATH = (
    OUTPUT_DIR
    / "e53r_4c3_representation_audit.json"
)


VAL_SEED = 20260923
AUDIT_SEED = 20260927

EXPECTED_VAL_SOURCES = 62
EXPECTED_VAL_ACTIVE = 124

# ------------------------------------------------------------
# Frozen E5.3R-4C-1 validation-only threshold.
#
# The value printed in the completed formal training result
# was approximately 0.960035.
#
# We additionally recover the higher-precision value from
# 4C-2 predictions if possible, WITHOUT re-optimizing it.
# ------------------------------------------------------------

FROZEN_DETECTION_THRESHOLD_APPROX = 0.960035

LOCALIZATION_THRESHOLD = 0.5

CONDITIONS = (
    "FULL",
    "VISUAL_ONLY",
    "WRONG_PAYLOAD",
    "EVIDENCE_ONLY_ABLATION",
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
# PATH
# ============================================================


def resolve_image_path(value):
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
# VALIDATION RECORDS
# ============================================================


def build_validation_records():
    if not MANIFEST_PATH.exists():
        raise RuntimeError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    df = pd.read_csv(
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
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "Manifest missing columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Explicitly exclude Test before source construction.
    # --------------------------------------------------------

    development_df = df[
        df["split"]
        .astype(str)
        .str.lower()
        .isin(
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
        record = {
            "sample_id":
                str(row["sample_id"]),

            "group":
                str(row["group"]),

            "image_path":
                resolve_image_path(
                    row["image_path"]
                ),
        }

        records.append(
            record
        )

    if len(records) != EXPECTED_VAL_SOURCES:
        raise RuntimeError(
            "Unexpected validation source count: "
            f"{len(records)}"
        )

    for record in records:
        path = Path(
            record["image_path"]
        )

        if not path.exists():
            raise RuntimeError(
                f"Missing validation image: {path}"
            )

    return records


# ============================================================
# CHECKPOINT
# ============================================================


def extract_state_dict(checkpoint):
    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            "Checkpoint must be a dictionary."
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

    if checkpoint:
        if all(
            torch.is_tensor(value)
            for value in checkpoint.values()
        ):
            return checkpoint

    raise RuntimeError(
        "Could not locate model state_dict."
    )


def build_model(device):
    if not JOINT_CHECKPOINT_PATH.exists():
        raise RuntimeError(
            "Joint checkpoint not found: "
            f"{JOINT_CHECKPOINT_PATH}"
        )

    if not STEGO_CHECKPOINT_PATH.exists():
        raise RuntimeError(
            "Stego checkpoint not found: "
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
        JOINT_CHECKPOINT_PATH,
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

    if model.frontend.frontend.decoder.training:
        raise RuntimeError(
            "Decoder unexpectedly in train mode."
        )

    if model.frontend.frontend.visual.vit.training:
        raise RuntimeError(
            "ViT unexpectedly in train mode."
        )

    return model


# ============================================================
# COMMON DOWNSTREAM FORWARD
# ============================================================


@torch.no_grad()
def downstream_forward(
    model,
    visual_tokens,
    patch_evidence,
):
    """
    Recompute:
        fusion
        graph
        GNN
        heads

    from explicitly supplied visual representation
    and explicitly supplied provenance evidence.
    """

    graph_frontend = (
        model.frontend
    )

    frontend = (
        graph_frontend.frontend
    )

    fused_tokens = (
        frontend.fusion(
            visual_tokens,
            patch_evidence,
        )
    )

    graph = (
        graph_frontend.graph_builder(
            fused_tokens,
            patch_evidence,
        )
    )

    graph_tokens = (
        graph_frontend.gnn(
            graph["node_features"],
            graph["edge_index"],
            graph["edge_attr"],
        )
    )

    head_output = (
        model.heads(
            graph_tokens
        )
    )

    return {
        "fused_tokens":
            fused_tokens,

        "graph_tokens":
            graph_tokens,

        "detection_logits":
            head_output[
                "detection_logits"
            ],

        "localization_grid_logits":
            head_output[
                "localization_grid_logits"
            ],
    }


# ============================================================
# REPRESENTATION EXTRACTION
# ============================================================


@torch.no_grad()
def extract_representations(
    model,
    received,
    payload,
):
    """
    Extract the normal visual representation and the normal
    provenance evidence exactly through SSPGFrontendV2.

    Also compute wrong-payload evidence from the SAME
    decoder probabilities.
    """

    frontend = (
        model.frontend.frontend
    )

    # --------------------------------------------------------
    # Frozen decoder.
    # --------------------------------------------------------

    decoder_logits = (
        frontend.decoder(
            received
        )
    )

    decoder_prob = torch.sigmoid(
        decoder_logits
    )

    # --------------------------------------------------------
    # Correct provenance evidence.
    # --------------------------------------------------------

    normal_evidence = (
        frontend.projector(
            decoder_prob,
            payload,
        )
    )

    normal_patch_evidence = (
        normal_evidence[
            "patch_evidence"
        ]
    )

    # --------------------------------------------------------
    # Wrong payload.
    # Same received image and SAME decoder output.
    # --------------------------------------------------------

    wrong_payload = (
        1.0
        - payload
    )

    wrong_evidence = (
        frontend.projector(
            decoder_prob,
            wrong_payload,
        )
    )

    wrong_patch_evidence = (
        wrong_evidence[
            "patch_evidence"
        ]
    )

    # --------------------------------------------------------
    # Visual path.
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

    return {
        "decoder_prob":
            decoder_prob,

        "visual_tokens":
            visual_tokens,

        "normal_patch_evidence":
            normal_patch_evidence,

        "wrong_patch_evidence":
            wrong_patch_evidence,
    }


# ============================================================
# FOUR CONDITIONS
# ============================================================


@torch.no_grad()
def forward_conditions(
    model,
    received,
    payload,
):
    rep = extract_representations(
        model,
        received,
        payload,
    )

    visual_tokens = (
        rep[
            "visual_tokens"
        ]
    )

    normal_evidence = (
        rep[
            "normal_patch_evidence"
        ]
    )

    wrong_evidence = (
        rep[
            "wrong_patch_evidence"
        ]
    )

    zero_evidence = torch.zeros_like(
        normal_evidence
    )

    zero_visual = torch.zeros_like(
        visual_tokens
    )

    # --------------------------------------------------------
    # FULL
    # --------------------------------------------------------

    full = downstream_forward(
        model,
        visual_tokens,
        normal_evidence,
    )

    # --------------------------------------------------------
    # VISUAL ONLY
    # --------------------------------------------------------

    visual_only = downstream_forward(
        model,
        visual_tokens,
        zero_evidence,
    )

    # --------------------------------------------------------
    # WRONG PAYLOAD
    # --------------------------------------------------------

    wrong_payload = downstream_forward(
        model,
        visual_tokens,
        wrong_evidence,
    )

    # --------------------------------------------------------
    # EVIDENCE-ONLY ABLATION
    #
    # Diagnostic OOD representation ablation only.
    # --------------------------------------------------------

    evidence_only = downstream_forward(
        model,
        zero_visual,
        normal_evidence,
    )

    return {
        "FULL":
            full,

        "VISUAL_ONLY":
            visual_only,

        "WRONG_PAYLOAD":
            wrong_payload,

        "EVIDENCE_ONLY_ABLATION":
            evidence_only,
    }


# ============================================================
# OUTPUT HELPERS
# ============================================================


def detection_logit(output):
    value = output[
        "detection_logits"
    ]

    return value.reshape(
        -1
    )


def localization_probability_256(
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

    return torch.sigmoid(
        logits_256
    )


# ============================================================
# FROZEN THRESHOLD RECOVERY
# ============================================================


def recover_frozen_threshold():
    """
    Prefer recovering the already-selected NORMAL threshold
    from the completed 4C-2 output.

    IMPORTANT:
        This function does NOT optimize a new threshold.
    """

    if not C4C2_PREDICTIONS_PATH.exists():
        print(
            "4C-2 predictions not found; "
            "using frozen approximate threshold "
            f"{FROZEN_DETECTION_THRESHOLD_APPROX:.6f}"
        )

        return float(
            FROZEN_DETECTION_THRESHOLD_APPROX
        )

    df = pd.read_csv(
        C4C2_PREDICTIONS_PATH
    )

    required = {
        "detection_label",
        "NORMAL_det_probability",
    }

    if not required.issubset(
        set(df.columns)
    ):
        print(
            "4C-2 predictions lack required columns; "
            "using frozen approximate threshold."
        )

        return float(
            FROZEN_DETECTION_THRESHOLD_APPROX
        )

    # --------------------------------------------------------
    # We intentionally do NOT recompute Youden here.
    #
    # The 4C-1 printed frozen threshold is used.
    # --------------------------------------------------------

    return float(
        FROZEN_DETECTION_THRESHOLD_APPROX
    )


# ============================================================
# METRICS
# ============================================================


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

    sensitivity = (
        float(tp)
        / float(tp + fn)
        if (tp + fn)
        else float("nan")
    )

    specificity = (
        float(tn)
        / float(tn + fp)
        if (tn + fp)
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


def global_localization_metrics(
    masks,
    probabilities,
):
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for mask, probability in zip(
        masks,
        probabilities,
    ):
        target = (
            np.asarray(mask)
            >= 0.5
        )

        prediction = (
            np.asarray(probability)
            >= LOCALIZATION_THRESHOLD
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

    pixel_f1 = (
        2.0 * tp / f1_den
        if f1_den > 0
        else 0.0
    )

    pixel_iou = (
        float(tp) / iou_den
        if iou_den > 0
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
# MAIN INFERENCE
# ============================================================


@torch.no_grad()
def run_audit(
    model,
    dataset,
    device,
):
    rows = []

    localization_store = {
        condition: {
            "mask": [],
            "probability": [],
        }
        for condition in CONDITIONS
    }

    contribution_rows = []

    print()
    print("=" * 76)
    print("4C-3 VALIDATION REPRESENTATION ABLATION")
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

        label = int(
            sample[
                "detection_label"
            ].item()
        )

        outputs = forward_conditions(
            model,
            received,
            payload,
        )

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

            "attack":
                str(
                    sample[
                        "attack"
                    ]
                ),

            "detection_label":
                label,

            "mask_area_fraction":
                float(
                    mask_np.mean()
                ),
        }

        loc_maps = {}

        for condition in CONDITIONS:
            output = outputs[
                condition
            ]

            logit = detection_logit(
                output
            )

            probability = torch.sigmoid(
                logit
            )

            loc_probability = (
                localization_probability_256(
                    model,
                    output,
                )
            )

            loc_np = (
                loc_probability[
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
            ] = float(
                logit[
                    0
                ].detach()
                .cpu()
                .item()
            )

            row[
                f"{condition}_det_probability"
            ] = float(
                probability[
                    0
                ].detach()
                .cpu()
                .item()
            )

            row[
                f"{condition}_loc_mean_probability"
            ] = float(
                loc_np.mean()
            )

            localization_store[
                condition
            ]["mask"].append(
                mask_np
            )

            localization_store[
                condition
            ]["probability"].append(
                loc_np
            )

            loc_maps[
                condition
            ] = loc_np

        # ----------------------------------------------------
        # Detection provenance contribution.
        #
        # Correct provenance evidence added to same visual
        # representation:
        #
        # FULL - VISUAL_ONLY
        # ----------------------------------------------------

        det_logit_contribution = (
            row[
                "FULL_det_logit"
            ]
            - row[
                "VISUAL_ONLY_det_logit"
            ]
        )

        det_probability_contribution = (
            row[
                "FULL_det_probability"
            ]
            - row[
                "VISUAL_ONLY_det_probability"
            ]
        )

        # ----------------------------------------------------
        # Localization provenance contribution map.
        # ----------------------------------------------------

        loc_contribution = (
            loc_maps[
                "FULL"
            ]
            - loc_maps[
                "VISUAL_ONLY"
            ]
        )

        if label == 1:
            inside = (
                mask_np
                >= 0.5
            )

            outside = np.logical_not(
                inside
            )

            if not inside.any():
                raise RuntimeError(
                    "Manipulated sample has empty mask."
                )

            inside_mean = float(
                loc_contribution[
                    inside
                ].mean()
            )

            outside_mean = float(
                loc_contribution[
                    outside
                ].mean()
            )

            spatial_contrast = (
                inside_mean
                - outside_mean
            )

            full_inside = float(
                loc_maps[
                    "FULL"
                ][inside].mean()
            )

            visual_inside = float(
                loc_maps[
                    "VISUAL_ONLY"
                ][inside].mean()
            )

        else:
            inside_mean = float(
                "nan"
            )

            outside_mean = float(
                loc_contribution.mean()
            )

            spatial_contrast = float(
                "nan"
            )

            full_inside = float(
                "nan"
            )

            visual_inside = float(
                "nan"
            )

        contribution_rows.append(
            {
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

                "attack":
                    str(
                        sample[
                            "attack"
                        ]
                    ),

                "detection_label":
                    label,

                "det_logit_contribution":
                    float(
                        det_logit_contribution
                    ),

                "det_probability_contribution":
                    float(
                        det_probability_contribution
                    ),

                "loc_contribution_inside":
                    inside_mean,

                "loc_contribution_outside":
                    outside_mean,

                "loc_spatial_contrast":
                    spatial_contrast,

                "full_loc_inside":
                    full_inside,

                "visual_only_loc_inside":
                    visual_inside,
            }
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
        pd.DataFrame(
            contribution_rows
        ),
        localization_store,
    )


# ============================================================
# SUMMARY
# ============================================================


def build_summary(
    predictions_df,
    localization_store,
    frozen_threshold,
):
    labels = (
        predictions_df[
            "detection_label"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    rows = []

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
            frozen_threshold,
        )

        loc = global_localization_metrics(
            localization_store[
                condition
            ]["mask"],

            localization_store[
                condition
            ]["probability"],
        )

        rows.append(
            {
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
                        frozen_threshold
                    ),

                "localization_threshold":
                    float(
                        LOCALIZATION_THRESHOLD
                    ),

                **det,
                **loc,
            }
        )

    summary = pd.DataFrame(
        rows
    )

    full = (
        summary[
            summary[
                "condition"
            ]
            == "FULL"
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
        summary[
            f"delta_{metric}_vs_full"
        ] = (
            summary[
                metric
            ]
            - float(
                full[
                    metric
                ]
            )
        )

    return summary


# ============================================================
# CONTRIBUTION SUMMARY
# ============================================================


def summarize_contribution(
    contribution_df,
):
    authentic = (
        contribution_df[
            contribution_df[
                "detection_label"
            ]
            == 0
        ]
    )

    manipulated = (
        contribution_df[
            contribution_df[
                "detection_label"
            ]
            == 1
        ]
    )

    result = {}

    for name, subset in (
        (
            "authentic",
            authentic,
        ),
        (
            "manipulated",
            manipulated,
        ),
    ):
        values = (
            subset[
                "det_logit_contribution"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        result[
            f"{name}_det_logit_contribution_mean"
        ] = float(
            np.mean(
                values
            )
        )

        result[
            f"{name}_det_logit_contribution_median"
        ] = float(
            np.median(
                values
            )
        )

    manipulated_contrast = (
        manipulated[
            "loc_spatial_contrast"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    result[
        "manipulated_loc_spatial_contrast_mean"
    ] = float(
        np.mean(
            manipulated_contrast
        )
    )

    result[
        "manipulated_loc_spatial_contrast_median"
    ] = float(
        np.median(
            manipulated_contrast
        )
    )

    result[
        "manipulated_loc_spatial_contrast_positive_fraction"
    ] = float(
        np.mean(
            manipulated_contrast
            > 0.0
        )
    )

    return result


# ============================================================
# ATTACK-WISE CONTRIBUTION
# ============================================================


def build_attackwise_contribution(
    contribution_df,
):
    manipulated = (
        contribution_df[
            contribution_df[
                "detection_label"
            ]
            == 1
        ]
    )

    rows = []

    for attack in (
        "erase",
        "copy_move",
        "roll",
    ):
        subset = (
            manipulated[
                manipulated[
                    "attack"
                ]
                == attack
            ]
        )

        if subset.empty:
            raise RuntimeError(
                f"No samples for attack={attack}"
            )

        det_values = (
            subset[
                "det_logit_contribution"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        inside_values = (
            subset[
                "loc_contribution_inside"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        outside_values = (
            subset[
                "loc_contribution_outside"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        contrast_values = (
            subset[
                "loc_spatial_contrast"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        rows.append(
            {
                "attack":
                    attack,

                "n_samples":
                    int(
                        len(
                            subset
                        )
                    ),

                "det_logit_contribution_mean":
                    float(
                        np.mean(
                            det_values
                        )
                    ),

                "det_logit_contribution_median":
                    float(
                        np.median(
                            det_values
                        )
                    ),

                "loc_contribution_inside_mean":
                    float(
                        np.mean(
                            inside_values
                        )
                    ),

                "loc_contribution_outside_mean":
                    float(
                        np.mean(
                            outside_values
                        )
                    ),

                "loc_spatial_contrast_mean":
                    float(
                        np.mean(
                            contrast_values
                        )
                    ),

                "loc_spatial_contrast_median":
                    float(
                        np.median(
                            contrast_values
                        )
                    ),

                "loc_spatial_contrast_positive_fraction":
                    float(
                        np.mean(
                            contrast_values
                            > 0.0
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# 4C-2 REPRODUCTION CHECK
# ============================================================


def compare_with_4c2(
    predictions_df,
):
    if not C4C2_PREDICTIONS_PATH.exists():
        print()
        print(
            "4C-2 reproduction comparison skipped: "
            "predictions file not found."
        )

        return None

    old = pd.read_csv(
        C4C2_PREDICTIONS_PATH
    )

    if len(old) != len(
        predictions_df
    ):
        raise RuntimeError(
            "4C-2 row count mismatch."
        )

    required = {
        "sample_key",
        "NORMAL_det_probability",
        "ZERO_EVIDENCE_det_probability",
        "WRONG_PAYLOAD_det_probability",
    }

    if not required.issubset(
        set(
            old.columns
        )
    ):
        print()
        print(
            "4C-2 reproduction comparison skipped: "
            "required columns absent."
        )

        return None

    old_keys = (
        old[
            "sample_key"
        ]
        .astype(str)
        .tolist()
    )

    new_keys = (
        predictions_df[
            "sample_key"
        ]
        .astype(str)
        .tolist()
    )

    if old_keys != new_keys:
        raise RuntimeError(
            "4C-2 / 4C-3 sample ordering mismatch."
        )

    comparisons = {
        "FULL_vs_4C2_NORMAL":
            (
                "FULL_det_probability",
                "NORMAL_det_probability",
            ),

        "VISUAL_ONLY_vs_4C2_ZERO_EVIDENCE":
            (
                "VISUAL_ONLY_det_probability",
                "ZERO_EVIDENCE_det_probability",
            ),

        "WRONG_PAYLOAD_vs_4C2_WRONG_PAYLOAD":
            (
                "WRONG_PAYLOAD_det_probability",
                "WRONG_PAYLOAD_det_probability",
            ),
    }

    result = {}

    print()
    print("=" * 76)
    print("4C-2 REPRODUCTION CHECK")
    print("=" * 76)

    for name, (
        new_column,
        old_column,
    ) in comparisons.items():
        new_values = (
            predictions_df[
                new_column
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        old_values = (
            old[
                old_column
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        max_abs_diff = float(
            np.max(
                np.abs(
                    new_values
                    - old_values
                )
            )
        )

        result[
            name
        ] = max_abs_diff

        print(
            f"{name:42s}: "
            f"max abs diff = "
            f"{max_abs_diff:.10f}"
        )

    return result


# ============================================================
# PRINT
# ============================================================


def print_summary(
    summary_df,
):
    print()
    print("=" * 76)
    print("REPRESENTATION ABLATION SUMMARY")
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
            "Localization Pixel F1    : "
            f"{row['pixel_f1']:.6f}"
        )

        print(
            "Localization Pixel IoU   : "
            f"{row['pixel_iou']:.6f}"
        )

        print(
            "Delta AUC vs FULL        : "
            f"{row['delta_roc_auc_vs_full']:+.6f}"
        )

        print(
            "Delta Loc F1 vs FULL     : "
            f"{row['delta_pixel_f1_vs_full']:+.6f}"
        )


def print_contribution(
    result,
):
    print()
    print("=" * 76)
    print("PROVENANCE CONTRIBUTION")
    print("=" * 76)

    print()
    print(
        "Detection logit contribution:"
    )

    print(
        "  Authentic mean   : "
        f"{result['authentic_det_logit_contribution_mean']:+.6f}"
    )

    print(
        "  Authentic median : "
        f"{result['authentic_det_logit_contribution_median']:+.6f}"
    )

    print(
        "  Manipulated mean : "
        f"{result['manipulated_det_logit_contribution_mean']:+.6f}"
    )

    print(
        "  Manipulated median: "
        f"{result['manipulated_det_logit_contribution_median']:+.6f}"
    )

    print()
    print(
        "Localization provenance spatial contribution:"
    )

    print(
        "  Mean inside-outside contrast   : "
        f"{result['manipulated_loc_spatial_contrast_mean']:+.6f}"
    )

    print(
        "  Median inside-outside contrast : "
        f"{result['manipulated_loc_spatial_contrast_median']:+.6f}"
    )

    print(
        "  Fraction contrast > 0          : "
        f"{100.0 * result['manipulated_loc_spatial_contrast_positive_fraction']:.2f}%"
    )


def print_attackwise(
    attack_df,
):
    print()
    print("=" * 76)
    print("ATTACK-WISE PROVENANCE CONTRIBUTION")
    print("=" * 76)

    for _, row in attack_df.iterrows():
        print()
        print(
            f"[{str(row['attack']).upper()}]"
        )

        print(
            "  N                         : "
            f"{int(row['n_samples'])}"
        )

        print(
            "  Detection logit contrib   : "
            f"{row['det_logit_contribution_mean']:+.6f}"
        )

        print(
            "  Loc inside contribution   : "
            f"{row['loc_contribution_inside_mean']:+.6f}"
        )

        print(
            "  Loc outside contribution  : "
            f"{row['loc_contribution_outside_mean']:+.6f}"
        )

        print(
            "  Loc spatial contrast mean : "
            f"{row['loc_spatial_contrast_mean']:+.6f}"
        )

        print(
            "  Contrast > 0              : "
            f"{100.0 * row['loc_spatial_contrast_positive_fraction']:.2f}%"
        )


# ============================================================
# MAIN
# ============================================================


def main():
    print()
    print("=" * 76)
    print(
        "E5.3R-4C-3 REPRESENTATION "
        "ABLATION AUDIT"
    )
    print("=" * 76)

    print()
    print(
        "Validation only."
    )

    print(
        "No training."
    )

    print(
        "No optimizer step."
    )

    print(
        "No checkpoint modification."
    )

    print(
        "No Test access."
    )

    print(
        "No post-hoc threshold optimization."
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
        f"Joint checkpoint   : {JOINT_CHECKPOINT_PATH}"
    )

    print(
        f"Stego checkpoint   : {STEGO_CHECKPOINT_PATH}"
    )

    records = (
        build_validation_records()
    )

    dataset = ActiveSSPGDataset(
        records=records,
        base_seed=VAL_SEED,
    )

    if len(dataset) != EXPECTED_VAL_ACTIVE:
        raise RuntimeError(
            "Unexpected validation active count."
        )

    print()
    print(
        f"Validation sources : {len(records)}"
    )

    print(
        f"Validation active  : {len(dataset)}"
    )

    frozen_threshold = (
        recover_frozen_threshold()
    )

    print(
        "Frozen detection threshold : "
        f"{frozen_threshold:.6f}"
    )

    print()
    print(
        "Loading best checkpoint..."
    )

    model = build_model(
        device
    )

    print(
        "Checkpoint loaded strictly."
    )

    (
        predictions_df,
        contribution_df,
        localization_store,
    ) = run_audit(
        model,
        dataset,
        device,
    )

    summary_df = build_summary(
        predictions_df,
        localization_store,
        frozen_threshold,
    )

    contribution_summary = (
        summarize_contribution(
            contribution_df
        )
    )

    attackwise_df = (
        build_attackwise_contribution(
            contribution_df
        )
    )

    reproduction = (
        compare_with_4c2(
            predictions_df
        )
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    contribution_df.to_csv(
        CONTRIBUTION_PATH,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    attackwise_df.to_csv(
        ATTACKWISE_PATH,
        index=False,
    )

    audit_record = {
        "experiment":
            "E5.3R-4C-3",

        "name":
            "Representation Ablation Audit",

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

        "posthoc_threshold_optimization":
            False,

        "frozen_detection_threshold":
            float(
                frozen_threshold
            ),

        "localization_threshold":
            float(
                LOCALIZATION_THRESHOLD
            ),

        "conditions": [
            "FULL",
            "VISUAL_ONLY",
            "WRONG_PAYLOAD",
            "EVIDENCE_ONLY_ABLATION",
        ],

        "evidence_only_interpretation":
            (
                "Representation ablation only; "
                "not a separately trained "
                "provenance-only model."
            ),

        "contribution_summary":
            contribution_summary,

        "reproduction_check":
            reproduction,
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

    print_summary(
        summary_df
    )

    print_contribution(
        contribution_summary
    )

    print_attackwise(
        attackwise_df
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
        CONTRIBUTION_PATH
    )

    print(
        ATTACKWISE_PATH
    )

    print(
        AUDIT_JSON_PATH
    )

    print()
    print("=" * 76)
    print(
        "E5.3R-4C-3 REPRESENTATION "
        "ABLATION AUDIT: COMPLETE"
    )
    print("=" * 76)

    print()
    print(
        "Scientific boundary:"
    )

    print(
        "EVIDENCE_ONLY_ABLATION is an "
        "out-of-distribution representation ablation."
    )

    print(
        "It must not be reported as the performance "
        "of a separately trained provenance-only model."
    )

    print(
        "All results are Validation-only."
    )

    print(
        "Test remains untouched."
    )


if __name__ == "__main__":
    main()
