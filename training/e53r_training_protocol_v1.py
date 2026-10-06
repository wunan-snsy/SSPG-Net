from dataclasses import dataclass, asdict
import json

import torch
import torch.nn.functional as F


# ============================================================
# E5.3R-4B-2
# FORMAL TRAINING PROTOCOL V1
#
# IMPORTANT:
#
# This file defines the training protocol BEFORE any formal
# E5.3R joint-model Train/Val result is observed.
#
# After protocol freeze:
#
#   - do not tune hyperparameters against validation results
#     under the same experiment ID;
#
#   - any scientifically justified change must receive a new
#     revision / experiment ID;
#
#   - Test data remain prohibited until final evaluation.
# ============================================================


@dataclass(frozen=True)
class E53RTrainingProtocolV1:

    # --------------------------------------------------------
    # Identity
    # --------------------------------------------------------

    experiment_id: str = (
        "E5.3R-4B-2-V1"
    )

    protocol_status: str = (
        "PREREGISTERED"
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    image_size: int = 256

    payload_bits: int = 128

    localization_grid_size: int = 14

    train_seed: int = 20260922

    val_seed: int = 20260923

    training_seed: int = 20260925

    # Test seed intentionally undefined.
    test_seed: object = None

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    adapter_bottleneck: int = 96

    fusion_hidden_dim: int = 96

    knn_k: int = 8

    gnn_hidden_dim: int = 256

    gnn_depth: int = 2

    # --------------------------------------------------------
    # Detection objective
    # --------------------------------------------------------

    detection_loss: str = (
        "BCEWithLogitsLoss"
    )

    # --------------------------------------------------------
    # Localization objective
    #
    # Localization:
    #
    #   0.5 * BCE
    # + 0.5 * SoftDiceLoss
    #
    # --------------------------------------------------------

    localization_bce_weight: float = 0.5

    localization_dice_weight: float = 0.5

    dice_epsilon: float = 1.0e-6

    # --------------------------------------------------------
    # Joint objective
    #
    # L_total =
    #
    #     1.0 * L_detection
    #   + 1.0 * L_localization
    #
    # --------------------------------------------------------

    detection_task_weight: float = 1.0

    localization_task_weight: float = 1.0

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer: str = "AdamW"

    learning_rate: float = 1.0e-4

    weight_decay: float = 1.0e-4

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    max_epochs: int = 20

    early_stopping_patience: int = 5

    min_delta: float = 0.0

    # Start conservatively on RTX 3060 Laptop GPU.
    batch_size: int = 1

    gradient_accumulation_steps: int = 4

    effective_batch_size: int = 4

    # --------------------------------------------------------
    # Checkpoint selection
    # --------------------------------------------------------

    checkpoint_metric: str = (
        "validation_joint_loss"
    )

    checkpoint_mode: str = "min"

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    detection_threshold_method: str = (
        "validation_youden_j"
    )

    localization_threshold: float = 0.5

    # --------------------------------------------------------
    # Test policy
    # --------------------------------------------------------

    test_access_before_final_freeze: bool = False

    test_threshold_tuning: bool = False

    test_checkpoint_selection: bool = False


PROTOCOL = E53RTrainingProtocolV1()


# ============================================================
# Localization target
# ============================================================

def mask_to_patch_target(
    mask_256,
    grid_size=None,
):
    """
    Convert pixel-level localization GT to a soft patch target.

    Input:
        [B,1,256,256]

    Output:
        [B,1,14,14]

    Uses area interpolation so each patch target represents
    approximate manipulated-area coverage.

    Targets remain in [0,1].
    """

    if grid_size is None:
        grid_size = (
            PROTOCOL.localization_grid_size
        )

    if mask_256.ndim != 4:

        raise RuntimeError(
            "Expected mask [B,1,H,W]."
        )

    if mask_256.shape[1] != 1:

        raise RuntimeError(
            "Localization mask must have "
            "one channel."
        )

    target = F.interpolate(
        mask_256.float(),
        size=(
            int(grid_size),
            int(grid_size),
        ),
        mode="area",
    )

    return target


# ============================================================
# Soft Dice loss
# ============================================================

def soft_dice_loss_from_logits(
    logits,
    target,
    epsilon=None,
):
    """
    Soft Dice loss.

    IMPORTANT:

    Dice is calculated per sample and then averaged.

    Authentic samples with all-zero target are handled with
    epsilon in numerator and denominator.

    This keeps the loss finite and gives a perfect score only
    when predicted positive mass also approaches zero.
    """

    if epsilon is None:
        epsilon = (
            PROTOCOL.dice_epsilon
        )

    if logits.shape != target.shape:

        raise RuntimeError(
            "Dice logits/target shape mismatch."
        )

    probabilities = torch.sigmoid(
        logits
    )

    probabilities = probabilities.flatten(
        start_dim=1
    )

    target = target.float().flatten(
        start_dim=1
    )

    intersection = (
        probabilities
        * target
    ).sum(
        dim=1
    )

    probability_mass = (
        probabilities.sum(
            dim=1
        )
    )

    target_mass = (
        target.sum(
            dim=1
        )
    )

    dice = (
        (
            2.0
            * intersection
        )
        + epsilon
    ) / (
        probability_mass
        + target_mass
        + epsilon
    )

    loss = (
        1.0
        - dice
    )

    return loss.mean()


# ============================================================
# Formal joint loss
# ============================================================

def compute_joint_loss(
    detection_logits,
    detection_target,
    localization_logits,
    localization_target,
):
    """
    Formal E5.3R V1 joint objective.

    Detection:
        BCEWithLogits

    Localization:
        0.5 BCE + 0.5 Soft Dice

    Total:
        1.0 Detection + 1.0 Localization
    """

    detection_target = (
        detection_target
        .float()
        .reshape_as(
            detection_logits
        )
    )

    if (
        localization_logits.shape
        != localization_target.shape
    ):

        raise RuntimeError(
            "Localization logits/target "
            "shape mismatch."
        )

    detection_bce = (
        F.binary_cross_entropy_with_logits(
            detection_logits,
            detection_target,
        )
    )

    localization_bce = (
        F.binary_cross_entropy_with_logits(
            localization_logits,
            localization_target.float(),
        )
    )

    localization_dice = (
        soft_dice_loss_from_logits(
            localization_logits,
            localization_target,
            epsilon=PROTOCOL.dice_epsilon,
        )
    )

    localization_loss = (
        PROTOCOL.localization_bce_weight
        * localization_bce
        +
        PROTOCOL.localization_dice_weight
        * localization_dice
    )

    total_loss = (
        PROTOCOL.detection_task_weight
        * detection_bce
        +
        PROTOCOL.localization_task_weight
        * localization_loss
    )

    return {
        "total_loss":
            total_loss,

        "detection_loss":
            detection_bce,

        "localization_loss":
            localization_loss,

        "localization_bce":
            localization_bce,

        "localization_dice":
            localization_dice,
    }


# ============================================================
# Optimizer
# ============================================================

def build_optimizer(
    model,
):
    """
    AdamW over trainable parameters only.
    """

    parameters = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    if len(parameters) == 0:

        raise RuntimeError(
            "No trainable parameters."
        )

    optimizer = torch.optim.AdamW(
        parameters,
        lr=PROTOCOL.learning_rate,
        weight_decay=PROTOCOL.weight_decay,
    )

    return optimizer


# ============================================================
# Protocol serialization
# ============================================================

def protocol_dict():

    return asdict(
        PROTOCOL
    )


def protocol_json():

    return json.dumps(
        protocol_dict(),
        indent=2,
        sort_keys=True,
    )


if __name__ == "__main__":

    print(
        protocol_json()
    )