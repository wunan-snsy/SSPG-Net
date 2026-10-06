from pathlib import Path
import sys
import hashlib

import torch


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from training.e53r_training_protocol_v1 import (
    PROTOCOL,
    protocol_json,
    mask_to_patch_target,
    soft_dice_loss_from_logits,
    compute_joint_loss,
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def main():

    print("=" * 78)

    print(
        "E5.3R-4B-2 FORMAL TRAINING "
        "PROTOCOL V1 FREEZE GATE"
    )

    print("=" * 78)

    print(
        "Device :",
        DEVICE,
    )

    # ========================================================
    # Protocol identity
    # ========================================================

    print()

    print("-" * 78)

    print(
        "PROTOCOL"
    )

    print("-" * 78)

    print(
        "Experiment ID              :",
        PROTOCOL.experiment_id,
    )

    print(
        "Status                     :",
        PROTOCOL.protocol_status,
    )

    print(
        "Image size                 :",
        PROTOCOL.image_size,
    )

    print(
        "Payload bits               :",
        PROTOCOL.payload_bits,
    )

    print(
        "Localization grid          :",
        PROTOCOL.localization_grid_size,
    )

    print(
        "Training seed              :",
        PROTOCOL.training_seed,
    )

    print(
        "Optimizer                  :",
        PROTOCOL.optimizer,
    )

    print(
        "Learning rate              :",
        PROTOCOL.learning_rate,
    )

    print(
        "Weight decay               :",
        PROTOCOL.weight_decay,
    )

    print(
        "Batch size                 :",
        PROTOCOL.batch_size,
    )

    print(
        "Gradient accumulation      :",
        PROTOCOL.gradient_accumulation_steps,
    )

    print(
        "Effective batch size       :",
        PROTOCOL.effective_batch_size,
    )

    print(
        "Maximum epochs             :",
        PROTOCOL.max_epochs,
    )

    print(
        "Early stopping patience    :",
        PROTOCOL.early_stopping_patience,
    )

    print(
        "Checkpoint metric          :",
        PROTOCOL.checkpoint_metric,
    )

    print(
        "Detection threshold        :",
        PROTOCOL.detection_threshold_method,
    )

    print(
        "Localization threshold     :",
        PROTOCOL.localization_threshold,
    )

    print(
        "Test seed                  :",
        PROTOCOL.test_seed,
    )

    # ========================================================
    # Fixed protocol invariants
    # ========================================================

    if PROTOCOL.test_seed is not None:

        raise RuntimeError(
            "Test seed must remain undefined."
        )

    if (
        PROTOCOL
        .test_access_before_final_freeze
    ):

        raise RuntimeError(
            "Protocol unexpectedly allows "
            "early Test access."
        )

    if PROTOCOL.test_threshold_tuning:

        raise RuntimeError(
            "Test threshold tuning must "
            "be prohibited."
        )

    if PROTOCOL.test_checkpoint_selection:

        raise RuntimeError(
            "Test checkpoint selection must "
            "be prohibited."
        )

    if (
        PROTOCOL.effective_batch_size
        !=
        PROTOCOL.batch_size
        *
        PROTOCOL.gradient_accumulation_steps
    ):

        raise RuntimeError(
            "Effective batch-size mismatch."
        )

    if (
        PROTOCOL.checkpoint_metric
        !=
        "validation_joint_loss"
    ):

        raise RuntimeError(
            "Checkpoint must be selected "
            "by validation joint loss."
        )

    print()

    print(
        "Protocol invariant gate : PASS"
    )

    # ========================================================
    # Localization target projection
    # ========================================================

    mask = torch.zeros(
        2,
        1,
        256,
        256,
        device=DEVICE,
    )

    mask[
        1,
        :,
        64:160,
        80:192,
    ] = 1.0

    target = mask_to_patch_target(
        mask
    )

    print()

    print("-" * 78)

    print(
        "LOCALIZATION TARGET"
    )

    print("-" * 78)

    print(
        "Input mask shape :",
        tuple(
            mask.shape
        ),
    )

    print(
        "Patch target     :",
        tuple(
            target.shape
        ),
    )

    print(
        "Target min       :",
        f"{target.min().item():.8f}",
    )

    print(
        "Target max       :",
        f"{target.max().item():.8f}",
    )

    if tuple(
        target.shape
    ) != (
        2,
        1,
        14,
        14,
    ):

        raise RuntimeError(
            "Patch target shape failure."
        )

    if target.min().item() < 0.0:

        raise RuntimeError(
            "Patch target below zero."
        )

    if target.max().item() > 1.0:

        raise RuntimeError(
            "Patch target above one."
        )

    # Authentic sample must remain zero.
    authentic_max = float(
        target[
            0
        ]
        .abs()
        .max()
        .item()
    )

    print(
        "Authentic target max abs :",
        f"{authentic_max:.12f}",
    )

    if authentic_max != 0.0:

        raise RuntimeError(
            "Authentic localization target "
            "must be zero."
        )

    print()

    print(
        "Localization target gate : PASS"
    )

    # ========================================================
    # Loss finite / backward gate
    # ========================================================

    torch.manual_seed(
        PROTOCOL.training_seed
    )

    detection_logits = torch.randn(
        2,
        device=DEVICE,
        requires_grad=True,
    )

    localization_logits = torch.randn(
        2,
        1,
        14,
        14,
        device=DEVICE,
        requires_grad=True,
    )

    detection_target = torch.tensor(
        [
            0.0,
            1.0,
        ],
        device=DEVICE,
    )

    losses = compute_joint_loss(
        detection_logits=
            detection_logits,

        detection_target=
            detection_target,

        localization_logits=
            localization_logits,

        localization_target=
            target,
    )

    print()

    print("-" * 78)

    print(
        "FORMAL LOSS"
    )

    print("-" * 78)

    for key in [
        "detection_loss",
        "localization_bce",
        "localization_dice",
        "localization_loss",
        "total_loss",
    ]:

        value = losses[
            key
        ]

        print(
            f"{key:22s}:",
            f"{value.item():.8f}",
        )

        if not torch.isfinite(
            value
        ):

            raise RuntimeError(
                f"{key} is non-finite."
            )

    losses[
        "total_loss"
    ].backward()

    detection_gradient = (
        detection_logits.grad
        is not None
        and bool(
            (
                detection_logits.grad.abs()
                > 0
            ).any()
        )
    )

    localization_gradient = (
        localization_logits.grad
        is not None
        and bool(
            (
                localization_logits.grad.abs()
                > 0
            ).any()
        )
    )

    print()

    print(
        "Detection-logit gradient    :",
        detection_gradient,
    )

    print(
        "Localization-logit gradient :",
        localization_gradient,
    )

    if not detection_gradient:

        raise RuntimeError(
            "Detection loss has no gradient."
        )

    if not localization_gradient:

        raise RuntimeError(
            "Localization loss has no gradient."
        )

    # ========================================================
    # Dice sanity gate
    # ========================================================

    perfect_positive_logits = torch.full(
        (
            1,
            1,
            14,
            14,
        ),
        20.0,
        device=DEVICE,
    )

    perfect_positive_target = torch.ones(
        1,
        1,
        14,
        14,
        device=DEVICE,
    )

    positive_dice_loss = (
        soft_dice_loss_from_logits(
            perfect_positive_logits,
            perfect_positive_target,
        )
    )

    perfect_negative_logits = torch.full(
        (
            1,
            1,
            14,
            14,
        ),
        -20.0,
        device=DEVICE,
    )

    perfect_negative_target = torch.zeros(
        1,
        1,
        14,
        14,
        device=DEVICE,
    )

    negative_dice_loss = (
        soft_dice_loss_from_logits(
            perfect_negative_logits,
            perfect_negative_target,
        )
    )

    print()

    print(
        "Near-perfect positive Dice loss :",
        f"{positive_dice_loss.item():.12f}",
    )

    print(
        "Near-perfect negative Dice loss :",
        f"{negative_dice_loss.item():.12f}",
    )

    if positive_dice_loss.item() > 1.0e-5:

        raise RuntimeError(
            "Positive Dice sanity failure."
        )

    if negative_dice_loss.item() > 0.5:

        raise RuntimeError(
            "Negative Dice sanity failure."
        )

    print()

    print(
        "Formal loss gate : PASS"
    )

    # ========================================================
    # Protocol fingerprint
    # ========================================================

    serialized = protocol_json()

    fingerprint = hashlib.sha256(
        serialized.encode(
            "utf-8"
        )
    ).hexdigest()

    print()

    print("-" * 78)

    print(
        "PROTOCOL FINGERPRINT"
    )

    print("-" * 78)

    print(
        "SHA256 :",
        fingerprint,
    )

    print()

    print(
        "This fingerprint identifies the "
        "pre-registered protocol configuration."
    )

    # ========================================================
    # Final
    # ========================================================

    print()

    print("=" * 78)

    print(
        "E5.3R-4B-2 FORMAL TRAINING "
        "PROTOCOL V1 FREEZE GATE: PASS"
    )

    print("=" * 78)

    print()

    print(
        "No model was trained."
    )

    print(
        "No optimizer step was performed."
    )

    print(
        "No IMD2020 image was loaded."
    )

    print(
        "No Test data was accessed."
    )


if __name__ == "__main__":

    main()