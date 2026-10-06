from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from models.sspg_joint_model_v2 import SSPGJointModelV2

from training.train_stego_structured_v33 import (
    embed_payload,
)

from training.e53r_training_protocol_v1 import (
    PROTOCOL,
    mask_to_patch_target,
    compute_joint_loss,
    build_optimizer,
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "stego_structured_v33_128bit_best.pth"
)


def trainable_snapshot(model):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def frozen_snapshot(model):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }


def changed(before, model):
    changed_names = []

    current = dict(model.named_parameters())

    for name, old_value in before.items():

        new_value = current[name].detach()

        if not torch.equal(
            old_value,
            new_value,
        ):
            changed_names.append(name)

    return changed_names


def main():

    print("=" * 78)
    print(
        "E5.3R-4C-0 FORMAL TRAINING "
        "SINGLE-STEP SAFETY GATE"
    )
    print("=" * 78)

    print("Device     :", DEVICE)
    print("Checkpoint :", CHECKPOINT)
    print("Protocol   :", PROTOCOL.experiment_id)

    if not CHECKPOINT.is_file():
        raise RuntimeError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    # ========================================================
    # Determinism
    # ========================================================

    torch.manual_seed(
        PROTOCOL.training_seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            PROTOCOL.training_seed
        )

    # ========================================================
    # Model
    # ========================================================

    model = SSPGJointModelV2(
        stego_checkpoint=str(CHECKPOINT),
        adapter_bottleneck=(
            PROTOCOL.adapter_bottleneck
        ),
        fusion_hidden_dim=(
            PROTOCOL.fusion_hidden_dim
        ),
        knn_k=PROTOCOL.knn_k,
        gnn_hidden_dim=(
            PROTOCOL.gnn_hidden_dim
        ),
        gnn_depth=PROTOCOL.gnn_depth,
    ).to(DEVICE)

    model.train()

    optimizer = build_optimizer(model)

    # ========================================================
    # Optimizer audit
    # ========================================================

    print()
    print("-" * 78)
    print("OPTIMIZER AUDIT")
    print("-" * 78)

    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    trainable_parameters = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }

    frozen_parameters = {
        id(parameter)
        for parameter in model.parameters()
        if not parameter.requires_grad
    }

    if optimizer_parameter_ids != trainable_parameters:
        raise RuntimeError(
            "Optimizer parameter set does not "
            "exactly match trainable parameters."
        )

    if (
        optimizer_parameter_ids
        & frozen_parameters
    ):
        raise RuntimeError(
            "Frozen parameter entered optimizer."
        )

    print(
        "Optimizer              :",
        optimizer.__class__.__name__,
    )

    print(
        "Learning rate          :",
        optimizer.param_groups[0]["lr"],
    )

    print(
        "Weight decay           :",
        optimizer.param_groups[0][
            "weight_decay"
        ],
    )

    print(
        "Trainable tensors      :",
        len(trainable_parameters),
    )

    print(
        "Frozen tensors         :",
        len(frozen_parameters),
    )

    print()
    print("Optimizer isolation : PASS")

    # ========================================================
    # Synthetic protected + manipulated sample
    #
    # No IMD2020.
    # No Test.
    # ========================================================

    original = torch.rand(
        1,
        3,
        256,
        256,
        device=DEVICE,
    )

    payload = torch.randint(
        0,
        2,
        (
            1,
            PROTOCOL.payload_bits,
        ),
        device=DEVICE,
    ).float()

    with torch.no_grad():
        protected = embed_payload(
            original,
            payload,
        )

    received = protected.clone()

    received[
        :,
        :,
        72:152,
        88:176,
    ] = 0.0

    mask_256 = torch.zeros(
        1,
        1,
        256,
        256,
        device=DEVICE,
    )

    mask_256[
        :,
        :,
        72:152,
        88:176,
    ] = 1.0

    detection_target = torch.ones(
        1,
        device=DEVICE,
    )

    localization_target = (
        mask_to_patch_target(
            mask_256
        )
    )

    # ========================================================
    # Parameter snapshots BEFORE update
    # ========================================================

    before_trainable = trainable_snapshot(
        model
    )

    before_frozen = frozen_snapshot(
        model
    )

    # ========================================================
    # Forward
    # ========================================================

    optimizer.zero_grad(
        set_to_none=True
    )

    output = model(
        received,
        payload,
    )

    losses = compute_joint_loss(
        detection_logits=(
            output["detection_logits"]
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

    total_loss = losses[
        "total_loss"
    ]

    print()
    print("-" * 78)
    print("FORWARD / LOSS")
    print("-" * 78)

    print(
        "Detection loss    :",
        f"{losses['detection_loss'].item():.8f}",
    )

    print(
        "Localization BCE  :",
        f"{losses['localization_bce'].item():.8f}",
    )

    print(
        "Localization Dice :",
        f"{losses['localization_dice'].item():.8f}",
    )

    print(
        "Localization loss :",
        f"{losses['localization_loss'].item():.8f}",
    )

    print(
        "Total loss        :",
        f"{total_loss.item():.8f}",
    )

    if not torch.isfinite(total_loss):
        raise RuntimeError(
            "Training loss is non-finite."
        )

    # ========================================================
    # Backward
    # ========================================================

    total_loss.backward()

    decoder = (
        model
        .frontend
        .frontend
        .decoder
    )

    vit = (
        model
        .frontend
        .frontend
        .visual
        .vit
    )

    def module_has_gradient(module):

        for parameter in module.parameters():

            if (
                parameter.grad is not None
                and bool(
                    (
                        parameter.grad.abs()
                        > 0
                    ).any()
                )
            ):
                return True

        return False

    if module_has_gradient(decoder):
        raise RuntimeError(
            "Gradient entered frozen decoder."
        )

    if module_has_gradient(vit):
        raise RuntimeError(
            "Gradient entered frozen ViT."
        )

    print()
    print("Backward isolation : PASS")

    # ========================================================
    # Exactly ONE optimizer step
    # ========================================================

    optimizer.step()

    # ========================================================
    # Parameter update audit
    # ========================================================

    changed_trainable = changed(
        before_trainable,
        model,
    )

    changed_frozen = changed(
        before_frozen,
        model,
    )

    print()
    print("-" * 78)
    print("PARAMETER UPDATE AUDIT")
    print("-" * 78)

    print(
        "Changed trainable tensors :",
        len(changed_trainable),
    )

    print(
        "Changed frozen tensors    :",
        len(changed_frozen),
    )

    if len(changed_trainable) == 0:
        raise RuntimeError(
            "Optimizer step changed no "
            "trainable parameters."
        )

    if len(changed_frozen) != 0:
        print(
            "Unexpected frozen changes:"
        )

        for name in changed_frozen:
            print("  ", name)

        raise RuntimeError(
            "Frozen parameters changed "
            "during optimizer step."
        )

    # ========================================================
    # Critical trainable components must update
    # ========================================================

    required_prefixes = {
        "adapter":
            "frontend.frontend.visual.adapter",

        "fusion":
            "frontend.frontend.fusion",

        "gnn":
            "frontend.gnn",

        "detection":
            "heads.detection",

        "localization":
            "heads.localization",
    }

    print()

    for label, prefix in (
        required_prefixes.items()
    ):

        component_changed = any(
            name.startswith(prefix)
            for name in changed_trainable
        )

        print(
            f"{label:14s} updated :",
            component_changed,
        )

        if not component_changed:
            raise RuntimeError(
                f"Expected trainable component "
                f"did not update: {label}"
            )

    # Adapter scale is a standalone parameter.
    adapter_scale_changed = any(
        name
        ==
        "frontend.frontend.visual.adapter_scale"
        for name in changed_trainable
    )

    print(
        f"{'adapter_scale':14s} updated :",
        adapter_scale_changed,
    )

    if not adapter_scale_changed:
        raise RuntimeError(
            "Adapter scale did not update."
        )

    print()
    print(
        "Single optimizer-step update : PASS"
    )

    # ========================================================
    # Frozen mode invariant AFTER optimizer step
    # ========================================================

    if decoder.training:
        raise RuntimeError(
            "Decoder entered train mode."
        )

    if vit.training:
        raise RuntimeError(
            "ViT entered train mode."
        )

    print(
        "Frozen mode after step       : PASS"
    )

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 78)

    print(
        "E5.3R-4C-0 FORMAL TRAINING "
        "SINGLE-STEP SAFETY GATE: PASS"
    )

    print("=" * 78)

    print()
    print(
        "Exactly one optimizer step "
        "was executed on synthetic data."
    )

    print(
        "No IMD2020 image was loaded."
    )

    print(
        "No checkpoint was saved."
    )

    print(
        "No validation metric was observed."
    )

    print(
        "No Test data was accessed."
    )


if __name__ == "__main__":
    main()