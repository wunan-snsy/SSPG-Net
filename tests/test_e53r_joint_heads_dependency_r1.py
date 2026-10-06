from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from models.provenance_heads_v2 import (
    JointDetectionLocalizationHeadsV2,
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def main():

    print("=" * 78)
    print(
        "E5.3R-4B-0-R1 JOINT HEAD "
        "DEPENDENCY GATE"
    )
    print("=" * 78)

    print(
        "Device :",
        DEVICE,
    )

    torch.manual_seed(
        20260924
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            20260924
        )

    model = (
        JointDetectionLocalizationHeadsV2(
            hidden_dim=256,
            detection_mlp_dim=256,
            detection_dropout=0.1,
            grid_size=14,
        )
        .to(DEVICE)
    )

    model.eval()

    # --------------------------------------------------------
    # Base graph tokens.
    # --------------------------------------------------------

    base = torch.randn(
        1,
        196,
        256,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # R1 perturbation.
    #
    # IMPORTANT:
    #
    # The previous gate added +0.5 to ALL 256 channels:
    #
    #     token -> token + constant
    #
    # LayerNorm is intentionally invariant to this type of
    # common-mode translation.
    #
    # R1 therefore uses a channel-structured perturbation:
    #
    #     channels 0:128   += 0.5
    #     channels 128:256 -= 0.5
    #
    # This changes the internal channel structure and cannot
    # be removed by LayerNorm as a simple additive offset.
    #
    # Model architecture is NOT changed.
    # --------------------------------------------------------

    changed = base.clone()

    changed[
        :,
        40:80,
        0:128,
    ] += 0.5

    changed[
        :,
        40:80,
        128:256,
    ] -= 0.5

    with torch.no_grad():

        out_base = model(
            base
        )

        out_changed = model(
            changed
        )

    detection_base = (
        out_base[
            "detection_logits"
        ]
    )

    detection_changed = (
        out_changed[
            "detection_logits"
        ]
    )

    localization_base = (
        out_base[
            "localization_grid_logits"
        ]
    )

    localization_changed = (
        out_changed[
            "localization_grid_logits"
        ]
    )

    detection_difference = float(
        (
            detection_base
            - detection_changed
        )
        .abs()
        .mean()
        .item()
    )

    localization_difference = float(
        (
            localization_base
            - localization_changed
        )
        .abs()
        .mean()
        .item()
    )

    token_difference = float(
        (
            base
            - changed
        )
        .abs()
        .mean()
        .item()
    )

    print()
    print("-" * 78)
    print(
        "STRUCTURED TOKEN PERTURBATION"
    )
    print("-" * 78)

    print(
        "Input token difference        :",
        f"{token_difference:.12f}",
    )

    print(
        "Detection output difference   :",
        f"{detection_difference:.12f}",
    )

    print(
        "Localization output difference:",
        f"{localization_difference:.12f}",
    )

    # --------------------------------------------------------
    # Structural gate only.
    #
    # No performance-dependent threshold is introduced.
    # We require only a genuine non-zero dependency.
    # --------------------------------------------------------

    if not torch.isfinite(
        detection_changed
    ).all():
        raise RuntimeError(
            "Detection output is non-finite."
        )

    if not torch.isfinite(
        localization_changed
    ).all():
        raise RuntimeError(
            "Localization output is non-finite."
        )

    if detection_difference <= 0.0:
        raise RuntimeError(
            "Detection head still shows no "
            "dependency on structured graph-token "
            "perturbation."
        )

    if localization_difference <= 0.0:
        raise RuntimeError(
            "Localization head still shows no "
            "dependency on structured graph-token "
            "perturbation."
        )

    # --------------------------------------------------------
    # Gradient dependency check.
    # --------------------------------------------------------

    model.train()

    graph_tokens = torch.randn(
        2,
        196,
        256,
        device=DEVICE,
        requires_grad=True,
    )

    output = model(
        graph_tokens
    )

    detection_target = torch.tensor(
        [
            0.0,
            1.0,
        ],
        device=DEVICE,
    )

    localization_target = torch.zeros(
        2,
        1,
        14,
        14,
        device=DEVICE,
    )

    localization_target[
        1,
        :,
        4:10,
        5:11,
    ] = 1.0

    detection_loss = (
        torch.nn.functional
        .binary_cross_entropy_with_logits(
            output[
                "detection_logits"
            ],
            detection_target,
        )
    )

    localization_loss = (
        torch.nn.functional
        .binary_cross_entropy_with_logits(
            output[
                "localization_grid_logits"
            ],
            localization_target,
        )
    )

    joint_loss = (
        detection_loss
        + localization_loss
    )

    model.zero_grad(
        set_to_none=True
    )

    joint_loss.backward()

    graph_gradient = (
        graph_tokens.grad is not None
        and bool(
            (
                graph_tokens.grad.abs()
                > 0
            ).any()
        )
    )

    detection_gradient = any(
        p.grad is not None
        and bool(
            (
                p.grad.abs()
                > 0
            ).any()
        )
        for p
        in model.detection.parameters()
    )

    localization_gradient = any(
        p.grad is not None
        and bool(
            (
                p.grad.abs()
                > 0
            ).any()
        )
        for p
        in model.localization.parameters()
    )

    print()
    print("-" * 78)
    print(
        "GRADIENT CONFIRMATION"
    )
    print("-" * 78)

    print(
        "Graph-token gradient  :",
        graph_gradient,
    )

    print(
        "Detection-head gradient:",
        detection_gradient,
    )

    print(
        "Localization-head gradient:",
        localization_gradient,
    )

    if not graph_gradient:
        raise RuntimeError(
            "Graph-token gradient failure."
        )

    if not detection_gradient:
        raise RuntimeError(
            "Detection-head gradient failure."
        )

    if not localization_gradient:
        raise RuntimeError(
            "Localization-head gradient failure."
        )

    print()
    print("=" * 78)
    print(
        "E5.3R-4B-0-R1 JOINT HEAD "
        "DEPENDENCY GATE: PASS"
    )
    print("=" * 78)

    print()
    print(
        "Model architecture was not changed."
    )

    print(
        "R1 corrects only the invalid "
        "LayerNorm-invariant perturbation "
        "used by the original dependency gate."
    )

    print(
        "No dataset image was accessed."
    )

    print(
        "No model was trained."
    )

    print(
        "No Test data was accessed."
    )


if __name__ == "__main__":
    main()