from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from models.provenance_heads_v2 import (
    JointDetectionLocalizationHeadsV2,
    upsample_localization_logits,
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def count_parameters(
    module,
):
    total = sum(
        p.numel()
        for p in module.parameters()
    )

    trainable = sum(
        p.numel()
        for p in module.parameters()
        if p.requires_grad
    )

    return (
        total,
        trainable,
    )


def main():

    print("=" * 78)

    print(
        "E5.3R-4B-0 JOINT DETECTION / "
        "LOCALIZATION HEAD ARCHITECTURE GATE"
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

    # --------------------------------------------------------
    # Instantiate heads.
    # --------------------------------------------------------

    model = (
        JointDetectionLocalizationHeadsV2(
            hidden_dim=256,
            detection_mlp_dim=256,
            detection_dropout=0.1,
            grid_size=14,
        )
        .to(
            DEVICE
        )
    )

    total_params, trainable_params = (
        count_parameters(
            model
        )
    )

    detection_total, detection_trainable = (
        count_parameters(
            model.detection
        )
    )

    localization_total, localization_trainable = (
        count_parameters(
            model.localization
        )
    )

    print()
    print("-" * 78)
    print(
        "PARAMETER AUDIT"
    )
    print("-" * 78)

    print(
        "Detection total/trainable    :",
        detection_total,
        "/",
        detection_trainable,
    )

    print(
        "Localization total/trainable :",
        localization_total,
        "/",
        localization_trainable,
    )

    print(
        "Joint total/trainable        :",
        total_params,
        "/",
        trainable_params,
    )

    if total_params <= 0:
        raise RuntimeError(
            "Heads contain no parameters."
        )

    if (
        total_params
        != trainable_params
    ):
        raise RuntimeError(
            "All head parameters should "
            "be trainable."
        )

    # --------------------------------------------------------
    # Synthetic graph tokens.
    # --------------------------------------------------------

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

    detection_logits = (
        output[
            "detection_logits"
        ]
    )

    node_logits = (
        output[
            "localization_node_logits"
        ]
    )

    grid_logits = (
        output[
            "localization_grid_logits"
        ]
    )

    print()
    print("-" * 78)
    print(
        "OUTPUT SHAPES"
    )
    print("-" * 78)

    print(
        "Input graph tokens       :",
        tuple(
            graph_tokens.shape
        ),
    )

    print(
        "Detection logits         :",
        tuple(
            detection_logits.shape
        ),
    )

    print(
        "Localization node logits :",
        tuple(
            node_logits.shape
        ),
    )

    print(
        "Localization grid logits :",
        tuple(
            grid_logits.shape
        ),
    )

    if tuple(
        detection_logits.shape
    ) != (
        2,
    ):
        raise RuntimeError(
            "Detection shape failure."
        )

    if tuple(
        node_logits.shape
    ) != (
        2,
        196,
    ):
        raise RuntimeError(
            "Node localization shape failure."
        )

    if tuple(
        grid_logits.shape
    ) != (
        2,
        1,
        14,
        14,
    ):
        raise RuntimeError(
            "Grid localization shape failure."
        )

    for tensor in (
        detection_logits,
        node_logits,
        grid_logits,
    ):
        if not torch.isfinite(
            tensor
        ).all():
            raise RuntimeError(
                "Non-finite head output."
            )

    print()
    print(
        "Shape / finite gate : PASS"
    )

    # --------------------------------------------------------
    # 256x256 projection.
    # --------------------------------------------------------

    pixel_logits = (
        upsample_localization_logits(
            grid_logits,
            output_size=256,
        )
    )

    print()
    print(
        "Upsampled localization :",
        tuple(
            pixel_logits.shape
        ),
    )

    if tuple(
        pixel_logits.shape
    ) != (
        2,
        1,
        256,
        256,
    ):
        raise RuntimeError(
            "Localization upsampling failure."
        )

    # --------------------------------------------------------
    # Gradient gate.
    #
    # Both tasks must affect graph tokens and their heads.
    # --------------------------------------------------------

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
            detection_logits,
            detection_target,
        )
    )

    localization_loss = (
        torch.nn.functional
        .binary_cross_entropy_with_logits(
            grid_logits,
            localization_target,
        )
    )

    loss = (
        detection_loss
        + localization_loss
    )

    model.zero_grad(
        set_to_none=True
    )

    if graph_tokens.grad is not None:
        graph_tokens.grad.zero_()

    loss.backward()

    graph_grad = (
        graph_tokens.grad
        is not None
        and bool(
            (
                graph_tokens.grad.abs()
                > 0
            ).any()
        )
    )

    detection_grad = any(
        p.grad is not None
        and bool(
            (
                p.grad.abs()
                > 0
            ).any()
        )
        for p in model.detection.parameters()
    )

    localization_grad = any(
        p.grad is not None
        and bool(
            (
                p.grad.abs()
                > 0
            ).any()
        )
        for p in model.localization.parameters()
    )

    print()
    print("-" * 78)
    print(
        "GRADIENT GATE"
    )
    print("-" * 78)

    print(
        "Graph-token gradient :",
        graph_grad,
    )

    print(
        "Detection-head grad  :",
        detection_grad,
    )

    print(
        "Localization-head grad:",
        localization_grad,
    )

    if not graph_grad:
        raise RuntimeError(
            "No gradient reaches graph tokens."
        )

    if not detection_grad:
        raise RuntimeError(
            "Detection head has no gradient."
        )

    if not localization_grad:
        raise RuntimeError(
            "Localization head has no gradient."
        )

    # --------------------------------------------------------
    # Task dependency gate.
    #
    # Perturb graph tokens and verify both task outputs change.
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        base = torch.randn(
            1,
            196,
            256,
            device=DEVICE,
        )

        changed = base.clone()

        changed[
            :,
            40:80,
            :,
        ] += 0.5

        out_a = model(
            base
        )

        out_b = model(
            changed
        )

        detection_difference = float(
            (
                out_a[
                    "detection_logits"
                ]
                -
                out_b[
                    "detection_logits"
                ]
            )
            .abs()
            .mean()
            .item()
        )

        localization_difference = float(
            (
                out_a[
                    "localization_grid_logits"
                ]
                -
                out_b[
                    "localization_grid_logits"
                ]
            )
            .abs()
            .mean()
            .item()
        )

    print()
    print("-" * 78)
    print(
        "GRAPH-TOKEN DEPENDENCY"
    )
    print("-" * 78)

    print(
        "Detection output difference    :",
        f"{detection_difference:.12f}",
    )

    print(
        "Localization output difference :",
        f"{localization_difference:.12f}",
    )

    if detection_difference <= 0:
        raise RuntimeError(
            "Detection head ignores graph tokens."
        )

    if localization_difference <= 0:
        raise RuntimeError(
            "Localization head ignores graph tokens."
        )

    print()
    print("=" * 78)

    print(
        "E5.3R-4B-0 JOINT HEAD "
        "ARCHITECTURE GATE: PASS"
    )

    print("=" * 78)

    print()
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