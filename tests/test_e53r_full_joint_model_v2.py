from pathlib import Path
import sys

import torch
import torch.nn.functional as F


# ============================================================
# Project root
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


# ============================================================
# Imports
# ============================================================

from models.sspg_joint_model_v2 import (
    SSPGJointModelV2,
)

from training.train_stego_structured_v33 import (
    embed_payload,
)


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "stego_structured_v33_128bit_best.pth"
)

SEED = 20260924


# ============================================================
# Utilities
# ============================================================

def count_trainable_parameters(
    module,
):
    return sum(
        p.numel()
        for p in module.parameters()
        if p.requires_grad
    )


def has_nonzero_gradient(
    module,
):
    for parameter in module.parameters():

        if not parameter.requires_grad:
            continue

        if parameter.grad is None:
            continue

        if bool(
            (
                parameter.grad.abs()
                > 0
            ).any()
        ):
            return True

    return False


def tensor_is_finite(
    tensor,
):
    return bool(
        torch.isfinite(
            tensor
        ).all()
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 78)

    print(
        "E5.3R-4B-1 FULL JOINT MODEL "
        "END-TO-END GATE"
    )

    print("=" * 78)

    print(
        "Device     :",
        DEVICE,
    )

    print(
        "Checkpoint :",
        CHECKPOINT,
    )

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not CHECKPOINT.is_file():

        raise RuntimeError(
            "Checkpoint not found: "
            f"{CHECKPOINT}"
        )

    # --------------------------------------------------------
    # Fixed seed
    # --------------------------------------------------------

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED
        )

    # ========================================================
    # Instantiate complete joint model
    # ========================================================

    model = SSPGJointModelV2(

        stego_checkpoint=str(
            CHECKPOINT
        ),

        adapter_bottleneck=96,

        fusion_hidden_dim=96,

        knn_k=8,

        gnn_hidden_dim=256,

        gnn_depth=2,

    ).to(
        DEVICE
    )

    # --------------------------------------------------------
    # Enter training mode.
    #
    # Existing validated frontend must keep:
    #
    # decoder.eval()
    # visual.vit.eval()
    #
    # while trainable components remain trainable.
    # --------------------------------------------------------

    model.train()

    # ========================================================
    # Resolve REAL validated module hierarchy
    # ========================================================

    decoder = (
        model
        .frontend
        .frontend
        .decoder
    )

    visual = (
        model
        .frontend
        .frontend
        .visual
    )

    vit = (
        model
        .frontend
        .frontend
        .visual
        .vit
    )

    adapter = (
        model
        .frontend
        .frontend
        .visual
        .adapter
    )

    adapter_scale = (
        model
        .frontend
        .frontend
        .visual
        .adapter_scale
    )

    fusion = (
        model
        .frontend
        .frontend
        .fusion
    )

    gnn = (
        model
        .frontend
        .gnn
    )

    detection_head = (
        model
        .heads
        .detection
    )

    localization_head = (
        model
        .heads
        .localization
    )

    # ========================================================
    # Parameter isolation
    # ========================================================

    print()

    print("-" * 78)

    print(
        "PARAMETER ISOLATION"
    )

    print("-" * 78)

    decoder_trainable = (
        count_trainable_parameters(
            decoder
        )
    )

    vit_trainable = (
        count_trainable_parameters(
            vit
        )
    )

    adapter_trainable = (
        count_trainable_parameters(
            adapter
        )
    )

    fusion_trainable = (
        count_trainable_parameters(
            fusion
        )
    )

    gnn_trainable = (
        count_trainable_parameters(
            gnn
        )
    )

    detection_trainable = (
        count_trainable_parameters(
            detection_head
        )
    )

    localization_trainable = (
        count_trainable_parameters(
            localization_head
        )
    )

    print(
        "Decoder trainable       :",
        decoder_trainable,
    )

    print(
        "ViT trainable           :",
        vit_trainable,
    )

    print(
        "Adapter trainable       :",
        adapter_trainable,
    )

    print(
        "Adapter scale trainable :",
        int(
            adapter_scale.requires_grad
        ),
    )

    print(
        "Fusion trainable        :",
        fusion_trainable,
    )

    print(
        "GNN trainable           :",
        gnn_trainable,
    )

    print(
        "Detection head          :",
        detection_trainable,
    )

    print(
        "Localization head       :",
        localization_trainable,
    )

    # --------------------------------------------------------
    # Frozen modules
    # --------------------------------------------------------

    if decoder_trainable != 0:

        raise RuntimeError(
            "V3.3 decoder must remain frozen."
        )

    if vit_trainable != 0:

        raise RuntimeError(
            "Pretrained ViT must remain frozen."
        )

    # --------------------------------------------------------
    # Trainable modules
    # --------------------------------------------------------

    if adapter_trainable <= 0:

        raise RuntimeError(
            "Token adapter must be trainable."
        )

    if not adapter_scale.requires_grad:

        raise RuntimeError(
            "Adapter scale must be trainable."
        )

    if fusion_trainable <= 0:

        raise RuntimeError(
            "Evidence fusion must be trainable."
        )

    if gnn_trainable <= 0:

        raise RuntimeError(
            "GNN must be trainable."
        )

    if detection_trainable <= 0:

        raise RuntimeError(
            "Detection head must be trainable."
        )

    if localization_trainable <= 0:

        raise RuntimeError(
            "Localization head must be trainable."
        )

    print()

    print(
        "Parameter isolation : PASS"
    )

    # ========================================================
    # Train / eval mode isolation
    # ========================================================

    print()

    print("-" * 78)

    print(
        "MODE ISOLATION"
    )

    print("-" * 78)

    print(
        "Whole model training :",
        model.training,
    )

    print(
        "Graph frontend       :",
        model.frontend.training,
    )

    print(
        "SSPG frontend        :",
        model.frontend.frontend.training,
    )

    print(
        "Visual wrapper       :",
        visual.training,
    )

    print(
        "Decoder training     :",
        decoder.training,
    )

    print(
        "ViT training         :",
        vit.training,
    )

    print(
        "Adapter training     :",
        adapter.training,
    )

    print(
        "Fusion training      :",
        fusion.training,
    )

    print(
        "GNN training         :",
        gnn.training,
    )

    print(
        "Detection head       :",
        detection_head.training,
    )

    print(
        "Localization head    :",
        localization_head.training,
    )

    if decoder.training:

        raise RuntimeError(
            "Frozen decoder entered train mode."
        )

    if vit.training:

        raise RuntimeError(
            "Frozen ViT entered train mode."
        )

    if not adapter.training:

        raise RuntimeError(
            "Adapter unexpectedly not in "
            "training mode."
        )

    if not fusion.training:

        raise RuntimeError(
            "Fusion unexpectedly not in "
            "training mode."
        )

    if not gnn.training:

        raise RuntimeError(
            "GNN unexpectedly not in "
            "training mode."
        )

    if not detection_head.training:

        raise RuntimeError(
            "Detection head unexpectedly not "
            "in training mode."
        )

    if not localization_head.training:

        raise RuntimeError(
            "Localization head unexpectedly not "
            "in training mode."
        )

    print()

    print(
        "Frozen mode isolation : PASS"
    )

    # ========================================================
    # Construct one synthetic ACTIVE SSPG sample
    #
    # IMPORTANT:
    #
    # This does NOT load IMD2020.
    # This does NOT access Test.
    #
    # The purpose is only:
    #
    # original
    # -> V3.3 embedding
    # -> controlled manipulation
    # -> complete joint model
    # -> joint gradient
    # ========================================================

    original = torch.rand(
        1,
        3,
        256,
        256,
        device=DEVICE,
        dtype=torch.float32,
    )

    payload = torch.randint(
        low=0,
        high=2,
        size=(
            1,
            128,
        ),
        device=DEVICE,
        dtype=torch.int64,
    ).float()

    # --------------------------------------------------------
    # Frozen deterministic V3.3 embedding
    # --------------------------------------------------------

    with torch.no_grad():

        protected = embed_payload(
            original,
            payload,
        )

    if tuple(
        protected.shape
    ) != (
        1,
        3,
        256,
        256,
    ):

        raise RuntimeError(
            "Unexpected protected image shape."
        )

    if not tensor_is_finite(
        protected
    ):

        raise RuntimeError(
            "Protected image contains "
            "non-finite values."
        )

    # --------------------------------------------------------
    # Controlled erase manipulation
    # --------------------------------------------------------

    received = (
        protected
        .clone()
    )

    received[
        :,
        :,
        80:160,
        96:176,
    ] = 0.0

    # --------------------------------------------------------
    # Ground-truth mask
    # --------------------------------------------------------

    mask_256 = torch.zeros(
        1,
        1,
        256,
        256,
        device=DEVICE,
        dtype=torch.float32,
    )

    mask_256[
        :,
        :,
        80:160,
        96:176,
    ] = 1.0

    detection_target = torch.ones(
        1,
        device=DEVICE,
        dtype=torch.float32,
    )

    # ========================================================
    # Complete forward
    # ========================================================

    output = model(
        received,
        payload,
    )

    # --------------------------------------------------------
    # Required outputs
    # --------------------------------------------------------

    required_keys = [

        "decoder_prob",

        "evidence_grid",

        "vit_evidence_grid",

        "patch_evidence",

        "visual_tokens",

        "fused_tokens",

        "node_features",

        "graph_tokens",

        "detection_logits",

        "localization_node_logits",

        "localization_grid_logits",

    ]

    for key in required_keys:

        if key not in output:

            raise RuntimeError(
                "Missing model output: "
                f"{key}"
            )

    # ========================================================
    # Forward shape audit
    # ========================================================

    print()

    print("-" * 78)

    print(
        "FULL FORWARD SHAPES"
    )

    print("-" * 78)

    for key in required_keys:

        print(
            f"{key:28s}:",
            tuple(
                output[
                    key
                ].shape
            ),
        )

    expected_shapes = {

        "decoder_prob":
            (
                1,
                128,
            ),

        "evidence_grid":
            (
                1,
                1,
                8,
                16,
            ),

        "vit_evidence_grid":
            (
                1,
                1,
                14,
                14,
            ),

        "patch_evidence":
            (
                1,
                196,
                1,
            ),

        "visual_tokens":
            (
                1,
                196,
                768,
            ),

        "fused_tokens":
            (
                1,
                196,
                768,
            ),

        "node_features":
            (
                1,
                196,
                771,
            ),

        "graph_tokens":
            (
                1,
                196,
                256,
            ),

        "detection_logits":
            (
                1,
            ),

        "localization_node_logits":
            (
                1,
                196,
            ),

        "localization_grid_logits":
            (
                1,
                1,
                14,
                14,
            ),

    }

    for key, expected_shape in (
        expected_shapes.items()
    ):

        actual_shape = tuple(
            output[
                key
            ].shape
        )

        if actual_shape != expected_shape:

            raise RuntimeError(
                f"{key} shape failure. "
                f"Expected {expected_shape}, "
                f"got {actual_shape}."
            )

        if not tensor_is_finite(
            output[
                key
            ]
        ):

            raise RuntimeError(
                f"{key} contains "
                "non-finite values."
            )

    # ========================================================
    # 14x14 -> 256x256 localization projection
    # ========================================================

    localization_logits_256 = (
        model.localization_logits_256(
            output[
                "localization_grid_logits"
            ]
        )
    )

    print(
        f"{'localization_logits_256':28s}:",
        tuple(
            localization_logits_256.shape
        ),
    )

    if tuple(
        localization_logits_256.shape
    ) != (
        1,
        1,
        256,
        256,
    ):

        raise RuntimeError(
            "Localization 256x256 "
            "projection shape failure."
        )

    if not tensor_is_finite(
        localization_logits_256
    ):

        raise RuntimeError(
            "Localization 256x256 logits "
            "contain non-finite values."
        )

    print()

    print(
        "Full forward shape/finite gate : PASS"
    )

    # ========================================================
    # Localization target projection
    #
    # IMPORTANT:
    #
    # This is used only for the gradient smoke test.
    #
    # Final training loss has NOT yet been frozen.
    # ========================================================

    localization_target_14 = (
        F.interpolate(
            mask_256,
            size=(
                14,
                14,
            ),
            mode="area",
        )
    )

    if tuple(
        localization_target_14.shape
    ) != (
        1,
        1,
        14,
        14,
    ):

        raise RuntimeError(
            "Localization target projection "
            "shape failure."
        )

    if (
        localization_target_14.min().item()
        < 0.0
    ):

        raise RuntimeError(
            "Localization target below zero."
        )

    if (
        localization_target_14.max().item()
        > 1.0
    ):

        raise RuntimeError(
            "Localization target above one."
        )

    print()

    print("-" * 78)

    print(
        "LOCALIZATION TARGET PROJECTION"
    )

    print("-" * 78)

    print(
        "Target shape :",
        tuple(
            localization_target_14.shape
        ),
    )

    print(
        "Target min   :",
        f"{localization_target_14.min().item():.8f}",
    )

    print(
        "Target max   :",
        f"{localization_target_14.max().item():.8f}",
    )

    print(
        "Target mean  :",
        f"{localization_target_14.mean().item():.8f}",
    )

    # ========================================================
    # Gradient-only joint loss
    #
    # IMPORTANT:
    #
    # These losses are NOT yet the final formal training
    # objective.
    #
    # They are used ONLY to prove that gradients can flow
    # through the entire trainable chain.
    # ========================================================

    detection_loss = (
        F.binary_cross_entropy_with_logits(
            output[
                "detection_logits"
            ],
            detection_target,
        )
    )

    localization_loss = (
        F.binary_cross_entropy_with_logits(
            output[
                "localization_grid_logits"
            ],
            localization_target_14,
        )
    )

    joint_loss = (
        detection_loss
        +
        localization_loss
    )

    print()

    print("-" * 78)

    print(
        "GRADIENT-ONLY JOINT LOSS"
    )

    print("-" * 78)

    print(
        "Detection BCE    :",
        f"{detection_loss.item():.8f}",
    )

    print(
        "Localization BCE :",
        f"{localization_loss.item():.8f}",
    )

    print(
        "Joint loss       :",
        f"{joint_loss.item():.8f}",
    )

    if not torch.isfinite(
        joint_loss
    ):

        raise RuntimeError(
            "Joint loss is non-finite."
        )

    # ========================================================
    # Backward
    # ========================================================

    model.zero_grad(
        set_to_none=True
    )

    joint_loss.backward()

    # --------------------------------------------------------
    # Frozen gradients
    # --------------------------------------------------------

    decoder_gradient = (
        has_nonzero_gradient(
            decoder
        )
    )

    vit_gradient = (
        has_nonzero_gradient(
            vit
        )
    )

    # --------------------------------------------------------
    # Trainable gradients
    # --------------------------------------------------------

    adapter_gradient = (
        has_nonzero_gradient(
            adapter
        )
    )

    adapter_scale_gradient = (

        adapter_scale.grad
        is not None

        and

        bool(
            (
                adapter_scale.grad.abs()
                > 0
            ).any()
        )
    )

    fusion_gradient = (
        has_nonzero_gradient(
            fusion
        )
    )

    gnn_gradient = (
        has_nonzero_gradient(
            gnn
        )
    )

    detection_gradient = (
        has_nonzero_gradient(
            detection_head
        )
    )

    localization_gradient = (
        has_nonzero_gradient(
            localization_head
        )
    )

    print()

    print("-" * 78)

    print(
        "END-TO-END GRADIENT FLOW"
    )

    print("-" * 78)

    print(
        "Decoder gradient           :",
        decoder_gradient,
    )

    print(
        "ViT gradient               :",
        vit_gradient,
    )

    print(
        "Adapter gradient           :",
        adapter_gradient,
    )

    print(
        "Adapter-scale gradient     :",
        adapter_scale_gradient,
    )

    print(
        "Evidence-fusion gradient   :",
        fusion_gradient,
    )

    print(
        "GNN gradient               :",
        gnn_gradient,
    )

    print(
        "Detection-head gradient    :",
        detection_gradient,
    )

    print(
        "Localization-head gradient :",
        localization_gradient,
    )

    # --------------------------------------------------------
    # Frozen modules MUST remain gradient-free
    # --------------------------------------------------------

    if decoder_gradient:

        raise RuntimeError(
            "Gradient entered frozen "
            "V3.3 decoder."
        )

    if vit_gradient:

        raise RuntimeError(
            "Gradient entered frozen "
            "pretrained ViT."
        )

    # --------------------------------------------------------
    # Trainable chain MUST receive gradients
    # --------------------------------------------------------

    if not adapter_gradient:

        raise RuntimeError(
            "Token adapter received "
            "no gradient."
        )

    if not adapter_scale_gradient:

        raise RuntimeError(
            "Adapter scale received "
            "no gradient."
        )

    if not fusion_gradient:

        raise RuntimeError(
            "Evidence fusion received "
            "no gradient."
        )

    if not gnn_gradient:

        raise RuntimeError(
            "GNN received no gradient."
        )

    if not detection_gradient:

        raise RuntimeError(
            "Detection head received "
            "no gradient."
        )

    if not localization_gradient:

        raise RuntimeError(
            "Localization head received "
            "no gradient."
        )

    print()

    print(
        "End-to-end gradient gate : PASS"
    )

    # ========================================================
    # Whole-chain provenance / payload dependency
    #
    # Same received image.
    #
    # Only supplied known payload changes.
    #
    # Expected:
    #
    # decoder output:
    #     identical
    #
    # evidence:
    #     changes
    #
    # graph representation:
    #     changes
    #
    # detection:
    #     changes
    #
    # localization:
    #     changes
    #
    # This is a structural dependency test only.
    # ========================================================

    model.eval()

    wrong_payload = (
        1.0
        -
        payload
    )

    with torch.no_grad():

        correct_output = model(
            received,
            payload,
        )

        wrong_output = model(
            received,
            wrong_payload,
        )

    # --------------------------------------------------------
    # Decoder receives image only.
    #
    # Supplied payload must therefore NOT alter decoder output.
    # --------------------------------------------------------

    decoder_difference = float(
        (
            correct_output[
                "decoder_prob"
            ]
            -
            wrong_output[
                "decoder_prob"
            ]
        )
        .abs()
        .max()
        .item()
    )

    # --------------------------------------------------------
    # Evidence explicitly depends on supplied known payload.
    # --------------------------------------------------------

    evidence_difference = float(
        (
            correct_output[
                "evidence_grid"
            ]
            -
            wrong_output[
                "evidence_grid"
            ]
        )
        .abs()
        .mean()
        .item()
    )

    # --------------------------------------------------------
    # Graph representation must inherit provenance dependency.
    # --------------------------------------------------------

    graph_difference = float(
        (
            correct_output[
                "graph_tokens"
            ]
            -
            wrong_output[
                "graph_tokens"
            ]
        )
        .abs()
        .mean()
        .item()
    )

    # --------------------------------------------------------
    # Task outputs must inherit graph dependency.
    # --------------------------------------------------------

    detection_difference = float(
        (
            correct_output[
                "detection_logits"
            ]
            -
            wrong_output[
                "detection_logits"
            ]
        )
        .abs()
        .mean()
        .item()
    )

    localization_difference = float(
        (
            correct_output[
                "localization_grid_logits"
            ]
            -
            wrong_output[
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
        "WHOLE-CHAIN PAYLOAD DEPENDENCY"
    )

    print("-" * 78)

    print(
        "Decoder difference      :",
        f"{decoder_difference:.12f}",
    )

    print(
        "Evidence difference     :",
        f"{evidence_difference:.12f}",
    )

    print(
        "Graph-token difference  :",
        f"{graph_difference:.12f}",
    )

    print(
        "Detection difference    :",
        f"{detection_difference:.12f}",
    )

    print(
        "Localization difference :",
        f"{localization_difference:.12f}",
    )

    # --------------------------------------------------------
    # Decoder invariance
    # --------------------------------------------------------

    if decoder_difference != 0.0:

        raise RuntimeError(
            "Decoder unexpectedly changed when "
            "only the supplied payload changed."
        )

    # --------------------------------------------------------
    # Provenance dependency
    # --------------------------------------------------------

    if evidence_difference <= 0.0:

        raise RuntimeError(
            "Reference-free evidence does not "
            "depend on supplied payload."
        )

    if graph_difference <= 0.0:

        raise RuntimeError(
            "Graph representation does not "
            "inherit provenance evidence."
        )

    if detection_difference <= 0.0:

        raise RuntimeError(
            "Detection output does not inherit "
            "provenance dependency."
        )

    if localization_difference <= 0.0:

        raise RuntimeError(
            "Localization output does not inherit "
            "provenance dependency."
        )

    print()

    print(
        "Whole-chain payload dependency : PASS"
    )

    # ========================================================
    # Final result
    # ========================================================

    print()

    print("=" * 78)

    print(
        "E5.3R-4B-1 FULL JOINT MODEL "
        "END-TO-END GATE: PASS"
    )

    print("=" * 78)

    print()

    print(
        "Validated chain:"
    )

    print(
        "received image"
    )

    print(
        "  -> frozen V3.3 decoder"
    )

    print(
        "  -> reference-free evidence"
    )

    print(
        "  -> frozen pretrained ViT"
    )

    print(
        "  -> trainable token adapter"
    )

    print(
        "  -> trainable evidence fusion"
    )

    print(
        "  -> deterministic graph construction"
    )

    print(
        "  -> trainable GNN"
    )

    print(
        "  -> detection head"
    )

    print(
        "  -> localization head"
    )

    print()

    print(
        "No IMD2020 dataset image was accessed."
    )

    print(
        "No optimizer step was performed."
    )

    print(
        "No model was trained."
    )

    print(
        "No Test data was accessed."
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "The BCE losses above are gradient-gate "
        "diagnostics only."
    )

    print(
        "They are NOT formal experimental results "
        "and are NOT the frozen training objective."
    )


if __name__ == "__main__":

    main()