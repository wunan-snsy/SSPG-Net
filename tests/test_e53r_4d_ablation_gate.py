"""
E5.3R-4D-0
Formal Ablation Architecture Gate
=================================

Revision:
    E5.3R-4D-0-R4

Purpose
-------
Final pre-training architecture gate for the formal SSPG-Net V2
training-level ablation study.

Variants
--------
FULL
    Visual + Provenance + Graph/GNN

NO_PROVENANCE
    Visual + Graph/GNN
    Payload-derived provenance absent

NO_GRAPH
    Visual + Provenance
    Graph/GNN absent

VISUAL_ONLY
    Visual only
    Provenance and Graph/GNN absent

R4 additions
------------
In addition to the R3 functional/causal checks, R4 verifies:

    - inactive provenance fusion is non-trainable
    - inactive provenance fusion remains in eval mode
    - formal trainable parameter count matches the preregistered
      architecture definition

Expected formal trainable parameter counts
------------------------------------------
FULL:
    1,416,490

NO_PROVENANCE:
    1,341,801

NO_GRAPH:
    555,812

VISUAL_ONLY:
    481,123

Scientific scope
----------------
This script:
    - does NOT access IMD2020
    - does NOT access Test
    - does NOT initialize an optimizer
    - does NOT train
    - does NOT select thresholds
    - does NOT save scientific checkpoints
    - does NOT report performance metrics
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch


# ======================================================================
# PROJECT ROOT
# ======================================================================

ROOT = Path(
    __file__
).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


from models.sspg_ablation_model_v2 import (
    SSPGAblationModelV2,
    VALID_ABLATIONS,
)


# ======================================================================
# PATHS
# ======================================================================

STEGO_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "stego_structured_v33_128bit_best.pth"
)


# ======================================================================
# DEVICE
# ======================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ======================================================================
# CONSTANTS
# ======================================================================

GATE_SEED = 20260926

BATCH_SIZE = 1

IMAGE_SIZE = 256

PAYLOAD_BITS = 128

VIT_TOKEN_COUNT = 196

VIT_TOKEN_DIM = 768

HEAD_TOKEN_DIM = 256

LOCALIZATION_GRID_SIZE = 14

NUMERICAL_NOISE_FLOOR = 1e-6

REPEAT_NOISE_MULTIPLIER = 5.0

PROVENANCE_DEPENDENCE_MIN = 1e-5


EXPECTED_TRAINABLE_PARAMETERS = {
    "full":
        1_416_490,

    "no_provenance":
        1_341_801,

    "no_graph":
        555_812,

    "visual_only":
        481_123,
}


EXPECTED_FUSION_TRAINABLE = {
    "full":
        74_689,

    "no_provenance":
        0,

    "no_graph":
        74_689,

    "visual_only":
        0,
}


# ======================================================================
# BASIC HELPERS
# ======================================================================

def set_gate_seed() -> None:

    torch.manual_seed(
        GATE_SEED
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            GATE_SEED
        )


def count_parameters(
    module,
) -> int:

    if module is None:
        return 0

    return sum(
        parameter.numel()
        for parameter
        in module.parameters()
    )


def count_trainable(
    module,
) -> int:

    if module is None:
        return 0

    return sum(
        parameter.numel()
        for parameter
        in module.parameters()
        if parameter.requires_grad
    )


def clear_gradients(
    model: torch.nn.Module,
) -> None:

    for parameter in model.parameters():
        parameter.grad = None


def any_trainable_gradient(
    module,
) -> bool:

    if module is None:
        return False

    for parameter in module.parameters():

        if not parameter.requires_grad:
            continue

        if parameter.grad is None:
            continue

        gradient = (
            parameter
            .grad
            .detach()
        )

        if not torch.isfinite(
            gradient
        ).all():

            raise RuntimeError(
                "Non-finite gradient detected."
            )

        if float(
            gradient
            .abs()
            .sum()
            .item()
        ) > 0.0:

            return True

    return False


# ======================================================================
# PRINT HELPERS
# ======================================================================

def print_bool(
    name: str,
    value: bool,
) -> None:

    print(
        f"{name:<40}: {bool(value)}"
    )


def print_int(
    name: str,
    value: int,
) -> None:

    print(
        f"{name:<40}: {int(value):,}"
    )


def print_float(
    name: str,
    value: float,
) -> None:

    print(
        f"{name:<40}: {float(value):.10f}"
    )


# ======================================================================
# TENSOR HELPERS
# ======================================================================

def assert_tensor_shape(
    tensor,
    expected_shape,
    name: str,
) -> None:

    if not isinstance(
        tensor,
        torch.Tensor,
    ):

        raise RuntimeError(
            f"{name} is not a tensor."
        )

    actual_shape = tuple(
        tensor.shape
    )

    expected_shape = tuple(
        expected_shape
    )

    if actual_shape != expected_shape:

        raise RuntimeError(
            f"{name} shape mismatch: "
            f"expected {expected_shape}, "
            f"got {actual_shape}."
        )

    if not torch.isfinite(
        tensor
    ).all():

        raise RuntimeError(
            f"{name} contains NaN or Inf."
        )


def max_abs_diff(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:

    return float(
        (
            a.detach()
            -
            b.detach()
        )
        .abs()
        .max()
        .item()
    )


def assert_exact_zero(
    tensor: torch.Tensor,
    name: str,
) -> None:

    maximum = float(
        tensor
        .detach()
        .abs()
        .max()
        .item()
    )

    if maximum != 0.0:

        raise RuntimeError(
            f"{name} expected exact zeros, "
            f"max abs value={maximum:.10f}."
        )


# ======================================================================
# SYNTHETIC INPUT
# ======================================================================

def make_synthetic_input() -> Tuple[
    torch.Tensor,
    torch.Tensor,
]:

    received = torch.rand(
        (
            BATCH_SIZE,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        dtype=torch.float32,
        device=DEVICE,
    )

    payload = torch.randint(
        low=0,
        high=2,
        size=(
            BATCH_SIZE,
            PAYLOAD_BITS,
        ),
        device=DEVICE,
        dtype=torch.int64,
    ).float()

    return (
        received,
        payload,
    )


# ======================================================================
# ARCHITECTURE FLAGS
# ======================================================================

def check_architecture_flags(
    model: SSPGAblationModelV2,
    variant: str,
) -> Tuple[
    bool,
    bool,
]:

    expected_provenance = (
        variant
        in (
            "full",
            "no_graph",
        )
    )

    expected_graph = (
        variant
        in (
            "full",
            "no_provenance",
        )
    )

    expected_visual_only = (
        variant
        ==
        "visual_only"
    )

    if (
        model.uses_provenance
        !=
        expected_provenance
    ):

        raise RuntimeError(
            f"{variant}: provenance flag mismatch."
        )

    if (
        model.uses_graph
        !=
        expected_graph
    ):

        raise RuntimeError(
            f"{variant}: graph flag mismatch."
        )

    if (
        model.is_visual_only
        !=
        expected_visual_only
    ):

        raise RuntimeError(
            f"{variant}: visual-only flag mismatch."
        )

    print_bool(
        "Uses provenance evidence",
        model.uses_provenance,
    )

    print_bool(
        "Uses graph/GNN",
        model.uses_graph,
    )

    print_bool(
        "Visual-only",
        model.is_visual_only,
    )

    return (
        expected_provenance,
        expected_graph,
    )


# ======================================================================
# FROZEN MODULE CHECK
# ======================================================================

def check_frozen_modules(
    model: SSPGAblationModelV2,
    variant: str,
):

    decoder = (
        model
        .frozen_decoder()
    )

    vit = (
        model
        .frozen_vit()
    )

    decoder_trainable = count_trainable(
        decoder
    )

    vit_trainable = count_trainable(
        vit
    )

    if decoder_trainable != 0:

        raise RuntimeError(
            f"{variant}: frozen decoder has "
            f"{decoder_trainable} trainable parameters."
        )

    if vit_trainable != 0:

        raise RuntimeError(
            f"{variant}: frozen ViT has "
            f"{vit_trainable} trainable parameters."
        )

    print_int(
        "Decoder trainable parameters",
        decoder_trainable,
    )

    print_int(
        "ViT trainable parameters",
        vit_trainable,
    )

    return (
        decoder,
        vit,
    )


# ======================================================================
# FUSION TRAINABILITY CHECK
# ======================================================================

def check_fusion_trainability(
    model: SSPGAblationModelV2,
    variant: str,
):

    fusion = (
        model
        .provenance_fusion()
    )

    total = count_parameters(
        fusion
    )

    trainable = count_trainable(
        fusion
    )

    expected_trainable = (
        EXPECTED_FUSION_TRAINABLE[
            variant
        ]
    )

    print_int(
        "Fusion total parameters",
        total,
    )

    print_int(
        "Fusion trainable parameters",
        trainable,
    )

    if trainable != expected_trainable:

        raise RuntimeError(
            f"{variant}: fusion trainable parameter mismatch: "
            f"expected {expected_trainable:,}, "
            f"got {trainable:,}."
        )

    return fusion


# ======================================================================
# FORMAL TRAINABLE PARAMETER CHECK
# ======================================================================

def check_formal_trainable_count(
    model: SSPGAblationModelV2,
    variant: str,
) -> None:

    actual = count_trainable(
        model
    )

    expected = (
        EXPECTED_TRAINABLE_PARAMETERS[
            variant
        ]
    )

    print_int(
        "Formal trainable parameters",
        actual,
    )

    print_int(
        "Expected trainable parameters",
        expected,
    )

    if actual != expected:

        raise RuntimeError(
            f"{variant}: formal trainable parameter mismatch: "
            f"expected {expected:,}, "
            f"got {actual:,}."
        )


# ======================================================================
# TRAIN/EVAL MODE CHECK
# ======================================================================

def check_training_modes(
    model: SSPGAblationModelV2,
    decoder,
    vit,
    fusion,
    variant: str,
) -> None:

    model.train()

    if not model.training:

        raise RuntimeError(
            f"{variant}: model failed to enter train mode."
        )

    if decoder.training:

        raise RuntimeError(
            f"{variant}: frozen decoder entered train mode."
        )

    if vit.training:

        raise RuntimeError(
            f"{variant}: frozen ViT entered train mode."
        )

    if not model.uses_provenance:

        if fusion.training:

            raise RuntimeError(
                f"{variant}: inactive provenance fusion "
                "entered train mode."
            )

    print_bool(
        "Model training mode",
        model.training,
    )

    print_bool(
        "Decoder training mode",
        decoder.training,
    )

    print_bool(
        "ViT training mode",
        vit.training,
    )

    print_bool(
        "Fusion training mode",
        fusion.training,
    )


# ======================================================================
# COMMON FORWARD CONTRACT
# ======================================================================

def check_common_forward_contract(
    model: SSPGAblationModelV2,
    output: Dict[str, object],
) -> None:

    required_keys = (
        "visual_tokens",
        "head_tokens",
        "detection_logits",
        "localization_grid_logits",
    )

    for key in required_keys:

        if key not in output:

            raise RuntimeError(
                f"Forward output missing '{key}'."
            )

    assert_tensor_shape(
        output[
            "visual_tokens"
        ],
        (
            BATCH_SIZE,
            VIT_TOKEN_COUNT,
            VIT_TOKEN_DIM,
        ),
        "visual_tokens",
    )

    assert_tensor_shape(
        output[
            "head_tokens"
        ],
        (
            BATCH_SIZE,
            VIT_TOKEN_COUNT,
            HEAD_TOKEN_DIM,
        ),
        "head_tokens",
    )

    assert_tensor_shape(
        output[
            "detection_logits"
        ],
        (
            BATCH_SIZE,
        ),
        "detection_logits",
    )

    assert_tensor_shape(
        output[
            "localization_grid_logits"
        ],
        (
            BATCH_SIZE,
            1,
            LOCALIZATION_GRID_SIZE,
            LOCALIZATION_GRID_SIZE,
        ),
        "localization_grid_logits",
    )

    localization_256 = (
        model
        .localization_logits_256(
            output[
                "localization_grid_logits"
            ]
        )
    )

    assert_tensor_shape(
        localization_256,
        (
            BATCH_SIZE,
            1,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        "localization_logits_256",
    )


# ======================================================================
# VARIANT-SPECIFIC FORWARD CONTRACT
# ======================================================================

def check_variant_forward_contract(
    output: Dict[str, object],
    variant: str,
) -> None:

    if variant == "full":

        assert_tensor_shape(
            output[
                "patch_evidence"
            ],
            (
                BATCH_SIZE,
                VIT_TOKEN_COUNT,
                1,
            ),
            "patch_evidence",
        )

        assert_tensor_shape(
            output[
                "graph_tokens"
            ],
            (
                BATCH_SIZE,
                VIT_TOKEN_COUNT,
                HEAD_TOKEN_DIM,
            ),
            "graph_tokens",
        )

        return

    if variant == "no_provenance":

        assert_tensor_shape(
            output[
                "patch_evidence"
            ],
            (
                BATCH_SIZE,
                VIT_TOKEN_COUNT,
                1,
            ),
            "patch_evidence",
        )

        assert_exact_zero(
            output[
                "patch_evidence"
            ],
            "NO_PROVENANCE patch_evidence",
        )

        assert_tensor_shape(
            output[
                "graph_tokens"
            ],
            (
                BATCH_SIZE,
                VIT_TOKEN_COUNT,
                HEAD_TOKEN_DIM,
            ),
            "graph_tokens",
        )

        return

    if variant == "no_graph":

        assert_tensor_shape(
            output[
                "patch_evidence"
            ],
            (
                BATCH_SIZE,
                VIT_TOKEN_COUNT,
                1,
            ),
            "patch_evidence",
        )

        if (
            output[
                "graph_tokens"
            ]
            is not None
        ):

            raise RuntimeError(
                "NO_GRAPH unexpectedly returned graph_tokens."
            )

        return

    if variant == "visual_only":

        if (
            output[
                "patch_evidence"
            ]
            is not None
        ):

            raise RuntimeError(
                "VISUAL_ONLY unexpectedly returned patch_evidence."
            )

        if (
            output[
                "graph_tokens"
            ]
            is not None
        ):

            raise RuntimeError(
                "VISUAL_ONLY unexpectedly returned graph_tokens."
            )

        return

    raise RuntimeError(
        f"Unknown variant: {variant}"
    )


# ======================================================================
# PAYLOAD CAUSALITY / INDEPENDENCE
# ======================================================================

def check_payload_dependency(
    model: SSPGAblationModelV2,
    received: torch.Tensor,
    payload: torch.Tensor,
    variant: str,
    expected_provenance: bool,
) -> None:

    model.eval()

    wrong_payload = (
        1.0
        -
        payload
    )

    with torch.no_grad():

        normal_a = model(
            received,
            payload,
        )

        normal_b = model(
            received,
            payload,
        )

        wrong = model(
            received,
            wrong_payload,
        )

    repeat_detection_diff = max_abs_diff(
        normal_a[
            "detection_logits"
        ],
        normal_b[
            "detection_logits"
        ],
    )

    repeat_localization_diff = max_abs_diff(
        normal_a[
            "localization_grid_logits"
        ],
        normal_b[
            "localization_grid_logits"
        ],
    )

    payload_detection_diff = max_abs_diff(
        normal_a[
            "detection_logits"
        ],
        wrong[
            "detection_logits"
        ],
    )

    payload_localization_diff = max_abs_diff(
        normal_a[
            "localization_grid_logits"
        ],
        wrong[
            "localization_grid_logits"
        ],
    )

    detection_noise_limit = max(
        NUMERICAL_NOISE_FLOOR,
        (
            REPEAT_NOISE_MULTIPLIER
            *
            repeat_detection_diff
        )
        +
        NUMERICAL_NOISE_FLOOR,
    )

    localization_noise_limit = max(
        NUMERICAL_NOISE_FLOOR,
        (
            REPEAT_NOISE_MULTIPLIER
            *
            repeat_localization_diff
        )
        +
        NUMERICAL_NOISE_FLOOR,
    )

    print_float(
        "Repeat detection max diff",
        repeat_detection_diff,
    )

    print_float(
        "Repeat localization max diff",
        repeat_localization_diff,
    )

    print_float(
        "Payload detection max diff",
        payload_detection_diff,
    )

    print_float(
        "Payload localization max diff",
        payload_localization_diff,
    )

    print_float(
        "Detection noise envelope",
        detection_noise_limit,
    )

    print_float(
        "Localization noise envelope",
        localization_noise_limit,
    )

    if expected_provenance:

        detection_signal = (
            payload_detection_diff
            >
            max(
                PROVENANCE_DEPENDENCE_MIN,
                detection_noise_limit,
            )
        )

        localization_signal = (
            payload_localization_diff
            >
            max(
                PROVENANCE_DEPENDENCE_MIN,
                localization_noise_limit,
            )
        )

        print_bool(
            "Detection payload signal",
            detection_signal,
        )

        print_bool(
            "Localization payload signal",
            localization_signal,
        )

        if not (
            detection_signal
            or
            localization_signal
        ):

            raise RuntimeError(
                f"{variant}: provenance-enabled architecture "
                "does not show payload dependence beyond "
                "numerical noise."
            )

        return

    detection_independent = (
        payload_detection_diff
        <=
        detection_noise_limit
    )

    localization_independent = (
        payload_localization_diff
        <=
        localization_noise_limit
    )

    print_bool(
        "Detection payload-independent",
        detection_independent,
    )

    print_bool(
        "Localization payload-independent",
        localization_independent,
    )

    if not detection_independent:

        raise RuntimeError(
            f"{variant}: detection changes beyond numerical "
            "noise despite provenance being disabled."
        )

    if not localization_independent:

        raise RuntimeError(
            f"{variant}: localization changes beyond numerical "
            "noise despite provenance being disabled."
        )


# ======================================================================
# GRADIENT CHECK
# ======================================================================

def check_gradient_contract(
    model: SSPGAblationModelV2,
    received: torch.Tensor,
    payload: torch.Tensor,
    decoder,
    vit,
    fusion,
    variant: str,
    expected_graph: bool,
    expected_provenance: bool,
) -> None:

    model.train()

    clear_gradients(
        model
    )

    output = model(
        received,
        payload,
    )

    synthetic_gate_loss = (
        output[
            "detection_logits"
        ].mean()
        +
        output[
            "localization_grid_logits"
        ].mean()
    )

    if not torch.isfinite(
        synthetic_gate_loss
    ):

        raise RuntimeError(
            f"{variant}: synthetic gate loss is non-finite."
        )

    synthetic_gate_loss.backward()

    decoder_has_gradient = (
        any_trainable_gradient(
            decoder
        )
    )

    vit_has_gradient = (
        any_trainable_gradient(
            vit
        )
    )

    adapter_has_gradient = (
        any_trainable_gradient(
            model.visual_adapter()
        )
    )

    fusion_has_gradient = (
        any_trainable_gradient(
            fusion
        )
    )

    heads_have_gradient = (
        any_trainable_gradient(
            model.heads
        )
    )

    if decoder_has_gradient:

        raise RuntimeError(
            f"{variant}: frozen decoder received gradient."
        )

    if vit_has_gradient:

        raise RuntimeError(
            f"{variant}: frozen ViT received gradient."
        )

    if not adapter_has_gradient:

        raise RuntimeError(
            f"{variant}: visual adapter received no gradient."
        )

    if not heads_have_gradient:

        raise RuntimeError(
            f"{variant}: task heads received no gradient."
        )

    if expected_provenance:

        if not fusion_has_gradient:

            raise RuntimeError(
                f"{variant}: active provenance fusion "
                "received no gradient."
            )

    else:

        if fusion_has_gradient:

            raise RuntimeError(
                f"{variant}: inactive provenance fusion "
                "unexpectedly received gradient."
            )

    print_bool(
        "Decoder has trainable gradient",
        decoder_has_gradient,
    )

    print_bool(
        "ViT has trainable gradient",
        vit_has_gradient,
    )

    print_bool(
        "Visual adapter has gradient",
        adapter_has_gradient,
    )

    print_bool(
        "Provenance fusion has gradient",
        fusion_has_gradient,
    )

    print_bool(
        "Prediction heads have gradient",
        heads_have_gradient,
    )

    # ------------------------------------------------------------------
    # GNN
    # ------------------------------------------------------------------

    if model.graph_frontend is not None:

        gnn_has_gradient = (
            any_trainable_gradient(
                model
                .graph_frontend
                .gnn
            )
        )

    else:

        gnn_has_gradient = False

    print_bool(
        "GNN has gradient",
        gnn_has_gradient,
    )

    if expected_graph:

        if not gnn_has_gradient:

            raise RuntimeError(
                f"{variant}: graph-enabled variant has "
                "no GNN gradient."
            )

    else:

        if gnn_has_gradient:

            raise RuntimeError(
                f"{variant}: graph-disabled variant unexpectedly "
                "has GNN gradient."
            )

    # ------------------------------------------------------------------
    # Graph-free projection
    # ------------------------------------------------------------------

    if model.token_projection is not None:

        projection_has_gradient = (
            any_trainable_gradient(
                model.token_projection
            )
        )

    else:

        projection_has_gradient = False

    print_bool(
        "Token projection has gradient",
        projection_has_gradient,
    )

    if expected_graph:

        if projection_has_gradient:

            raise RuntimeError(
                f"{variant}: graph-based variant unexpectedly "
                "uses graph-free projection."
            )

    else:

        if not projection_has_gradient:

            raise RuntimeError(
                f"{variant}: graph-free projection received "
                "no gradient."
            )


# ======================================================================
# PARAMETER REPORT
# ======================================================================

def print_parameter_report(
    model: SSPGAblationModelV2,
) -> None:

    print()

    print_int(
        "Total model parameters",
        count_parameters(
            model
        ),
    )

    print_int(
        "Trainable model parameters",
        count_trainable(
            model
        ),
    )

    print_int(
        "Prediction-head parameters",
        count_parameters(
            model.heads
        ),
    )

    print_int(
        "Prediction-head trainable",
        count_trainable(
            model.heads
        ),
    )

    print_int(
        "Fusion parameters",
        count_parameters(
            model.provenance_fusion()
        ),
    )

    print_int(
        "Fusion trainable",
        count_trainable(
            model.provenance_fusion()
        ),
    )

    if model.graph_frontend is not None:

        print_int(
            "GNN parameters",
            count_parameters(
                model
                .graph_frontend
                .gnn
            ),
        )

        print_int(
            "GNN trainable",
            count_trainable(
                model
                .graph_frontend
                .gnn
            ),
        )

    else:

        print_int(
            "GNN parameters",
            0,
        )

        print_int(
            "GNN trainable",
            0,
        )

    if model.token_projection is not None:

        print_int(
            "Token-projection parameters",
            count_parameters(
                model.token_projection
            ),
        )

        print_int(
            "Token-projection trainable",
            count_trainable(
                model.token_projection
            ),
        )

    else:

        print_int(
            "Token-projection parameters",
            0,
        )

        print_int(
            "Token-projection trainable",
            0,
        )


# ======================================================================
# ONE VARIANT
# ======================================================================

def test_variant(
    variant: str,
) -> None:

    print()
    print("=" * 78)
    print(
        f"VARIANT: {variant.upper()}"
    )
    print("=" * 78)

    set_gate_seed()

    model = SSPGAblationModelV2(
        stego_checkpoint=str(
            STEGO_CHECKPOINT
        ),
        ablation=variant,
        adapter_bottleneck=96,
        fusion_hidden_dim=96,
        knn_k=8,
        gnn_hidden_dim=256,
        gnn_depth=2,
    ).to(
        DEVICE
    )

    (
        expected_provenance,
        expected_graph,
    ) = check_architecture_flags(
        model,
        variant,
    )

    (
        decoder,
        vit,
    ) = check_frozen_modules(
        model,
        variant,
    )

    fusion = check_fusion_trainability(
        model,
        variant,
    )

    check_formal_trainable_count(
        model,
        variant,
    )

    check_training_modes(
        model,
        decoder,
        vit,
        fusion,
        variant,
    )

    (
        received,
        payload,
    ) = make_synthetic_input()

    output = model(
        received,
        payload,
    )

    check_common_forward_contract(
        model,
        output,
    )

    check_variant_forward_contract(
        output,
        variant,
    )

    check_payload_dependency(
        model,
        received,
        payload,
        variant,
        expected_provenance,
    )

    check_gradient_contract(
        model,
        received,
        payload,
        decoder,
        vit,
        fusion,
        variant,
        expected_graph,
        expected_provenance,
    )

    print_parameter_report(
        model
    )

    print()
    print(
        f"{variant.upper()} ARCHITECTURE GATE: PASS"
    )

    del output
    del received
    del payload
    del model

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:

    print()
    print("=" * 78)
    print(
        "E5.3R-4D-0 FORMAL ABLATION ARCHITECTURE GATE"
    )
    print("=" * 78)

    print(
        "Revision                   : E5.3R-4D-0-R4"
    )

    print(
        "Device                     :",
        DEVICE,
    )

    print(
        "Stego checkpoint           :",
        STEGO_CHECKPOINT,
    )

    print(
        "IMD2020 accessed           : False"
    )

    print(
        "Test accessed              : False"
    )

    print(
        "Optimizer initialized      : False"
    )

    print(
        "Training performed         : False"
    )

    if not STEGO_CHECKPOINT.is_file():

        raise RuntimeError(
            "Missing frozen V3.3 checkpoint: "
            f"{STEGO_CHECKPOINT}"
        )

    expected_variants = (
        "full",
        "no_provenance",
        "no_graph",
        "visual_only",
    )

    if tuple(
        VALID_ABLATIONS
    ) != expected_variants:

        raise RuntimeError(
            "VALID_ABLATIONS mismatch: "
            f"expected {expected_variants}, "
            f"got {tuple(VALID_ABLATIONS)}."
        )

    for variant in VALID_ABLATIONS:

        test_variant(
            variant
        )

    print()
    print("=" * 78)
    print(
        "E5.3R-4D-0-R4 FINAL GATE: PASS"
    )
    print("=" * 78)

    print()

    print(
        "FULL          : "
        "Visual + Provenance + Graph/GNN"
    )

    print(
        "NO_PROVENANCE : "
        "Visual + Graph/GNN; "
        "provenance absent and inactive fusion frozen"
    )

    print(
        "NO_GRAPH      : "
        "Visual + Provenance; "
        "Graph/GNN absent"
    )

    print(
        "VISUAL_ONLY   : "
        "Visual only; provenance absent; "
        "Graph/GNN absent; inactive fusion frozen"
    )

    print()

    print(
        "Expected trainable parameters:"
    )

    print(
        "FULL          : 1,416,490"
    )

    print(
        "NO_PROVENANCE : 1,341,801"
    )

    print(
        "NO_GRAPH      :   555,812"
    )

    print(
        "VISUAL_ONLY   :   481,123"
    )

    print()

    print(
        "IMD2020 accessed       : False"
    )

    print(
        "Test accessed          : False"
    )

    print(
        "Optimizer initialized  : False"
    )

    print(
        "Training performed     : False"
    )


if __name__ == "__main__":

    main()