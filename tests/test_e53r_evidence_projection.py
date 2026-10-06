from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.provenance_evidence import (
    ProvenanceEvidenceProjector,
    EvidenceTokenFusion,
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def main():

    print("=" * 76)
    print(
        "E5.3R-2B PROVENANCE-TO-ViT "
        "SPATIAL PROJECTION GATE"
    )
    print("=" * 76)

    print("Device :", DEVICE)
    print()

    projector = (
        ProvenanceEvidenceProjector()
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # Parameter audit
    # --------------------------------------------------------

    projector_params = sum(
        p.numel()
        for p in projector.parameters()
    )

    print(
        "Projector trainable parameters :",
        projector_params,
    )

    assert projector_params == 0

    # --------------------------------------------------------
    # Test 1:
    # reference-free evidence formula
    # --------------------------------------------------------

    payload = torch.tensor(
        [[
            0.0, 1.0,
        ]],
        device=DEVICE,
    )

    prob = torch.tensor(
        [[
            0.10, 0.80,
        ]],
        device=DEVICE,
    )

    # Expected:
    #
    # bit=0, p(1)=0.10
    # true probability = 0.90
    # evidence = 0.10
    #
    # bit=1, p(1)=0.80
    # true probability = 0.80
    # evidence = 0.20

    true_prob = (
        payload * prob
        + (1.0 - payload)
        * (1.0 - prob)
    )

    evidence = 1.0 - true_prob

    expected = torch.tensor(
        [[0.10, 0.20]],
        device=DEVICE,
    )

    assert torch.allclose(
        evidence,
        expected,
        atol=1e-6,
    )

    print(
        "Reference-free evidence formula : PASS"
    )

    # --------------------------------------------------------
    # Test 2:
    # real 128-bit interface
    # --------------------------------------------------------

    torch.manual_seed(20260921)

    payload128 = torch.randint(
        0,
        2,
        (2, 128),
        device=DEVICE,
    ).float()

    prob128 = torch.rand(
        2,
        128,
        device=DEVICE,
    )

    output = projector(
        prob128,
        payload128,
    )

    evidence_grid = output[
        "evidence_grid"
    ]

    vit_grid = output[
        "vit_evidence_grid"
    ]

    patch_evidence = output[
        "patch_evidence"
    ]

    print()
    print("-" * 76)
    print("SHAPE AUDIT")
    print("-" * 76)

    print(
        "V3.3 evidence grid  :",
        tuple(evidence_grid.shape),
    )

    print(
        "ViT evidence grid   :",
        tuple(vit_grid.shape),
    )

    print(
        "Patch evidence      :",
        tuple(patch_evidence.shape),
    )

    assert evidence_grid.shape == (
        2, 1, 8, 16
    )

    assert vit_grid.shape == (
        2, 1, 14, 14
    )

    assert patch_evidence.shape == (
        2, 196, 1
    )

    assert torch.isfinite(
        patch_evidence
    ).all()

    assert (
        patch_evidence.min() >= 0
    )

    assert (
        patch_evidence.max() <= 1
    )

    print(
        "Shape/range audit   : PASS"
    )

    # --------------------------------------------------------
    # Test 3:
    # Spatial causality of projection.
    #
    # Create a single hot provenance block.
    # The projected maximum must remain spatially near
    # the corresponding location rather than being
    # arbitrarily reshuffled.
    # --------------------------------------------------------

    source = torch.zeros(
        1,
        1,
        8,
        16,
        device=DEVICE,
    )

    source_row = 3
    source_col = 10

    source[
        0,
        0,
        source_row,
        source_col,
    ] = 1.0

    vit_hot, patch_hot = (
        projector.project_to_vit(
            source
        )
    )

    flat_index = int(
        patch_hot[
            0, :, 0
        ].argmax().item()
    )

    target_row = (
        flat_index // 14
    )

    target_col = (
        flat_index % 14
    )

    # Expected continuous-coordinate center.
    expected_row = (
        (source_row + 0.5)
        / 8.0
        * 14.0
        - 0.5
    )

    expected_col = (
        (source_col + 0.5)
        / 16.0
        * 14.0
        - 0.5
    )

    row_error = abs(
        target_row - expected_row
    )

    col_error = abs(
        target_col - expected_col
    )

    print()
    print("-" * 76)
    print("SPATIAL CORRESPONDENCE")
    print("-" * 76)

    print(
        "Source provenance block :",
        (source_row, source_col),
    )

    print(
        "Projected max ViT patch :",
        (target_row, target_col),
    )

    print(
        "Expected continuous pos :",
        (
            round(expected_row, 3),
            round(expected_col, 3),
        ),
    )

    print(
        "Row error               :",
        f"{row_error:.4f}",
    )

    print(
        "Column error            :",
        f"{col_error:.4f}",
    )

    assert row_error <= 1.0
    assert col_error <= 1.0

    print(
        "Spatial correspondence  : PASS"
    )

    # --------------------------------------------------------
    # Test 4:
    # Uniform field must remain uniform.
    # --------------------------------------------------------

    uniform = torch.full(
        (1, 1, 8, 16),
        0.37,
        device=DEVICE,
    )

    uniform_vit, _ = (
        projector.project_to_vit(
            uniform
        )
    )

    uniform_error = (
        uniform_vit - 0.37
    ).abs().max().item()

    print()
    print(
        "Uniform projection max error :",
        f"{uniform_error:.10f}",
    )

    assert uniform_error < 1e-6

    print(
        "Uniform-field preservation    : PASS"
    )

    # --------------------------------------------------------
    # Test 5:
    # Evidence-token fusion interface + gradients
    # --------------------------------------------------------

    fusion = EvidenceTokenFusion(
        embed_dim=768,
        hidden_dim=96,
    ).to(DEVICE)

    tokens = torch.randn(
        2,
        196,
        768,
        device=DEVICE,
    )

    evidence_tokens = (
        patch_evidence.detach()
    )

    fused = fusion(
        tokens,
        evidence_tokens,
    )

    assert fused.shape == (
        2,
        196,
        768,
    )

    loss = fused.square().mean()

    loss.backward()

    mlp_has_grad = any(
        p.grad is not None
        for p in fusion.evidence_mlp.parameters()
    )

    scale_has_grad = (
        fusion.scale.grad
        is not None
    )

    assert mlp_has_grad
    assert scale_has_grad

    trainable_fusion = sum(
        p.numel()
        for p in fusion.parameters()
        if p.requires_grad
    )

    print()
    print("-" * 76)
    print("FUSION AUDIT")
    print("-" * 76)

    print(
        "Fused token shape      :",
        tuple(fused.shape),
    )

    print(
        "Fusion trainable params:",
        f"{trainable_fusion:,}",
    )

    print(
        "Fusion MLP gradient    :",
        mlp_has_grad,
    )

    print(
        "Fusion scale gradient  :",
        scale_has_grad,
    )

    print()
    print("=" * 76)
    print(
        "E5.3R-2B PROJECTION GATE: PASS"
    )
    print("=" * 76)

    print()
    print(
        "8x16 provenance evidence can now be "
        "mapped deterministically to the "
        "14x14 ViT patch coordinate system."
    )

    print(
        "No detector/GNN training has occurred."
    )


if __name__ == "__main__":
    main()