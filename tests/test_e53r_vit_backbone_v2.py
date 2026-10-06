from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.vit_backbone_v2 import (
    ProvenanceViTBackbone,
    count_parameters,
    MODEL_NAME,
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main():

    print("=" * 76)
    print(
        "E5.3R-2A PRETRAINED ViT BACKBONE PRECHECK"
    )
    print("=" * 76)

    print("Device :", DEVICE)
    print("Model  :", MODEL_NAME)
    print()

    print(
        "Loading pretrained weights..."
    )

    model = ProvenanceViTBackbone(
        adapter_bottleneck=96,
        pretrained=True,
    ).to(DEVICE)

    model.eval()

    total, trainable = count_parameters(
        model
    )

    vit_total = sum(
        p.numel()
        for p in model.vit.parameters()
    )

    vit_trainable = sum(
        p.numel()
        for p in model.vit.parameters()
        if p.requires_grad
    )

    adapter_trainable = sum(
        p.numel()
        for p in model.adapter.parameters()
        if p.requires_grad
    )

    scale_trainable = (
        model.adapter_scale.numel()
        if model.adapter_scale.requires_grad
        else 0
    )

    print()
    print("-" * 76)
    print("PARAMETER AUDIT")
    print("-" * 76)

    print(
        "Total parameters           :",
        f"{total:,}",
    )

    print(
        "Total trainable parameters :",
        f"{trainable:,}",
    )

    print(
        "ViT parameters             :",
        f"{vit_total:,}",
    )

    print(
        "ViT trainable              :",
        f"{vit_trainable:,}",
    )

    print(
        "Adapter trainable          :",
        f"{adapter_trainable:,}",
    )

    print(
        "Scale trainable            :",
        scale_trainable,
    )

    assert vit_trainable == 0

    assert trainable == (
        adapter_trainable
        + scale_trainable
    )

    print()
    print(
        "Frozen ViT audit           : PASS"
    )

    # --------------------------------------------------------
    # Interface test
    # --------------------------------------------------------

    torch.manual_seed(20260921)

    x = torch.rand(
        2,
        3,
        224,
        224,
        device=DEVICE,
    )

    with torch.no_grad():
        tokens = model(x)

    print()
    print("-" * 76)
    print("TOKEN INTERFACE")
    print("-" * 76)

    print(
        "Input shape                :",
        tuple(x.shape),
    )

    print(
        "Output token shape         :",
        tuple(tokens.shape),
    )

    print(
        "Embedding dimension        :",
        model.embed_dim,
    )

    assert tokens.shape == (
        2,
        196,
        768,
    )

    finite = bool(
        torch.isfinite(tokens)
        .all()
        .item()
    )

    assert finite

    print(
        "Token shape                : PASS"
    )

    print(
        "Finite-value check         : PASS"
    )

    # --------------------------------------------------------
    # Verify adapter receives gradients while ViT does not.
    # --------------------------------------------------------

    model.train()

    x = torch.rand(
        1,
        3,
        224,
        224,
        device=DEVICE,
    )

    tokens = model(x)

    loss = (
        tokens.square().mean()
    )

    loss.backward()

    vit_has_grad = any(
        p.grad is not None
        for p in model.vit.parameters()
    )

    adapter_has_grad = any(
        p.grad is not None
        for p in model.adapter.parameters()
    )

    scale_has_grad = (
        model.adapter_scale.grad
        is not None
    )

    print()
    print("-" * 76)
    print("GRADIENT AUDIT")
    print("-" * 76)

    print(
        "ViT receives gradient      :",
        vit_has_grad,
    )

    print(
        "Adapter receives gradient  :",
        adapter_has_grad,
    )

    print(
        "Scale receives gradient    :",
        scale_has_grad,
    )

    assert not vit_has_grad
    assert adapter_has_grad
    assert scale_has_grad

    print()
    print("=" * 76)
    print(
        "E5.3R-2A BACKBONE GATE: PASS"
    )
    print("=" * 76)

    print()
    print(
        "Pretrained ViT-B/16 is loaded "
        "and frozen."
    )

    print(
        "Only the lightweight token adapter "
        "is trainable."
    )

    print(
        "No detector or localization model "
        "has been trained."
    )


if __name__ == "__main__":
    main()