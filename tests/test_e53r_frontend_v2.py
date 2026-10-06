from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.sspg_frontend_v2 import (
    SSPGFrontendV2,
)

from training.train_stego_structured_v33 import (
    embed_payload,
    make_payload,
)


CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "stego_structured_v33_128bit_best.pth"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def count_trainable(module):

    return sum(
        p.numel()
        for p in module.parameters()
        if p.requires_grad
    )


def has_gradient(module):

    return any(
        p.grad is not None
        for p in module.parameters()
    )


def main():

    print("=" * 78)
    print(
        "E5.3R-2C END-TO-END FRONTEND / "
        "GRADIENT ISOLATION AUDIT"
    )
    print("=" * 78)

    print("Device     :", DEVICE)
    print("Checkpoint :", CHECKPOINT)
    print()

    model = SSPGFrontendV2(
        stego_checkpoint=CHECKPOINT,
        adapter_bottleneck=96,
        fusion_hidden_dim=96,
    ).to(DEVICE)

    # --------------------------------------------------------
    # Parameter audit
    # --------------------------------------------------------

    decoder_trainable = count_trainable(
        model.decoder
    )

    vit_trainable = count_trainable(
        model.visual.vit
    )

    adapter_trainable = count_trainable(
        model.visual.adapter
    )

    adapter_scale_trainable = (
        model.visual.adapter_scale.numel()
        if model.visual.adapter_scale.requires_grad
        else 0
    )

    fusion_trainable = count_trainable(
        model.fusion
    )

    projector_trainable = count_trainable(
        model.projector
    )

    total_trainable = count_trainable(
        model
    )

    print("-" * 78)
    print("PARAMETER AUDIT")
    print("-" * 78)

    print(
        "Decoder trainable        :",
        decoder_trainable,
    )

    print(
        "ViT trainable            :",
        vit_trainable,
    )

    print(
        "Adapter trainable        :",
        f"{adapter_trainable:,}",
    )

    print(
        "Adapter scale trainable  :",
        adapter_scale_trainable,
    )

    print(
        "Fusion trainable         :",
        f"{fusion_trainable:,}",
    )

    print(
        "Projector trainable      :",
        projector_trainable,
    )

    print(
        "TOTAL trainable          :",
        f"{total_trainable:,}",
    )

    assert decoder_trainable == 0
    assert vit_trainable == 0
    assert projector_trainable == 0

    expected_trainable = (
        adapter_trainable
        + adapter_scale_trainable
        + fusion_trainable
    )

    assert (
        total_trainable
        == expected_trainable
    )

    print()
    print(
        "Parameter isolation      : PASS"
    )

    # --------------------------------------------------------
    # Training-mode isolation
    # --------------------------------------------------------

    model.train()

    assert model.training
    assert not model.decoder.training
    assert not model.visual.vit.training

    print()
    print("-" * 78)
    print("MODE AUDIT")
    print("-" * 78)

    print(
        "Frontend training mode   :",
        model.training,
    )

    print(
        "Decoder training mode    :",
        model.decoder.training,
    )

    print(
        "ViT training mode        :",
        model.visual.vit.training,
    )

    print(
        "Frozen-module mode audit : PASS"
    )

    # --------------------------------------------------------
    # Real V3.3 protected input
    #
    # This is synthetic random image data for interface
    # verification only. It is NOT an experiment result.
    # --------------------------------------------------------

    torch.manual_seed(20260921)

    original = torch.rand(
        1,
        3,
        256,
        256,
        device=DEVICE,
    )

    payload = make_payload(
        1,
        DEVICE,
    )

    protected = embed_payload(
        original,
        payload,
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    output = model(
        protected,
        payload,
    )

    print()
    print("-" * 78)
    print("FORWARD INTERFACE")
    print("-" * 78)

    for key in (
        "decoder_logits",
        "decoder_prob",
        "evidence_grid",
        "vit_evidence_grid",
        "patch_evidence",
        "visual_tokens",
        "fused_tokens",
    ):

        print(
            f"{key:20s}:",
            tuple(output[key].shape),
        )

    assert output[
        "decoder_logits"
    ].shape == (1, 128)

    assert output[
        "decoder_prob"
    ].shape == (1, 128)

    assert output[
        "evidence_grid"
    ].shape == (1, 1, 8, 16)

    assert output[
        "vit_evidence_grid"
    ].shape == (1, 1, 14, 14)

    assert output[
        "patch_evidence"
    ].shape == (1, 196, 1)

    assert output[
        "visual_tokens"
    ].shape == (1, 196, 768)

    assert output[
        "fused_tokens"
    ].shape == (1, 196, 768)

    for value in output.values():

        assert torch.isfinite(
            value
        ).all()

    print()
    print(
        "Shape/finite audit       : PASS"
    )

    # --------------------------------------------------------
    # Gradient isolation
    #
    # Simulate a downstream graph/detection loss.
    # --------------------------------------------------------

    model.zero_grad(
        set_to_none=True
    )

    loss = (
        output[
            "fused_tokens"
        ].square().mean()
    )

    loss.backward()

    decoder_grad = has_gradient(
        model.decoder
    )

    vit_grad = has_gradient(
        model.visual.vit
    )

    adapter_grad = has_gradient(
        model.visual.adapter
    )

    adapter_scale_grad = (
        model.visual.adapter_scale.grad
        is not None
    )

    fusion_grad = has_gradient(
        model.fusion
    )

    print()
    print("-" * 78)
    print("GRADIENT AUDIT")
    print("-" * 78)

    print(
        "Decoder receives grad    :",
        decoder_grad,
    )

    print(
        "ViT receives grad        :",
        vit_grad,
    )

    print(
        "Adapter receives grad    :",
        adapter_grad,
    )

    print(
        "Adapter scale gets grad  :",
        adapter_scale_grad,
    )

    print(
        "Fusion receives grad     :",
        fusion_grad,
    )

    assert not decoder_grad
    assert not vit_grad

    assert adapter_grad
    assert adapter_scale_grad
    assert fusion_grad

    print()
    print(
        "Gradient isolation       : PASS"
    )

    # --------------------------------------------------------
    # Evidence dependency audit
    #
    # Change ONLY the known payload while keeping the
    # received image fixed.
    #
    # Decoder output must remain identical.
    # Evidence/fused tokens must change.
    # --------------------------------------------------------

    wrong_payload = (
        1.0 - payload
    )

    with torch.no_grad():

        correct = model(
            protected,
            payload,
        )

        wrong = model(
            protected,
            wrong_payload,
        )

    decoder_difference = (
        correct["decoder_prob"]
        - wrong["decoder_prob"]
    ).abs().max().item()

    evidence_difference = (
        correct["patch_evidence"]
        - wrong["patch_evidence"]
    ).abs().mean().item()

    fused_difference = (
        correct["fused_tokens"]
        - wrong["fused_tokens"]
    ).abs().mean().item()

    print()
    print("-" * 78)
    print("EVIDENCE DEPENDENCY AUDIT")
    print("-" * 78)

    print(
        "Decoder difference       :",
        f"{decoder_difference:.12f}",
    )

    print(
        "Evidence difference      :",
        f"{evidence_difference:.8f}",
    )

    print(
        "Fused-token difference   :",
        f"{fused_difference:.8f}",
    )

    assert decoder_difference == 0.0

    assert evidence_difference > 0.0

    assert fused_difference > 0.0

    print()
    print(
        "Payload/evidence dependency: PASS"
    )

    print()
    print("=" * 78)

    print(
        "E5.3R-2C FRONTEND GATE: PASS"
    )

    print("=" * 78)

    print()
    print(
        "Frozen V3.3 decoder -> reference-free "
        "evidence -> spatial projection -> "
        "frozen pretrained ViT -> trainable "
        "adapter/fusion is operational."
    )

    print()
    print(
        "No detector, localization head, or GNN "
        "has been trained."
    )


if __name__ == "__main__":
    main()