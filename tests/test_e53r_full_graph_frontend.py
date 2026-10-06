from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.sspg_graph_frontend_v2 import (
    SSPGGraphFrontendV2,
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


def has_grad(module):

    return any(
        p.grad is not None
        for p in module.parameters()
        if p.requires_grad
    )


def has_any_grad(module):

    return any(
        p.grad is not None
        for p in module.parameters()
    )


def main():

    print("=" * 78)
    print(
        "E5.3R-3C FULL SSPG GRAPH FRONTEND "
        "INTEGRATION / GRADIENT GATE"
    )
    print("=" * 78)

    print("Device     :", DEVICE)
    print("Checkpoint :", CHECKPOINT)
    print()

    torch.manual_seed(20260921)

    model = SSPGGraphFrontendV2(
        stego_checkpoint=CHECKPOINT,
        adapter_bottleneck=96,
        fusion_hidden_dim=96,
        knn_k=8,
        gnn_hidden_dim=256,
        gnn_depth=2,
    ).to(DEVICE)

    # --------------------------------------------------------
    # Parameter audit
    # --------------------------------------------------------

    decoder_trainable = count_trainable(
        model.frontend.decoder
    )

    vit_trainable = count_trainable(
        model.frontend.visual.vit
    )

    adapter_trainable = count_trainable(
        model.frontend.visual.adapter
    )

    adapter_scale_trainable = (
        model.frontend.visual.adapter_scale.numel()
        if model.frontend.visual.adapter_scale.requires_grad
        else 0
    )

    fusion_trainable = count_trainable(
        model.frontend.fusion
    )

    projector_trainable = count_trainable(
        model.frontend.projector
    )

    builder_trainable = count_trainable(
        model.graph_builder
    )

    gnn_trainable = count_trainable(
        model.gnn
    )

    total_trainable = count_trainable(
        model
    )

    expected_trainable = (
        adapter_trainable
        + adapter_scale_trainable
        + fusion_trainable
        + gnn_trainable
    )

    print("-" * 78)
    print("PARAMETER AUDIT")
    print("-" * 78)

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
        f"{adapter_trainable:,}",
    )

    print(
        "Adapter scale           :",
        adapter_scale_trainable,
    )

    print(
        "Fusion trainable        :",
        f"{fusion_trainable:,}",
    )

    print(
        "Projector trainable     :",
        projector_trainable,
    )

    print(
        "Graph builder trainable :",
        builder_trainable,
    )

    print(
        "GNN trainable           :",
        f"{gnn_trainable:,}",
    )

    print(
        "TOTAL trainable         :",
        f"{total_trainable:,}",
    )

    assert decoder_trainable == 0
    assert vit_trainable == 0
    assert projector_trainable == 0
    assert builder_trainable == 0

    assert total_trainable == expected_trainable

    print()
    print(
        "Parameter isolation     : PASS"
    )

    # --------------------------------------------------------
    # Mode audit
    # --------------------------------------------------------

    model.train()

    assert model.training
    assert not model.frontend.decoder.training
    assert not model.frontend.visual.vit.training

    print()
    print("-" * 78)
    print("MODE AUDIT")
    print("-" * 78)

    print(
        "Whole model training    :",
        model.training,
    )

    print(
        "Decoder training        :",
        model.frontend.decoder.training,
    )

    print(
        "ViT training            :",
        model.frontend.visual.vit.training,
    )

    print(
        "Frozen mode isolation   : PASS"
    )

    # --------------------------------------------------------
    # Protected synthetic interface input.
    # This is NOT an experiment result.
    # --------------------------------------------------------

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

    output = model(
        protected,
        payload,
    )

    print()
    print("-" * 78)
    print("FULL FORWARD AUDIT")
    print("-" * 78)

    tensor_shapes = (
        "decoder_prob",
        "evidence_grid",
        "vit_evidence_grid",
        "patch_evidence",
        "visual_tokens",
        "fused_tokens",
        "node_features",
        "graph_tokens",
    )

    for key in tensor_shapes:

        print(
            f"{key:20s}:",
            tuple(output[key].shape),
        )

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

    assert output[
        "node_features"
    ].shape == (1, 196, 771)

    assert output[
        "graph_tokens"
    ].shape == (1, 196, 256)

    assert len(
        output["edge_index"]
    ) == 1

    assert len(
        output["edge_attr"]
    ) == 1

    edge_count = output[
        "edge_index"
    ][0].shape[1]

    print(
        "Directed graph edges    :",
        edge_count,
    )

    for key in tensor_shapes:

        assert torch.isfinite(
            output[key]
        ).all()

    assert torch.isfinite(
        output["edge_attr"][0]
    ).all()

    print(
        "Full forward audit      : PASS"
    )

    # --------------------------------------------------------
    # Gradient isolation.
    #
    # A dummy downstream loss is placed ONLY on graph_tokens.
    # --------------------------------------------------------

    model.zero_grad(
        set_to_none=True
    )

    loss = (
        output["graph_tokens"]
        .square()
        .mean()
    )

    loss.backward()

    decoder_grad = has_any_grad(
        model.frontend.decoder
    )

    vit_grad = has_any_grad(
        model.frontend.visual.vit
    )

    adapter_grad = has_grad(
        model.frontend.visual.adapter
    )

    adapter_scale_grad = (
        model.frontend.visual.adapter_scale.grad
        is not None
    )

    fusion_grad = has_grad(
        model.frontend.fusion
    )

    gnn_grad = has_grad(
        model.gnn
    )

    print()
    print("-" * 78)
    print("FULL GRADIENT AUDIT")
    print("-" * 78)

    print(
        "Decoder receives grad   :",
        decoder_grad,
    )

    print(
        "ViT receives grad       :",
        vit_grad,
    )

    print(
        "Adapter receives grad   :",
        adapter_grad,
    )

    print(
        "Adapter scale gets grad :",
        adapter_scale_grad,
    )

    print(
        "Fusion receives grad    :",
        fusion_grad,
    )

    print(
        "GNN receives grad       :",
        gnn_grad,
    )

    assert not decoder_grad
    assert not vit_grad

    assert adapter_grad
    assert adapter_scale_grad
    assert fusion_grad
    assert gnn_grad

    print()
    print(
        "End-to-end gradient flow: PASS"
    )

    # --------------------------------------------------------
    # Provenance dependency through the WHOLE graph chain.
    #
    # Keep the received image identical.
    # Change only known payload.
    #
    # Decoder must remain identical.
    # Evidence and final graph representation must change.
    # --------------------------------------------------------

    model.eval()

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

    graph_difference = (
        correct["graph_tokens"]
        - wrong["graph_tokens"]
    ).abs().mean().item()

    print()
    print("-" * 78)
    print("FULL PROVENANCE DEPENDENCY")
    print("-" * 78)

    print(
        "Decoder difference      :",
        f"{decoder_difference:.12f}",
    )

    print(
        "Evidence difference     :",
        f"{evidence_difference:.8f}",
    )

    print(
        "Graph-token difference  :",
        f"{graph_difference:.8f}",
    )

    assert decoder_difference == 0.0
    assert evidence_difference > 0.0
    assert graph_difference > 0.0

    print()
    print(
        "Whole-chain provenance dependency: PASS"
    )

    print()
    print("=" * 78)

    print(
        "E5.3R-3C FULL GRAPH FRONTEND GATE: PASS"
    )

    print("=" * 78)

    print()
    print(
        "The complete SSPG representation path is "
        "operational:"
    )

    print(
        "V3.3 decoder -> reference-free evidence -> "
        "ViT/adaptation -> evidence fusion -> "
        "provenance graph -> evidence-aware GNN."
    )

    print()
    print(
        "No detector/localization head has been "
        "trained and no Test split has been accessed."
    )


if __name__ == "__main__":
    main()