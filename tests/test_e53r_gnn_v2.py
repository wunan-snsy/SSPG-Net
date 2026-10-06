from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.graph_builder_v2 import (
    ProvenanceGraphBuilder,
)

from models.provenance_gnn_v2 import (
    ProvenanceGNNV2,
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def grad_exists(module):

    return any(
        p.grad is not None
        for p in module.parameters()
        if p.requires_grad
    )


def main():

    print("=" * 78)
    print(
        "E5.3R-3B EVIDENCE-AWARE GNN "
        "MESSAGE-PASSING GATE"
    )
    print("=" * 78)

    print("Device     :", DEVICE)
    print("Nodes      : 196")
    print("Node dim   : 771")
    print("Edge dim   : 4")
    print("Hidden dim : 256")
    print("GNN depth  : 2")
    print()

    torch.manual_seed(20260921)

    builder = (
        ProvenanceGraphBuilder(
            rows=14,
            cols=14,
            knn_k=8,
        )
        .to(DEVICE)
    )

    gnn = (
        ProvenanceGNNV2(
            node_dim=771,
            hidden_dim=256,
            edge_dim=4,
            depth=2,
        )
        .to(DEVICE)
    )

    builder_params = sum(
        p.numel()
        for p in builder.parameters()
    )

    gnn_trainable = sum(
        p.numel()
        for p in gnn.parameters()
        if p.requires_grad
    )

    print(
        "Graph builder trainable :",
        builder_params,
    )

    print(
        "GNN trainable           :",
        f"{gnn_trainable:,}",
    )

    assert builder_params == 0
    assert gnn_trainable > 0

    # --------------------------------------------------------
    # Synthetic interface data only.
    # No dataset evaluation.
    # --------------------------------------------------------

    tokens = torch.randn(
        2,
        196,
        768,
        device=DEVICE,
        requires_grad=True,
    )

    evidence = torch.rand(
        2,
        196,
        1,
        device=DEVICE,
        requires_grad=True,
    )

    graph = builder(
        tokens,
        evidence,
    )

    output = gnn(
        graph["node_features"],
        graph["edge_index"],
        graph["edge_attr"],
    )

    print()
    print("-" * 78)
    print("FORWARD AUDIT")
    print("-" * 78)

    print(
        "Input node shape  :",
        tuple(
            graph[
                "node_features"
            ].shape
        ),
    )

    print(
        "Output node shape :",
        tuple(output.shape),
    )

    assert output.shape == (
        2,
        196,
        256,
    )

    assert torch.isfinite(
        output
    ).all()

    print(
        "Forward shape     : PASS"
    )

    # --------------------------------------------------------
    # Gradient audit
    # --------------------------------------------------------

    loss = output.square().mean()

    loss.backward()

    token_grad = (
        tokens.grad is not None
    )

    evidence_grad = (
        evidence.grad is not None
    )

    gnn_grad = grad_exists(
        gnn
    )

    print()
    print("-" * 78)
    print("GRADIENT AUDIT")
    print("-" * 78)

    print(
        "Token gradient    :",
        token_grad,
    )

    print(
        "Evidence gradient :",
        evidence_grad,
    )

    print(
        "GNN gradient      :",
        gnn_grad,
    )

    assert token_grad
    assert evidence_grad
    assert gnn_grad

    assert torch.isfinite(
        tokens.grad
    ).all()

    assert torch.isfinite(
        evidence.grad
    ).all()

    print(
        "Gradient flow     : PASS"
    )

    # --------------------------------------------------------
    # Edge-attribute dependency
    #
    # Keep:
    #   node features identical
    #   topology identical
    #
    # Change ONLY edge attributes.
    #
    # GNN output must change.
    # --------------------------------------------------------

    gnn.eval()

    with torch.no_grad():

        baseline = gnn(
            graph["node_features"],
            graph["edge_index"],
            graph["edge_attr"],
        )

        zero_edge_attr = [
            torch.zeros_like(x)
            for x in graph["edge_attr"]
        ]

        no_edge_information = gnn(
            graph["node_features"],
            graph["edge_index"],
            zero_edge_attr,
        )

    edge_dependency = (
        baseline
        - no_edge_information
    ).abs().mean().item()

    print()
    print("-" * 78)
    print("EDGE-ATTRIBUTE DEPENDENCY")
    print("-" * 78)

    print(
        "Mean output difference :",
        f"{edge_dependency:.10f}",
    )

    assert edge_dependency > 0.0

    print(
        "Edge attributes affect GNN : PASS"
    )

    # --------------------------------------------------------
    # Evidence-specific edge dependency
    #
    # Preserve:
    #   spatial distance
    #   cosine similarity
    #
    # Remove ONLY:
    #   source evidence
    #   target evidence
    # --------------------------------------------------------

    evidence_removed_attrs = []

    for attr in graph[
        "edge_attr"
    ]:

        modified = attr.clone()

        modified[:, 2] = 0.0
        modified[:, 3] = 0.0

        evidence_removed_attrs.append(
            modified
        )

    with torch.no_grad():

        no_edge_evidence = gnn(
            graph["node_features"],
            graph["edge_index"],
            evidence_removed_attrs,
        )

    evidence_edge_dependency = (
        baseline
        - no_edge_evidence
    ).abs().mean().item()

    print()
    print(
        "Evidence-edge difference :",
        f"{evidence_edge_dependency:.10f}",
    )

    assert (
        evidence_edge_dependency
        > 0.0
    )

    print(
        "Evidence edge attrs used : PASS"
    )

    # --------------------------------------------------------
    # Topology dependency
    #
    # Keep node features and edge attributes from
    # a deterministic subset of the graph.
    #
    # Remove half the edges.
    # Output must change.
    # --------------------------------------------------------

    half_edges = []
    half_attrs = []

    for edges, attrs in zip(
        graph["edge_index"],
        graph["edge_attr"],
    ):

        keep = torch.arange(
            0,
            edges.shape[1],
            2,
            device=DEVICE,
        )

        half_edges.append(
            edges[:, keep]
        )

        half_attrs.append(
            attrs[keep]
        )

    with torch.no_grad():

        reduced_graph_output = gnn(
            graph["node_features"],
            half_edges,
            half_attrs,
        )

    topology_dependency = (
        baseline
        - reduced_graph_output
    ).abs().mean().item()

    print()
    print("-" * 78)
    print("TOPOLOGY DEPENDENCY")
    print("-" * 78)

    print(
        "Mean output difference :",
        f"{topology_dependency:.10f}",
    )

    assert topology_dependency > 0.0

    print(
        "Graph topology affects GNN : PASS"
    )

    print()
    print("=" * 78)

    print(
        "E5.3R-3B GNN GATE: PASS"
    )

    print("=" * 78)

    print()
    print(
        "Evidence-aware graph message passing "
        "is operational."
    )

    print(
        "No dataset training, detector head, "
        "or localization head has been used."
    )


if __name__ == "__main__":
    main()