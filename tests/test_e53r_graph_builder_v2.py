from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.graph_builder_v2 import (
    ProvenanceGraphBuilder,
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


def main():

    print("=" * 78)
    print(
        "E5.3R-3A PROVENANCE GRAPH "
        "CONSTRUCTION GATE"
    )
    print("=" * 78)

    print("Device :", DEVICE)
    print("Grid   : 14 x 14")
    print("Nodes  : 196")
    print("kNN k  : 8")
    print()

    builder = ProvenanceGraphBuilder(
        rows=14,
        cols=14,
        knn_k=8,
    ).to(DEVICE)

    params = sum(
        p.numel()
        for p in builder.parameters()
    )

    print(
        "Graph-builder trainable parameters :",
        params,
    )

    assert params == 0

    torch.manual_seed(20260921)

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
    )

    output = builder(
        tokens,
        evidence,
    )

    nodes = output["node_features"]

    print()
    print("-" * 78)
    print("NODE AUDIT")
    print("-" * 78)

    print(
        "Node feature shape :",
        tuple(nodes.shape),
    )

    assert nodes.shape == (
        2,
        196,
        771,
    )

    print(
        "Node feature audit : PASS"
    )

    print()
    print("-" * 78)
    print("EDGE AUDIT")
    print("-" * 78)

    for b in range(2):

        edges = output[
            "edge_index"
        ][b]

        attrs = output[
            "edge_attr"
        ][b]

        assert edges.ndim == 2
        assert edges.shape[0] == 2

        assert attrs.shape == (
            edges.shape[1],
            4,
        )

        assert (
            edges.min() >= 0
        )

        assert (
            edges.max() < 196
        )

        self_edges = (
            edges[0] == edges[1]
        ).sum().item()

        assert self_edges == 0

        unique_edges = torch.unique(
            edges.t(),
            dim=0,
        )

        assert (
            unique_edges.shape[0]
            == edges.shape[1]
        )

        assert torch.isfinite(
            attrs
        ).all()

        print(
            f"Graph {b}: "
            f"{edges.shape[1]} directed edges"
        )

        print(
            "  self edges       :",
            self_edges,
        )

        print(
            "  edge attr shape  :",
            tuple(attrs.shape),
        )

    print(
        "Edge validity audit : PASS"
    )

    # --------------------------------------------------------
    # Spatial graph exact count
    # --------------------------------------------------------

    spatial_count = (
        builder.spatial_edge_index
        .shape[1]
    )

    print()
    print(
        "Spatial directed edges :",
        spatial_count,
    )

    # 14x14 8-neighbour directed grid:
    # horizontal = 14*13*2
    # vertical   = 14*13*2
    # diagonals  = 13*13*4
    expected_spatial = (
        14 * 13 * 2
        + 14 * 13 * 2
        + 13 * 13 * 4
    )

    print(
        "Expected spatial edges :",
        expected_spatial,
    )

    assert (
        spatial_count
        == expected_spatial
    )

    print(
        "Spatial topology audit : PASS"
    )

    # --------------------------------------------------------
    # Coordinate audit
    # --------------------------------------------------------

    coords = builder.coordinates

    assert coords.shape == (
        196,
        2,
    )

    assert coords.min() > 0
    assert coords.max() < 1

    print()
    print(
        "Coordinate range        :",
        (
            float(coords.min()),
            float(coords.max()),
        ),
    )

    print(
        "Coordinate audit        : PASS"
    )

    # --------------------------------------------------------
    # kNN topology determinism
    # --------------------------------------------------------

    out2 = builder(
        tokens,
        evidence,
    )

    for b in range(2):

        assert torch.equal(
            output["edge_index"][b],
            out2["edge_index"][b],
        )

    print()
    print(
        "Topology determinism    : PASS"
    )

    # --------------------------------------------------------
    # Gradient audit
    #
    # Node/edge features can participate in downstream
    # optimization, but discrete kNN topology itself is
    # detached.
    # --------------------------------------------------------

    loss = (
        nodes.square().mean()
        + sum(
            x.square().mean()
            for x in output["edge_attr"]
        )
    )

    loss.backward()

    token_grad = (
        tokens.grad is not None
    )

    assert token_grad

    assert torch.isfinite(
        tokens.grad
    ).all()

    print(
        "Feature gradient audit  : PASS"
    )

    print()
    print("=" * 78)
    print(
        "E5.3R-3A GRAPH GATE: PASS"
    )
    print("=" * 78)

    print()
    print(
        "The graph contains 196 ViT patch nodes, "
        "deterministic spatial edges, and "
        "feature-space kNN edges."
    )

    print(
        "No GNN, detector, or localization head "
        "has been trained."
    )


if __name__ == "__main__":
    main()