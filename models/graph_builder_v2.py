import torch
import torch.nn as nn
import torch.nn.functional as F


class ProvenanceGraphBuilder(nn.Module):
    """
    E5.3R provenance graph builder.

    Nodes:
        14 x 14 ViT patches = 196 nodes

    Edges:
        1. deterministic 8-neighbour spatial edges
        2. cosine-similarity kNN edges

    Graph construction itself has no trainable parameters.

    IMPORTANT:
        kNN topology is constructed from detached token
        representations. Graph topology therefore does not
        participate in gradient optimization.
    """

    def __init__(
        self,
        rows=14,
        cols=14,
        knn_k=8,
    ):
        super().__init__()

        self.rows = rows
        self.cols = cols
        self.num_nodes = rows * cols
        self.knn_k = knn_k

        coords = []

        for r in range(rows):
            for c in range(cols):

                y = (
                    (r + 0.5) / rows
                )

                x = (
                    (c + 0.5) / cols
                )

                coords.append(
                    [x, y]
                )

        coords = torch.tensor(
            coords,
            dtype=torch.float32,
        )

        self.register_buffer(
            "coordinates",
            coords,
        )

        spatial_edges = []

        offsets = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )

        for r in range(rows):
            for c in range(cols):

                source = (
                    r * cols + c
                )

                for dr, dc in offsets:

                    rr = r + dr
                    cc = c + dc

                    if (
                        0 <= rr < rows
                        and 0 <= cc < cols
                    ):

                        target = (
                            rr * cols + cc
                        )

                        spatial_edges.append(
                            [source, target]
                        )

        spatial_edges = torch.tensor(
            spatial_edges,
            dtype=torch.long,
        ).t().contiguous()

        self.register_buffer(
            "spatial_edge_index",
            spatial_edges,
        )

    def _knn_edges(
        self,
        tokens,
    ):
        """
        tokens:
            [B, N, C]

        returns:
            list of B edge_index tensors,
            each [2, N*k]
        """

        if tokens.ndim != 3:
            raise RuntimeError(
                "tokens must be [B,N,C]"
            )

        B, N, _ = tokens.shape

        if N != self.num_nodes:
            raise RuntimeError(
                f"Expected {self.num_nodes} nodes, "
                f"got {N}"
            )

        if self.knn_k >= N:
            raise RuntimeError(
                "knn_k must be smaller than "
                "number of nodes."
            )

        # Topology is intentionally non-differentiable.
        detached = tokens.detach()

        normalized = F.normalize(
            detached,
            p=2,
            dim=-1,
        )

        similarity = torch.matmul(
            normalized,
            normalized.transpose(1, 2),
        )

        # Never select self.
        eye = torch.eye(
            N,
            device=tokens.device,
            dtype=torch.bool,
        ).unsqueeze(0)

        similarity = similarity.masked_fill(
            eye,
            float("-inf"),
        )

        neighbours = torch.topk(
            similarity,
            k=self.knn_k,
            dim=-1,
            largest=True,
        ).indices

        result = []

        sources = torch.arange(
            N,
            device=tokens.device,
        ).view(N, 1).expand(
            N,
            self.knn_k,
        )

        for b in range(B):

            edge_index = torch.stack(
                [
                    sources.reshape(-1),
                    neighbours[b].reshape(-1),
                ],
                dim=0,
            )

            result.append(edge_index)

        return result

    @staticmethod
    def _unique_edges(edge_index):
        """
        Remove duplicate directed edges.
        """

        pairs = edge_index.t()

        pairs = torch.unique(
            pairs,
            dim=0,
        )

        return (
            pairs.t()
            .contiguous()
        )

    def forward(
        self,
        tokens,
        patch_evidence,
    ):
        """
        tokens:
            [B,196,768]

        patch_evidence:
            [B,196,1]

        returns:
            node_features:
                [B,196,771]

            graphs:
                list of edge_index [2,E]

            edge attributes:
                list of [E,4]

        Node feature:
            fused token 768
            evidence 1
            coordinate 2

        Edge attribute:
            spatial distance
            token cosine similarity
            source evidence
            target evidence
        """

        if tokens.shape[:2] != (
            patch_evidence.shape[:2]
        ):
            raise RuntimeError(
                "Token/evidence mismatch."
            )

        B, N, _ = tokens.shape

        if N != self.num_nodes:
            raise RuntimeError(
                "Unexpected node count."
            )

        coords = (
            self.coordinates
            .to(tokens.device)
            .unsqueeze(0)
            .expand(B, -1, -1)
        )

        node_features = torch.cat(
            [
                tokens,
                patch_evidence,
                coords,
            ],
            dim=-1,
        )

        knn_graphs = self._knn_edges(
            tokens
        )

        graphs = []
        edge_attributes = []

        spatial = (
            self.spatial_edge_index
            .to(tokens.device)
        )

        normalized = F.normalize(
            tokens,
            p=2,
            dim=-1,
        )

        for b in range(B):

            combined = torch.cat(
                [
                    spatial,
                    knn_graphs[b],
                ],
                dim=1,
            )

            combined = self._unique_edges(
                combined
            )

            src = combined[0]
            dst = combined[1]

            spatial_distance = (
                coords[b, src]
                - coords[b, dst]
            ).square().sum(
                dim=-1,
                keepdim=True,
            ).sqrt()

            cosine_similarity = (
                normalized[b, src]
                * normalized[b, dst]
            ).sum(
                dim=-1,
                keepdim=True,
            )

            source_evidence = (
                patch_evidence[
                    b, src, :
                ]
            )

            target_evidence = (
                patch_evidence[
                    b, dst, :
                ]
            )

            attributes = torch.cat(
                [
                    spatial_distance,
                    cosine_similarity,
                    source_evidence,
                    target_evidence,
                ],
                dim=-1,
            )

            graphs.append(
                combined
            )

            edge_attributes.append(
                attributes
            )

        return {
            "node_features":
                node_features,

            "edge_index":
                graphs,

            "edge_attr":
                edge_attributes,

            "coordinates":
                coords,
        }