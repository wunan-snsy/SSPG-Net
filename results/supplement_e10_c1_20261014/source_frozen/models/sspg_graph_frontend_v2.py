import torch
import torch.nn as nn

from models.sspg_frontend_v2 import SSPGFrontendV2
from models.graph_builder_v2 import ProvenanceGraphBuilder
from models.provenance_gnn_v2 import ProvenanceGNNV2


class SSPGGraphFrontendV2(nn.Module):
    """
    Full E5.3R graph frontend.

    received image + known payload
        -> frozen V3.3 decoder
        -> reference-free evidence
        -> frozen pretrained ViT-B/16
        -> trainable adapter
        -> evidence fusion
        -> provenance graph
        -> evidence-aware GNN

    Frozen:
        V3.3 decoder
        pretrained ViT-B/16

    Trainable:
        ViT token adapter
        evidence-token fusion
        provenance GNN

    No detection/localization heads are included here.
    """

    def __init__(
        self,
        stego_checkpoint,
        adapter_bottleneck=96,
        fusion_hidden_dim=96,
        knn_k=8,
        gnn_hidden_dim=256,
        gnn_depth=2,
    ):
        super().__init__()

        self.frontend = SSPGFrontendV2(
            stego_checkpoint=stego_checkpoint,
            adapter_bottleneck=adapter_bottleneck,
            fusion_hidden_dim=fusion_hidden_dim,
        )

        self.graph_builder = ProvenanceGraphBuilder(
            rows=14,
            cols=14,
            knn_k=knn_k,
        )

        self.gnn = ProvenanceGNNV2(
            node_dim=771,
            hidden_dim=gnn_hidden_dim,
            edge_dim=4,
            depth=gnn_depth,
        )

    def train(self, mode=True):

        super().train(mode)

        # SSPGFrontendV2.train() already enforces these,
        # but keep the invariant explicit here too.
        self.frontend.decoder.eval()
        self.frontend.visual.vit.eval()

        return self

    def forward(
        self,
        received,
        payload,
    ):

        front = self.frontend(
            received,
            payload,
        )

        graph = self.graph_builder(
            front["fused_tokens"],
            front["patch_evidence"],
        )

        graph_tokens = self.gnn(
            graph["node_features"],
            graph["edge_index"],
            graph["edge_attr"],
        )

        return {
            **front,

            "node_features":
                graph["node_features"],

            "edge_index":
                graph["edge_index"],

            "edge_attr":
                graph["edge_attr"],

            "coordinates":
                graph["coordinates"],

            "graph_tokens":
                graph_tokens,
        }