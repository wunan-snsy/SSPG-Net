import torch
import torch.nn as nn

from models.sspg_graph_frontend_v2 import (
    SSPGGraphFrontendV2,
)

from models.provenance_heads_v2 import (
    JointDetectionLocalizationHeadsV2,
    upsample_localization_logits,
)


# ============================================================
# E5.3R-4B
# Joint SSPG Detection + Localization Model
#
# Existing validated chain:
#
# received image + known payload
#       |
#       v
# SSPGGraphFrontendV2
#       |
#       v
# graph_tokens [B,196,256]
#       |
#       +-------------------------+
#       |                         |
#       v                         v
# Detection Head          Localization Head
#
#
# Frozen by existing frontend:
#     V3.3 decoder
#     pretrained ViT
#
# Trainable:
#     ViT token adapter
#     adapter scale
#     evidence fusion
#     GNN
#     detection head
#     localization head
#
#
# E5.3R-4B-1-R2
#
# IMPORTANT:
#     SSPGGraphFrontendV2 already owns and enforces the
#     frozen-module train/eval invariant.
#
#     This wrapper therefore does NOT reach into the
#     internal decoder/ViT hierarchy in train().
#
#     This avoids duplicating internal implementation
#     assumptions in the joint wrapper.
# ============================================================


class SSPGJointModelV2(nn.Module):

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

        # ----------------------------------------------------
        # Existing validated graph frontend.
        #
        # Real constructor:
        #
        # SSPGGraphFrontendV2(
        #     stego_checkpoint,
        #     adapter_bottleneck=96,
        #     fusion_hidden_dim=96,
        #     knn_k=8,
        #     gnn_hidden_dim=256,
        #     gnn_depth=2,
        # )
        # ----------------------------------------------------

        self.frontend = SSPGGraphFrontendV2(
            stego_checkpoint=stego_checkpoint,
            adapter_bottleneck=adapter_bottleneck,
            fusion_hidden_dim=fusion_hidden_dim,
            knn_k=knn_k,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_depth=gnn_depth,
        )

        # ----------------------------------------------------
        # Joint task heads.
        #
        # Input dimension is exactly the GNN output dimension.
        # ----------------------------------------------------

        self.heads = JointDetectionLocalizationHeadsV2(
            hidden_dim=gnn_hidden_dim,
            detection_mlp_dim=256,
            detection_dropout=0.1,
            grid_size=14,
        )

    def train(
        self,
        mode=True,
    ):
        """
        Set model training mode.

        SSPGGraphFrontendV2.train() already guarantees:

            frontend.decoder.eval()
            frontend.visual.vit.eval()

        Therefore no duplicated internal attribute access
        is performed here.
        """

        super().train(mode)

        # Defensive verification only.
        #
        # Do not modify module state here beyond what the
        # validated frontend already performs.

        if self.frontend.frontend.decoder.training:
            raise RuntimeError(
                "Frozen V3.3 decoder unexpectedly "
                "entered train mode."
            )

        if self.frontend.frontend.visual.vit.training:
            raise RuntimeError(
                "Frozen pretrained ViT unexpectedly "
                "entered train mode."
            )

        return self

    def forward(
        self,
        received,
        payload,
    ):
        # ----------------------------------------------------
        # Existing graph frontend.
        # ----------------------------------------------------

        graph_output = self.frontend(
            received,
            payload,
        )

        if "graph_tokens" not in graph_output:
            raise RuntimeError(
                "SSPGGraphFrontendV2 did not return "
                "'graph_tokens'."
            )

        graph_tokens = graph_output[
            "graph_tokens"
        ]

        # ----------------------------------------------------
        # Structural checks.
        # ----------------------------------------------------

        if graph_tokens.ndim != 3:
            raise RuntimeError(
                "Expected graph_tokens [B,N,C], "
                f"got {tuple(graph_tokens.shape)}."
            )

        if graph_tokens.shape[1] != 196:
            raise RuntimeError(
                "Expected 196 graph nodes for the "
                "14x14 ViT patch grid, "
                f"got {graph_tokens.shape[1]}."
            )

        expected_hidden_dim = (
            self.heads.detection.hidden_dim
        )

        if (
            graph_tokens.shape[2]
            != expected_hidden_dim
        ):
            raise RuntimeError(
                "GNN/head hidden dimension mismatch: "
                f"GNN returned {graph_tokens.shape[2]}, "
                f"head expects {expected_hidden_dim}."
            )

        # ----------------------------------------------------
        # Detection + localization.
        # ----------------------------------------------------

        head_output = self.heads(
            graph_tokens
        )

        # Preserve all graph frontend outputs.
        output = dict(
            graph_output
        )

        # Add task outputs.
        output.update(
            head_output
        )

        return output

    @staticmethod
    def localization_logits_256(
        localization_grid_logits,
    ):
        """
        Deterministically project localization logits
        from 14x14 to 256x256.

        No sigmoid.
        No threshold selection.
        """

        return upsample_localization_logits(
            localization_grid_logits,
            output_size=256,
        )