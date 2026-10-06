"""
E5.3R-4D Formal Training-Level Ablation Models
==============================================

Revision:
    E5.3R-4D-0-R4

Purpose
-------
Formal retraining ablations for SSPG-Net V2.

Variants
--------
full
    Visual + provenance evidence + Graph/GNN.

no_provenance
    Visual + Graph/GNN.
    Payload-derived provenance evidence is excluded from the prediction
    path from the beginning of training.

no_graph
    Visual + provenance evidence.
    Graph construction and GNN are excluded from the prediction path
    from the beginning of training.

visual_only
    Visual representation only.
    No provenance evidence and no Graph/GNN contribution.

Important
---------
These are training-level ablations.

They are NOT the inference-time interventions used in E5.3R-4C-2 or
E5.3R-4C-3.

Frozen components:
    V3.3 decoder
    pretrained ViT

R4 cleanup
----------
For formal ablations in which provenance is absent:

    no_provenance
    visual_only

the registered evidence-fusion module is explicitly frozen with
requires_grad=False.

This does not change the forward path.

It prevents inactive provenance parameters from being included in the
formal trainable-parameter count or a future optimizer.

Expected trainable parameter counts
-----------------------------------
full:
    1,416,490

no_provenance:
    1,341,801

no_graph:
    555,812

visual_only:
    481,123

This module contains:
    no dataset loading
    no optimizer
    no checkpoint selection
    no threshold selection
    no Test access
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sspg_frontend_v2 import SSPGFrontendV2
from models.sspg_graph_frontend_v2 import SSPGGraphFrontendV2
from models.provenance_heads_v2 import (
    JointDetectionLocalizationHeadsV2,
    upsample_localization_logits,
)


# ======================================================================
# FORMAL ABLATION REGISTRY
# ======================================================================

VALID_ABLATIONS = (
    "full",
    "no_provenance",
    "no_graph",
    "visual_only",
)


# ======================================================================
# GRAPH-FREE TOKEN PROJECTION
# ======================================================================

class TokenProjection768To256(nn.Module):
    """
    Explicit dimensionality projection for graph-free ablations.

    Input:
        [B, 196, 768]

    Output:
        [B, 196, 256]

    This module performs no message passing.

    It must not be interpreted as a graph or GNN substitute.
    """

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 256,
    ) -> None:

        super().__init__()

        self.input_dim = int(
            input_dim
        )

        self.output_dim = int(
            output_dim
        )

        self.norm = nn.LayerNorm(
            self.input_dim
        )

        self.proj = nn.Linear(
            self.input_dim,
            self.output_dim,
        )

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> torch.Tensor:

        if tokens.ndim != 3:
            raise RuntimeError(
                "TokenProjection768To256 expects [B,N,C]."
            )

        if tokens.shape[-1] != self.input_dim:
            raise RuntimeError(
                "TokenProjection768To256 dimension mismatch: "
                f"expected {self.input_dim}, "
                f"got {tokens.shape[-1]}."
            )

        x = self.norm(
            tokens
        )

        x = self.proj(
            x
        )

        return x


# ======================================================================
# HELPERS
# ======================================================================

def freeze_module(
    module: Optional[nn.Module],
) -> None:
    """
    Set requires_grad=False for every parameter in a module.

    This function does not alter module topology and does not remove the
    module from the model.
    """

    if module is None:
        return

    for parameter in module.parameters():
        parameter.requires_grad = False


# ======================================================================
# FORMAL ABLATION MODEL
# ======================================================================

class SSPGAblationModelV2(nn.Module):

    def __init__(
        self,
        stego_checkpoint: str,
        ablation: str,
        adapter_bottleneck: int = 96,
        fusion_hidden_dim: int = 96,
        knn_k: int = 8,
        gnn_hidden_dim: int = 256,
        gnn_depth: int = 2,
    ) -> None:

        super().__init__()

        ablation = str(
            ablation
        ).strip().lower()

        if ablation not in VALID_ABLATIONS:
            raise ValueError(
                f"Unknown ablation '{ablation}'. "
                f"Expected one of {VALID_ABLATIONS}."
            )

        self.ablation = ablation

        self.adapter_bottleneck = int(
            adapter_bottleneck
        )

        self.fusion_hidden_dim = int(
            fusion_hidden_dim
        )

        self.knn_k = int(
            knn_k
        )

        self.gnn_hidden_dim = int(
            gnn_hidden_dim
        )

        self.gnn_depth = int(
            gnn_depth
        )

        # ==============================================================
        # GRAPH-BASED VARIANTS
        # ==============================================================

        if self.ablation in (
            "full",
            "no_provenance",
        ):

            self.graph_frontend: Optional[
                SSPGGraphFrontendV2
            ] = SSPGGraphFrontendV2(
                stego_checkpoint=stego_checkpoint,
                adapter_bottleneck=self.adapter_bottleneck,
                fusion_hidden_dim=self.fusion_hidden_dim,
                knn_k=self.knn_k,
                gnn_hidden_dim=self.gnn_hidden_dim,
                gnn_depth=self.gnn_depth,
            )

        else:

            self.graph_frontend = None

        # ==============================================================
        # GRAPH-FREE VARIANTS
        # ==============================================================

        if self.ablation in (
            "no_graph",
            "visual_only",
        ):

            self.frontend: Optional[
                SSPGFrontendV2
            ] = SSPGFrontendV2(
                stego_checkpoint=stego_checkpoint,
                adapter_bottleneck=self.adapter_bottleneck,
                fusion_hidden_dim=self.fusion_hidden_dim,
            )

            self.token_projection: Optional[
                TokenProjection768To256
            ] = TokenProjection768To256(
                input_dim=768,
                output_dim=self.gnn_hidden_dim,
            )

        else:

            self.frontend = None
            self.token_projection = None

        # ==============================================================
        # COMMON TASK HEADS
        # ==============================================================

        self.heads = JointDetectionLocalizationHeadsV2(
            hidden_dim=self.gnn_hidden_dim,
            detection_mlp_dim=256,
            detection_dropout=0.1,
            grid_size=14,
        )

        # ==============================================================
        # R4 FORMAL INACTIVE-MODULE FREEZING
        #
        # The provenance fusion module is registered by SSPGFrontendV2.
        #
        # For no_provenance and visual_only, this module is not part of
        # the prediction path and must therefore not remain trainable.
        # ==============================================================

        if self.ablation in (
            "no_provenance",
            "visual_only",
        ):

            freeze_module(
                self._active_frontend().fusion
            )

    # ==================================================================
    # ARCHITECTURE FLAGS
    # ==================================================================

    @property
    def uses_provenance(
        self,
    ) -> bool:

        return self.ablation in (
            "full",
            "no_graph",
        )

    @property
    def uses_graph(
        self,
    ) -> bool:

        return self.ablation in (
            "full",
            "no_provenance",
        )

    @property
    def is_visual_only(
        self,
    ) -> bool:

        return (
            self.ablation
            ==
            "visual_only"
        )

    # ==================================================================
    # FRONTEND ACCESS
    # ==================================================================

    def _active_frontend(
        self,
    ) -> SSPGFrontendV2:

        if self.graph_frontend is not None:

            return (
                self.graph_frontend
                .frontend
            )

        if self.frontend is not None:

            return self.frontend

        raise RuntimeError(
            "No active SSPG frontend."
        )

    def frozen_decoder(
        self,
    ) -> nn.Module:

        return (
            self._active_frontend()
            .decoder
        )

    def frozen_vit(
        self,
    ) -> nn.Module:

        return (
            self._active_frontend()
            .visual
            .vit
        )

    def visual_adapter(
        self,
    ) -> nn.Module:

        return (
            self._active_frontend()
            .visual
            .adapter
        )

    def provenance_fusion(
        self,
    ) -> nn.Module:

        return (
            self._active_frontend()
            .fusion
        )

    # ==================================================================
    # TRAIN MODE SAFETY
    # ==================================================================

    def train(
        self,
        mode: bool = True,
    ):

        super().train(
            mode
        )

        frontend = (
            self._active_frontend()
        )

        # Frozen V3.3 decoder.
        frontend.decoder.eval()

        # Frozen pretrained ViT.
        frontend.visual.vit.eval()

        # In provenance-disabled variants, the registered fusion module
        # is inactive and frozen. Keep it in eval mode as well.
        if not self.uses_provenance:
            frontend.fusion.eval()

        if frontend.decoder.training:
            raise RuntimeError(
                "Frozen V3.3 decoder entered training mode."
            )

        if frontend.visual.vit.training:
            raise RuntimeError(
                "Frozen pretrained ViT entered training mode."
            )

        if (
            not self.uses_provenance
            and
            frontend.fusion.training
        ):
            raise RuntimeError(
                "Inactive provenance fusion entered training mode."
            )

        return self

    # ==================================================================
    # FULL
    # ==================================================================

    def _forward_full(
        self,
        received: torch.Tensor,
        payload: torch.Tensor,
    ) -> Dict[str, object]:

        if self.graph_frontend is None:
            raise RuntimeError(
                "FULL has no graph frontend."
            )

        graph_output = self.graph_frontend(
            received,
            payload,
        )

        if "graph_tokens" not in graph_output:
            raise RuntimeError(
                "FULL graph frontend did not return graph_tokens."
            )

        graph_tokens = graph_output[
            "graph_tokens"
        ]

        if graph_tokens.ndim != 3:
            raise RuntimeError(
                "FULL expected graph_tokens [B,N,C]."
            )

        if graph_tokens.shape[1] != 196:
            raise RuntimeError(
                "FULL expected 196 graph nodes."
            )

        if graph_tokens.shape[2] != self.gnn_hidden_dim:
            raise RuntimeError(
                "FULL GNN/head hidden dimension mismatch."
            )

        head_output = self.heads(
            graph_tokens
        )

        output = dict(
            graph_output
        )

        output.update(
            head_output
        )

        output[
            "head_tokens"
        ] = graph_tokens

        output[
            "ablation"
        ] = self.ablation

        return output

    # ==================================================================
    # NO PROVENANCE
    # ==================================================================

    def _forward_no_provenance(
        self,
        received: torch.Tensor,
        payload: torch.Tensor,
    ) -> Dict[str, object]:
        """
        Visual + Graph/GNN.

        Payload-derived provenance is absent.

        The decoder is not called.
        Evidence projection is not called.
        Evidence-token fusion is not called.

        The graph builder retains its validated feature schema, so the
        provenance-related graph fields receive deterministic zeros.
        """

        del payload

        if self.graph_frontend is None:
            raise RuntimeError(
                "NO_PROVENANCE has no graph frontend."
            )

        frontend = (
            self.graph_frontend
            .frontend
        )

        visual_input = F.interpolate(
            received,
            size=(
                224,
                224,
            ),
            mode="bilinear",
            align_corners=False,
        )

        visual_tokens = (
            frontend
            .visual(
                visual_input
            )
        )

        if visual_tokens.ndim != 3:
            raise RuntimeError(
                "NO_PROVENANCE expected visual_tokens [B,N,C]."
            )

        if visual_tokens.shape[1] != 196:
            raise RuntimeError(
                "NO_PROVENANCE expected 196 visual tokens."
            )

        if visual_tokens.shape[2] != 768:
            raise RuntimeError(
                "NO_PROVENANCE expected 768-D visual tokens."
            )

        batch_size = (
            visual_tokens.shape[0]
        )

        zero_patch_evidence = torch.zeros(
            (
                batch_size,
                196,
                1,
            ),
            dtype=visual_tokens.dtype,
            device=visual_tokens.device,
        )

        graph_output = (
            self.graph_frontend
            .graph_builder(
                visual_tokens,
                zero_patch_evidence,
            )
        )

        required_graph_keys = (
            "node_features",
            "edge_index",
            "edge_attr",
        )

        for key in required_graph_keys:

            if key not in graph_output:
                raise RuntimeError(
                    "NO_PROVENANCE graph builder "
                    f"did not return '{key}'."
                )

        node_features = graph_output[
            "node_features"
        ]

        edge_index = graph_output[
            "edge_index"
        ]

        edge_attr = graph_output[
            "edge_attr"
        ]

        graph_tokens = (
            self.graph_frontend
            .gnn(
                node_features,
                edge_index,
                edge_attr,
            )
        )

        if graph_tokens.ndim != 3:
            raise RuntimeError(
                "NO_PROVENANCE expected graph_tokens [B,N,C]."
            )

        if graph_tokens.shape[1] != 196:
            raise RuntimeError(
                "NO_PROVENANCE expected 196 graph tokens."
            )

        if graph_tokens.shape[2] != self.gnn_hidden_dim:
            raise RuntimeError(
                "NO_PROVENANCE GNN/head dimension mismatch."
            )

        head_output = self.heads(
            graph_tokens
        )

        output: Dict[str, object] = {
            "visual_tokens":
                visual_tokens,

            "patch_evidence":
                zero_patch_evidence,

            "fused_tokens":
                visual_tokens,

            "node_features":
                node_features,

            "edge_index":
                edge_index,

            "edge_attr":
                edge_attr,

            "graph_tokens":
                graph_tokens,

            "head_tokens":
                graph_tokens,

            "ablation":
                self.ablation,
        }

        output.update(
            head_output
        )

        return output

    # ==================================================================
    # NO GRAPH
    # ==================================================================

    def _forward_no_graph(
        self,
        received: torch.Tensor,
        payload: torch.Tensor,
    ) -> Dict[str, object]:
        """
        Visual + provenance.

        Graph construction and GNN are absent.

        Fused 768-D tokens are mapped to 256-D with the explicit
        graph-free token projection.
        """

        if self.frontend is None:
            raise RuntimeError(
                "NO_GRAPH has no SSPG frontend."
            )

        if self.token_projection is None:
            raise RuntimeError(
                "NO_GRAPH has no token projection."
            )

        frontend_output = self.frontend(
            received,
            payload,
        )

        if "fused_tokens" not in frontend_output:
            raise RuntimeError(
                "NO_GRAPH frontend did not return fused_tokens."
            )

        fused_tokens = frontend_output[
            "fused_tokens"
        ]

        if fused_tokens.ndim != 3:
            raise RuntimeError(
                "NO_GRAPH expected fused_tokens [B,N,C]."
            )

        if fused_tokens.shape[1] != 196:
            raise RuntimeError(
                "NO_GRAPH expected 196 fused tokens."
            )

        if fused_tokens.shape[2] != 768:
            raise RuntimeError(
                "NO_GRAPH expected 768-D fused tokens."
            )

        head_tokens = (
            self.token_projection(
                fused_tokens
            )
        )

        if head_tokens.shape[1] != 196:
            raise RuntimeError(
                "NO_GRAPH expected 196 projected tokens."
            )

        if head_tokens.shape[2] != self.gnn_hidden_dim:
            raise RuntimeError(
                "NO_GRAPH projection/head dimension mismatch."
            )

        head_output = self.heads(
            head_tokens
        )

        output = dict(
            frontend_output
        )

        output.update(
            head_output
        )

        output[
            "graph_tokens"
        ] = None

        output[
            "head_tokens"
        ] = head_tokens

        output[
            "ablation"
        ] = self.ablation

        return output

    # ==================================================================
    # VISUAL ONLY
    # ==================================================================

    def _forward_visual_only(
        self,
        received: torch.Tensor,
        payload: torch.Tensor,
    ) -> Dict[str, object]:
        """
        Visual-only formal retraining baseline.

        Payload is ignored.

        Decoder:
            not called

        Evidence projection:
            not called

        Evidence fusion:
            not called

        Graph builder:
            not called

        GNN:
            not called
        """

        del payload

        if self.frontend is None:
            raise RuntimeError(
                "VISUAL_ONLY has no SSPG frontend."
            )

        if self.token_projection is None:
            raise RuntimeError(
                "VISUAL_ONLY has no token projection."
            )

        visual_input = F.interpolate(
            received,
            size=(
                224,
                224,
            ),
            mode="bilinear",
            align_corners=False,
        )

        visual_tokens = (
            self.frontend
            .visual(
                visual_input
            )
        )

        if visual_tokens.ndim != 3:
            raise RuntimeError(
                "VISUAL_ONLY expected visual_tokens [B,N,C]."
            )

        if visual_tokens.shape[1] != 196:
            raise RuntimeError(
                "VISUAL_ONLY expected 196 visual tokens."
            )

        if visual_tokens.shape[2] != 768:
            raise RuntimeError(
                "VISUAL_ONLY expected 768-D visual tokens."
            )

        head_tokens = (
            self.token_projection(
                visual_tokens
            )
        )

        if head_tokens.shape[1] != 196:
            raise RuntimeError(
                "VISUAL_ONLY expected 196 projected tokens."
            )

        if head_tokens.shape[2] != self.gnn_hidden_dim:
            raise RuntimeError(
                "VISUAL_ONLY projection/head dimension mismatch."
            )

        head_output = self.heads(
            head_tokens
        )

        output: Dict[str, object] = {
            "visual_tokens":
                visual_tokens,

            "patch_evidence":
                None,

            "fused_tokens":
                visual_tokens,

            "graph_tokens":
                None,

            "head_tokens":
                head_tokens,

            "ablation":
                self.ablation,
        }

        output.update(
            head_output
        )

        return output

    # ==================================================================
    # PUBLIC FORWARD
    # ==================================================================

    def forward(
        self,
        received: torch.Tensor,
        payload: torch.Tensor,
    ) -> Dict[str, object]:

        if received.ndim != 4:
            raise RuntimeError(
                "received must have shape [B,3,H,W]."
            )

        if received.shape[1] != 3:
            raise RuntimeError(
                "received must contain exactly 3 RGB channels."
            )

        if payload.ndim != 2:
            raise RuntimeError(
                "payload must have shape [B,128]."
            )

        if payload.shape[1] != 128:
            raise RuntimeError(
                "payload must contain exactly 128 bits."
            )

        if received.shape[0] != payload.shape[0]:
            raise RuntimeError(
                "received/payload batch-size mismatch."
            )

        if self.ablation == "full":

            return self._forward_full(
                received,
                payload,
            )

        if self.ablation == "no_provenance":

            return self._forward_no_provenance(
                received,
                payload,
            )

        if self.ablation == "no_graph":

            return self._forward_no_graph(
                received,
                payload,
            )

        if self.ablation == "visual_only":

            return self._forward_visual_only(
                received,
                payload,
            )

        raise RuntimeError(
            f"Unhandled ablation variant: {self.ablation}"
        )

    # ==================================================================
    # LOCALIZATION PROJECTION
    # ==================================================================

    @staticmethod
    def localization_logits_256(
        localization_grid_logits: torch.Tensor,
    ) -> torch.Tensor:

        return upsample_localization_logits(
            localization_grid_logits,
            output_size=256,
        )