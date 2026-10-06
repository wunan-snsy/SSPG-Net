import torch
import torch.nn as nn

from .stego_encoder import StegoEncoder
from .stego_decoder import StegoDecoder
from .adapter import ContentAgnosticAdapter
from .vit_backbone import TinyViTBackbone
from .graph_builder import grid_edges
from .hierarchical_gnn import HierarchicalGNN

from preprocessing.embed import reference_message
from preprocessing.tamper import synthetic_tamper
from preprocessing.residual import residual_map


class SSPGNet(nn.Module):

    def __init__(self, cfg):

        super().__init__()

        self.cfg = cfg

        patch_size = cfg["patch_size"]
        image_size = cfg["image_size"]
        embed_dim = cfg["embed_dim"]

        # ====================================================
        # Self-steganographic components
        # ====================================================

        self.encoder = StegoEncoder(
            cfg["stego_strength"]
        )

        self.decoder = StegoDecoder()

        # ====================================================
        # Content-agnostic adapter
        # ====================================================

        self.adapter = ContentAgnosticAdapter(
            3,
            cfg["adapter_bottleneck"],
        )

        # ====================================================
        # Backbone
        #
        # 3 RGB channels + 1 residual channel
        # ====================================================

        self.backbone = TinyViTBackbone(
            4,
            image_size,
            patch_size,
            embed_dim,
            cfg["depth"],
            cfg["num_heads"],
        )

        # ====================================================
        # Hierarchical GNN
        # ====================================================

        self.gnn = HierarchicalGNN(
            embed_dim,
            cfg["gnn_hidden"],
            cfg["gnn_layers"],
        )

        hidden = cfg["gnn_hidden"]

        # ====================================================
        # Heads
        # ====================================================

        self.det = nn.Linear(
            hidden,
            1,
        )

        self.loc = nn.Linear(
            hidden,
            1,
        )

        self.prov = nn.Sequential(

            nn.Linear(
                hidden * 2,
                hidden,
            ),

            nn.GELU(),

            nn.Linear(
                hidden,
                1,
            ),
        )

    # ========================================================
    # Shared prediction path
    # ========================================================

    def _predict_from_fused(
        self,
        fused,
    ):

        patches = self.backbone(
            fused
        )

        batch_size, num_nodes, _ = (
            patches.shape
        )

        side = int(
            num_nodes ** 0.5
        )

        if (
            side * side
            != num_nodes
        ):

            raise RuntimeError(
                "Patch graph requires a square "
                f"grid, but N={num_nodes}."
            )

        base_edge = grid_edges(
            side,
            side,
            fused.device,
        )

        node_outputs = []
        edge_outputs = []

        for b in range(
            batch_size
        ):

            z = self.gnn(
                patches[b],
                base_edge,
            )

            node_outputs.append(
                z
            )

            source_nodes = z[
                base_edge[0]
            ]

            target_nodes = z[
                base_edge[1]
            ]

            edge_features = torch.cat(
                [
                    source_nodes,
                    target_nodes,
                ],
                dim=-1,
            )

            edge_logits = self.prov(
                edge_features
            ).squeeze(-1)

            edge_outputs.append(
                edge_logits
            )

        nodes = torch.stack(
            node_outputs,
            dim=0,
        )

        provenance_logits = torch.stack(
            edge_outputs,
            dim=0,
        )

        # ----------------------------------------------------
        # Image-level detection
        # ----------------------------------------------------

        graph_feature = nodes.mean(
            dim=1
        )

        detection_logits = self.det(
            graph_feature
        ).squeeze(-1)

        # ----------------------------------------------------
        # Patch-level localization
        # ----------------------------------------------------

        localization_logits = self.loc(
            nodes
        ).squeeze(-1)

        return {

            "patch_features":
                patches,

            "node_features":
                nodes,

            "edge_index":
                base_edge,

            "detection_logits":
                detection_logits,

            "localization_logits":
                localization_logits,

            "provenance_logits":
                provenance_logits,
        }

    # ========================================================
    # ACTIVE SSPG MODE
    # ========================================================

    def forward_synthetic(
        self,
        image,
        tamper_probability=0.5,
    ):

        # ----------------------------------------------------
        # Step 1:
        # deterministic self-reference
        #
        # Engineering assumption V1.
        # ----------------------------------------------------

        reference = reference_message(
            image
        )

        # ----------------------------------------------------
        # Step 2:
        # protect image using self-steganography
        # ----------------------------------------------------

        stego = self.encoder(
            image,
            reference,
        )

        # ----------------------------------------------------
        # Step 3:
        # Active protocol attack generator
        #
        # Now returns:
        #
        # tampered
        # GT mask
        # detection GT label
        # attack type
        # ----------------------------------------------------

        (
            tampered,
            mask,
            labels,
            attack_types,
        ) = synthetic_tamper(
            stego,
            tamper_probability=(
                tamper_probability
            ),
        )

        # ----------------------------------------------------
        # Step 4:
        # decode self-reference
        # ----------------------------------------------------

        decoded = self.decoder(
            tampered
        )

        # ----------------------------------------------------
        # Step 5:
        # compare recovered reference against original
        # reference
        # ----------------------------------------------------

        residual = residual_map(
            decoded,
            reference,
        )

        # ----------------------------------------------------
        # Step 6:
        # content branch
        # ----------------------------------------------------

        adapted = self.adapter(
            tampered
        )

        # ----------------------------------------------------
        # Step 7:
        # RGB/content features + provenance residual
        # ----------------------------------------------------

        fused = torch.cat(
            [
                adapted,
                residual,
            ],
            dim=1,
        )

        predictions = (
            self._predict_from_fused(
                fused
            )
        )

        # ----------------------------------------------------
        # Return complete ACTIVE protocol record
        # ----------------------------------------------------

        outputs = {

            "mode":
                "synthetic",

            "input_image":
                image,

            "reference":
                reference,

            "stego":
                stego,

            "tampered":
                tampered,

            "mask":
                mask,

            "labels":
                labels,

            "attack_types":
                attack_types,

            "decoded":
                decoded,

            "residual":
                residual,
        }

        outputs.update(
            predictions
        )

        return outputs

    # ========================================================
    # EXTERNAL / PASSIVE FALLBACK MODE
    # ========================================================

    def forward_external(
        self,
        image,
    ):

        # ----------------------------------------------------
        # IMPORTANT
        #
        # The supplied external image must be analyzed exactly
        # as provided.
        #
        # No synthetic tampering.
        # No fake active residual.
        # ----------------------------------------------------

        adapted = self.adapter(
            image
        )

        batch_size = image.shape[0]
        height = image.shape[2]
        width = image.shape[3]

        # ----------------------------------------------------
        # Passive fallback has no valid self-steganographic
        # residual because an ordinary external IMD2020 image
        # was never protected by this SSPG encoder.
        #
        # Therefore use an explicit zero residual channel.
        # ----------------------------------------------------

        residual = torch.zeros(
            (
                batch_size,
                1,
                height,
                width,
            ),
            device=image.device,
            dtype=image.dtype,
        )

        fused = torch.cat(
            [
                adapted,
                residual,
            ],
            dim=1,
        )

        predictions = (
            self._predict_from_fused(
                fused
            )
        )

        outputs = {

            "mode":
                "external",

            "input_image":
                image,

            "residual":
                residual,
        }

        outputs.update(
            predictions
        )

        return outputs

    # ========================================================
    # Main forward dispatcher
    # ========================================================

    def forward(
        self,
        image,
        mode="synthetic",
        tamper_probability=0.5,
    ):

        if mode == "synthetic":

            return self.forward_synthetic(
                image,
                tamper_probability=(
                    tamper_probability
                ),
            )

        if mode == "external":

            return self.forward_external(
                image
            )

        raise ValueError(
            "Unknown SSPGNet mode: "
            f"{mode}. "
            "Expected 'synthetic' "
            "or 'external'."
        )