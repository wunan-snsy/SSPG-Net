import torch
import torch.nn as nn
import torch.nn.functional as F


class ProvenanceEvidenceProjector(nn.Module):
    """
    Reference-free provenance evidence construction
    and spatial projection.

    Input:
        decoded_prob : [B, 128]
        payload      : [B, 128]

    V3.3 evidence:
        [B, 8, 16]

    Spatial projection:
        8x16
          -> continuous image plane
          -> 14x14 ViT patch grid

    Output:
        evidence_grid   : [B, 1, 8, 16]
        patch_evidence  : [B, 196, 1]

    No trainable parameters.
    """

    def __init__(
        self,
        provenance_rows=8,
        provenance_cols=16,
        vit_rows=14,
        vit_cols=14,
    ):
        super().__init__()

        self.provenance_rows = provenance_rows
        self.provenance_cols = provenance_cols

        self.vit_rows = vit_rows
        self.vit_cols = vit_cols

        self.payload_bits = (
            provenance_rows
            * provenance_cols
        )

        self.vit_tokens = (
            vit_rows
            * vit_cols
        )

    def construct_evidence(
        self,
        decoded_prob,
        payload,
    ):
        if decoded_prob.ndim != 2:
            raise RuntimeError(
                "decoded_prob must be [B,128], "
                f"got {decoded_prob.shape}"
            )

        if payload.shape != decoded_prob.shape:
            raise RuntimeError(
                "payload shape must equal "
                "decoded_prob shape."
            )

        if decoded_prob.shape[1] != self.payload_bits:
            raise RuntimeError(
                f"Expected {self.payload_bits} bits, "
                f"got {decoded_prob.shape[1]}"
            )

        # Probability assigned by decoder
        # to the known TRUE embedded bit.
        true_prob = (
            payload * decoded_prob
            + (1.0 - payload)
            * (1.0 - decoded_prob)
        )

        evidence = (
            1.0 - true_prob
        )

        evidence_grid = evidence.view(
            -1,
            1,
            self.provenance_rows,
            self.provenance_cols,
        )

        return evidence_grid

    def project_to_vit(
        self,
        evidence_grid,
    ):
        """
        Area interpolation is used because each output
        ViT patch should represent the spatially averaged
        provenance evidence over its image region.

        This is a deterministic coordinate projection,
        not a learned mapping.
        """

        if evidence_grid.ndim != 4:
            raise RuntimeError(
                "evidence_grid must be [B,1,H,W]."
            )

        if evidence_grid.shape[1:] != (
            1,
            self.provenance_rows,
            self.provenance_cols,
        ):
            raise RuntimeError(
                "Unexpected provenance grid shape: "
                f"{evidence_grid.shape}"
            )

        vit_grid = F.interpolate(
            evidence_grid,
            size=(
                self.vit_rows,
                self.vit_cols,
            ),
            mode="area",
        )

        patch_evidence = (
            vit_grid
            .flatten(2)
            .transpose(1, 2)
        )

        return vit_grid, patch_evidence

    def forward(
        self,
        decoded_prob,
        payload,
    ):
        evidence_grid = (
            self.construct_evidence(
                decoded_prob,
                payload,
            )
        )

        (
            vit_grid,
            patch_evidence,
        ) = self.project_to_vit(
            evidence_grid
        )

        return {
            "evidence_grid":
                evidence_grid,

            "vit_evidence_grid":
                vit_grid,

            "patch_evidence":
                patch_evidence,
        }


class EvidenceTokenFusion(nn.Module):
    """
    Fuse scalar provenance evidence into ViT patch tokens.

    ViT token:
        [B,196,768]

    Evidence:
        [B,196,1]

    Evidence is first projected into token space and then
    added through a learnable residual scale.

    This module is trainable.
    """

    def __init__(
        self,
        embed_dim=768,
        hidden_dim=96,
    ):
        super().__init__()

        self.evidence_mlp = nn.Sequential(
            nn.Linear(
                1,
                hidden_dim,
            ),
            nn.GELU(),

            nn.Linear(
                hidden_dim,
                embed_dim,
            ),
        )

        # Start close to visual-only representation.
        self.scale = nn.Parameter(
            torch.tensor(0.1)
        )

    def forward(
        self,
        tokens,
        patch_evidence,
    ):
        if tokens.ndim != 3:
            raise RuntimeError(
                "tokens must be [B,N,C]."
            )

        if patch_evidence.ndim != 3:
            raise RuntimeError(
                "patch_evidence must be [B,N,1]."
            )

        if tokens.shape[:2] != (
            patch_evidence.shape[:2]
        ):
            raise RuntimeError(
                "Token/evidence spatial mismatch: "
                f"{tokens.shape} vs "
                f"{patch_evidence.shape}"
            )

        evidence_embedding = (
            self.evidence_mlp(
                patch_evidence
            )
        )

        fused = (
            tokens
            + self.scale
            * evidence_embedding
        )

        return fused