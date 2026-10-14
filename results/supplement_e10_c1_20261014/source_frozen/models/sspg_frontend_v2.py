import torch
import torch.nn as nn
import torch.nn.functional as F

from training.train_stego_structured_v33 import (
    StructuredStegoDecoderV33,
)

from models.vit_backbone_v2 import (
    ProvenanceViTBackbone,
)

from models.provenance_evidence import (
    ProvenanceEvidenceProjector,
    EvidenceTokenFusion,
)


class SSPGFrontendV2(nn.Module):
    """
    E5.3R SSPG frontend V2.

    received image [B,3,256,256]
              |
              +--> frozen V3.3 decoder
              |          |
              |      [B,128] probabilities
              |          +
              |      known payload
              |          |
              |     reference-free evidence
              |          |
              |       8x16 -> 14x14
              |
              +--> resize 224x224
                         |
                    frozen ViT-B/16
                         |
                    trainable adapter
                         |
                    [B,196,768]
                         +
                    evidence fusion
                         |
                    fused tokens

    Frozen:
        V3.3 decoder
        pretrained ViT

    Trainable:
        ViT token adapter
        evidence fusion
    """

    def __init__(
        self,
        stego_checkpoint,
        adapter_bottleneck=96,
        fusion_hidden_dim=96,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Frozen V3.3 decoder
        # ----------------------------------------------------

        checkpoint = torch.load(
            stego_checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        if checkpoint.get("version") != "V3.3":
            raise RuntimeError(
                "Expected V3.3 checkpoint."
            )

        if checkpoint.get("payload_bits") != 128:
            raise RuntimeError(
                "Expected 128-bit V3.3 checkpoint."
            )

        self.decoder = (
            StructuredStegoDecoderV33()
        )

        self.decoder.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)

        self.decoder.eval()

        # ----------------------------------------------------
        # Frozen ViT + trainable token adapter
        # ----------------------------------------------------

        self.visual = ProvenanceViTBackbone(
            adapter_bottleneck=adapter_bottleneck,
            pretrained=True,
        )

        # ----------------------------------------------------
        # Deterministic evidence projection
        # ----------------------------------------------------

        self.projector = (
            ProvenanceEvidenceProjector()
        )

        # ----------------------------------------------------
        # Trainable evidence-token fusion
        # ----------------------------------------------------

        self.fusion = EvidenceTokenFusion(
            embed_dim=768,
            hidden_dim=fusion_hidden_dim,
        )

    def train(self, mode=True):
        """
        Keep frozen decoder and frozen ViT in eval mode
        even when the overall frontend enters train mode.

        Adapter and fusion remain trainable.
        """

        super().train(mode)

        self.decoder.eval()
        self.visual.vit.eval()

        return self

    def forward(
        self,
        received,
        payload,
    ):

        if received.ndim != 4:
            raise RuntimeError(
                "received must be [B,3,H,W]."
            )

        if received.shape[1] != 3:
            raise RuntimeError(
                "received must have 3 RGB channels."
            )

        if payload.ndim != 2:
            raise RuntimeError(
                "payload must be [B,128]."
            )

        if payload.shape[1] != 128:
            raise RuntimeError(
                "payload must contain 128 bits."
            )

        if payload.shape[0] != received.shape[0]:
            raise RuntimeError(
                "Batch mismatch between received "
                "image and payload."
            )

        # ----------------------------------------------------
        # 1. Frozen V3.3 decoder
        #
        # no_grad is intentional:
        # decoder is an immutable measurement device.
        # ----------------------------------------------------

        with torch.no_grad():

            decoder_logits = self.decoder(
                received
            )

            decoder_prob = torch.sigmoid(
                decoder_logits
            )

        # ----------------------------------------------------
        # 2. Reference-free provenance evidence
        # ----------------------------------------------------

        evidence = self.projector(
            decoder_prob,
            payload,
        )

        patch_evidence = evidence[
            "patch_evidence"
        ]

        # ----------------------------------------------------
        # 3. Visual path
        #
        # ViT-B/16 expects 224x224.
        # ----------------------------------------------------

        visual_input = F.interpolate(
            received,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        visual_tokens = self.visual(
            visual_input
        )

        # ----------------------------------------------------
        # 4. Evidence-token fusion
        # ----------------------------------------------------

        fused_tokens = self.fusion(
            visual_tokens,
            patch_evidence,
        )

        return {
            "decoder_logits":
                decoder_logits,

            "decoder_prob":
                decoder_prob,

            "evidence_grid":
                evidence["evidence_grid"],

            "vit_evidence_grid":
                evidence["vit_evidence_grid"],

            "patch_evidence":
                patch_evidence,

            "visual_tokens":
                visual_tokens,

            "fused_tokens":
                fused_tokens,
        }