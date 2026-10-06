"""Formally trained provenance-only baseline for E3."""
import torch
import torch.nn as nn

from training.train_stego_structured_v33 import StructuredStegoDecoderV33
from models.provenance_evidence import ProvenanceEvidenceProjector
from models.provenance_heads_v2 import JointDetectionLocalizationHeadsV2, upsample_localization_logits


class ProvenanceOnlyModel(nn.Module):
    def __init__(self, stego_checkpoint, hidden_dim=256):
        super().__init__()
        ckpt = torch.load(stego_checkpoint, map_location="cpu", weights_only=False)
        if ckpt.get("version") != "V3.3" or ckpt.get("payload_bits") != 128:
            raise RuntimeError("Expected frozen 128-bit V3.3 checkpoint")
        self.decoder = StructuredStegoDecoderV33()
        self.decoder.load_state_dict(ckpt["model_state_dict"], strict=True)
        for p in self.decoder.parameters(): p.requires_grad_(False)
        self.decoder.eval()
        self.projector = ProvenanceEvidenceProjector()
        # Do not apply LayerNorm to a scalar channel: it would map every
        # provenance value to zero and make the baseline input-independent.
        self.evidence_encoder = nn.Sequential(nn.Linear(1, 96), nn.GELU(),
                                              nn.Linear(96, hidden_dim), nn.LayerNorm(hidden_dim))
        self.heads = JointDetectionLocalizationHeadsV2(hidden_dim=hidden_dim,
            detection_mlp_dim=256, detection_dropout=0.1, grid_size=14)

    def train(self, mode=True):
        super().train(mode); self.decoder.eval(); return self

    def forward(self, received, payload):
        with torch.no_grad():
            decoder_logits = self.decoder(received)
            decoder_prob = torch.sigmoid(decoder_logits)
        evidence = self.projector(decoder_prob, payload)
        tokens = self.evidence_encoder(evidence["patch_evidence"])
        out = self.heads(tokens)
        out.update(evidence)
        out["decoder_logits"] = decoder_logits
        out["decoder_prob"] = decoder_prob
        out["head_tokens"] = tokens
        return out

    @staticmethod
    def localization_logits_256(logits):
        return upsample_localization_logits(logits, output_size=256)
