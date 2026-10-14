import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# E5.3R-4B-0
# Joint Detection + Localization Heads
#
# Input:
#     graph_tokens [B, 196, 256]
#
# Detection:
#     mean pooling + max pooling
#     -> 512-D
#     -> MLP
#     -> image-level logit
#
# Localization:
#     per-node Linear
#     -> 196 logits
#     -> 14 x 14
#
# IMPORTANT:
#     Outputs are RAW LOGITS.
#     Sigmoid belongs to evaluation/inference, not here.
# ============================================================


class DetectionHeadV2(nn.Module):

    def __init__(
        self,
        hidden_dim=256,
        mlp_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        self.hidden_dim = int(
            hidden_dim
        )

        self.norm = nn.LayerNorm(
            self.hidden_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                self.hidden_dim * 2,
                mlp_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                mlp_dim,
                1,
            ),
        )

    def forward(
        self,
        graph_tokens,
    ):
        if graph_tokens.ndim != 3:
            raise RuntimeError(
                "DetectionHeadV2 expects "
                "[B,N,C]."
            )

        if graph_tokens.shape[-1] != self.hidden_dim:
            raise RuntimeError(
                "DetectionHeadV2 hidden "
                "dimension mismatch."
            )

        x = self.norm(
            graph_tokens
        )

        mean_feature = x.mean(
            dim=1
        )

        max_feature = x.max(
            dim=1
        ).values

        pooled = torch.cat(
            [
                mean_feature,
                max_feature,
            ],
            dim=-1,
        )

        logits = self.classifier(
            pooled
        ).squeeze(
            -1
        )

        return logits


class LocalizationHeadV2(nn.Module):

    def __init__(
        self,
        hidden_dim=256,
        grid_size=14,
    ):
        super().__init__()

        self.hidden_dim = int(
            hidden_dim
        )

        self.grid_size = int(
            grid_size
        )

        self.expected_nodes = (
            self.grid_size
            * self.grid_size
        )

        self.norm = nn.LayerNorm(
            self.hidden_dim
        )

        self.classifier = nn.Linear(
            self.hidden_dim,
            1,
        )

    def forward(
        self,
        graph_tokens,
    ):
        if graph_tokens.ndim != 3:
            raise RuntimeError(
                "LocalizationHeadV2 expects "
                "[B,N,C]."
            )

        batch_size = (
            graph_tokens.shape[0]
        )

        nodes = (
            graph_tokens.shape[1]
        )

        channels = (
            graph_tokens.shape[2]
        )

        if nodes != self.expected_nodes:
            raise RuntimeError(
                "LocalizationHeadV2 expected "
                f"{self.expected_nodes} nodes, "
                f"got {nodes}."
            )

        if channels != self.hidden_dim:
            raise RuntimeError(
                "LocalizationHeadV2 hidden "
                "dimension mismatch."
            )

        x = self.norm(
            graph_tokens
        )

        node_logits = self.classifier(
            x
        ).squeeze(
            -1
        )

        grid_logits = node_logits.reshape(
            batch_size,
            1,
            self.grid_size,
            self.grid_size,
        )

        return {
            "node_logits":
                node_logits,

            "grid_logits":
                grid_logits,
        }


class JointDetectionLocalizationHeadsV2(
    nn.Module
):

    def __init__(
        self,
        hidden_dim=256,
        detection_mlp_dim=256,
        detection_dropout=0.1,
        grid_size=14,
    ):
        super().__init__()

        self.detection = DetectionHeadV2(
            hidden_dim=hidden_dim,
            mlp_dim=detection_mlp_dim,
            dropout=detection_dropout,
        )

        self.localization = LocalizationHeadV2(
            hidden_dim=hidden_dim,
            grid_size=grid_size,
        )

    def forward(
        self,
        graph_tokens,
    ):
        detection_logits = (
            self.detection(
                graph_tokens
            )
        )

        localization = (
            self.localization(
                graph_tokens
            )
        )

        return {
            "detection_logits":
                detection_logits,

            "localization_node_logits":
                localization[
                    "node_logits"
                ],

            "localization_grid_logits":
                localization[
                    "grid_logits"
                ],
        }


def upsample_localization_logits(
    grid_logits,
    output_size=256,
):
    """
    Deterministic visualization/evaluation projection.

    IMPORTANT:
        Input is logits, not probabilities.

    Output:
        [B,1,256,256] by default.
    """

    if grid_logits.ndim != 4:
        raise RuntimeError(
            "Expected localization grid "
            "logits [B,1,H,W]."
        )

    return F.interpolate(
        grid_logits,
        size=(
            int(output_size),
            int(output_size),
        ),
        mode="bilinear",
        align_corners=False,
    )