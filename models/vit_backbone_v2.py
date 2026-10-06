import torch
import torch.nn as nn
import timm


MODEL_NAME = (
    "vit_base_patch16_224."
    "augreg_in21k_ft_in1k"
)


class ProvenanceViTBackbone(nn.Module):
    """
    E5.3R ViT backbone.

    Frozen pretrained ViT-B/16
        +
    trainable lightweight token adapter.

    Expected input:
        [B, 3, 224, 224]

    Expected patch tokens:
        [B, 196, 768]

    The CLS token is removed before returning.

    IMPORTANT:
    The pretrained ViT itself is frozen.
    Only the adapter is trainable.
    """

    def __init__(
        self,
        adapter_bottleneck=96,
        pretrained=True,
    ):
        super().__init__()

        self.model_name = MODEL_NAME

        self.vit = timm.create_model(
            MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
        )

        # Freeze entire pretrained backbone.
        for parameter in self.vit.parameters():
            parameter.requires_grad_(False)

        embed_dim = int(
            self.vit.embed_dim
        )

        self.embed_dim = embed_dim

        # Residual bottleneck token adapter.
        self.adapter = nn.Sequential(
            nn.LayerNorm(embed_dim),

            nn.Linear(
                embed_dim,
                adapter_bottleneck,
            ),

            nn.GELU(),

            nn.Linear(
                adapter_bottleneck,
                embed_dim,
            ),
        )

        # Start close to identity.
        self.adapter_scale = nn.Parameter(
            torch.tensor(0.1)
        )

    def forward(self, x):

        # Frozen ViT.
        with torch.no_grad():
            tokens = self.vit.forward_features(
                x
            )

        # timm ViT normally returns:
        # [B, 197, 768]
        # CLS + 196 patch tokens.

        if tokens.ndim != 3:
            raise RuntimeError(
                "Expected ViT token tensor "
                f"[B,N,C], got {tokens.shape}"
            )

        # Remove prefix tokens such as CLS.
        num_prefix_tokens = int(
            getattr(
                self.vit,
                "num_prefix_tokens",
                1,
            )
        )

        patch_tokens = tokens[
            :,
            num_prefix_tokens:,
            :
        ]

        if patch_tokens.shape[1] != 196:
            raise RuntimeError(
                "Expected 196 ViT-B/16 patch "
                "tokens for 224x224 input, got "
                f"{patch_tokens.shape}"
            )

        if patch_tokens.shape[2] != self.embed_dim:
            raise RuntimeError(
                "Unexpected embedding dimension: "
                f"{patch_tokens.shape}"
            )

        adapted = (
            patch_tokens
            + self.adapter_scale
            * self.adapter(patch_tokens)
        )

        return adapted


def count_parameters(module):

    total = sum(
        p.numel()
        for p in module.parameters()
    )

    trainable = sum(
        p.numel()
        for p in module.parameters()
        if p.requires_grad
    )

    return total, trainable