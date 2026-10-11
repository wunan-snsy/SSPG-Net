"""Capacity-matched, tokenwise control for the frozen SSPG-Net graph ablation.

The two residual MLP blocks act on each of the 196 tokens independently.
They never construct edges or exchange information across tokens.
"""

import torch
from torch import nn

from models.sspg_ablation_model_v2 import SSPGAblationModelV2


class TokenwiseResidualMLP(nn.Module):
    def __init__(self, width=256, expansion=837):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, expansion),
            nn.GELU(),
            nn.Linear(expansion, width),
        )

    def forward(self, tokens):
        return tokens + self.net(tokens)


class SSPGParamMatchedNoGraphV2(SSPGAblationModelV2):
    def __init__(self, stego_checkpoint, ablation="matched_no_graph", **kwargs):
        if ablation != "matched_no_graph":
            raise ValueError("Expected matched_no_graph")
        super().__init__(stego_checkpoint=stego_checkpoint, ablation="no_graph", **kwargs)
        self.tokenwise_capacity = nn.Sequential(
            TokenwiseResidualMLP(width=self.gnn_hidden_dim),
            TokenwiseResidualMLP(width=self.gnn_hidden_dim),
        )

    def _forward_no_graph(self, received: torch.Tensor, payload: torch.Tensor):
        front = self.frontend(received, payload)
        tokens = front["fused_tokens"]
        if tokens.ndim != 3 or tokens.shape[1:] != (196, 768):
            raise RuntimeError("Expected 196 fused tokens with 768 channels")
        tokens = self.token_projection(tokens)
        tokens = self.tokenwise_capacity(tokens)
        output = dict(front)
        output.update(self.heads(tokens))
        output.update(graph_tokens=None, head_tokens=tokens, ablation="matched_no_graph")
        return output
