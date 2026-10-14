"""E10: local spatial interaction control with a matched trainable budget."""
import torch
from torch import nn
from models.sspg_ablation_model_v2 import SSPGAblationModelV2

EXPANSION = 832  # Selected by parameter arithmetic only, before any E10 run.


class ResidualSpatialMixer(nn.Module):
    """Channel-only LayerNorm and a local 3x3 spatial convolution per block."""
    def __init__(self, width=256, expansion=EXPANSION):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.expand = nn.Conv2d(width, expansion, 1)
        self.activation = nn.GELU()
        self.project = nn.Conv2d(expansion, width, 1)

    def forward(self, grid):
        normalized = self.norm(grid.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return grid + self.project(self.activation(self.expand(self.depthwise(normalized))))


class SpatialTokenMixer(nn.Module):
    def __init__(self, width=256, expansion=EXPANSION):
        super().__init__()
        self.width = width
        self.blocks = nn.Sequential(ResidualSpatialMixer(width, expansion),
                                    ResidualSpatialMixer(width, expansion))

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1:] != (196, self.width):
            raise RuntimeError('Expected [B,196,256] projected tokens')
        grid = tokens.transpose(1, 2).reshape(tokens.shape[0], self.width, 14, 14)
        return self.blocks(grid).flatten(2).transpose(1, 2)


class SSPGSpatialMixerMatchedV2(SSPGAblationModelV2):
    model_name = 'SPATIAL_MIXER_MATCHED'

    def __init__(self, stego_checkpoint, expansion=EXPANSION, **kwargs):
        super().__init__(stego_checkpoint=stego_checkpoint, ablation='no_graph', **kwargs)
        self.spatial_mixer = SpatialTokenMixer(self.gnn_hidden_dim, expansion)

    def _forward_no_graph(self, received: torch.Tensor, payload: torch.Tensor):
        front = self.frontend(received, payload)
        tokens = self.spatial_mixer(self.token_projection(front['fused_tokens']))
        output = dict(front)
        output.update(self.heads(tokens))
        output.update(graph_tokens=None, head_tokens=tokens, ablation='spatial_mixer_matched')
        return output
