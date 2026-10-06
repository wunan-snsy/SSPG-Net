import torch.nn as nn
from preprocessing.embed import embed_message

class StegoEncoder(nn.Module):
    def __init__(self, strength=0.02): super().__init__(); self.strength=strength
    def forward(self, image, message): return embed_message(image, message, self.strength)
