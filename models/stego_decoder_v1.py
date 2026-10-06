import torch.nn as nn

class StegoDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # ASSUMPTION: lightweight learned decoder for reference implementation.
        self.net=nn.Sequential(nn.Conv2d(3,16,3,1,1),nn.GELU(),nn.Conv2d(16,1,3,1,1),nn.Sigmoid())
    def forward(self,x): return self.net(x)
