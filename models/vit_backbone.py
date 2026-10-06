import torch
import torch.nn as nn

class TinyViTBackbone(nn.Module):
    # ASSUMPTION: compact ViT for smoke testing; production experiments can swap to ViT-B/16.
    def __init__(self, in_ch=4, image_size=128, patch_size=16, dim=128, depth=2, heads=4):
        super().__init__(); self.patch_size=patch_size
        self.patch=nn.Conv2d(in_ch,dim,patch_size,patch_size)
        n=(image_size//patch_size)**2
        self.pos=nn.Parameter(torch.zeros(1,n,dim))
        layer=nn.TransformerEncoderLayer(dim,heads,dim*4,batch_first=True,norm_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(layer,depth); self.norm=nn.LayerNorm(dim)
    def forward(self,x):
        z=self.patch(x).flatten(2).transpose(1,2); z=z+self.pos[:,:z.size(1)]
        return self.norm(self.enc(z))
