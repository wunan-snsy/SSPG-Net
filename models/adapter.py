import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentAgnosticAdapter(nn.Module):
    def __init__(self, channels=3, bottleneck=48):
        super().__init__()
        lap=torch.tensor([[0.,-1.,0.],[-1.,4.,-1.],[0.,-1.,0.]])
        self.register_buffer('kernel', lap.view(1,1,3,3).repeat(channels,1,1,1))
        self.down=nn.Conv2d(channels,bottleneck,1); self.up=nn.Conv2d(bottleneck,channels,1)
        self.alpha=nn.Parameter(torch.tensor(0.1))
    def forward(self,x):
        hp=F.conv2d(x,self.kernel,padding=1,groups=x.shape[1])
        hp=F.instance_norm(hp)
        return x+self.alpha*self.up(F.gelu(self.down(hp)))
