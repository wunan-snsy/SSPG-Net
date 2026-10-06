import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self,a,b):
        super().__init__();self.x=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU())
    def forward(self,x):return self.x(x)
class UNetLocalizationBaseline(nn.Module):
    def __init__(self):
        super().__init__();self.a=Conv(3,32);self.b=Conv(32,64);self.c=Conv(64,128);self.p=nn.MaxPool2d(2);self.u2=nn.ConvTranspose2d(128,64,2,2);self.d2=Conv(128,64);self.u1=nn.ConvTranspose2d(64,32,2,2);self.d1=Conv(64,32);self.out=nn.Conv2d(32,1,1)
    def forward(self,x):
        a=self.a(x);b=self.b(self.p(a));c=self.c(self.p(b));d=self.d2(torch.cat([self.u2(c),b],1));return self.out(self.d1(torch.cat([self.u1(d),a],1)))
