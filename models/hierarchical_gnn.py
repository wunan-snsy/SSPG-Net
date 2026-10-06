import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class HierarchicalGNN(nn.Module):
    def __init__(self, dim=128, hidden=128, layers=2):
        super().__init__(); self.layers=nn.ModuleList([GCNConv(dim if i==0 else hidden,hidden) for i in range(layers)])
        self.act=nn.GELU()
    def forward(self,x,edge_index):
        for g in self.layers: x=self.act(g(x,edge_index))
        return x
