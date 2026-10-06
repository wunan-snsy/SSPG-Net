import torch

def grid_edges(hp, wp, device):
    # ASSUMPTION: 4-neighbour directed patch graph for V1 smoke test.
    edges=[]
    for r in range(hp):
        for c in range(wp):
            i=r*wp+c
            for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                rr,cc=r+dr,c+dc
                if 0<=rr<hp and 0<=cc<wp: edges.append((i,rr*wp+cc))
    return torch.tensor(edges,device=device,dtype=torch.long).t().contiguous()
