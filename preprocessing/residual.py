import torch

def residual_map(decoded, reference, eps=1e-6):
    r=(decoded-reference).abs()
    mx=r.amax(dim=(2,3), keepdim=True)
    return r/(mx+eps)
