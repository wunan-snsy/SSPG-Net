import torch
import torch.nn.functional as F

def patch_targets(mask, patch_size):
    return F.avg_pool2d(mask, kernel_size=patch_size, stride=patch_size).flatten(1)
