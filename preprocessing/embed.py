import torch
import torch.nn.functional as F

def reference_message(x):
    # ASSUMPTION: deterministic grayscale reference, resized to image resolution.
    return (0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3]).clamp(0,1)

def embed_message(image, message, strength=0.02):
    # Smoke-test differentiable self-steganographic proxy. Not a claim of the paper's exact encoder.
    centered = message - message.mean(dim=(2,3), keepdim=True)
    return (image + strength * centered.repeat(1,3,1,1)).clamp(0,1)
