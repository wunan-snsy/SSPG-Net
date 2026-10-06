import torch.nn.functional as F
def detection_loss(logits,target): return F.binary_cross_entropy_with_logits(logits,target)
