import torch
import torch.nn.functional as F
def localization_loss(logits,target):
    bce=F.binary_cross_entropy_with_logits(logits,target)
    p=torch.sigmoid(logits); inter=(p*target).sum(); dice=1-(2*inter+1)/(p.sum()+target.sum()+1)
    return bce+dice
