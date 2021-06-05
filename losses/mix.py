from losses.ohem import ohem_loss
import numpy as np
import torch
from torch import nn


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
    
def cutmix(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    targets = [targets, shuffled_targets, lam]
    return data, targets

def mixup(data, targets, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = data.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * data + (1 - lam) * data[index, :]
    y_a, y_b = targets, targets[index]
    return mixed_x, y_a, y_b, lam
    
def cutmix_criterion(preds, targets, criterion, rate=0.7):
    preds, targets = cutmix(preds, targets)
    targets1, targets2, lam = targets[0], targets[1], targets[2]
    return lam * ohem_loss(rate, criterion, preds, targets1) + (1 - lam) * ohem_loss(rate, criterion, preds, targets2)

class MixupLoss(nn.Module):
    def __init__(self, criterion, rate=0.7):
        super(MixupLoss, self).__init__()
        self.criterion = criterion
        self.rate = rate
    
    def forward(self, preds, targets):
        targets1, targets2, lam = targets[0], targets[1], targets[2]
        return lam * ohem_loss(self.rate, self.criterion, preds, targets1) + (1 - lam) * ohem_loss(self.rate, self.criterion, preds, targets2)
