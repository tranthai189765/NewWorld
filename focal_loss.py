import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight  # class weights
        self.reduction = reduction

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)               # [B, C]
        prob = torch.exp(log_prob)                            # [B, C]
        target_one_hot = F.one_hot(target, num_classes=input.size(-1))  # [B, C]
        
        focal_weight = (1 - prob) ** self.gamma               # [B, C]
        loss = -target_one_hot * focal_weight * log_prob      # [B, C]

        if self.weight is not None:
            weight = self.weight.unsqueeze(0)                 # [1, C]
            loss = loss * weight

        loss = loss.sum(dim=1)  # sum over classes

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss