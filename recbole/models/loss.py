import torch.nn as nn
import torch

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        self.gamma = 1e-10

    def forward(self, pos_scores, neg_scores):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_scores - neg_scores)).mean()
        return loss

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, scores, targets):
        return self.loss_fct(scores, targets)
