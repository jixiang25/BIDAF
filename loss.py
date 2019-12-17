import torch
import torch.nn as nn


class MCLoss(nn.Module):
    def __init__(self):
        super(MCLoss, self).__init__()

    def forward(self, prob_start, prob_end, answer):
        batch_size = prob_start.shape[0]
        indices = [idx for idx in range(batch_size)]
        p1 = torch.log(prob_start[indices, answer[0]])
        p2 = torch.log(prob_end[indices, answer[1]])
        loss_value = torch.sum(p1 + p2)
        return -loss_value
