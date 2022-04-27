import torch
from torch import nn


class DirectionLoss(nn.Module):
    def __init__(self):
        super(DirectionLoss, self).__init__()
        self.loss_func = torch.nn.BCELoss(reduction='none')

    def forward(self, y_pred, y_gt):
        # direction: Bx37xBHxBW
        y_pred = torch.softmax(y_pred, 1)
        pred_mask = (1 - y_gt[:, 0]).unsqueeze(1)
        direction_loss = self.loss_func(y_pred, y_gt)
        direction_loss = (direction_loss * pred_mask).sum() / (pred_mask.sum() * direction_loss.shape[1] + 1e-6)
        return direction_loss
