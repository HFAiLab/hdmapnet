import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, loss_config):
        super(CrossEntropyLoss, self).__init__()
        if getattr(loss_config, 'weight', None):
            self.loss_func = torch.nn.CrossEntropyLoss(torch.tensor(loss_config.weight))
        elif getattr(loss_config, 'ignore_index', None):
            self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=loss_config.ignore_index)
        else:
            self.loss_func = torch.nn.CrossEntropyLoss()
        self.out_dim = loss_config.out_dim

    def forward(self, y_pred, y_gt):
        y_gt = torch.permute(y_gt, [0, 2, 3, 1]).contiguous()
        y_pred = torch.permute(y_pred, [0, 2, 3, 1]).contiguous()
        y_gt = torch.argmax(y_gt, dim=-1)
        y_pred = y_pred.view(-1, self.out_dim)
        y_gt = y_gt.view(-1, 1).squeeze()
        return self.loss_func(y_pred, y_gt)
