import torch


class SemanticLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SemanticLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    def forward(self, y_pred, y_gt):
        loss = self.loss_fn(y_pred, y_gt)
        return loss
