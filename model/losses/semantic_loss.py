import torch


class SemanticLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SemanticLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt)
        return loss
