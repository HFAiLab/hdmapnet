from torch import nn

from .contrastive_loss import InstanceLoss
from .cross_entropy_loss import CrossEntropyLoss


class Criterion(nn.Module):
    def __init__(self, loss_config):
        super(Criterion, self).__init__()
        self.loss_config = loss_config
        self.weight_dict = {}
        print('loss_config', loss_config)
        self.semantic_loss = CrossEntropyLoss(loss_config.semantic)
        self.direction_loss = CrossEntropyLoss(loss_config.direction)
        self.instance_loss = InstanceLoss(loss_config.instance)
        self.semantic_weight = loss_config.semantic.loss_weight
        self.direction_weight = loss_config.direction.loss_weight
        self.instance_weight = loss_config.instance.loss_weight

    def forward(self, pred_dict, data_dict):
        loss = self.semantic_weight * self.semantic_loss(pred_dict['semantic'], data_dict['semantic_gt'].float())
        direction_loss = self.direction_weight * self.direction_loss(pred_dict['direction'], data_dict['direction_gt'])
        instance_loss = self.instance_weight * self.instance_loss(pred_dict['instance'], data_dict['instance_gt'])[0]
        return loss + direction_loss + instance_loss
