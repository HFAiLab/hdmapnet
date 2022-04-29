from torch import nn
import torch
from .instance_loss import InstanceLoss
from .direction_loss import DirectionLoss
from .semantic_loss import SemanticLoss


class Criterion(nn.Module):
    def __init__(self, model_conf):
        super(Criterion, self).__init__()
        self.seg_loss_fn = SemanticLoss(model_conf.hdmap_loss.pos_weight)
        self.emb_loss_fn = InstanceLoss(
            model_conf.embedded_dim, model_conf.hdmap_loss.delta_v,
            model_conf.hdmap_loss.delta_d)
        self.dir_loss_fn = DirectionLoss()
        self.model_conf = model_conf

    def forward(self, pred_dict, data_dict):
        semantic, embedding, direction = pred_dict['semantic'], pred_dict['instance'], pred_dict['direction']
        seg_loss = self.seg_loss_fn(semantic, data_dict['semantic_gt'].float())
        var_loss, dist_loss, reg_loss = self.emb_loss_fn(embedding, data_dict['instance_gt'])
        direction_loss = self.dir_loss_fn(direction, data_dict['direction_gt'])
        final_loss = \
            seg_loss * self.model_conf.hdmap_loss.scale_seg + \
            var_loss * self.model_conf.hdmap_loss.scale_var + \
            dist_loss * self.model_conf.hdmap_loss.scale_dist + \
            direction_loss * self.model_conf.hdmap_loss.scale_direction
        return final_loss
