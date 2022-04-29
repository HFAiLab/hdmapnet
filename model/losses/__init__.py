from .criterion import Criterion as CriterionHDMap
from .criterion_basic import Criterion as CriterionBasic


def build_criterion(model_conf):
    if model_conf.loss_type == 'basic':
        return CriterionBasic(model_conf.basic_loss)
    else:
        return CriterionHDMap(model_conf)
