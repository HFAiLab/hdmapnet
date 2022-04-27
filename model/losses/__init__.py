from .criterion import Criterion


def build_losses(heads_config, device):
    return Criterion(heads_config, device)
