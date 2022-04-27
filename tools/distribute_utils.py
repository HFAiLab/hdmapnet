import torch
import os
import numpy as np
import random
import torch.distributed as dist


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_config(local_rank, runtime_config):
    runtime_config.rank = int(os.environ["RANK"])
    runtime_config.world_size = int(os.environ['WORLD_SIZE'])
    runtime_config.gpu = local_rank
    runtime_config.gpus = torch.cuda.device_count()  # gpus per node
    ip = os.environ.get("MASTER_IP", None)
    runtime_config.ip = ip if ip else "127.0.0.1"
    runtime_config.port = os.environ["MASTER_PORT"]


def init_distributed_mode(runtime_config):
    if runtime_config.rank in [0, -1]:
        print(runtime_config)
    init_method = f"tcp://{runtime_config.ip}:{runtime_config.port}"
    print('| distributed init (rank {}): {}'.format(runtime_config.rank, init_method))
    torch.distributed.init_process_group(
        backend=runtime_config.dist_backend, init_method=init_method,
        world_size=runtime_config.world_size * runtime_config.gpus,
        rank=runtime_config.rank * runtime_config.gpus + runtime_config.gpu
    )
    torch.cuda.set_device(runtime_config.gpu)
    torch.distributed.barrier()
    setup_for_distributed(runtime_config.rank == 0)


def fix_seed(train_cfg):
    seed = train_cfg.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_train_model_state(epoch, lr_scheduler, model, model_name, optimizer, output_dir, runtime_config):
    state = {
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        "epoch": epoch,
        'args': runtime_config,
    }
    print("save to", f"{output_dir / model_name}")
    save_on_master(state, output_dir / model_name)
