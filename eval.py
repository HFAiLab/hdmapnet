import hfai
from torch.multiprocessing import Process

hfai.client.bind_hf_except_hook(Process)

from pathlib import Path
import torch
import yaml
from easydict import EasyDict
from engine import main

if __name__ == '__main__':

    cfg_file = 'configs/default.yaml'
    with open(cfg_file, 'r') as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    if config.runtime.output_dir:
        Path(config.runtime.output_dir).mkdir(parents=True, exist_ok=True)
    ngpus = torch.cuda.device_count()
    config.runtime.eval = True
    print('ngpus', ngpus)
    print(config)
    torch.multiprocessing.spawn(main, args=(config,), nprocs=ngpus)
