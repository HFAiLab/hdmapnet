import hfai_env

from engine import main

hfai_env.set_env("hdmn")
import hfai
from torch.multiprocessing import Process

hfai.client.bind_hf_except_hook(Process)
import argparse
from pathlib import Path
import torch
import yaml
from easydict import EasyDict
from tools.model_downloader import download_models

if __name__ == '__main__':
    download_models()

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--version", type=str, default=None)
    args = parser.parse_args()

    cfg_file = 'configs/default.yaml'
    with open(cfg_file, 'r') as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    if args.output_dir:
        config.runtime.output_dir = args.output_dir
    if args.version:
        config.data.version = args.version
    if config.runtime.output_dir:
        Path(config.runtime.output_dir).mkdir(parents=True, exist_ok=True)
    ngpus = torch.cuda.device_count()
    config.runtime.eval = False
    print('ngpus', ngpus)
    print(config)
    torch.multiprocessing.spawn(main, args=(config,), nprocs=ngpus)
