import torch
from torch.utils.data import DistributedSampler
from ffrecord.torch import DataLoader
from hfai import datasets
from .image import Transform


def compile_data(data_conf):
    version = data_conf['version'].split('-')[-1]
    transform = Transform(data_conf)
    dataset_train = datasets.NuScenes(split='train', transform=transform, version=version)
    dataset_val = datasets.NuScenes(split='val', transform=transform, version=version)
    return dataset_train, dataset_val


def compile_dataloader(data_conf, dataset_train, dataset_val):
    sampler_train = DistributedSampler(dataset_train)
    sampler_val = DistributedSampler(dataset_val, shuffle=False)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, data_conf.batch_size, drop_last=True
    )
    train_loader = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        num_workers=data_conf.num_workers,
        pin_memory=data_conf.pin_memory
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=data_conf.eval_batch_size,
        sampler=sampler_val,
        drop_last=False,
        num_workers=data_conf.num_workers,
        pin_memory=data_conf.pin_memory
    )
    return sampler_train, train_loader, val_loader
