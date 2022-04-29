from pathlib import Path
from time import time
import os
import hfai
import numpy as np
import torch
import logging
import sys
from data import compile_data, compile_dataloader
from evaluation.iou import eval_iou, get_batch_iou, onehot_encoding
from model import compile_model
from tools import distribute_utils
from model.losses import build_criterion


def main(local_rank, config):
    torch.cuda.set_device(local_rank)
    # init distributed mode
    print('init distribution')
    distribute_utils.init_distributed_config(local_rank, config.runtime)
    distribute_utils.init_distributed_mode(config.runtime)
    rank = distribute_utils.get_rank()
    print(f'rank {rank}, local_rank {local_rank}')
    if rank in [-1, 0]:
        print(f'logging to {os.path.join(config.runtime.output_dir, "results.log")}')
        logging.basicConfig(
            filename=os.path.join(config.runtime.output_dir, "results.log"),
            filemode='w',
            format='%(asctime)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    # fix the seed for reproducibility
    logging.info('Fix seed ...')
    distribute_utils.fix_seed(config.runtime)

    # init data
    logging.info('Loading data ...')
    dataset_train, dataset_val = compile_data(config.data)
    sampler_train, train_loader, val_loader = compile_dataloader(config.data, dataset_train, dataset_val)

    # load model
    logging.info('Loading model ...')
    model = compile_model(config.data, config.model)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model to distributed mode
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank],
        find_unused_parameters=True)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'number of params: {n_parameters}')

    optimization_config = config.optimization
    optimizer = torch.optim.Adam(
        model_without_ddp.parameters(),
        lr=optimization_config['lr'],
        weight_decay=optimization_config['weight_decay']
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, optimization_config['lr_drop_step'],
                                                   optimization_config['lr_drop_rate'])

    # init losses
    criterion = build_criterion(config.model)
    criterion.cuda()

    # load if latest.pt model exists
    output_dir = Path(config.runtime.output_dir)
    config.runtime.start_epoch = 0
    if Path(output_dir / "latest.pt").exists():
        config.runtime.resume = Path(output_dir / "latest.pt")
    if config.runtime.resume:
        checkpoint = torch.load(config.runtime.resume)
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not config.runtime.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            config.runtime.start_epoch = checkpoint['epoch'] + 1

    if config.runtime.eval:
        if config.runtime.visualizer:
            logging.info(f'visualize samples saved in {config.runtime.visualize_dir}')
        iou = eval_iou(model, val_loader, visualizer=config.runtime.visualizer)
        logging.info(
            f"EVAL:    "
            f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}")
        return

    model.train()
    batch_len = len(train_loader)
    for epoch in range(config.runtime.start_epoch, config.runtime.nepochs):
        logging.info(f"epoch {epoch}")
        sampler_train.set_epoch(epoch)
        iter_t = time()
        for batchi, data_dict in enumerate(train_loader):
            data_dict = {k: v.cuda() for k, v in data_dict.items()}

            load_data_t = time()
            optimizer.zero_grad()

            semantic, embedding, direction = model(data_dict)
            forward_t = time()

            pred_dict = {
                'semantic': semantic,
                'direction': direction,
                'instance': embedding
            }
            final_loss = criterion(pred_dict, data_dict)
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimization.max_grad_norm)
            optimizer.step()
            backward_t = time()

            if rank in [-1, 0] and batchi % (batch_len // 30 + 1) == 0:
                intersects, union = get_batch_iou(onehot_encoding(semantic), data_dict['semantic_gt'].float())
                iou = intersects / (union + 1e-7)
                logging.info(
                    f"TRAIN[{epoch:>3d}]: [{batchi:>4d}/{batch_len - 1}]    "
                    f"Data: {load_data_t - iter_t:>3.3f}    "
                    f"Forward: {forward_t - load_data_t:>3.3f}    "
                    f"Total: {backward_t - iter_t:>3.3f}    "
                    f"Loss: {final_loss.item():>3.2f}    "
                    f'LR: {optimizer.param_groups[0]["lr"]}    '
                    f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}"
                )

            iter_t = time()

        lr_scheduler.step()

        # save checkpoint if going to suspend
        if rank in [-1, 0] and hfai.receive_suspend_command():
            model_name = "latest.pt"
            distribute_utils.save_train_model_state(epoch, lr_scheduler, model, model_name, optimizer, output_dir,
                                                    config.runtime)
            time.sleep(5)
            hfai.go_suspend()

        # normal save
        if rank in [-1, 0] and epoch % config.runtime.eval_and_save_gap == 0:
            model_name = f"{epoch:04d}.pt"
            before_eval_t = time()
            iou = eval_iou(model, val_loader)
            after_eval_t = time()
            logging.info(
                f"EVAL[{epoch:>3d}]:    "
                f"TIME: {after_eval_t - before_eval_t:.2f}    "
                f"IOU: {np.array2string(iou[1:].numpy(), precision=3, floatmode='fixed')}"
            )

            model.train()
            distribute_utils.save_train_model_state(
                epoch, lr_scheduler, model,
                model_name, optimizer, output_dir,
                config.runtime
            )
