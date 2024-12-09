import argparse
import datetime
from pathlib import Path
import time
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb

import models_mri
from engine_pretrain import train_one_epoch, validate
import timm.optim.optim_factory as optim_factory
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.dataloader import NiftiDataset, monai_transform  # 使用自定义 dataloader 和 transform


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    # General parameters
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations')

    # Model parameters
    parser.add_argument('--model', default='mae3d_vit_base_patch16', type=str, help='Name of model to train')
    parser.add_argument('--input_size', type=int, nargs=3, default=(176, 224, 160), help='Input size as three integers: height, width, depth')
    parser.add_argument('--mask_ratio', default=0.7, type=float, help='Masking ratio')
    parser.add_argument('--norm_pix_loss', action='store_true', help='Use normalized pixels for loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--blr', type=float, default=1e-4, help='Base learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Warmup epochs')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data', type=str, help='Dataset path')
    parser.add_argument('--output_dir', default='./output_dir', help='Path to save checkpoints and logs')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='Resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int, help='Number of DataLoader workers')
    parser.add_argument('--pin_mem', action='store_true', help='Pin memory in DataLoader')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    return parser

def setup_distributed_training(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.local_rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.distributed = True
    else:
        args.distributed = False

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        torch.distributed.barrier()
        print(f"Distributed training on rank {args.local_rank} with {args.world_size} processes.")
    else:
        print("Single process training.")
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

def main(args):
    setup_distributed_training(args)

    print(f'Job directory: {os.path.dirname(os.path.realpath(__file__))}')
    print(f'Arguments: {args}')

    device = args.device
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Data transforms using build_transform from dataloader.py
    transform_train = monai_transform(spatial_size = args.input_size)
    transform_val = monai_transform(spatial_size = args.input_size)

    # Dataset and DataLoader
    dataset_train = NiftiDataset(root_path=os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_val = NiftiDataset(root_path=os.path.join(args.data_path, 'val'), transform=transform_val)

    sampler_train = torch.utils.data.RandomSampler(dataset_train) if not args.distributed else torch.utils.data.DistributedSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size,
                                                    num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, num_workers=args.num_workers,
                                                  pin_memory=args.pin_mem, drop_last=False)

    # Model and optimizer
    model = models_mri.__dict__[args.model](img_size = args.input_size, norm_pix_loss=args.norm_pix_loss).to(device)
    model_without_ddp = model.module if hasattr(model, "module") else model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True
        )

    optimizer = torch.optim.AdamW(optim_factory.param_groups_weight_decay(model, args.weight_decay),
                                  lr=args.lr or args.blr * args.batch_size / 256, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    if args.resume:
        misc.load_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if misc.is_main_process():
        wandb.init(project="hokusaimae3d", config=vars(args))

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler, log_writer=None, args=args)
        if misc.is_main_process():
            wandb.log({f'train_{k}': v for k, v in train_stats.items()}, step=epoch)

        # 保存每个 epoch 的模型
        if args.output_dir:
            epoch_output_dir = Path(args.output_dir) / f"epoch_{epoch}"
            epoch_output_dir.mkdir(parents=True, exist_ok=True)
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch
            )

    total_time = time.time() - start_time
    print(f'Training completed in {str(datetime.timedelta(seconds=int(total_time)))}')

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)