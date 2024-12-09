import math
import sys
from typing import Iterable
import torch
import gc
import utils.misc as misc
import utils.lr_sched as lr_sched
import os
import nibabel as nib

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 20

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print(f'log_dir: {log_writer.log_dir}')

    for data_iter_step, (samples, affine,_) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # 每个迭代的学习率调整
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        # 混合精度计算
        with torch.amp.autocast('cuda'):
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)  # Forward pass

        loss_value = loss.item()

        # 防止异常
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        # 梯度累积
        loss = loss / accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        # 清理显存和内存
        if (data_iter_step + 1) % accum_iter == 0:
            del samples, loss  # 删除当前 batch 数据和 loss
            torch.cuda.empty_cache()  # 清空显存缓存
            gc.collect()  # 垃圾回收

        # 同步 GPU
        torch.cuda.synchronize()

        # 日志更新
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # TensorBoard 日志记录
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # 汇总统计信息
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def validate(model: torch.nn.Module, data_loader: Iterable, args, device: torch.device):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Validation:'

    with torch.no_grad():
        for samples, affine, _ in metric_logger.log_every(data_loader, 20, header):
            samples = samples.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

            loss_value = loss.item()
            metric_logger.update(loss=loss_value)

            # 清理显存和内存
            del samples, loss
            torch.cuda.empty_cache()
            gc.collect()

    # 汇总统计信息
    metric_logger.synchronize_between_processes()
    print("Validation stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def test(model: torch.nn.Module, 
         data_loader: Iterable, 
         args, 
         device: torch.device, 
         output_dir: str = './test_results'):
    """
    测试模型并保存预测结果。

    Args:
        model (torch.nn.Module): 加载的模型。
        data_loader (Iterable): 测试数据 DataLoader。
        args: 参数对象，包含 `mask_ratio` 等信息。
        device (torch.device): 运行设备。
        output_dir (str): 保存测试结果的目录。
    """
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Testing:'

    # 创建结果保存目录
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (samples, affine, target_paths) in enumerate(metric_logger.log_every(data_loader, 20, header)):
            samples = samples.to(device, non_blocking=True)
            
            # 模型预测
            with torch.amp.autocast('cuda'):
                loss, _, outputs = model(samples, mask_ratio=args.mask_ratio)

            # 记录损失
            loss_value = loss.item()
            metric_logger.update(loss=loss_value)

            # 保存预测结果为 .nii.gz 文件
            for i, output in enumerate(outputs):
                output_file = os.path.join(output_dir, f"output_{idx}_{i}.nii.gz")
                save_tensor_as_nii(output, target_paths[i], output_file)
                print(f"Saved prediction to {output_file}")

            # 清理显存和内存
            del samples, loss, outputs
            torch.cuda.empty_cache()
            gc.collect()

    # 汇总统计信息
    metric_logger.synchronize_between_processes()
    print("Testing stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def save_tensor_as_nii(tensor, reference_file, output_file):
    """
    保存张量为 .nii.gz 文件
    tensor: [C, D, H, W] 或 [D, H, W]
    reference_file: 用于参考 affine 的 .nii.gz 文件
    output_file: 输出 .nii.gz 文件路径
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # 取第一个通道
    tensor = tensor.cpu().numpy()

    affine = nib.load(reference_file).affine
    nib.save(nib.Nifti1Image(tensor, affine), output_file)
