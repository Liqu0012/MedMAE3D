# Copyright (c) Cyril Zakka.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Modified from:
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/MAE
# --------------------------------------------------------

# import os
# from PIL import Image
# from torchvision import transforms
# from utils.dataloader import VideoFrameDataset

# from timm.data import create_transform
# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# def build_dataset(is_train, args):
#     transform = build_transform(is_train, args)

#     root = os.path.join(args.data_path, 'train' if is_train else 'val')
#     ledger = os.path.join(args.data_path, 'ledger.csv')
#     dataset = VideoFrameDataset(root_path=root, annotationfile_path=ledger,
#                  num_segments= args.num_segments,
#                  frames_per_segment = args.frames_per_segment,
#                  transform = transform,
#                  test_mode = not is_train)
#     print(dataset)
#     return dataset


# def build_transform(is_train, args):
#     mean = IMAGENET_DEFAULT_MEAN
#     std = IMAGENET_DEFAULT_STD
#     # train transform
#     if is_train:
#         # this should always dispatch to transforms_imagenet_train
#         transform = create_transform(
#             input_size=args.input_size,
#             is_training=True,
#             color_jitter=args.color_jitter,
#             auto_augment=args.aa,
#             interpolation='bicubic',
#             re_prob=args.reprob,
#             re_mode=args.remode,
#             re_count=args.recount,
#             mean=mean,
#             std=std,
#         )
#         return transform

#     # eval transform
#     t = []
#     if args.input_size <= 224:
#         crop_pct = 224 / 256
#     else:
#         crop_pct = 1.0
#     size = int(args.input_size / crop_pct)
#     t.append(
#         transforms.Resize(size, interpolation=Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
#     )
#     t.append(transforms.CenterCrop(args.input_size))

#     t.append(transforms.ToTensor())
#     t.append(transforms.Normalize(mean, std))
#     return transforms.Compose(t)
import os
from utils.dataloader import NiftiDataset
import torchio as tio  # 使用 torchio 处理 3D 图像

def build_dataset(is_train, args):
    """Build dataset for 3D MRI data."""
    transform = build_transform(is_train)

    # 使用适合3D MRI数据的路径
    root_path = os.path.join(args.data_path, 'train' if is_train else 'val')

    # 构建 NiftiDataset
    dataset = NiftiDataset(root_path=root_path, transform=transform)
    
    print(dataset)
    return dataset


def build_transform(is_train):
    """Build transformation pipeline for 3D MRI images."""
    transform_list = []

    if is_train:
        # 添加训练时的数据增强
        transform_list.append(tio.RandomAffine(scales=(0.9, 1.1), degrees=10))
        transform_list.append(tio.RandomFlip(axes=(0, 1, 2)))

    # 使用 Resize 调整大小，并使用 CropOrPad 确保图像完全匹配 224x224x224
    transform_list.append(tio.Resize((224, 224, 224)))  # 调整为 224x224x224
    transform_list.append(tio.CropOrPad((224, 224, 224)))  # 确保所有数据裁剪或填充到 224x224x224
    transform_list.append(tio.RescaleIntensity(out_min_max=(0, 1)))  # 强度归一化到 [0, 1]
    transform_list.append(tio.ZNormalization())  # Z-score 归一化

    return tio.Compose(transform_list)

# Example usage:
if __name__ == '__main__':
    class Args:
        data_path = './biobankT1'  # 使用绝对路径

    args = Args()

    # Build training dataset
    train_dataset = build_dataset(is_train=True, args=args)

    # Build validation dataset
    val_dataset = build_dataset(is_train=False, args=args)

    # To use the dataset with a DataLoader:
    from torch.utils.data import DataLoader

    # 在Windows中，如果你使用num_workers > 0，需要使用__name__保护
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # You can now iterate over train_loader and val_loader
    for data, label in train_loader:
        print(data.shape, label)