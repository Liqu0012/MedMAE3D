import os
import shutil
import random

# 定义源文件夹路径
source_dir = './biobankT1/'  # 原数据文件夹
train_dir = './data/train/'  # 训练集目标文件夹
val_dir = './data/val/'  # 验证集目标文件夹

# 确保目标文件夹存在
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 获取所有文件夹（每个个体的文件夹）
patient_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]

# 打乱文件夹顺序
random.shuffle(patient_folders)

# 按照8:2比例随机划分
split_point = int(len(patient_folders) * 0.8)
train_folders = patient_folders[:split_point]
val_folders = patient_folders[split_point:]

# 将文件夹移动到train和val文件夹中
for folder in train_folders:
    shutil.move(os.path.join(source_dir, folder), os.path.join(train_dir, folder))

for folder in val_folders:
    shutil.move(os.path.join(source_dir, folder), os.path.join(val_dir, folder))

print(f"已随机将 {len(train_folders)} 个文件夹分配到训练集，{len(val_folders)} 个文件夹分配到验证集。")
