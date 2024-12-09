import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.segmentation_model import MAE3DSegmentationModel
from datasets.dataset_loader import MRI3DDataset
from config import Config

def train():
    # 配置
    cfg = Config()
    
    # 数据集加载
    train_dataset = MRI3DDataset(cfg.DATA_PATH)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    
    # 加载预训练 MAE 模型
    mae_checkpoint = torch.load(cfg.CHECKPOINT_PATH, map_location='cpu')
    mae_model = MAEModel()  # 假设你已定义 MAE 模型
    mae_model.load_state_dict(mae_checkpoint['model'])

    # 初始化分割模型
    model = MAE3DSegmentationModel(mae_model, cfg.NUM_CLASSES).to('cuda')

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    
    # 训练
    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to('cuda'), masks.to('cuda')

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{cfg.NUM_EPOCHS}], Loss: {loss.item()}")

if __name__ == "__main__":
    train()
