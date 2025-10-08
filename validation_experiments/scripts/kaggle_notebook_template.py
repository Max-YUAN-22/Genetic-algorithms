#!/usr/bin/env python3
"""
Kaggle Notebook模板 - RSNA Intracranial Aneurysm Detection
基于现有0.69代码，集成我们的多模态融合模型
"""

# =============================================================================
# 导入必要的库
# =============================================================================
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 数据加载和预处理
# =============================================================================
class RSNADataset(Dataset):
    """RSNA数据集类"""
    def __init__(self, data_dir, mode='train', transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """加载样本列表"""
        # 这里需要根据实际数据格式调整
        samples = []
        # 示例实现
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # 实现数据加载逻辑
        pass

# =============================================================================
# 多模态融合模型
# =============================================================================
class MultimodalFusionModel(nn.Module):
    """多模态融合模型 - 基于我们的框架"""
    def __init__(self, num_classes=14):
        super().__init__()
        
        # 多模态特征提取器
        self.cta_extractor = self._create_feature_extractor()
        self.mra_extractor = self._create_feature_extractor()
        self.t1_extractor = self._create_feature_extractor()
        self.t2_extractor = self._create_feature_extractor()
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(256 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def _create_feature_extractor(self):
        """创建特征提取器"""
        return nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    
    def forward(self, cta, mra, t1, t2):
        """前向传播"""
        # 特征提取
        cta_feat = self.cta_extractor(cta)
        mra_feat = self.mra_extractor(mra)
        t1_feat = self.t1_extractor(t1)
        t2_feat = self.t2_extractor(t2)
        
        # 特征融合
        features = torch.stack([cta_feat, mra_feat, t1_feat, t2_feat], dim=1)
        
        # 注意力机制
        attended_features, _ = self.attention(features, features, features)
        
        # 全局平均池化
        fused_features = attended_features.mean(dim=1)
        
        # 分类
        output = self.fusion(fused_features)
        
        return output

# =============================================================================
# 训练和验证函数
# =============================================================================
def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        # 数据移动到设备
        cta, mra, t1, t2 = data
        cta, mra, t1, t2 = cta.to(device), mra.to(device), t1.to(device), t2.to(device)
        target = target.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(cta, mra, t1, t2)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            # 数据移动到设备
            cta, mra, t1, t2 = data
            cta, mra, t1, t2 = cta.to(device), mra.to(device), t1.to(device), t2.to(device)
            target = target.to(device)
            
            # 前向传播
            output = model(cta, mra, t1, t2)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            
            # 收集预测结果
            predictions = torch.sigmoid(output)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    # 计算AUC
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    auc_scores = []
    for i in range(all_targets.shape[1]):
        if np.sum(all_targets[:, i]) > 0:  # 确保有正样本
            auc = roc_auc_score(all_targets[:, i], all_predictions[:, i])
            auc_scores.append(auc)
    
    mean_auc = np.mean(auc_scores) if auc_scores else 0.0
    
    return total_loss / len(dataloader), mean_auc

# =============================================================================
# 主函数
# =============================================================================
def main():
    """主函数"""
    print("🚀 开始RSNA Intracranial Aneurysm Detection验证...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用设备: {device}")
    
    # 数据路径
    data_dir = "/kaggle/input/rsna-intracranial-aneurysm-detection"
    
    # 创建模型
    model = MultimodalFusionModel(num_classes=14).to(device)
    print("✅ 模型创建完成")
    
    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 数据加载器
    # 这里需要根据实际数据格式调整
    train_dataset = RSNADataset(data_dir, mode='train')
    val_dataset = RSNADataset(data_dir, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print("✅ 数据加载器创建完成")
    
    # 训练循环
    best_auc = 0.0
    for epoch in range(10):  # 快速验证，只训练10个epoch
        print(f"\n📊 Epoch {epoch+1}/10")
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"训练损失: {train_loss:.4f}")
        
        # 验证
        val_loss, val_auc = validate(model, val_loader, criterion, device)
        print(f"验证损失: {val_loss:.4f}")
        print(f"验证AUC: {val_auc:.4f}")
        
        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"🎉 新的最佳AUC: {best_auc:.4f}")
    
    print(f"\n🎯 最终最佳AUC: {best_auc:.4f}")
    
    # 生成提交文件
    generate_submission(model, device)
    
    return best_auc

def generate_submission(model, device):
    """生成提交文件"""
    print("📝 生成提交文件...")
    
    # 加载测试数据
    test_dataset = RSNADataset("/kaggle/input/rsna-intracranial-aneurysm-detection", mode='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model.eval()
    all_predictions = []
    all_ids = []
    
    with torch.no_grad():
        for data, ids in test_loader:
            # 数据移动到设备
            cta, mra, t1, t2 = data
            cta, mra, t1, t2 = cta.to(device), mra.to(device), t1.to(device), t2.to(device)
            
            # 前向传播
            output = model(cta, mra, t1, t2)
            predictions = torch.sigmoid(output)
            
            all_predictions.append(predictions.cpu().numpy())
            all_ids.extend(ids)
    
    # 创建提交文件
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    submission_data = {
        "ID": all_ids,
        "Aneurysm Present": all_predictions[:, 0],
        "Location 1": all_predictions[:, 1],
        "Location 2": all_predictions[:, 2],
        "Location 3": all_predictions[:, 3],
        "Location 4": all_predictions[:, 4],
        "Location 5": all_predictions[:, 5],
        "Location 6": all_predictions[:, 6],
        "Location 7": all_predictions[:, 7],
        "Location 8": all_predictions[:, 8],
        "Location 9": all_predictions[:, 9],
        "Location 10": all_predictions[:, 10],
        "Location 11": all_predictions[:, 11],
        "Location 12": all_predictions[:, 12],
        "Location 13": all_predictions[:, 13]
    }
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv("submission.csv", index=False)
    print("✅ 提交文件已生成: submission.csv")

# =============================================================================
# 运行主函数
# =============================================================================
if __name__ == "__main__":
    best_auc = main()
    print(f"\n🎉 验证完成！最佳AUC: {best_auc:.4f}")
    print("📁 提交文件: submission.csv")
    print("📁 最佳模型: best_model.pth")
