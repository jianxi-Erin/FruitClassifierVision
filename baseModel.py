# 定义cnn卷积神经网络模型(优化后)
import torch.nn as nn

# CNN模型(优化)
class FruitCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # 第一层卷积（添加BatchNorm）
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 第二层卷积（添加BatchNorm）
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 第三层卷积（添加BatchNorm和额外Dropout）
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)  # 新增卷积层后的Dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*8*8, 1024),  # 扩大中间层维度
            nn.ReLU(),
            nn.Dropout(0.6),  # 增强Dropout比例
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        # 前向传播
        x = self.features(x)  # 特征提取
        x = x.view(x.size(0), -1)  # 将特征图展平为一维向量
        return self.classifier(x)  # 分类
