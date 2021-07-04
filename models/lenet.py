import torch
from torch import nn


class LeNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_class: int = 10) -> None:
        super().__init__()

        # 特征提取包括两个卷积层。
        # 每个卷积块中的基本单元是一个卷积层、一个 sigmoid 激活函数和平均池化层。
        # 注意，虽然 ReLU 和最大池化层更有效，但它们在 20 世纪 90 年代还没有出现。
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, num_class)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
