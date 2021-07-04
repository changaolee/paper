import torch
from torch import nn
from models.common.residual import Residual


class ResNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_class: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self.resnet_block(64, 64, 2, first_block=True),
            self.resnet_block(64, 128, 2),
            self.resnet_block(128, 256, 2),
            self.resnet_block(256, 512, 2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_class)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    @classmethod
    def resnet_block(cls, input_channels, num_channels, num_residuals, first_block=False):
        blocks = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blocks.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
            else:
                blocks.append(Residual(num_channels, num_channels))
        return nn.Sequential(*blocks)
