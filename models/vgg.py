import torch
from torch import nn


class VGG(nn.Module):
    def __init__(self, in_channels: int = 1, conv_arch: tuple = None, num_class: int = 10) -> None:
        super().__init__()

        if not conv_arch:
            conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

        conv_blks = []
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(self.vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        self.features = nn.Sequential(*conv_blks)
        self.classifier = nn.Sequential(
            nn.Linear(conv_arch[-1][1] * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_class)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    @classmethod
    def vgg_block(cls, num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            # 堆叠卷积层，第一层做通道变换，之后保持不变
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        # 最后增加 MaxPooling 层，特征图高宽减半
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)
