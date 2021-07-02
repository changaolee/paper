import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self, num_class: int = 10) -> None:
        super().__init__()

        # 由于早期 GPU 显存有限，原版的 AlexNet 采用了双数据流设计，使得每个 GPU 只负责存储和计算模型的一半参数。
        # 现在 GPU 显存相对充裕，所以我们现在很少需要跨 GPU 分解模型（因此，我们的 AlexNet 模型在这方面与原始论文稍有不同）。
        self.features = nn.Sequential(
            # 这里，我们使用一个 11 x 11 的更大窗口来捕捉对象。
            # 同时，步幅为 4，以减少输出的高度和宽度。
            # 另外，输出通道的数目远大于 LeNet。
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 减小卷积窗口，使用填充为 2 来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 使用三个连续的卷积层和较小的卷积窗口。
            # 除了最后的卷积层，输出通道的数量进一步增加。
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # 这里，全连接层的输出数量是 LeNet 中的好几倍，使用 dropout 层来减轻过拟合。
        self.classifier = nn.Sequential(
            nn.Linear(6400, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_class)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
