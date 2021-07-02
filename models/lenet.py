import torch
from torch import nn
from torchsummary import summary
from library.utils import DataLoader, Utils


class LeNet(nn.Module):
    def __init__(self, num_class: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
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


if __name__ == '__main__':
    net = LeNet()
    # summary(net, (1, 28, 28))

    batch_size = 256
    train_iter, test_iter = DataLoader().load_data_fashion_mnist(batch_size=batch_size)

    lr, num_epochs = 0.9, 10
    Utils.train(net, train_iter, test_iter, num_epochs, lr, Utils.try_gpu())
