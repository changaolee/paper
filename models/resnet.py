import torch
from torch import nn
from common.residual import Residual


class ResNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_class: int = 10) -> None:
        super().__init__()
