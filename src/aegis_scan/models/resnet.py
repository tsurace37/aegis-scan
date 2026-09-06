"""
Model architecture shared by stages 03 and 04.

A compact, hand-written ResNet rather than `torchvision.models.resnet18`:
torchvision's ResNets assume 224x224 ImageNet-sized inputs (their stem
alone downsamples 4x before the first residual block, which would shrink
a 28x28 PneumoniaMNIST image to nothing). This one adapts to whatever
image size and channel count stage 01's loaders hand it -- 1-channel
28x28 chest X-rays or 3-channel 32x32 CIFAR-10 -- and reduces to a fixed
feature size at the end with an adaptive average pool, so the same class
trains on either dataset unchanged.

The layers are named (`stem`, `layer1`, `layer2`, `layer3`, `avgpool`,
`fc`) on purpose: stage 04 hooks a layer by name
(`model.get_submodule("layer3")`) to pull out intermediate activations,
so having stable, predictable names here is what makes that possible
without stage 04 needing to know anything about this file's internals.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a skip connection (He et al., 2015).

    `stride=2` halves the spatial resolution while doubling channels --
    used between layer1/2/3 below instead of a separate pooling layer,
    which is the standard ResNet way of downsampling.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # identity shortcut only works when shape doesn't change; otherwise
        # a 1x1 conv reshapes the skip connection to match
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class SmallResNet(nn.Module):
    """A small ResNet for the 28x28-32x32 images this project's benchmarks use.

    Architecture: a stem conv, three residual blocks that progressively
    downsample (stride 1, 2, 2) while doubling channels, a global average
    pool, and a linear classifier head. `base_channels` controls capacity
    (default 32 keeps this fast to train on CPU, which matters since the
    project's own sandbox has no GPU).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2, base_channels: int = 32) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock(base_channels, base_channels, stride=1)
        self.layer2 = ResidualBlock(base_channels, base_channels * 2, stride=2)
        self.layer3 = ResidualBlock(base_channels * 2, base_channels * 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
