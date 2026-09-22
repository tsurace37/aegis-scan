"""
Named-architecture registry that stage 03 (training) and stage 04
(activation extraction, via `load_checkpoint`) both go through, so
neither is hardwired to one specific `nn.Module` subclass.

`small_resnet` is the default and the only architecture this project's
published results were trained with -- adding a name here, or passing
`--arch` at train time, doesn't change that unless it's chosen
explicitly. A checkpoint saved before this registry existed has no
`arch` field at all; `train.py`'s `load_checkpoint` treats that absence
as `small_resnet` rather than raising an error, so every checkpoint
already produced by this project (including the ones behind the
paper's published results) keeps loading unchanged.
"""

from __future__ import annotations

from typing import Callable

from torch import nn

from .resnet import SmallResNet
from .resnet18_small import build_resnet18

ARCHITECTURES: dict[str, Callable[[int, int], nn.Module]] = {
    "small_resnet": lambda in_channels, num_classes: SmallResNet(
        in_channels=in_channels, num_classes=num_classes
    ),
    "resnet18": build_resnet18,
}

DEFAULT_ARCHITECTURE = "small_resnet"


def build_model(arch: str, in_channels: int, num_classes: int) -> nn.Module:
    """Construct a fresh, untrained model for the named architecture.

    Raises a specific `ValueError` naming the available choices on an
    unknown architecture name, rather than a bare `KeyError` -- this is
    also what protects `load_checkpoint` from silently mis-loading a
    checkpoint saved by some future architecture this version of the
    code doesn't know about.
    """
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown architecture '{arch}' -- available: {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[arch](in_channels, num_classes)
