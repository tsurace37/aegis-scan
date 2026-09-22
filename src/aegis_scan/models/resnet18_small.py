"""
A second, off-the-shelf architecture (torchvision's ResNet18), adapted
for small images the way common CIFAR-ResNet recipes do.

Exists to prove stage 03 (training) and stage 04 (activation
extraction, via `load_checkpoint`) aren't hardwired to one bespoke
network -- selecting `--arch resnet18` at train time runs this instead,
with no other pipeline code needing to change. `SmallResNet` stays the
default: it's the architecture this project's own published results
were trained with, so existing checkpoints and the paper's numbers
remain reproducible exactly as before, whether or not this file exists.

torchvision's stock `resnet18` assumes ImageNet-sized (224x224) input:
its stem (a 7x7 stride-2 conv followed by a stride-2 max-pool)
downsamples 4x before the first residual block, which would shrink a
28x28 or 32x32 image to almost nothing before any real feature
extraction happens. The fix used here -- a 3x3 stride-1 stem conv, and
dropping the max-pool -- is the standard adaptation used throughout the
CIFAR-ResNet literature for exactly this reason (e.g. the original
ResNet paper's own CIFAR-10 variant uses a 3x3 stride-1 stem); it is not
a bespoke invention for this project.

Because this keeps torchvision's own `layer1`/`layer2`/`layer3`/`layer4`
submodule names, stage 04's `--layer` argument (e.g. `--layer layer3`)
still resolves to a real submodule here, the same way it does for
`SmallResNet`'s `layer1`/`layer2`/`layer3` -- though note this
architecture has four residual stages where `SmallResNet` has three, so
`layer3` means "third of four" here versus "last" on `SmallResNet`; the
two aren't guaranteed to be the equivalent depth in the network.
"""

from __future__ import annotations

from torch import nn
from torchvision.models import resnet18


def build_resnet18(in_channels: int, num_classes: int) -> nn.Module:
    """Build a CIFAR-adapted ResNet18 sized for `in_channels`/`num_classes`.

    Weights are always randomly initialized (`weights=None`): this
    project trains from scratch on its own (possibly poisoned) data by
    design -- loading ImageNet-pretrained weights here would mean the
    model enters training already "knowing" features from millions of
    unrelated images, which is a different, uncontrolled experiment
    from the one this pipeline is built to run.
    """
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
