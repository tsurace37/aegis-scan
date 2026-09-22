"""Tests for the named-architecture registry (models/registry.py).

Checks the registry mechanics themselves (both architectures build,
produce the right output shape, and an unknown name fails clearly) --
training-loop-level integration is covered separately in test_train.py.
"""

import pytest
import torch

from aegis_scan.models.registry import ARCHITECTURES, DEFAULT_ARCHITECTURE, build_model


def test_default_architecture_is_small_resnet():
    assert DEFAULT_ARCHITECTURE == "small_resnet"
    assert DEFAULT_ARCHITECTURE in ARCHITECTURES


def test_both_architectures_registered():
    assert set(ARCHITECTURES) == {"small_resnet", "resnet18"}


@pytest.mark.parametrize("arch", ["small_resnet", "resnet18"])
@pytest.mark.parametrize("in_channels,size", [(1, 16), (3, 20)])
def test_build_model_output_shape(arch, in_channels, size):
    num_classes = 5
    model = build_model(arch, in_channels=in_channels, num_classes=num_classes)
    model.eval()
    x = torch.rand(4, in_channels, size, size)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (4, num_classes)


def test_build_model_rejects_unknown_arch():
    with pytest.raises(ValueError, match="unknown architecture"):
        build_model("not_a_real_arch", in_channels=1, num_classes=2)


def test_resnet18_has_named_layers_for_stage04_hooking():
    """Stage 04 hooks a submodule by string name (e.g. `--layer layer3`).
    Confirms the adapted resnet18 keeps torchvision's own layer1-4
    names, so the same --layer values used for SmallResNet resolve here
    too (see resnet18_small.py's docstring for the depth caveat).
    """
    model = build_model("resnet18", in_channels=3, num_classes=2)
    for name in ["layer1", "layer2", "layer3", "layer4"]:
        assert model.get_submodule(name) is not None
