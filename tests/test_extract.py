"""Tests for stage 04 (activation extraction)."""

import numpy as np
import pytest
import torch

from aegis_scan.activations.extract import extract_activations
from aegis_scan.models.resnet import SmallResNet


def make_images(n=20, channels=1, size=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, channels, size, size), dtype=np.float32)


def test_extract_returns_one_row_per_sample():
    model = SmallResNet(in_channels=1, num_classes=2, base_channels=8)
    images = make_images(n=20)

    activations = extract_activations(model, images, layer_name="layer3", batch_size=7)

    assert activations.shape[0] == 20
    assert activations.ndim == 2


def test_extract_respects_batch_size_boundaries():
    # 17 samples with batch_size=5 forces a ragged last batch (2 samples) --
    # this would silently drop or misalign samples if the DataLoader/hook
    # wiring assumed every batch was full-sized
    model = SmallResNet(in_channels=1, num_classes=2, base_channels=8)
    images = make_images(n=17)

    activations = extract_activations(model, images, layer_name="layer1", batch_size=5)
    assert activations.shape[0] == 17


def test_different_layers_give_different_feature_sizes():
    model = SmallResNet(in_channels=1, num_classes=2, base_channels=8)
    images = make_images(n=6)

    layer1_acts = extract_activations(model, images, layer_name="layer1")
    layer3_acts = extract_activations(model, images, layer_name="layer3")

    # layer3 has more channels but a smaller spatial footprint than layer1;
    # they should not coincidentally produce identical feature-vector sizes
    assert layer1_acts.shape[1] != layer3_acts.shape[1]


def test_invalid_layer_name_raises_with_helpful_message():
    model = SmallResNet(in_channels=1, num_classes=2, base_channels=8)
    images = make_images(n=4)

    with pytest.raises(ValueError, match="no submodule named"):
        extract_activations(model, images, layer_name="not_a_real_layer")


def test_hook_is_removed_after_extraction():
    # if the forward hook weren't cleaned up, a second, unrelated forward
    # pass on the same model would still trigger it and silently grow a
    # module-level list forever
    model = SmallResNet(in_channels=1, num_classes=2, base_channels=8)
    images = make_images(n=4)

    extract_activations(model, images, layer_name="layer2")

    layer2 = model.get_submodule("layer2")
    assert len(layer2._forward_hooks) == 0

    with torch.no_grad():
        model(torch.from_numpy(images))  # should not raise or misbehave
