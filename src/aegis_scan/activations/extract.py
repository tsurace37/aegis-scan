"""
Stage 04 -- extract intermediate-layer activations via forward hooks.

Both of stage 05's detection methods need per-sample *feature vectors*
from inside the trained model, not its final class predictions:

- Spectral signature analysis (Tran et al., 2018) looks for an outlier
  singular-value direction in a layer's activation covariance -- poisoned
  samples' activations cluster along a direction clean samples don't use.
- Activation clustering (Chen et al., 2018) runs unsupervised clustering
  (k-means/PCA) directly on a layer's per-sample activations and expects
  poisoned samples to separate into their own small cluster.

Both papers use a late, pre-classifier layer for this (deep enough to
reflect what the network actually learned, not just low-level pixel
statistics). This module gets those activations out of a trained
`SmallResNet` with a `forward hook` -- a function PyTorch calls with a
layer's output every time data flows through it -- attached to that
layer for the duration of one inference pass, then removed. That keeps
`models/resnet.py` a normal, unmodified classifier: nothing about its
`forward()` needs to know it's being probed for this project's own
detection purposes, which also means this same approach will work
against a different or larger architecture later without stage 05
needing to change at all.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def extract_activations(
    model: nn.Module,
    images: np.ndarray,
    layer_name: str = "layer3",
    batch_size: int = 128,
    device: str = "cpu",
) -> np.ndarray:
    """Return `layer_name`'s output for every sample in `images`, as one row per sample.

    `layer_name` is resolved with `model.get_submodule`, so it accepts
    any dotted attribute path PyTorch itself understands (e.g.
    `"layer3"`, or `"layer3.conv2"` for something inside a block) -- this
    isn't hardcoded to `SmallResNet` specifically. A raised `ValueError`
    lists every available submodule name so a typo is easy to fix without
    reading the model source.

    Each sample's activation tensor (which for a conv layer is a 3-D
    feature map, channels x height x width) is flattened to a 1-D vector
    before being returned, because both downstream detection methods
    (spectral signatures, activation clustering) are defined over
    per-sample feature *vectors*, not spatial feature maps.
    """
    device_t = torch.device(device)
    model = model.to(device_t)
    model.eval()

    try:
        layer = model.get_submodule(layer_name)
    except AttributeError as exc:
        available = [name for name, _ in model.named_modules() if name]
        raise ValueError(f"no submodule named {layer_name!r}; available layers: {available}") from exc

    captured: list[torch.Tensor] = []

    def _hook(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        captured.append(output.detach().reshape(output.shape[0], -1).cpu())

    handle = layer.register_forward_hook(_hook)
    try:
        loader = DataLoader(TensorDataset(torch.from_numpy(images)), batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for (xb,) in loader:
                model(xb.to(device_t))
    finally:
        # always detach the hook, even if a batch raises -- leaving it
        # attached would silently keep capturing (and leaking memory) on
        # every later use of this model, including stage 05's own
        # inference passes
        handle.remove()

    return torch.cat(captured, dim=0).numpy()
