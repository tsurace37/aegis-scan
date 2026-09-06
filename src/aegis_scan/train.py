"""
Stage 03 -- train a classifier on stage 02's (possibly poisoned) output.

This is a deliberately ordinary supervised-training loop: the point of
this project isn't a novel training procedure, it's what happens *after*
training (stages 04-08), which need a real trained model whose behavior
reflects whatever poison was mixed into its training data. A backdoored
model is one thing when it's just an `.npz` file of altered pixels; it's
a different thing once a network has actually learned "this trigger
pattern means target_label" as a shortcut -- that learned association is
what stage 04's activations and stage 05's detectors are looking for
evidence of.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models.resnet import SmallResNet


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 42
    device: str = "cpu"


@dataclass
class TrainResult:
    model: nn.Module
    history: list[dict[str, float]] = field(default_factory=list)
    in_channels: int = 0
    num_classes: int = 0


def train_classifier(
    images: np.ndarray,
    labels: np.ndarray,
    config: TrainConfig | None = None,
) -> TrainResult:
    """Train a `SmallResNet` on `images`/`labels` (stage 02's output shape) and return it.

    `in_channels` and `num_classes` are read off the data itself rather
    than passed in separately, so this works unchanged whether it's
    handed the 1-channel healthcare set or the 3-channel benchmark set --
    matching stage 01's design goal of both datasets sharing one shape.
    """
    config = config or TrainConfig()
    torch.manual_seed(config.seed)

    in_channels = images.shape[1]
    num_classes = int(labels.max()) + 1
    device = torch.device(config.device)

    model = SmallResNet(in_channels=in_channels, num_classes=num_classes).to(device)

    dataset = TensorDataset(torch.from_numpy(images), torch.from_numpy(labels))
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(config.epochs):
        total_loss = 0.0
        correct = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(dim=1) == yb).sum().item()

        n = len(dataset)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / n,
                "train_acc": correct / n,
            }
        )

    return TrainResult(model=model, history=history, in_channels=in_channels, num_classes=num_classes)


def save_checkpoint(result: TrainResult, path: str | Path) -> None:
    """Save weights plus the architecture metadata `load_checkpoint` needs to rebuild the model.

    Saving just `model.state_dict()` isn't enough on its own -- reloading
    it later requires re-constructing a `SmallResNet` with the exact same
    `in_channels`/`num_classes` it was trained with first. Bundling those
    two numbers into the checkpoint means stage 04 (or anyone else) can
    load a trained model from disk without having to separately remember
    or re-derive which dataset it came from.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "model_state_dict": result.model.state_dict(),
        "in_channels": result.in_channels,
        "num_classes": result.num_classes,
        "history": result.history,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> nn.Module:
    """Rebuild a `SmallResNet` from a checkpoint saved by `save_checkpoint`, in eval mode."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = SmallResNet(in_channels=checkpoint["in_channels"], num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
