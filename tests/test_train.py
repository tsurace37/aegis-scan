"""Tests for stage 03 (training). Uses tiny synthetic data and 2 epochs
so these run in a couple seconds on CPU -- they check that the training
loop, checkpointing, and reload path are wired correctly, not that the
model reaches any particular accuracy.
"""

import numpy as np
import torch

from aegis_scan.train import TrainConfig, load_checkpoint, save_checkpoint, train_classifier


def make_toy_data(n=64, channels=1, size=16, num_classes=2, seed=0):
    rng = np.random.default_rng(seed)
    images = rng.random((n, channels, size, size), dtype=np.float32)
    labels = rng.integers(0, num_classes, size=n).astype(np.int64)
    return images, labels


def test_train_runs_and_produces_history():
    images, labels = make_toy_data()
    result = train_classifier(images, labels, config=TrainConfig(epochs=2, batch_size=16, seed=1))

    assert len(result.history) == 2
    for row in result.history:
        assert set(row) == {"epoch", "train_loss", "train_acc"}
        assert row["train_loss"] >= 0
        assert 0.0 <= row["train_acc"] <= 1.0


def test_model_shape_matches_data():
    images, labels = make_toy_data(channels=3, size=20, num_classes=4)
    result = train_classifier(images, labels, config=TrainConfig(epochs=1, batch_size=8))

    assert result.in_channels == 3
    assert result.num_classes == 4

    model = result.model
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(images[:5]))
    assert logits.shape == (5, 4)


def test_reproducible_with_same_seed():
    images, labels = make_toy_data(n=32)
    r1 = train_classifier(images, labels, config=TrainConfig(epochs=2, batch_size=8, seed=7))
    r2 = train_classifier(images, labels, config=TrainConfig(epochs=2, batch_size=8, seed=7))

    assert r1.history == r2.history


def test_checkpoint_round_trip_preserves_predictions(tmp_path):
    images, labels = make_toy_data(n=32)
    result = train_classifier(images, labels, config=TrainConfig(epochs=1, batch_size=8, seed=3))

    ckpt_path = tmp_path / "model.pt"
    save_checkpoint(result, ckpt_path)
    assert ckpt_path.exists()

    reloaded = load_checkpoint(ckpt_path)
    with torch.no_grad():
        original_out = result.model.eval()(torch.from_numpy(images))
        reloaded_out = reloaded(torch.from_numpy(images))

    assert torch.allclose(original_out, reloaded_out, atol=1e-6)
