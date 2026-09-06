"""
Stage 01 -- source datasets.

Loads two image-classification datasets into one common, "poisonable"
format:

  * a healthcare-imaging benchmark (PneumoniaMNIST -- a lightweight
    stand-in for the fuller ChestX-ray14 benchmark used in the
    methodology paper; same modality, same binary framing, orders of
    magnitude faster to iterate on while the pipeline is being built)
  * a non-healthcare benchmark (CIFAR-10), to test whether a detection
    method that works on chest X-rays generalizes to a completely
    different image domain

Both loaders return a `PoisonableDataset`: a plain container holding
images as a float32 NCHW numpy array (values in [0, 1]) and integer
labels, independent of whichever library the data came from. Stage 02
(poison injection) and stage 03 (training) are written against this one
shape, so they don't need to know or care which dataset produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PoisonableDataset:
    """A dataset in the common shape the rest of the pipeline expects.

    images: float32 array, shape (N, C, H, W), values in [0, 1]
    labels: int64 array, shape (N,)
    """

    images: np.ndarray
    labels: np.ndarray
    name: str
    class_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.images.ndim != 4:
            raise ValueError(f"images must be NCHW (4D), got shape {self.images.shape}")
        if len(self.images) != len(self.labels):
            raise ValueError(
                f"images and labels length mismatch: {len(self.images)} vs {len(self.labels)}"
            )
        if self.images.dtype != np.float32:
            raise ValueError(f"images must be float32, got {self.images.dtype}")

    def __len__(self) -> int:
        return len(self.labels)


def load_healthcare_dataset(split: str = "train", download_root: str = "./data/medmnist") -> PoisonableDataset:
    """Chest X-ray images, normal vs. pneumonia (PneumoniaMNIST, 28x28 grayscale).

    Standing in for ChestX-ray14 during pipeline development -- same
    healthcare-imaging modality named in the methodology paper, but small
    enough to train against repeatedly while stages 03-08 are being built.
    Swapping in the full ChestX-ray14 benchmark later only requires a new
    loader with this same return shape; nothing downstream changes.
    """
    from pathlib import Path

    from medmnist import PneumoniaMNIST

    Path(download_root).mkdir(parents=True, exist_ok=True)
    ds = PneumoniaMNIST(split=split, download=True, root=download_root)
    images = ds.imgs.astype(np.float32) / 255.0  # (N, 28, 28), uint8 -> float32 [0,1]
    images = images[:, None, :, :]  # add channel dim -> (N, 1, 28, 28)
    labels = ds.labels.squeeze().astype(np.int64)
    return PoisonableDataset(
        images=images,
        labels=labels,
        name="pneumonia_mnist",
        class_names=["normal", "pneumonia"],
    )


def load_synthetic_dataset(
    split: str = "train", n: int = 1000, size: int = 28, num_classes: int = 2, seed: int = 0
) -> PoisonableDataset:
    """Randomly generated images -- no download, no network required.

    Not one of the two real benchmarks the methodology paper evaluates
    against; this exists so the pipeline can be run and tested end to end
    (locally, in CI, or in a network-restricted environment) without
    depending on an external download succeeding.
    """
    rng = np.random.default_rng(seed if split == "train" else seed + 1)
    images = rng.random((n, 1, size, size), dtype=np.float32)
    labels = rng.integers(0, num_classes, size=n).astype(np.int64)
    return PoisonableDataset(
        images=images,
        labels=labels,
        name="synthetic",
        class_names=[f"class_{i}" for i in range(num_classes)],
    )


def load_benchmark_dataset(split: str = "train", download_root: str = "./data") -> PoisonableDataset:
    """CIFAR-10 -- the non-healthcare benchmark, to test cross-sector generalization.

    Deliberately CIFAR-10 rather than CIFAR-10-C: CIFAR-10-C is a
    corruption-robustness benchmark (blur, noise, weather effects), a
    different question from backdoor detection. The spectral-signature
    and activation-clustering literature this project builds on
    (Tran et al. 2018; Chen et al. 2018) both benchmark against plain
    CIFAR-10 with an injected trigger, which is what this loader
    provides poison injection something faithful to compare against.
    """
    import torchvision

    ds = torchvision.datasets.CIFAR10(root=download_root, train=(split == "train"), download=True)
    images = ds.data.astype(np.float32) / 255.0  # (N, 32, 32, 3), uint8 -> float32 [0,1]
    images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
    labels = np.array(ds.targets, dtype=np.int64)
    return PoisonableDataset(images=images, labels=labels, name="cifar10", class_names=list(ds.classes))
