"""
Stage 02 -- poison injection.

Simulates a classic backdoor attack (BadNets-style, Gu et al. 2017): a
small fixed visual trigger is stamped onto a subset of training images,
and each poisoned image's label is flipped to a single target class.
Real poisoned healthcare data isn't available or desirable to go looking
for, so this project makes its own -- and because it makes the poison
itself, it also knows *exactly* which samples were altered. That list
(`PoisonResult.poison_mask`) is the ground truth this project's own
detection methods are never shown; it only reappears at evaluation
(stage 07), where detection accuracy gets checked against a known
answer instead of being self-reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..datasets.loaders import PoisonableDataset


@dataclass
class SquareTrigger:
    """A small solid square stamped in the bottom-right corner of an image.

    The simplest trigger pattern in the backdoor literature, and
    deliberately so for a first version: it's easy to verify by eye,
    easy to unit-test precisely, and both spectral-signature analysis
    and activation-clustering detection (stage 05) are reported in the
    literature to catch this pattern reliably -- which makes it a fair
    baseline before testing against subtler, harder-to-spot triggers.
    """

    size: int = 3
    intensity: float = 1.0
    margin: int = 2

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Return a copy of `image` (C, H, W) with the trigger stamped on."""
        stamped = image.copy()
        h, w = stamped.shape[-2:]
        y0 = h - self.margin - self.size
        x0 = w - self.margin - self.size
        if y0 < 0 or x0 < 0:
            raise ValueError(
                f"trigger (size={self.size}, margin={self.margin}) doesn't fit an "
                f"image of height/width {h}x{w}"
            )
        stamped[..., y0 : y0 + self.size, x0 : x0 + self.size] = self.intensity
        return stamped


@dataclass
class PoisonResult:
    images: np.ndarray  # (N, C, H, W) -- poisoned copy of the input images
    labels: np.ndarray  # (N,) -- poisoned copy of the input labels
    poison_mask: np.ndarray  # (N,) bool -- the ground truth: True where this sample was altered
    poison_rate: float  # actual achieved rate (may differ slightly from the requested one; see below)
    target_label: int


def inject_poison(
    dataset: PoisonableDataset,
    poison_rate: float,
    target_label: int,
    trigger: SquareTrigger | None = None,
    seed: int = 42,
) -> PoisonResult:
    """Poison `poison_rate` fraction of `dataset` and return the result plus ground truth.

    Only samples whose *original* label is not already `target_label` are
    eligible to be poisoned -- stamping a trigger on a sample that's
    already the target class doesn't test whether the trigger caused the
    misclassification, since nothing about its label actually changed.
    This mirrors how the backdoor-detection literature sets up the
    attack, and it's also why `poison_rate` here is computed against the
    full dataset, but the requested count is drawn only from eligible
    (non-target-class) samples -- if the target class is already common,
    the achieved rate can come in slightly under the requested one.

    Deterministic for a given `seed`: the same dataset, rate, target
    label, and seed always poison the same indices.
    """
    if not 0.0 <= poison_rate < 1.0:
        raise ValueError(f"poison_rate must be in [0, 1), got {poison_rate}")
    trigger = trigger or SquareTrigger()

    rng = np.random.default_rng(seed)
    n = len(dataset)
    n_requested = int(round(n * poison_rate))

    eligible = np.flatnonzero(dataset.labels != target_label)
    n_poison = min(n_requested, len(eligible))
    poison_idx = rng.choice(eligible, size=n_poison, replace=False)

    images = dataset.images.copy()
    labels = dataset.labels.copy()
    poison_mask = np.zeros(n, dtype=bool)

    for idx in poison_idx:
        images[idx] = trigger.apply(images[idx])
        labels[idx] = target_label
        poison_mask[idx] = True

    return PoisonResult(
        images=images,
        labels=labels,
        poison_mask=poison_mask,
        poison_rate=float(poison_mask.mean()),
        target_label=target_label,
    )
