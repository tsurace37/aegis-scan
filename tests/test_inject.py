"""Tests for stage 02 (poison injection). Uses a small synthetic dataset
so these run fast, offline, and without depending on stage 01's real
downloads.
"""

import numpy as np

from aegis_scan.datasets.loaders import PoisonableDataset
from aegis_scan.poison.inject import SquareTrigger, inject_poison


def make_toy_dataset(n=200, size=16, seed=0) -> PoisonableDataset:
    rng = np.random.default_rng(seed)
    images = rng.random((n, 1, size, size), dtype=np.float32)
    labels = rng.integers(0, 2, size=n).astype(np.int64)
    return PoisonableDataset(images=images, labels=labels, name="toy")


def test_poison_rate_is_approximately_correct():
    ds = make_toy_dataset(n=500)
    result = inject_poison(ds, poison_rate=0.1, target_label=0, seed=1)
    assert abs(result.poison_rate - 0.1) < 0.02


def test_poisoned_samples_get_target_label():
    ds = make_toy_dataset(n=300)
    result = inject_poison(ds, poison_rate=0.2, target_label=1, seed=2)
    assert np.all(result.labels[result.poison_mask] == 1)


def test_only_trigger_region_changes():
    ds = make_toy_dataset(n=50)
    trigger = SquareTrigger(size=3, margin=1)
    result = inject_poison(ds, poison_rate=0.3, target_label=0, trigger=trigger, seed=3)
    poisoned_idx = np.flatnonzero(result.poison_mask)
    assert len(poisoned_idx) > 0

    idx = poisoned_idx[0]
    diff = np.abs(result.images[idx] - ds.images[idx])
    changed_pixels = np.count_nonzero(diff.sum(axis=0))
    assert changed_pixels == trigger.size * trigger.size


def test_clean_samples_unchanged():
    ds = make_toy_dataset(n=50)
    result = inject_poison(ds, poison_rate=0.3, target_label=0, seed=4)
    clean_idx = np.flatnonzero(~result.poison_mask)
    assert np.allclose(result.images[clean_idx], ds.images[clean_idx])
    assert np.array_equal(result.labels[clean_idx], ds.labels[clean_idx])


def test_reproducible_with_same_seed():
    ds = make_toy_dataset(n=200)
    r1 = inject_poison(ds, poison_rate=0.1, target_label=0, seed=7)
    r2 = inject_poison(ds, poison_rate=0.1, target_label=0, seed=7)
    assert np.array_equal(r1.poison_mask, r2.poison_mask)


def test_never_poisons_samples_already_the_target_class():
    ds = make_toy_dataset(n=300)
    result = inject_poison(ds, poison_rate=0.2, target_label=0, seed=5)
    poisoned_idx = np.flatnonzero(result.poison_mask)
    # every poisoned sample's ORIGINAL label must not already be the target
    assert np.all(ds.labels[poisoned_idx] != 0)


def test_rejects_invalid_poison_rate():
    ds = make_toy_dataset(n=50)
    for bad_rate in (-0.1, 1.0, 1.5):
        try:
            inject_poison(ds, poison_rate=bad_rate, target_label=0, seed=1)
            raise AssertionError(f"expected ValueError for poison_rate={bad_rate}")
        except ValueError:
            pass
