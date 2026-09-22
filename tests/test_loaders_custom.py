"""Tests for `load_custom_dataset` (the generalization path for any
dataset not covered by the built-in healthcare/benchmark/synthetic
loaders). Uses small locally-written .npz files so these run fast and
offline.
"""

import numpy as np
import pytest

from aegis_scan.datasets.loaders import load_custom_dataset


def write_npz(path, images, labels):
    np.savez_compressed(path, images=images, labels=labels)


def test_loads_a_well_formed_npz(tmp_path):
    images = np.random.default_rng(0).random((20, 3, 24, 24), dtype=np.float32)
    labels = np.array([0, 1, 2] * 6 + [0, 1], dtype=np.int64)
    path = tmp_path / "mydata.npz"
    write_npz(path, images, labels)

    ds = load_custom_dataset(str(path))

    assert len(ds) == 20
    assert ds.images.shape == (20, 3, 24, 24)
    assert ds.images.dtype == np.float32
    assert ds.labels.dtype == np.int64
    assert ds.name == "mydata"
    assert ds.class_names == ["0", "1", "2"]


def test_split_argument_is_accepted_but_has_no_effect(tmp_path):
    images = np.random.default_rng(1).random((10, 1, 16, 16), dtype=np.float32)
    labels = np.zeros(10, dtype=np.int64)
    path = tmp_path / "d.npz"
    write_npz(path, images, labels)

    train_ds = load_custom_dataset(str(path), split="train")
    test_ds = load_custom_dataset(str(path), split="test")
    assert np.array_equal(train_ds.images, test_ds.images)


def test_coerces_dtypes_that_are_close_but_not_exact(tmp_path):
    # int labels stored as int32, images stored as float64 -- both
    # common outputs of an ordinary preprocessing script -- should be
    # coerced rather than rejected.
    images = np.random.default_rng(2).random((5, 1, 8, 8)).astype(np.float64)
    labels = np.array([0, 1, 0, 1, 1], dtype=np.int32)
    path = tmp_path / "coerce.npz"
    write_npz(path, images, labels)

    ds = load_custom_dataset(str(path))
    assert ds.images.dtype == np.float32
    assert ds.labels.dtype == np.int64


def test_missing_images_key_raises_clear_error(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, labels=np.zeros(5, dtype=np.int64))

    with pytest.raises(ValueError, match="'images' and 'labels'"):
        load_custom_dataset(str(path))


def test_missing_labels_key_raises_clear_error(tmp_path):
    path = tmp_path / "bad2.npz"
    np.savez_compressed(path, images=np.zeros((5, 1, 8, 8), dtype=np.float32))

    with pytest.raises(ValueError, match="'images' and 'labels'"):
        load_custom_dataset(str(path))


def test_malformed_shape_raises_via_poisonable_dataset_validation(tmp_path):
    # 3D images (missing the channel dim) -- PoisonableDataset's own
    # __post_init__ validation should catch this, not a numpy crash
    # several stages later.
    path = tmp_path / "bad_shape.npz"
    np.savez_compressed(
        path,
        images=np.zeros((5, 8, 8), dtype=np.float32),
        labels=np.zeros(5, dtype=np.int64),
    )

    with pytest.raises(ValueError, match="NCHW"):
        load_custom_dataset(str(path))
