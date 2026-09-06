"""Tests for stage 05a (spectral signature analysis)."""

import numpy as np

from aegis_scan.detect.spectral import spectral_signature_scores


def make_activations_with_outliers(n_clean=90, n_outlier=10, dim=8, seed=0):
    """One class: `n_clean` samples near the origin, `n_outlier` samples
    pushed far out along a single shared direction -- the kind of
    one-directional fingerprint a learned backdoor trigger is expected
    to leave behind.
    """
    rng = np.random.default_rng(seed)
    clean = rng.normal(loc=0.0, scale=0.1, size=(n_clean, dim))
    direction = np.zeros(dim)
    direction[0] = 1.0
    outliers = rng.normal(loc=0.0, scale=0.1, size=(n_outlier, dim)) + 5.0 * direction

    activations = np.vstack([clean, outliers]).astype(np.float32)
    is_outlier = np.array([False] * n_clean + [True] * n_outlier)
    return activations, is_outlier


def test_outliers_score_higher_than_clean_samples():
    activations, is_outlier = make_activations_with_outliers()
    labels = np.zeros(len(activations), dtype=np.int64)  # single class

    scores = spectral_signature_scores(activations, labels)

    assert scores[is_outlier].mean() > scores[~is_outlier].mean() * 10


def test_scores_are_computed_independently_per_class():
    # class 0: has outliers. class 1: pure noise, no outliers at all.
    # class 1's scores shouldn't be inflated just because class 0 has a
    # strong signal -- each class gets its own SVD.
    outlier_acts, is_outlier = make_activations_with_outliers(n_clean=90, n_outlier=10, seed=1)
    clean_acts, _ = make_activations_with_outliers(n_clean=50, n_outlier=0, seed=2)

    activations = np.vstack([outlier_acts, clean_acts])
    labels = np.array([0] * len(outlier_acts) + [1] * len(clean_acts))

    scores = spectral_signature_scores(activations, labels)

    class0_outlier_scores = scores[: len(outlier_acts)][is_outlier]
    class1_scores = scores[len(outlier_acts) :]

    assert class0_outlier_scores.mean() > class1_scores.max()


def test_returns_one_score_per_sample():
    activations, _ = make_activations_with_outliers(n_clean=20, n_outlier=5)
    labels = np.zeros(len(activations), dtype=np.int64)

    scores = spectral_signature_scores(activations, labels)
    assert scores.shape == (len(activations),)


def test_single_sample_class_scores_zero_not_error():
    activations = np.random.default_rng(0).normal(size=(5, 4)).astype(np.float32)
    labels = np.array([0, 0, 0, 0, 1])  # class 1 has exactly one sample

    scores = spectral_signature_scores(activations, labels)
    assert scores[-1] == 0.0


def test_rejects_mismatched_lengths():
    activations = np.zeros((10, 4), dtype=np.float32)
    labels = np.zeros(5, dtype=np.int64)
    try:
        spectral_signature_scores(activations, labels)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
