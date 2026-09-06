"""Tests for stage 05b (activation clustering)."""

import numpy as np

from aegis_scan.detect.clustering import activation_clustering_flags


def make_two_cluster_class(n_majority=90, n_minority=10, dim=8, seed=0):
    """One class made of two well-separated sub-populations, at a realistic
    imbalance -- the majority standing in for genuine samples, the small
    minority standing in for a poisoned subset riding the same label.
    """
    rng = np.random.default_rng(seed)
    majority = rng.normal(loc=0.0, scale=0.3, size=(n_majority, dim))
    minority = rng.normal(loc=8.0, scale=0.3, size=(n_minority, dim))

    activations = np.vstack([majority, minority]).astype(np.float32)
    is_minority = np.array([False] * n_majority + [True] * n_minority)
    return activations, is_minority


def test_minority_cluster_gets_flagged():
    activations, is_minority = make_two_cluster_class()
    labels = np.zeros(len(activations), dtype=np.int64)

    flags = activation_clustering_flags(activations, labels, pca_components=5, seed=1)

    assert np.array_equal(flags, is_minority)


def test_flags_computed_independently_per_class():
    # class 0 has a real minority sub-population; class 1 is one uniform
    # blob and should come back with no flags at all
    act0, is_minority0 = make_two_cluster_class(n_majority=90, n_minority=10, seed=2)
    act1 = np.random.default_rng(3).normal(loc=0.0, scale=0.3, size=(40, 8)).astype(np.float32)

    activations = np.vstack([act0, act1])
    labels = np.array([0] * len(act0) + [1] * len(act1))

    flags = activation_clustering_flags(activations, labels, pca_components=5, seed=1)

    assert np.array_equal(flags[: len(act0)], is_minority0)
    assert not flags[len(act0) :].any()


def test_returns_boolean_array_matching_input_length():
    activations, _ = make_two_cluster_class(n_majority=20, n_minority=5)
    labels = np.zeros(len(activations), dtype=np.int64)

    flags = activation_clustering_flags(activations, labels)
    assert flags.shape == (len(activations),)
    assert flags.dtype == bool


def test_evenly_split_class_gets_no_flags():
    # a genuinely uniform population still gets cut ~50/50 by k=2 --
    # that even split is itself the signal there's no real minority to find
    activations = np.random.default_rng(4).normal(loc=0.0, scale=1.0, size=(80, 8)).astype(np.float32)
    labels = np.zeros(len(activations), dtype=np.int64)

    flags = activation_clustering_flags(activations, labels, pca_components=5, seed=1)
    assert not flags.any()


def test_tiny_class_is_skipped_without_error():
    activations = np.random.default_rng(0).normal(size=(6, 4)).astype(np.float32)
    labels = np.array([0, 0, 0, 0, 0, 1])  # class 1 has a single sample, below the size-4 minimum

    flags = activation_clustering_flags(activations, labels)
    assert not flags[-1]  # the tiny class's sample is never flagged


def test_rejects_mismatched_lengths():
    activations = np.zeros((10, 4), dtype=np.float32)
    labels = np.zeros(5, dtype=np.int64)
    try:
        activation_clustering_flags(activations, labels)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
