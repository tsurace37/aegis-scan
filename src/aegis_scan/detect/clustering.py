"""
Stage 05b -- activation clustering (Chen et al., 2018, "Detecting
Backdoor Attacks on Deep Neural Networks by Activation Clustering",
arXiv:1811.03728).

A different premise from spectral signatures, deliberately run
independently rather than combined into one method: a backdoored
network still has to represent "this is a poisoned sample wearing class
c's label" *somehow* internally, even though it outputs class c just
like a genuinely-class-c sample would. Chen et al.'s idea is that this
produces two distinguishable sub-populations within a single class's
activations -- the genuine samples, and the poisoned ones riding along
on the trigger -- that a simple unsupervised clustering step can
separate, without ever being told which is which.

Also run **per class**, for the same reason as spectral signatures:
"this class secretly contains two different kinds of activation
patterns" is a per-class question. Clustering all classes together would
mostly just rediscover the classes themselves.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def activation_clustering_flags(
    activations: np.ndarray,
    labels: np.ndarray,
    pca_components: int = 10,
    seed: int = 42,
    max_minority_fraction: float = 0.35,
) -> np.ndarray:
    """Flag each sample's membership in the smaller of two per-class activation clusters.

    Within each class: reduce the activation vectors to `pca_components`
    dimensions with PCA (the raw activation vectors from a late
    conv layer -- 6,272-dimensional in this project's default
    configuration -- are far too high-dimensional and noisy for k-means'
    distance metric to behave well on directly), then run k-means with
    k=2. Chen et al.'s method flags the **smaller** of the two resulting
    clusters as the suspected-poisoned one -- following the same
    assumption stage 02's `poison_rate` argument encodes: a real
    poisoning attack alters a minority of a class's samples, not most of
    them, so the genuine population should be the majority cluster.

    k-means with k=2 always returns *some* split, even for a class with
    no real sub-population -- run it on one uniform blob and it still
    cuts it roughly in half. `max_minority_fraction` guards against
    mistaking that artifact for a real finding: the smaller cluster is
    only flagged when it's actually a *minority* (at most 35% of the
    class by default, comfortably above this project's tested poisoning
    rates of 1-10% but well below an even 50/50 split). A class that
    splits close to evenly gets no flags at all -- that even split is
    itself evidence there's no distinguishable poisoned subgroup to find.

    A class with fewer than 4 samples is skipped (too few points for a
    meaningful 2-cluster split) and none of its samples are flagged --
    this only matters for tiny toy datasets, not real ones.
    """
    if len(activations) != len(labels):
        raise ValueError(f"activations and labels length mismatch: {len(activations)} vs {len(labels)}")

    flags = np.zeros(len(activations), dtype=bool)

    for class_label in np.unique(labels):
        idx = np.flatnonzero(labels == class_label)
        if len(idx) < 4:
            continue

        class_activations = activations[idx].astype(np.float64)

        n_components = min(pca_components, len(idx) - 1, class_activations.shape[1])
        reduced = PCA(n_components=n_components, random_state=seed).fit_transform(class_activations)

        kmeans = KMeans(n_clusters=2, random_state=seed, n_init=10).fit(reduced)
        cluster_sizes = np.bincount(kmeans.labels_)
        minority_cluster = int(np.argmin(cluster_sizes))
        minority_fraction = cluster_sizes[minority_cluster] / cluster_sizes.sum()

        if minority_fraction > max_minority_fraction:
            continue  # the split isn't lopsided enough to call either side a minority

        flags[idx[kmeans.labels_ == minority_cluster]] = True

    return flags
