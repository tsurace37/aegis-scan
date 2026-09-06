"""
Stage 05a -- spectral signature analysis (Tran, Li & Madry, 2018,
"Spectral Signatures in Backdoor Attacks", arXiv:1811.00636).

The premise: a backdoor trigger gives the network an easy, unnatural
shortcut ("this square pattern means class 0") that clean samples never
rely on. Learning that shortcut leaves a detectable fingerprint in a late
layer's activations -- poisoned samples end up correlated with each
other along one particular direction in activation space that clean
samples of the same class don't share. This module finds that direction
and scores every sample by how strongly it lies along it.

Deliberately run **per class, independently**: the fingerprint is a
property of "the target class's activations look unnaturally
one-directional," which only shows up when a class's samples are
examined on their own. Pooling every class into one shared SVD would
dilute a genuine one-class signal against the much larger variance
between unrelated classes.
"""

from __future__ import annotations

import numpy as np


def spectral_signature_scores(activations: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return one outlier score per sample -- higher means more suspicious.

    For each class: center that class's activation vectors, take the
    *top right singular vector* of the centered matrix (the direction of
    greatest variance -- computed via SVD rather than an explicit
    covariance-matrix eigendecomposition, which is both faster and more
    numerically stable for the tall, thin matrices a class's activations
    usually form), and score each sample by its squared projection onto
    that direction. This follows Tran et al.'s method directly: if a
    subset of samples was poisoned, that subset is usually exactly what
    pulls the top singular direction into existence in the first place,
    so it's also what scores highest against it.

    Scores are only comparable *within* a class -- a score of 4.0 in
    class 0 and 4.0 in class 1 aren't measuring the same thing, since
    each class gets its own direction and its own scale. Stage 06
    (fusion) is where per-class scores get reconciled into one
    per-sample number.
    """
    if len(activations) != len(labels):
        raise ValueError(f"activations and labels length mismatch: {len(activations)} vs {len(labels)}")

    scores = np.zeros(len(activations), dtype=np.float64)

    for class_label in np.unique(labels):
        idx = np.flatnonzero(labels == class_label)
        if len(idx) < 2:
            # a single-sample class has no variance to find a direction
            # in -- leave its score at 0 rather than dividing by nothing
            continue

        class_activations = activations[idx].astype(np.float64)
        centered = class_activations - class_activations.mean(axis=0, keepdims=True)

        # full_matrices=False: only the top min(n_samples, n_features)
        # singular vectors are needed, not the full square factorization
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        top_direction = vt[0]

        projection = centered @ top_direction
        scores[idx] = projection**2

    return scores
