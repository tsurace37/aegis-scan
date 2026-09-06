"""Tests for stage 06 (fusion)."""

import numpy as np
import pytest

from aegis_scan.fuse import percentile_rank_per_class, fuse_scores


def test_percentile_rank_is_zero_to_one_within_each_class():
    values = np.array([10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0])
    labels = np.array([0, 0, 0, 0, 1, 1, 1])

    ranks = percentile_rank_per_class(values, labels)

    assert ranks.min() >= 0.0
    assert ranks.max() <= 1.0
    # smallest and largest in each class should land exactly at the ends
    assert ranks[0] == 0.0  # value 10, smallest in class 0
    assert ranks[3] == 1.0  # value 40, largest in class 0
    assert ranks[4] == 0.0  # value 1, smallest in class 1
    assert ranks[6] == 1.0  # value 3, largest in class 1


def test_percentile_rank_single_sample_class_is_zero_not_error():
    values = np.array([5.0, 1.0, 2.0])
    labels = np.array([0, 1, 1])  # class 0 has exactly one sample

    ranks = percentile_rank_per_class(values, labels)
    assert ranks[0] == 0.0


def test_sample_flagged_by_both_scores_higher_than_flagged_by_neither():
    n = 20
    labels = np.zeros(n, dtype=np.int64)
    spectral_scores = np.zeros(n)
    clustering_flags = np.zeros(n, dtype=bool)

    # sample 0: high spectral score AND clustering flag -- should score highest
    spectral_scores[0] = 100.0
    clustering_flags[0] = True
    # sample 1: neither -- should score lowest
    spectral_scores[1] = 0.01

    result = fuse_scores(spectral_scores, clustering_flags, labels)

    assert result.fused_scores[0] > result.fused_scores[1]
    assert result.fused_scores[0] == pytest.approx(1.0)  # top spectral rank (1.0) + flagged (1.0), averaged


def test_agreement_requires_both_detectors():
    n = 10
    labels = np.zeros(n, dtype=np.int64)
    spectral_scores = np.arange(n, dtype=np.float64)  # ranks 0.0 .. 1.0 evenly spaced
    clustering_flags = np.zeros(n, dtype=bool)
    clustering_flags[-1] = True  # only the top-ranked spectral sample is also clustering-flagged

    result = fuse_scores(spectral_scores, clustering_flags, labels, spectral_percentile_cutoff=0.9)

    assert result.agreement[-1]  # top sample: high spectral rank AND clustering flag -> agreement
    assert not result.agreement[:-1].any()  # nobody else agrees
    assert result.fraction_flagged_by_both == pytest.approx(1 / n)


def test_fraction_flagged_by_either_is_at_least_flagged_by_both():
    n = 30
    rng = np.random.default_rng(0)
    labels = np.zeros(n, dtype=np.int64)
    spectral_scores = rng.random(n)
    clustering_flags = rng.random(n) > 0.7

    result = fuse_scores(spectral_scores, clustering_flags, labels)

    assert result.fraction_flagged_by_either >= result.fraction_flagged_by_both


def test_mean_fused_score_matches_manual_computation():
    labels = np.array([0, 0, 0, 1, 1, 1])
    spectral_scores = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    clustering_flags = np.array([False, False, True, True, False, False])

    result = fuse_scores(spectral_scores, clustering_flags, labels)
    assert result.mean_fused_score == pytest.approx(result.fused_scores.mean())


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        fuse_scores(
            spectral_scores=np.zeros(5),
            clustering_flags=np.zeros(5, dtype=bool),
            labels=np.zeros(4, dtype=np.int64),
        )
