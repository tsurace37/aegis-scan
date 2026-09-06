"""Tests for stage 07 (evaluate)."""

import numpy as np
import pytest

from aegis_scan.evaluate import binary_detection_metrics, evaluate, score_based_metrics


def test_binary_detection_metrics_perfect_detector():
    poison_mask = np.array([True, True, False, False, False])
    flags = np.array([True, True, False, False, False])

    m = binary_detection_metrics(flags, poison_mask)

    assert m.tp == 2 and m.fp == 0 and m.fn == 0 and m.tn == 3
    assert m.tpr == pytest.approx(1.0)
    assert m.fpr == pytest.approx(0.0)
    assert m.precision == pytest.approx(1.0)


def test_binary_detection_metrics_flags_nothing():
    poison_mask = np.array([True, True, False, False])
    flags = np.array([False, False, False, False])

    m = binary_detection_metrics(flags, poison_mask)

    assert m.tp == 0 and m.fn == 2 and m.fp == 0 and m.tn == 2
    assert m.tpr == pytest.approx(0.0)
    assert m.fpr == pytest.approx(0.0)
    # precision is undefined (nothing flagged) -- 0/0
    assert np.isnan(m.precision)


def test_binary_detection_metrics_all_clean_gives_nan_tpr():
    # no truly poisoned samples at all -- tpr (recall) is undefined
    poison_mask = np.array([False, False, False])
    flags = np.array([True, False, False])

    m = binary_detection_metrics(flags, poison_mask)

    assert np.isnan(m.tpr)
    assert m.fpr == pytest.approx(1 / 3)  # fp=1, tn=2
    assert m.precision == pytest.approx(0.0)


def test_binary_detection_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        binary_detection_metrics(np.zeros(3, dtype=bool), np.zeros(4, dtype=bool))


def test_score_based_metrics_perfect_separation():
    poison_mask = np.array([False, False, False, True, True])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.8])  # both poisoned samples score highest

    m = score_based_metrics(scores, poison_mask)

    assert m.auroc == pytest.approx(1.0)
    assert m.average_precision == pytest.approx(1.0)
    assert m.top_k_recall == pytest.approx(1.0)


def test_score_based_metrics_random_scores_worse_than_perfect():
    n = 200
    rng = np.random.default_rng(0)
    poison_mask = np.zeros(n, dtype=bool)
    poison_mask[:20] = True
    scores = rng.random(n)  # unrelated to poison_mask

    m = score_based_metrics(scores, poison_mask)

    assert 0.0 <= m.auroc <= 1.0
    assert m.auroc < 0.9  # should be roughly 0.5, well below a strong detector


def test_score_based_metrics_all_poisoned_returns_nan_not_error():
    poison_mask = np.array([True, True, True])
    scores = np.array([0.1, 0.5, 0.9])

    m = score_based_metrics(scores, poison_mask)

    assert np.isnan(m.auroc)
    assert np.isnan(m.average_precision)
    assert m.top_k_recall == pytest.approx(1.0)  # k == n, everything flagged is poisoned


def test_score_based_metrics_none_poisoned_returns_nan_not_error():
    poison_mask = np.array([False, False, False])
    scores = np.array([0.1, 0.5, 0.9])

    m = score_based_metrics(scores, poison_mask)

    assert np.isnan(m.auroc)
    assert np.isnan(m.average_precision)
    assert np.isnan(m.top_k_recall)  # k == 0, nothing to recall


def test_score_based_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        score_based_metrics(np.zeros(3), np.zeros(4, dtype=bool))


def test_evaluate_with_no_optional_inputs_still_reports_poison_rate():
    poison_mask = np.array([True, False, False, False])

    report = evaluate(poison_mask)

    assert report.poison_rate == pytest.approx(0.25)
    assert report.clustering is None
    assert report.agreement is None
    assert report.spectral is None
    assert report.fused is None


def test_evaluate_wires_clustering_and_agreement():
    poison_mask = np.array([True, True, False, False])
    clustering_flags = np.array([True, False, False, False])
    agreement_flags = np.array([True, True, False, False])

    report = evaluate(poison_mask, clustering_flags=clustering_flags, agreement_flags=agreement_flags)

    assert report.clustering.tp == 1
    assert report.agreement.tp == 2
    assert report.spectral is None
    assert report.fused is None


def test_evaluate_wires_fused_scores():
    poison_mask = np.array([False, False, True, True])
    fused_scores = np.array([0.1, 0.2, 0.8, 0.9])

    report = evaluate(poison_mask, fused_scores=fused_scores)

    assert report.fused.auroc == pytest.approx(1.0)
    assert report.clustering is None


def test_evaluate_spectral_requires_labels():
    poison_mask = np.array([True, False, True, False])
    spectral_scores = np.array([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError):
        evaluate(poison_mask, spectral_scores=spectral_scores)


def test_evaluate_spectral_uses_per_class_percentile_rank():
    # class 0: sample 0 (poisoned) has the lowest raw score but is still
    # the top-ranked sample *within its class*, so per-class ranking
    # should let it be perfectly detected even though a naive global
    # threshold on raw scores would miss it entirely.
    labels = np.array([0, 0, 1, 1])
    poison_mask = np.array([True, False, True, False])
    spectral_scores = np.array([2.0, 1.0, 100.0, 1.0])

    report = evaluate(poison_mask, labels=labels, spectral_scores=spectral_scores)

    assert report.spectral.auroc == pytest.approx(1.0)
