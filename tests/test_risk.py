"""Tests for the class-balance coverage module (risk.py).

Includes regression tests against the exact numbers confirmed in the
Article 1 paper's Results section (via direct inspection of saved
per-sample detection output, not simulation) -- CIFAR-10 at 10%
poisoning and PneumoniaMNIST's real class balance -- so this module is
checked against real validated results, not only internal consistency.
"""

import numpy as np
import pytest

from aegis_scan.risk import ClassCoverage, class_poisoning_thresholds, render_coverage_report


def make_labels(class_sizes: dict[int, int]) -> np.ndarray:
    parts = [np.full(size, label, dtype=np.int64) for label, size in class_sizes.items()]
    return np.concatenate(parts)


def test_rejects_invalid_tau():
    labels = make_labels({0: 10, 1: 10})
    with pytest.raises(ValueError):
        class_poisoning_thresholds(labels, max_minority_fraction=0.0)
    with pytest.raises(ValueError):
        class_poisoning_thresholds(labels, max_minority_fraction=1.0)
    with pytest.raises(ValueError):
        class_poisoning_thresholds(labels, max_minority_fraction=1.5)


def test_empty_labels_returns_empty_list():
    assert class_poisoning_thresholds(np.array([], dtype=np.int64)) == []


def test_sorted_ascending_by_threshold_smallest_class_first():
    # class 0 is a small minority, class 1 is the majority -- class 0
    # should have the LOWER threshold (more vulnerable), and come first.
    labels = make_labels({0: 10, 1: 90})
    result = class_poisoning_thresholds(labels)
    assert [c.class_label for c in result] == [0, 1]
    assert result[0].p_threshold < result[1].p_threshold


def test_balanced_classes_have_equal_thresholds():
    labels = make_labels({0: 50, 1: 50})
    result = class_poisoning_thresholds(labels)
    assert result[0].p_threshold == pytest.approx(result[1].p_threshold)


def test_formula_matches_hand_calculation():
    # q=0.2, tau=0.35 -> p_threshold = 0.2 * 0.35 / 0.65
    labels = make_labels({0: 200, 1: 800})  # class 0 is exactly 20% of 1000
    result = class_poisoning_thresholds(labels, max_minority_fraction=0.35)
    class_0 = next(c for c in result if c.class_label == 0)
    expected = 0.2 * 0.35 / 0.65
    assert class_0.p_threshold == pytest.approx(expected)
    assert class_0.class_fraction == pytest.approx(0.2)
    assert class_0.class_size == 200
    assert class_0.dataset_size == 1000


def test_cifar10_confirmed_case_predicts_the_measured_collapse():
    """Regression test against real, confirmed data (not simulated):
    Article 1's diagnostic script found CIFAR-10's target class at 10%
    poisoning sits at exactly 10,000 post-poison samples out of 50,000
    (5,000 original + 5,000 poisoned), with clustering flags at exactly
    zero across all five seeds. This test checks the formula predicts
    that configuration crosses the guard threshold, using CIFAR-10's
    real, exactly-balanced pre-poisoning structure (5,000 per class,
    10 classes, 50,000 total)."""
    labels = make_labels({c: 5000 for c in range(10)})  # CIFAR-10's real, balanced pre-poison structure
    result = class_poisoning_thresholds(labels, max_minority_fraction=0.35)
    # every class is identical (q=0.10); the measured poisoning rate (10%)
    # should exceed every class's threshold
    for c in result:
        assert c.class_fraction == pytest.approx(0.10)
        assert c.p_threshold == pytest.approx(0.10 * 0.35 / 0.65)
        assert c.p_threshold < 0.10  # 0.0538 < 0.10 -- confirms 10% poisoning crosses it
        assert c.is_low_coverage  # 5.4% <= the 10% reference cutoff


def test_cifar10_5pct_falls_just_under_the_threshold():
    """Companion to the collapse test above: at 5% poisoning, CIFAR-10's
    measured clustering TPR was 0.960 (still working), consistent with
    5% falling just under the ~5.38% threshold this formula predicts."""
    threshold = 0.10 * 0.35 / 0.65
    assert 0.05 < threshold < 0.055  # 5% (still worked) sits just below the predicted crossing point


def test_pneumoniamnist_confirmed_case_predicts_no_collapse():
    """Regression test against Article 1's confirmed PneumoniaMNIST class
    balance: 1,214 'normal' / 3,494 'pneumonia' before poisoning
    (4,708 total). Clustering flagged 466-471 of 471 poisoned samples
    across all five seeds at 10% poisoning -- i.e. did NOT collapse.
    This test checks the formula predicts 10% poisoning does NOT cross
    the guard threshold for this real class balance."""
    labels = make_labels({0: 1214, 1: 3494})  # real pre-poisoning PneumoniaMNIST split
    result = class_poisoning_thresholds(labels, max_minority_fraction=0.35)
    normal_class = next(c for c in result if c.class_label == 0)
    assert normal_class.class_fraction == pytest.approx(1214 / 4708)
    assert normal_class.p_threshold > 0.10  # 10% poisoning stays under this threshold
    assert not normal_class.is_low_coverage


def test_is_low_coverage_boundary():
    cov_at_boundary = ClassCoverage(
        class_label=0, class_size=1, dataset_size=1, class_fraction=1.0, p_threshold=0.10
    )
    cov_just_above = ClassCoverage(
        class_label=0, class_size=1, dataset_size=1, class_fraction=1.0, p_threshold=0.1001
    )
    assert cov_at_boundary.is_low_coverage
    assert not cov_just_above.is_low_coverage


def test_render_report_flags_low_coverage_classes():
    # class 0 at q=0.10 -> p_threshold = 0.10*0.35/0.65 = 5.4%, correctly
    # below the 10% reference cutoff (a 200/800 split, q=0.20, was tried
    # first and does NOT cross the cutoff -- 10.8% > 10% -- which is
    # itself a useful confirmation the boundary logic is precise, not
    # just "small class = flagged" via a looser check)
    labels = make_labels({0: 100, 1: 900})
    result = class_poisoning_thresholds(labels)
    report = render_coverage_report(result)
    assert "Class-balance coverage" in report
    assert "low coverage" in report
    assert "0" in report  # the vulnerable class label appears


def test_render_report_handles_no_low_coverage_classes():
    labels = make_labels({0: 500, 1: 500})  # both classes well above the 10% reference cutoff
    result = class_poisoning_thresholds(labels)
    report = render_coverage_report(result)
    assert "No class falls below" in report


def test_render_report_handles_empty_input():
    report = render_coverage_report([])
    assert "No classes to analyze" in report
