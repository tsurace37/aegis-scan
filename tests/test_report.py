"""Tests for stage 08 (report)."""

import math

from aegis_scan.report import (
    ATLAS_TECHNIQUES,
    NIST_AI_RMF_MEASURE,
    AssuranceReport,
    generate_report,
    render_markdown,
)


def _fake_evaluation(**overrides):
    base = {
        "poison_rate": 0.1,
        "clustering": None,
        "agreement": None,
        "spectral": None,
        "fused": None,
    }
    base.update(overrides)
    return base


def test_generate_report_includes_framework_mappings():
    evaluation = _fake_evaluation()
    report = generate_report(evaluation, dataset="synthetic_10pct")

    assert report.dataset == "synthetic_10pct"
    assert report.atlas_techniques == ATLAS_TECHNIQUES
    assert report.nist_ai_rmf_measure == NIST_AI_RMF_MEASURE
    assert report.evaluation is evaluation


def test_generate_report_atlas_ids_are_the_verified_ones():
    # Locks in the exact technique IDs this project verified against
    # MITRE ATLAS's own data (not a secondhand summary) -- a change here
    # should be deliberate, not an accidental edit.
    ids = {t["id"] for t in ATLAS_TECHNIQUES}
    assert ids == {"AML.T0020", "AML.T0059", "AML.T0018"}


def test_nist_measure_id_is_2_7():
    assert NIST_AI_RMF_MEASURE["id"] == "MEASURE 2.7"


def test_headline_prefers_fused_over_everything_else():
    evaluation = _fake_evaluation(
        fused={"auroc": 0.97, "average_precision": 0.9, "top_k_recall": 0.84},
        agreement={"tp": 1, "fp": 0, "fn": 1, "tn": 1, "tpr": 0.5, "fpr": 0.0, "precision": 1.0},
        clustering={"tp": 2, "fp": 1, "fn": 0, "tn": 1, "tpr": 1.0, "fpr": 0.5, "precision": 0.667},
        spectral={"auroc": 0.7, "average_precision": 0.5, "top_k_recall": 0.5},
    )
    report = generate_report(evaluation)
    assert "fused detector" in report.headline.lower()
    assert "0.970" in report.headline


def test_headline_falls_back_to_agreement_when_fused_missing():
    evaluation = _fake_evaluation(
        agreement={"tp": 1, "fp": 0, "fn": 1, "tn": 1, "tpr": 0.5, "fpr": 0.0, "precision": 1.0},
    )
    report = generate_report(evaluation)
    assert "both detectors" in report.headline.lower()


def test_headline_falls_back_to_clustering_when_only_clustering_present():
    evaluation = _fake_evaluation(
        clustering={"tp": 2, "fp": 1, "fn": 0, "tn": 1, "tpr": 1.0, "fpr": 0.5, "precision": 0.667},
    )
    report = generate_report(evaluation)
    assert "clustering alone" in report.headline.lower()


def test_headline_falls_back_to_spectral_when_only_spectral_present():
    evaluation = _fake_evaluation(spectral={"auroc": 0.85, "average_precision": 0.6, "top_k_recall": 0.7})
    report = generate_report(evaluation)
    assert "spectral signature analysis alone" in report.headline.lower()


def test_headline_handles_nothing_supplied():
    report = generate_report(_fake_evaluation())
    assert "no detector outputs" in report.headline.lower()


def test_headline_handles_nan_auroc_gracefully():
    evaluation = _fake_evaluation(fused={"auroc": float("nan"), "average_precision": float("nan"), "top_k_recall": 1.0})
    report = generate_report(evaluation)
    assert "not applicable" in report.headline.lower()


def test_render_markdown_includes_dataset_and_poison_rate():
    evaluation = _fake_evaluation(poison_rate=0.05)
    report = generate_report(evaluation, dataset="healthcare_5pct")
    md = render_markdown(report)

    assert "healthcare_5pct" in md
    assert "5.000%" in md
    assert "# aegis-scan Assurance Report" in md


def test_render_markdown_includes_atlas_and_nist_sections():
    report = generate_report(_fake_evaluation())
    md = render_markdown(report)

    assert "MITRE ATLAS mapping" in md
    assert "AML.T0020" in md
    assert "NIST AI RMF mapping" in md
    assert "MEASURE 2.7" in md


def test_render_markdown_only_includes_metrics_that_are_present():
    evaluation = _fake_evaluation(
        fused={"auroc": 0.97, "average_precision": 0.9, "top_k_recall": 0.84},
    )
    report = generate_report(evaluation)
    md = render_markdown(report)

    assert "Fused score" in md
    assert "Activation clustering:" not in md
    assert "Both detectors agree" not in md
    assert "Spectral signature analysis:" not in md
