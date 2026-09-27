"""Tests for stage 08 (report)."""

import math

from aegis_scan.report import (
    ATLAS_TECHNIQUES,
    NIST_AI_RMF_MEASURE,
    AssuranceReport,
    generate_report,
    render_html,
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


def test_report_omits_coverage_section_when_no_labels_supplied():
    """Default behavior (no labels) must stay exactly as before this
    feature was added -- no coverage section, no class_coverage field
    populated."""
    report = generate_report(_fake_evaluation())
    assert report.class_coverage is None
    assert "Class-balance coverage" not in render_markdown(report)


def test_report_includes_coverage_section_when_labels_supplied():
    """Integration test for the report.py <-> risk.py wiring itself,
    not just each module in isolation: a small, deliberately imbalanced
    label array should produce both a populated class_coverage list and
    a rendered section naming the vulnerable class."""
    import numpy as np

    labels = np.array([0] * 10 + [1] * 90)  # class 0 is a 10% minority -> low coverage
    report = generate_report(_fake_evaluation(), labels=labels)

    assert report.class_coverage is not None
    assert len(report.class_coverage) == 2
    assert report.class_coverage[0].class_label == 0  # smallest/most vulnerable class sorts first

    md = render_markdown(report)
    assert "Class-balance coverage" in md
    assert "low coverage" in md


def test_render_html_is_a_self_contained_document():
    report = generate_report(_fake_evaluation(poison_rate=0.05), dataset="healthcare_5pct")
    doc = render_html(report)

    assert doc.startswith("<!doctype html>")
    assert "<html" in doc and "</html>" in doc
    assert "<style>" in doc  # styling is inlined, not linked
    assert "http://" not in doc and "https://" not in doc  # nothing loaded from the network
    assert "healthcare_5pct" in doc
    assert "5.000%" in doc


def test_render_html_includes_atlas_and_nist_sections():
    report = generate_report(_fake_evaluation())
    doc = render_html(report)

    assert "MITRE ATLAS mapping" in doc
    assert "AML.T0020" in doc
    assert "NIST AI RMF mapping" in doc
    assert "MEASURE 2.7" in doc


def test_render_html_only_includes_metrics_that_are_present():
    evaluation = _fake_evaluation(
        fused={"auroc": 0.97, "average_precision": 0.9, "top_k_recall": 0.84},
    )
    report = generate_report(evaluation)
    doc = render_html(report)

    assert "Fused score" in doc
    assert "Activation clustering:" not in doc
    assert "Both detectors agree" not in doc
    assert "Spectral signature analysis:" not in doc


def test_render_html_omits_coverage_section_when_no_labels_supplied():
    report = generate_report(_fake_evaluation())
    assert "Class-balance coverage" not in render_html(report)


def test_render_html_includes_coverage_table_when_labels_supplied():
    import numpy as np

    labels = np.array([0] * 10 + [1] * 90)  # class 0 is a 10% minority -> low coverage
    report = generate_report(_fake_evaluation(), labels=labels)
    doc = render_html(report)

    assert "Class-balance coverage" in doc
    assert "<table>" in doc
    assert "low coverage" in doc
    assert 'class="low-coverage"' in doc


def test_render_html_escapes_untrusted_dataset_name():
    """The dataset name and headline flow into the HTML unescaped strings
    would be a stored-XSS-in-a-local-file bug; a dataset name containing
    markup must come out escaped, not as live HTML."""
    report = generate_report(_fake_evaluation(), dataset="<script>alert(1)</script>")
    doc = render_html(report)

    assert "<script>alert(1)</script>" not in doc
    assert "&lt;script&gt;" in doc


def test_render_html_and_render_markdown_agree_on_headline_and_poison_rate():
    """Both renderers are views of the same AssuranceReport -- they should
    never disagree on the actual numbers, only on formatting."""
    evaluation = _fake_evaluation(
        poison_rate=0.1,
        fused={"auroc": 0.913, "average_precision": 0.8, "top_k_recall": 0.75},
    )
    report = generate_report(evaluation, dataset="synthetic_10pct")

    md = render_markdown(report)
    doc = render_html(report)

    assert "10.000%" in md and "10.000%" in doc
    assert "0.913" in md and "0.913" in doc
