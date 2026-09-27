"""
Stage 08 -- translate stage 07's evaluation numbers into a report a
security or compliance reviewer can read without first learning this
project's internals: MITRE ATLAS technique IDs for the attack being
tested against, and the NIST AI RMF "Measure" function subcategory this
kind of testing satisfies.

This stage runs no new analysis of its own. Every number in the report
traces back to something stage 07 already computed against ground
truth -- this is a translation layer, not a new source of evidence.
Framework IDs and quoted text below were checked against MITRE ATLAS's
own technique data (atlas.mitre.org / mitre-atlas/atlas-data) and NIST
AI 100-1 directly, not secondhand summaries, because getting a citation
wrong in a security report is worse than leaving it out.
"""

from __future__ import annotations

import html as html_lib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .risk import ClassCoverage, class_poisoning_thresholds, render_coverage_report

# MITRE ATLAS techniques relevant to what this project tests for. Each
# entry's `relevance` says precisely how it maps to aegis-scan's own
# attack model (a BadNets-style trigger stamped into training images,
# with the label flipped) -- not a generic "this technique exists"
# gesture, since a too-loose mapping is exactly what NIST AI 100-1's
# own guidance and USCIS's 2025 policy update warn against.
ATLAS_TECHNIQUES = [
    {
        "id": "AML.T0020",
        "name": "Poison Training Data",
        "relevance": (
            "The primary technique aegis-scan is built around: stage 02 "
            "simulates this attack directly (modifying a subset of training "
            "images and their labels), and stages 05-07 are the detection "
            "and validation layer for it."
        ),
    },
    {
        "id": "AML.T0059",
        "name": "Erode Dataset Integrity",
        "relevance": (
            "A poisoned dataset is a targeted special case of this broader "
            "technique: the altered subset isn't random corruption, it's "
            "crafted to survive training and re-emerge as a backdoor -- "
            "which is why detecting it needs more than basic data-quality "
            "checks."
        ),
    },
    {
        "id": "AML.T0018",
        "name": "Backdoor ML Model",
        "relevance": (
            "The end state of a successful poisoning attack: a persistent, "
            "hidden change in the trained model's behavior. aegis-scan "
            "doesn't inspect model weights directly the way this "
            "technique's 'Poison ML Model' sub-technique (AML.T0018.000) "
            "describes -- it infers the same outcome indirectly, from how "
            "poisoned training data reshapes a layer's activations. Name "
            "verified directly against current MITRE ATLAS technique data "
            "(atlas.mitre.org) as of this fix, since an earlier version of "
            "this file used a non-standard label."
        ),
    },
]

# NIST AI RMF 1.0 (NIST AI 100-1), Table 3, MEASURE function.
NIST_AI_RMF_MEASURE = {
    "id": "MEASURE 2.7",
    "text": "AI system security and resilience -- as identified in the MAP function -- are evaluated and documented.",
    "how_this_report_satisfies_it": (
        "This report is exactly that evaluation and documentation, scoped "
        "to one named, testable security risk (data poisoning / backdoor "
        "attacks): quantified detection performance -- TPR, FPR, AUROC -- "
        "measured against a known ground truth, not a policy statement "
        "that testing happened somewhere. A governance framework can point "
        "an auditor here as the technical evidence MEASURE 2.7 asks an "
        "organization to produce."
    ),
}


def _auroc_band(auroc: float) -> str:
    """A standard, widely-used qualitative reading of an AUROC value -- not this project's own invention."""
    if auroc != auroc:  # NaN
        return "not applicable"
    if auroc >= 0.9:
        return "excellent"
    if auroc >= 0.8:
        return "good"
    if auroc >= 0.7:
        return "fair"
    return "poor"


def _build_headline(evaluation: dict[str, Any]) -> str:
    """Pick the strongest available evidence and state it in one plain sentence.

    Prefers the fused continuous score (stage 06's combined signal) since
    it's already been shown to outperform either detector alone; falls
    back to whatever was actually supplied to stage 07, since --detect
    and --fuse are both optional there.
    """
    fused = evaluation.get("fused")
    if fused is not None:
        return (
            f"The fused detector (spectral signature analysis + activation clustering combined) "
            f"achieved AUROC={fused['auroc']:.3f} ({_auroc_band(fused['auroc'])}) and caught "
            f"{fused['top_k_recall']:.1%} of truly poisoned samples when flagging exactly as many "
            f"samples as were actually poisoned."
        )
    agreement = evaluation.get("agreement")
    if agreement is not None:
        return (
            f"Samples flagged by both detectors (the highest-confidence signal) were correct "
            f"{agreement['precision']:.1%} of the time (precision), catching {agreement['tpr']:.1%} "
            f"of truly poisoned samples."
        )
    clustering = evaluation.get("clustering")
    if clustering is not None:
        return (
            f"Activation clustering alone caught {clustering['tpr']:.1%} of truly poisoned samples "
            f"with a {clustering['fpr']:.1%} false-positive rate."
        )
    spectral = evaluation.get("spectral")
    if spectral is not None:
        return (
            f"Spectral signature analysis alone achieved AUROC={spectral['auroc']:.3f} "
            f"({_auroc_band(spectral['auroc'])}) against ground truth."
        )
    return "No detector outputs were supplied to stage 07 -- this report only reflects the dataset's poison rate."


@dataclass
class AssuranceReport:
    generated_at: str
    dataset: str
    evaluation: dict[str, Any]
    atlas_techniques: list[dict[str, str]]
    nist_ai_rmf_measure: dict[str, str]
    headline: str
    class_coverage: list[ClassCoverage] | None = None


def generate_report(
    evaluation: dict[str, Any],
    *,
    dataset: str = "unspecified",
    labels: np.ndarray | None = None,
) -> AssuranceReport:
    """Wrap stage 07's evaluation dict (its saved JSON, loaded back in) with framework mappings and a headline.

    `evaluation` is exactly the dict `aegis-scan evaluate` saves: `poison_rate`
    plus optional `clustering`/`agreement`/`spectral`/`fused` metric dicts.

    `labels` is optional and unrelated to stage 07's output -- when
    supplied (e.g. read back from `aegis-scan inject`'s saved `labels`
    array), a class-balance coverage section (see `risk.py`) is computed
    and included. This is deliberately independent of everything else in
    this report: it doesn't need a trained model, activations, or a
    detection run, only the dataset's class distribution, so it's
    available even for a report generated before those stages ever run.
    Omitting `labels` (the default) preserves this function's existing
    behavior exactly, for callers that don't have that array on hand.
    """
    return AssuranceReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        dataset=dataset,
        evaluation=evaluation,
        atlas_techniques=ATLAS_TECHNIQUES,
        nist_ai_rmf_measure=NIST_AI_RMF_MEASURE,
        headline=_build_headline(evaluation),
        class_coverage=class_poisoning_thresholds(labels) if labels is not None else None,
    )


def render_markdown(report: AssuranceReport) -> str:
    """Render an `AssuranceReport` as a self-contained Markdown document for a human reviewer."""
    lines = [
        "# aegis-scan Assurance Report",
        "",
        f"**Dataset:** {report.dataset}  ",
        f"**Generated:** {report.generated_at}  ",
        f"**Poison rate (ground truth):** {report.evaluation['poison_rate']:.3%}",
        "",
        "## Finding",
        "",
        report.headline,
        "",
        "## Detection metrics",
        "",
    ]

    ev = report.evaluation
    if ev.get("clustering") is not None:
        m = ev["clustering"]
        lines.append(
            f"- **Activation clustering:** TPR={m['tpr']:.1%}, FPR={m['fpr']:.1%}, "
            f"precision={m['precision']:.1%} (tp={m['tp']}, fp={m['fp']}, fn={m['fn']}, tn={m['tn']})"
        )
    if ev.get("agreement") is not None:
        m = ev["agreement"]
        lines.append(
            f"- **Both detectors agree:** TPR={m['tpr']:.1%}, FPR={m['fpr']:.1%}, "
            f"precision={m['precision']:.1%} (tp={m['tp']}, fp={m['fp']}, fn={m['fn']}, tn={m['tn']})"
        )
    if ev.get("spectral") is not None:
        m = ev["spectral"]
        lines.append(
            f"- **Spectral signature analysis:** AUROC={m['auroc']:.3f} ({_auroc_band(m['auroc'])}), "
            f"average precision={m['average_precision']:.3f}, top-k recall={m['top_k_recall']:.1%}"
        )
    if ev.get("fused") is not None:
        m = ev["fused"]
        lines.append(
            f"- **Fused score (spectral + clustering combined):** AUROC={m['auroc']:.3f} "
            f"({_auroc_band(m['auroc'])}), average precision={m['average_precision']:.3f}, "
            f"top-k recall={m['top_k_recall']:.1%}"
        )

    lines += [
        "",
        "## MITRE ATLAS mapping",
        "",
        "The attack this report tests for maps to the following ATLAS techniques:",
        "",
    ]
    for t in report.atlas_techniques:
        lines.append(f"- **[{t['id']}] {t['name']}** -- {t['relevance']}")

    lines += [
        "",
        "## NIST AI RMF mapping",
        "",
        f"**{report.nist_ai_rmf_measure['id']}:** \"{report.nist_ai_rmf_measure['text']}\"",
        "",
        report.nist_ai_rmf_measure["how_this_report_satisfies_it"],
        "",
    ]

    if report.class_coverage is not None:
        lines.append(render_coverage_report(report.class_coverage))

    return "\n".join(lines)


def _render_coverage_html(coverages: list[ClassCoverage]) -> str:
    """HTML counterpart of `risk.render_coverage_report` -- same content,
    same wording, as a table instead of a Markdown pipe-table."""
    if not coverages:
        return "<h2>Class-balance coverage</h2>\n<p>No classes to analyze (empty dataset).</p>\n"

    rows = []
    for c in coverages:
        row_class = ' class="low-coverage"' if c.is_low_coverage else ""
        flag = " ⚠️ low coverage" if c.is_low_coverage else ""
        rows.append(
            f"<tr{row_class}><td>{html_lib.escape(str(c.class_label))}</td>"
            f"<td>{c.class_size}</td><td>{c.class_fraction:.1%}</td>"
            f"<td>{c.p_threshold:.1%}{flag}</td></tr>"
        )

    low = [c for c in coverages if c.is_low_coverage]
    if low:
        worst = low[0]
        summary = (
            f"<p><strong>{len(low)} of {len(coverages)} classes</strong> have a threshold at or "
            f"below 10% -- the highest poisoning rate this project's own "
            f"evaluation has validated (see the Article 1 paper). Class "
            f"{html_lib.escape(str(worst.class_label))}, the smallest, could have its guard "
            f"suppressed by poisoning as little as {worst.p_threshold:.1%} of "
            f"the whole dataset. This is a structural property of the "
            f"detector applied to this class distribution, not a measured "
            f"result on this specific dataset's content.</p>"
        )
    else:
        summary = (
            "<p>No class falls below the 10% reference threshold. This does not "
            "mean detection is guaranteed at any poisoning rate -- only that "
            "no class's size alone is known, from this analysis, to blind the "
            "clustering guard within the range this project has tested.</p>"
        )

    return (
        "<h2>Class-balance coverage</h2>\n"
        "<p>Estimated poisoning rate (as a fraction of the whole dataset) at "
        "which activation clustering's minority-cluster guard would stop "
        "flagging each class, assuming the class sizes below reflect an "
        "unpoisoned baseline -- if this dataset is already partially "
        "poisoned, these thresholds are optimistic (see the risk module's "
        "docstring for why).</p>\n"
        '<table>\n<thead><tr><th>Class</th><th>Size</th><th>Share of dataset</th>'
        "<th>Poisoning-rate threshold</th></tr></thead>\n"
        f"<tbody>\n{''.join(rows)}\n</tbody>\n</table>\n"
        f"{summary}\n"
    )


def render_html(report: AssuranceReport) -> str:
    """Render an `AssuranceReport` as a self-contained HTML document: same
    content and section order as `render_markdown`, styled for reading in
    a browser or printing to PDF (File > Print > Save as PDF). No external
    stylesheets, scripts, or fonts are loaded, so the file works offline
    and doesn't depend on a PDF library with system dependencies.
    """
    esc = html_lib.escape
    ev = report.evaluation

    metric_items = []
    if ev.get("clustering") is not None:
        m = ev["clustering"]
        metric_items.append(
            f"<li><strong>Activation clustering:</strong> TPR={m['tpr']:.1%}, FPR={m['fpr']:.1%}, "
            f"precision={m['precision']:.1%} (tp={m['tp']}, fp={m['fp']}, fn={m['fn']}, tn={m['tn']})</li>"
        )
    if ev.get("agreement") is not None:
        m = ev["agreement"]
        metric_items.append(
            f"<li><strong>Both detectors agree:</strong> TPR={m['tpr']:.1%}, FPR={m['fpr']:.1%}, "
            f"precision={m['precision']:.1%} (tp={m['tp']}, fp={m['fp']}, fn={m['fn']}, tn={m['tn']})</li>"
        )
    if ev.get("spectral") is not None:
        m = ev["spectral"]
        metric_items.append(
            f"<li><strong>Spectral signature analysis:</strong> AUROC={m['auroc']:.3f} "
            f"({_auroc_band(m['auroc'])}), average precision={m['average_precision']:.3f}, "
            f"top-k recall={m['top_k_recall']:.1%}</li>"
        )
    if ev.get("fused") is not None:
        m = ev["fused"]
        metric_items.append(
            f"<li><strong>Fused score (spectral + clustering combined):</strong> AUROC={m['auroc']:.3f} "
            f"({_auroc_band(m['auroc'])}), average precision={m['average_precision']:.3f}, "
            f"top-k recall={m['top_k_recall']:.1%}</li>"
        )
    metrics_html = "\n".join(metric_items) if metric_items else "<li>No detector metrics available.</li>"

    atlas_items = "\n".join(
        f"<li><strong>[{esc(t['id'])}] {esc(t['name'])}</strong> -- {esc(t['relevance'])}</li>"
        for t in report.atlas_techniques
    )

    coverage_html = _render_coverage_html(report.class_coverage) if report.class_coverage is not None else ""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>aegis-scan Assurance Report -- {esc(report.dataset)}</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; max-width: 820px;
         margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; line-height: 1.5; }}
  h1 {{ border-bottom: 3px solid #1a1a1a; padding-bottom: 0.3rem; }}
  h2 {{ margin-top: 2rem; border-bottom: 1px solid #ccc; padding-bottom: 0.2rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }}
  th {{ background: #f0f0f0; }}
  .meta {{ color: #444; }}
  .headline {{ font-size: 1.05rem; background: #f7f7f7; border-left: 4px solid #333; padding: 0.8rem 1rem; }}
  tr.low-coverage {{ color: #a30000; font-weight: 600; }}
  @media print {{ body {{ margin: 0; max-width: none; }} }}
</style>
</head>
<body>
<h1>aegis-scan Assurance Report</h1>
<p class="meta">
  <strong>Dataset:</strong> {esc(report.dataset)}<br>
  <strong>Generated:</strong> {esc(report.generated_at)}<br>
  <strong>Poison rate (ground truth):</strong> {ev['poison_rate']:.3%}
</p>

<h2>Finding</h2>
<p class="headline">{esc(report.headline)}</p>

<h2>Detection metrics</h2>
<ul>
{metrics_html}
</ul>

<h2>MITRE ATLAS mapping</h2>
<p>The attack this report tests for maps to the following ATLAS techniques:</p>
<ul>
{atlas_items}
</ul>

<h2>NIST AI RMF mapping</h2>
<p><strong>{esc(report.nist_ai_rmf_measure['id'])}:</strong> &quot;{esc(report.nist_ai_rmf_measure['text'])}&quot;</p>
<p>{esc(report.nist_ai_rmf_measure['how_this_report_satisfies_it'])}</p>

{coverage_html}</body>
</html>
"""
