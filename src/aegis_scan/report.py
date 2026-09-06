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

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

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
        "name": "Manipulate AI Model",
        "relevance": (
            "The end state of a successful poisoning attack: a persistent, "
            "hidden change in the trained model's behavior. aegis-scan "
            "doesn't inspect model weights directly the way this "
            "technique's 'Poison AI Model' sub-technique (AML.T0018.000) "
            "describes -- it infers the same outcome indirectly, from how "
            "poisoned training data reshapes a layer's activations."
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


def generate_report(evaluation: dict[str, Any], *, dataset: str = "unspecified") -> AssuranceReport:
    """Wrap stage 07's evaluation dict (its saved JSON, loaded back in) with framework mappings and a headline.

    `evaluation` is exactly the dict `aegis-scan evaluate` saves: `poison_rate`
    plus optional `clustering`/`agreement`/`spectral`/`fused` metric dicts.
    """
    return AssuranceReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        dataset=dataset,
        evaluation=evaluation,
        atlas_techniques=ATLAS_TECHNIQUES,
        nist_ai_rmf_measure=NIST_AI_RMF_MEASURE,
        headline=_build_headline(evaluation),
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

    return "\n".join(lines)
