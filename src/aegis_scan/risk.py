"""
Class-balance coverage analysis -- a pre-flight check computable from
dataset labels alone, no trained model, activations, or detection run
required.

Motivation. Activation clustering's minority-cluster guard (see
`detect/clustering.py`'s `max_minority_fraction`) suppresses its own
flag whenever a class's poisoned subset is not a clear minority of that
class. This is a structural property of the detector's design, not a
failure specific to one dataset: CIFAR-10's evenly balanced classes hit
this exact blind spot at a measured 10% poisoning rate (target-class
contamination 50%, confirmed via direct inspection of saved per-sample
outputs to fall to exactly zero true positives across five independent
seeds, in every class, not just the poisoned one). PneumoniaMNIST's
imbalanced classes did not hit the same wall at the same poisoning
rate, and the difference traces to class balance, not dataset size --
see the Article 1 paper's Results section for the full derivation this
module implements.

What this answers. For any dataset's class distribution, before
spending any compute training a model or running detection: which
classes, and at what poisoning rate (as a fraction of the *whole*
dataset), would push that class's contamination past the guard's
threshold and into this blind spot?

The formula. Writing p for the poisoning rate (fraction of the full
dataset relabeled into a target class) and q for that class's fraction
of the dataset *before* poisoning, a class that receives poisoning ends
up q+p of the dataset, of which p is poisoned -- so the within-class
contaminated fraction is p/(q+p). Solving for the p at which this
fraction exactly equals the guard's threshold tau gives the poisoning
rate at which the guard stops firing for that class:

    p_threshold = q * tau / (1 - tau)

Smaller classes (smaller q) have a smaller p_threshold: less poisoning,
as a fraction of the whole dataset, is needed to push a minority class
past the guard than a majority class. This is the actionable finding --
a dataset's smallest classes are structurally its weakest points against
this specific detector, regardless of anything about the attack itself.

Disclosed assumption, stated plainly rather than buried. This treats
the dataset's CURRENT observed class distribution as an unpoisoned
baseline. If the dataset handed to this function is already partially
poisoned, every threshold below is optimistic: the true remaining
headroom before the guard breaks is smaller than reported, since some
of a class's current size may already be attacker-controlled. This
function reasons only about class *sizes*; it has no way to detect
that on its own, and does not claim to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClassCoverage:
    """Coverage analysis for one class."""

    class_label: int
    class_size: int
    dataset_size: int
    class_fraction: float  # q: this class's share of the dataset, as observed
    p_threshold: float  # fraction of the FULL dataset that would need to be
    # poisoned into this class to push its contamination to exactly tau

    @property
    def is_low_coverage(self) -> bool:
        """True when this class's threshold sits at or below every poisoning
        rate this project has empirically tested (1%, 5%, 10%) -- i.e. a
        real-world poisoning campaign well within the range already shown
        to be achievable would be enough to blind the guard for this class.
        10% is used as the cutoff since it is the highest rate validated
        in Article 1; this is a documented, arbitrary-but-justified choice,
        not a claim that rates above 10% are somehow unrealistic."""
        return self.p_threshold <= 0.10


def class_poisoning_thresholds(
    labels: np.ndarray,
    max_minority_fraction: float = 0.35,
) -> list[ClassCoverage]:
    """Compute the per-class poisoning-rate threshold at which
    `activation_clustering_flags`'s minority-cluster guard (see
    `detect/clustering.py`, same `max_minority_fraction` default) stops
    being able to flag that class at all.

    Returns one `ClassCoverage` per distinct label in `labels`, sorted by
    ascending `p_threshold` -- the most structurally vulnerable class
    first, since that's the one worth a reader's attention.

    Raises ValueError if `max_minority_fraction` is not strictly between
    0 and 1 (at 0, no class could ever be poisoned without immediately
    exceeding the guard; at 1 or above, the guard would never fire
    regardless of contamination, making every threshold below undefined).
    """
    if not (0.0 < max_minority_fraction < 1.0):
        raise ValueError(f"max_minority_fraction must be in (0, 1), got {max_minority_fraction}")

    n_total = len(labels)
    if n_total == 0:
        return []

    tau = max_minority_fraction
    results = []
    for class_label in np.unique(labels):
        class_size = int(np.sum(labels == class_label))
        q = class_size / n_total
        p_threshold = q * tau / (1.0 - tau)
        results.append(
            ClassCoverage(
                class_label=int(class_label),
                class_size=class_size,
                dataset_size=n_total,
                class_fraction=q,
                p_threshold=p_threshold,
            )
        )

    return sorted(results, key=lambda c: c.p_threshold)


def render_coverage_report(coverages: list[ClassCoverage]) -> str:
    """Render `class_poisoning_thresholds`'s output as a self-contained
    Markdown section, most-vulnerable-class first, with the assumption
    this analysis rests on stated plainly rather than left implicit."""
    if not coverages:
        return "## Class-balance coverage\n\nNo classes to analyze (empty dataset).\n"

    lines = [
        "## Class-balance coverage",
        "",
        (
            "Estimated poisoning rate (as a fraction of the whole dataset) at "
            "which activation clustering's minority-cluster guard would stop "
            "flagging each class, assuming the class sizes below reflect an "
            "unpoisoned baseline -- if this dataset is already partially "
            "poisoned, these thresholds are optimistic (see this module's "
            "docstring for why)."
        ),
        "",
        "| Class | Size | Share of dataset | Poisoning-rate threshold |",
        "|---|---|---|---|",
    ]
    for c in coverages:
        flag = " \u26a0\ufe0f low coverage" if c.is_low_coverage else ""
        lines.append(
            f"| {c.class_label} | {c.class_size} | {c.class_fraction:.1%} | "
            f"{c.p_threshold:.1%}{flag} |"
        )

    low = [c for c in coverages if c.is_low_coverage]
    lines.append("")
    if low:
        worst = low[0]
        lines.append(
            f"**{len(low)} of {len(coverages)} classes** have a threshold at or "
            f"below 10% -- the highest poisoning rate this project's own "
            f"evaluation has validated (see the Article 1 paper). Class "
            f"{worst.class_label}, the smallest, could have its guard "
            f"suppressed by poisoning as little as {worst.p_threshold:.1%} of "
            f"the whole dataset. This is a structural property of the "
            f"detector applied to this class distribution, not a measured "
            f"result on this specific dataset's content."
        )
    else:
        lines.append(
            "No class falls below the 10% reference threshold. This does not "
            "mean detection is guaranteed at any poisoning rate -- only that "
            "no class's size alone is known, from this analysis, to blind the "
            "clustering guard within the range this project has tested."
        )

    return "\n".join(lines) + "\n"
