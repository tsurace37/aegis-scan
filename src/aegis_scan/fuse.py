"""
Stage 06 -- fuse stage 05's two independent detectors into one per-sample
score and a handful of model-level risk statistics.

Spectral signature analysis and activation clustering look for different
kinds of evidence and disagree about individual samples reasonably
often -- that's expected, not a bug, since each one can miss things the
other catches. This stage doesn't try to decide which method is "right"
for a given sample; it combines what both found, and treats the case
where they *agree* as the strongest signal of all, since two
differently-motivated methods independently pointing at the same sample
is far less likely by chance than either one alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def percentile_rank_per_class(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Rescale `values` to a 0-1 rank within each class, computed independently per class.

    Spectral signature scores aren't comparable across classes -- each
    class gets its own SVD in stage 05, so a score of 40 in one class
    and 40 in another don't mean the same thing. Ranking within each
    class instead (0 = smallest in its class, 1 = largest) puts every
    class on the same footing before fusing with the clustering signal,
    without assuming anything about how many samples are actually
    poisoned anywhere -- it's a relative ordering, not a threshold.

    Public (not a fuse.py-only helper) because stage 07 needs the exact
    same rescaling to evaluate spectral scores fairly -- computing a
    single global AUROC on raw spectral scores would be distorted by
    each class's own arbitrary scale, the same problem fusion solves.
    """
    ranks = np.zeros(len(values), dtype=np.float64)
    for class_label in np.unique(labels):
        idx = np.flatnonzero(labels == class_label)
        if len(idx) <= 1:
            continue  # nothing to rank a single sample against
        rank_order = np.argsort(np.argsort(values[idx]))
        ranks[idx] = rank_order / (len(idx) - 1)
    return ranks


@dataclass
class FusionResult:
    fused_scores: np.ndarray  # per-sample, in [0, 1]
    agreement: np.ndarray  # per-sample bool: both detectors flagged it
    mean_fused_score: float
    fraction_flagged_by_either: float
    fraction_flagged_by_both: float


def fuse_scores(
    spectral_scores: np.ndarray,
    clustering_flags: np.ndarray,
    labels: np.ndarray,
    spectral_percentile_cutoff: float = 0.9,
) -> FusionResult:
    """Combine stage 05's two detector outputs into one score per sample, plus model-level stats.

    `fused_scores` is a simple 50/50 average of each detector's signal,
    put on the same 0-1 footing first: the spectral score's per-class
    percentile rank, and the clustering flag as 0.0 or 1.0. A sample
    both methods agree is suspicious lands near 1.0; a sample neither
    flags lands near 0.0; a sample only one method catches lands in
    between -- worth a closer look, but not as strong as agreement.

    `spectral_percentile_cutoff` (default: the top 10% within each
    class) only decides whether spectral analysis counts as having
    "flagged" a sample for the *agreement* statistics below -- it never
    changes `fused_scores` itself, which stays a continuous rank.

    Model-level stats: `fraction_flagged_by_both` (agreement between
    the two methods) is this project's headline risk number, precisely
    because it's the harder bar to clear by chance. `fraction_flagged_
    by_either` is the more permissive, higher-recall companion --
    useful context, but expected to run higher on its own.
    """
    if not (len(spectral_scores) == len(clustering_flags) == len(labels)):
        raise ValueError(
            f"spectral_scores ({len(spectral_scores)}), clustering_flags ({len(clustering_flags)}), "
            f"and labels ({len(labels)}) must all be the same length"
        )

    spectral_rank = percentile_rank_per_class(spectral_scores, labels)
    clustering_score = clustering_flags.astype(np.float64)

    fused_scores = 0.5 * spectral_rank + 0.5 * clustering_score

    spectral_flag = spectral_rank >= spectral_percentile_cutoff
    agreement = spectral_flag & clustering_flags

    return FusionResult(
        fused_scores=fused_scores,
        agreement=agreement,
        mean_fused_score=float(fused_scores.mean()),
        fraction_flagged_by_either=float((spectral_flag | clustering_flags).mean()),
        fraction_flagged_by_both=float(agreement.mean()),
    )
