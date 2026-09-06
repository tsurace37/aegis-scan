"""
Stage 07 -- evaluate detection accuracy against stage 02's ground truth.

Every stage from 03 onward was deliberately built to never see
`poison_mask`: stage 03 trained on the poisoned data as if it were
genuine, stages 05-06 scored samples with no idea which ones were
altered. This is the one stage in the whole pipeline allowed to look at
it. That's not a contradiction of the "blind" design -- checking a
detector's performance afterward, and letting the detector see the
answer key while it works, are two different things. Only the second
one would be cheating.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .fuse import percentile_rank_per_class


@dataclass
class BinaryMetrics:
    """Confusion-matrix metrics for a yes/no detector output (a flag array)."""

    tp: int
    fp: int
    fn: int
    tn: int
    tpr: float  # recall -- of the truly poisoned samples, how many were caught
    fpr: float  # of the truly clean samples, how many were wrongly flagged
    precision: float  # of what was flagged, how many were actually poisoned


def binary_detection_metrics(flags: np.ndarray, poison_mask: np.ndarray) -> BinaryMetrics:
    """Score a boolean flag array (e.g. stage 05's clustering flags, or stage 06's agreement) against the ground truth."""
    if len(flags) != len(poison_mask):
        raise ValueError(f"flags ({len(flags)}) and poison_mask ({len(poison_mask)}) must be the same length")

    flags = flags.astype(bool)
    poison_mask = poison_mask.astype(bool)

    tp = int(np.sum(flags & poison_mask))
    fp = int(np.sum(flags & ~poison_mask))
    fn = int(np.sum(~flags & poison_mask))
    tn = int(np.sum(~flags & ~poison_mask))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    return BinaryMetrics(tp=tp, fp=fp, fn=fn, tn=tn, tpr=tpr, fpr=fpr, precision=precision)


@dataclass
class ScoreMetrics:
    """Ranking-quality metrics for a continuous detector output (a score array)."""

    auroc: float
    average_precision: float
    top_k_recall: float  # recall if exactly as many samples were flagged as are truly poisoned


def score_based_metrics(scores: np.ndarray, poison_mask: np.ndarray) -> ScoreMetrics:
    """Score a continuous array (e.g. stage 06's fused_scores) against the ground truth.

    `top_k_recall` answers a specific, interpretable question: if you
    flagged exactly as many samples as are actually poisoned (an
    "oracle" threshold no real deployment would know to pick), what
    fraction of the truly poisoned ones would be in that top-k? It's a
    useful companion to AUROC/average precision because it's easy to
    reason about directly -- "92 of the 100 poisoned samples were in
    the top 100 by score" -- without needing a threshold-selection
    story.
    """
    if len(scores) != len(poison_mask):
        raise ValueError(f"scores ({len(scores)}) and poison_mask ({len(poison_mask)}) must be the same length")

    poison_mask = poison_mask.astype(bool)
    k = int(poison_mask.sum())
    n = len(poison_mask)

    if 0 < k < n:
        auroc = float(roc_auc_score(poison_mask, scores))
        average_precision = float(average_precision_score(poison_mask, scores))
    else:
        # AUROC/AP are undefined when every sample (or no sample) is poisoned --
        # there's no "other class" to rank against
        auroc = float("nan")
        average_precision = float("nan")

    if k > 0:
        top_k_idx = np.argsort(-scores)[:k]
        top_k_recall = float(poison_mask[top_k_idx].sum() / k)
    else:
        top_k_recall = float("nan")

    return ScoreMetrics(auroc=auroc, average_precision=average_precision, top_k_recall=top_k_recall)


@dataclass
class EvaluationReport:
    poison_rate: float
    clustering: BinaryMetrics | None = None
    agreement: BinaryMetrics | None = None
    spectral: ScoreMetrics | None = None
    fused: ScoreMetrics | None = None


def evaluate(
    poison_mask: np.ndarray,
    labels: np.ndarray | None = None,
    spectral_scores: np.ndarray | None = None,
    clustering_flags: np.ndarray | None = None,
    fused_scores: np.ndarray | None = None,
    agreement_flags: np.ndarray | None = None,
) -> EvaluationReport:
    """Score whichever of stage 05/06's outputs are supplied against `poison_mask`.

    Every argument except `poison_mask` is optional, so this works
    whether it's handed just stage 05's raw outputs, just stage 06's
    fused ones, or (typically) both -- comparing them side by side is
    exactly how a gap like the one found during stage 06's own testing
    (the "agreement" flag being zero-false-positive but lower-recall
    than the continuous fused score) gets surfaced instead of hidden
    behind a single reported number.

    `spectral_scores` needs `labels` alongside it: raw spectral scores
    are only comparable within a class (each gets its own SVD in stage
    05), so they're converted to the same per-class percentile rank
    fusion already uses before being scored here -- otherwise a global
    AUROC over classes with wildly different score scales would be
    meaningless.
    """
    poison_mask = poison_mask.astype(bool)
    report = EvaluationReport(poison_rate=float(poison_mask.mean()))

    if clustering_flags is not None:
        report.clustering = binary_detection_metrics(clustering_flags, poison_mask)

    if agreement_flags is not None:
        report.agreement = binary_detection_metrics(agreement_flags, poison_mask)

    if spectral_scores is not None:
        if labels is None:
            raise ValueError("labels are required to evaluate spectral_scores -- they're only comparable within a class")
        spectral_rank = percentile_rank_per_class(spectral_scores, labels)
        report.spectral = score_based_metrics(spectral_rank, poison_mask)

    if fused_scores is not None:
        report.fused = score_based_metrics(fused_scores, poison_mask)

    return report
