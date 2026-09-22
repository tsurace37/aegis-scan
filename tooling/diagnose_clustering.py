#!/usr/bin/env python3
"""
Recovers per-class, per-seed clustering diagnostics from FILES YOU
ALREADY HAVE -- no re-running the pipeline needed. Answers three of the
open verification items in the paper's Discussion section directly:

  1. At CIFAR-10 10% poisoning: are clustering flags literally zero for
     every sample, or nonzero-but-always-wrong? (Changes what the
     fused-AUROC-equals-spectral-AUROC claim actually implies.)
  2. At 0% poisoning (both datasets): exact clustering false-positive
     count and rate, per seed -- not just "0.000 on the seed we checked."
  3. Per-class flagged counts, as evidence for (not proof of) the
     contamination-guard mechanism described in the paper.

How it works: detect.npz (from `aegis-scan detect`) already saves
per-sample `clustering_flags` and `labels`. inject.npz (from
`aegis-scan inject`) already saves the per-sample ground-truth
`poison_mask`. Cross-referencing the three needs no new model runs,
just reading files stage 05 and stage 02 already wrote.

Usage (from the aegis-scan repo root):
    python3 tooling/diagnose_clustering.py runs/
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_run(run_dir: Path):
    """Returns (poison_mask, clustering_flags, labels), or (None, reason)
    where reason explains exactly what's missing -- silently skipping
    was a real bug in the previous version of this script: it made a
    missing detect.npz indistinguishable from a run that never existed."""
    inject_path = run_dir / "inject.npz"
    detect_path = run_dir / "detect.npz"
    missing = [str(p.name) for p in (inject_path, detect_path) if not p.exists()]
    if missing:
        return None, f"missing {', '.join(missing)}"
    with np.load(inject_path) as npz:
        poison_mask = npz["poison_mask"].astype(bool)
    with np.load(detect_path) as npz:
        clustering_flags = npz["clustering_flags"].astype(bool)
        labels = npz["labels"]
    if not (len(poison_mask) == len(clustering_flags) == len(labels)):
        return None, "length mismatch between inject.npz and detect.npz"
    return (poison_mask, clustering_flags, labels), None


def main(root: str) -> None:
    run_dirs = sorted(p.parent for p in Path(root).glob("*/inject.npz"))
    if not run_dirs:
        sys.exit(f"No */inject.npz found under {root}")

    by_config: dict[tuple[str, str], list] = defaultdict(list)
    skipped = []
    for run_dir in run_dirs:
        m = re.match(r"(?P<ds>.+)_r(?P<rate>[\d.]+)_s(?P<seed>\d+)$", run_dir.name)
        if not m:
            continue
        result, reason = load_run(run_dir)
        if result is None:
            skipped.append((run_dir.name, reason))
            continue
        by_config[(m["ds"], m["rate"])].append((m["seed"], *result))

    if skipped:
        print("=" * 70)
        print(f"SKIPPED {len(skipped)} run director{'y' if len(skipped)==1 else 'ies'} "
              f"(present under {root}, but incomplete):")
        print("=" * 70)
        for name, reason in skipped:
            print(f"  {name}: {reason}")
        print()

    print("=" * 70)
    print("PART 1: overall clustering flag counts, all configurations")
    print("=" * 70)
    for (ds, rate), seeds in sorted(by_config.items()):
        for seed, poison_mask, flags, labels in sorted(seeds, key=lambda x: x[0]):
            tp = int((flags & poison_mask).sum())
            fp = int((flags & ~poison_mask).sum())
            n_flagged = int(flags.sum())
            n_total = len(flags)
            n_poison = int(poison_mask.sum())
            fpr = fp / max(1, (~poison_mask).sum())
            print(
                f"{ds:12s} r={rate:>5s} s={seed}: "
                f"flagged={n_flagged:5d}/{n_total:5d}  tp={tp:4d}  fp={fp:4d}  "
                f"fpr={fpr:.4f}  (poisoned in ground truth: {n_poison})"
            )

    print()
    print("=" * 70)
    print("PART 2: per-class breakdown for the highest-poisoning-rate")
    print("configurations (verification item 1 -- is CIFAR-10 clustering")
    print("really all-zero, or just always-wrong, at 10% poisoning?)")
    print("=" * 70)
    for (ds, rate), seeds in sorted(by_config.items()):
        if rate not in ("0.10",):
            continue
        for seed, poison_mask, flags, labels in sorted(seeds, key=lambda x: x[0]):
            print(f"\n{ds} r={rate} s={seed}, per class:")
            for c in np.unique(labels):
                idx = labels == c
                n_c = int(idx.sum())
                flagged_c = int((flags & idx).sum())
                poisoned_c = int((poison_mask & idx).sum())
                tp_c = int((flags & poison_mask & idx).sum())
                fp_c = int((flags & ~poison_mask & idx).sum())
                marker = "  <- target class, expect high contamination" if poisoned_c > 0 else ""
                print(
                    f"    class {c}: n={n_c:5d}  poisoned={poisoned_c:5d} "
                    f"({100*poisoned_c/max(1,n_c):.1f}%)  flagged={flagged_c:5d}  "
                    f"tp={tp_c:4d}  fp={fp_c:4d}{marker}"
                )

    print()
    print("=" * 70)
    print("PART 3: 0% control rows, exact clustering FPR per seed")
    print("(verification item 2)")
    print("=" * 70)
    for (ds, rate), seeds in sorted(by_config.items()):
        if rate != "0.00":
            continue
        for seed, poison_mask, flags, labels in sorted(seeds, key=lambda x: x[0]):
            assert poison_mask.sum() == 0, "0% run has nonzero poison_mask -- check inject.npz"
            fp = int(flags.sum())  # every flag on a 0%-poisoned run is a false positive
            fpr = fp / len(flags)
            print(f"{ds:12s} s={seed}: fp={fp:5d} / {len(flags):5d}  fpr={fpr:.5f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs")