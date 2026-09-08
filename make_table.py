#!/usr/bin/env python3
"""
Turn a directory of aegis-scan evaluate reports into the LaTeX table body
for the Results section.

Usage:
    python3 make_table.py runs/

Expects subdirectories named like  <dataset>_r<rate>_s<seed>/report.json
(matching the runner skeleton in fill-markers-guide.md).

Metric key matching is deliberately loose, because the exact key names in
your report.json are unknown here. RUN THIS ON ONE REPORT FIRST and check
the "matched keys" printout before trusting all 30.
"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# candidate substrings, in priority order, for each column
WANTED = {
    "asr":              [("attack_success_rate",), ("asr",)],
    "clean_acc":        [("clean_accuracy",), ("clean", "acc")],
    "spectral_auroc":   [("spectral", "auroc")],
    "clustering_auroc": [("cluster", "auroc"), ("cluster", "tpr")],
    "fused_auroc":      [("fus", "auroc")],
    "agreement_prec":   [("agree", "precision"), ("both", "precision")],
    # FPR is well-defined even when TPR/AUROC/precision are not (0%
    # poisoning has zero true positives but plenty of true negatives,
    # so fp/(fp+tn) is a real number). This is the metric that actually
    # answers "does this detector false-positive on clean data" -- the
    # whole point of a 0%-poisoning control row.
    "clustering_fpr":   [("clustering", "fpr")],
}


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix.rstrip(".")] = float(obj)
    return out


def pick(flat, patterns):
    for pat in patterns:
        for key, val in flat.items():
            low = key.lower()
            if all(p in low for p in pat):
                return key, val
    return None, None


def fmt(vals):
    """Format a list of per-seed values as mean +/- sd.

    Distinguishes two genuinely different situations that both look
    like "no number" at a glance:
      - vals is empty -> "---"  (not yet measured / key not found)
      - vals is non-empty but every entry is NaN -> "n/a (undefined)"
        (the metric is mathematically undefined for this configuration,
        e.g. AUROC/TPR at 0% poisoning, where there are zero positive
        samples to compute a rate against -- this is expected, not
        missing data, and the paper's Table 1 caption should say so)
      - a MIX of real and NaN values -> average over the real ones only,
        and note how many seeds were dropped, rather than silently
        contaminating the mean or crashing (Python's statistics.stdev
        raises AttributeError on an all-NaN input -- this is a real bug
        in earlier versions of this script, not a hypothetical one).
    """
    if not vals:
        return "---"
    finite = [v for v in vals if v == v]  # v == v is False only for NaN
    if not finite:
        return "n/a"
    if len(finite) == 1:
        suffix = f" (n=1/{len(vals)})" if len(finite) < len(vals) else ""
        return f"{finite[0]:.3f}{suffix}"
    suffix = f" (n={len(finite)}/{len(vals)})" if len(finite) < len(vals) else ""
    return f"{statistics.mean(finite):.3f} $\\pm$ {statistics.stdev(finite):.3f}{suffix}"


def main(root):
    reports = sorted(Path(root).glob("*/report.json"))
    if not reports:
        sys.exit(f"No */report.json found under {root}")

    groups = defaultdict(lambda: defaultdict(list))
    matched_keys, missing = {}, set()

    for rp in reports:
        m = re.match(r"(?P<ds>.+)_r(?P<rate>[\d.]+)_s(?P<seed>\d+)$", rp.parent.name)
        if not m:
            print(f"  skipping unrecognised dir name: {rp.parent.name}")
            continue
        flat = flatten(json.loads(rp.read_text()))
        asr_path = rp.parent / "asr.json"
        if asr_path.exists():
            flat.update(flatten(json.loads(asr_path.read_text())))
        else:
            print(f"  WARNING: no asr.json next to {rp} -- run measure_asr.py for this configuration")
        key = (m["ds"], float(m["rate"]))
        for col, pats in WANTED.items():
            src, val = pick(flat, pats)
            if val is None:
                missing.add(col)
            else:
                matched_keys.setdefault(col, src)
                groups[key][col].append(val)

    print("matched keys (CHECK THESE):")
    for col, src in sorted(matched_keys.items()):
        print(f"  {col:18s} <- {src}")
    for col in sorted(missing - set(matched_keys)):
        print(f"  {col:18s} <- NOT FOUND: add it to the report, or the column stays ---")

    print("\n% ---- paste over the placeholder rows in Table 1 ----")
    pretty = {"healthcare": "PneumoniaMNIST", "benchmark": "CIFAR-10"}
    for (ds, rate) in sorted(groups, key=lambda k: (k[0], k[1])):
        g = groups[(ds, rate)]
        n = max((len(v) for v in g.values()), default=0)
        cells = " & ".join(
            fmt(g.get(c, []))
            for c in ["asr", "clean_acc", "spectral_auroc", "clustering_auroc",
                      "fused_auroc", "agreement_prec"]
        )
        extra = fmt(g.get("clustering_fpr", []))
        print(f"{pretty.get(ds, ds)} & {rate*100:.0f}\\% & {cells} \\\\  % n={n} seeds, clustering FPR={extra}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs")