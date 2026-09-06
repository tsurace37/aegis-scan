"""
Command-line entry point.

`aegis-scan inject` runs stages 01-02 end to end: load a dataset, inject a
synthetic backdoor at a given poisoning rate, and save the result --
images, labels, and the ground-truth poison mask -- to disk, so stage 03
(training) can start from a fixed, reproducible artifact instead of
re-running injection every time.

Later stages (train, detect, evaluate, report) will be added as further
subcommands here as they're built.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .datasets.loaders import load_benchmark_dataset, load_healthcare_dataset, load_synthetic_dataset
from .poison.inject import SquareTrigger, inject_poison

DATASET_LOADERS = {
    "healthcare": load_healthcare_dataset,
    "benchmark": load_benchmark_dataset,
    "synthetic": load_synthetic_dataset,
}


def cmd_inject(args: argparse.Namespace) -> None:
    loader = DATASET_LOADERS[args.dataset]

    print(f"[01] loading '{args.dataset}' dataset (split={args.split})...")
    ds = loader(split=args.split)
    print(f"     {len(ds)} images, shape {ds.images.shape[1:]}, classes={ds.class_names}")

    print(
        f"[02] injecting backdoor: rate={args.rate}, target_label={args.target_label}, "
        f"trigger_size={args.trigger_size}, seed={args.seed}..."
    )
    trigger = SquareTrigger(size=args.trigger_size)
    result = inject_poison(
        ds, poison_rate=args.rate, target_label=args.target_label, trigger=trigger, seed=args.seed
    )
    print(
        f"     poisoned {result.poison_mask.sum()} / {len(result.poison_mask)} samples "
        f"({result.poison_rate:.3%}, requested {args.rate:.3%})"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        images=result.images,
        labels=result.labels,
        poison_mask=result.poison_mask,
        dataset=ds.name,
        poison_rate=result.poison_rate,
        target_label=result.target_label,
    )
    print(f"[--] saved to {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aegis-scan",
        description="Data-poisoning / backdoor detection scanner for ML classification pipelines.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_inject = sub.add_parser("inject", help="Stages 01-02: load a dataset and inject a synthetic backdoor.")
    p_inject.add_argument("--dataset", choices=list(DATASET_LOADERS), required=True)
    p_inject.add_argument("--split", default="train")
    p_inject.add_argument("--rate", type=float, required=True, help="Poisoning rate, e.g. 0.05 for 5%%")
    p_inject.add_argument("--target-label", type=int, default=0, dest="target_label")
    p_inject.add_argument("--trigger-size", type=int, default=3, dest="trigger_size")
    p_inject.add_argument("--seed", type=int, default=42)
    p_inject.add_argument("--out", required=True, help="Output .npz path")
    p_inject.set_defaults(func=cmd_inject)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
