"""
Command-line entry point.

`aegis-scan inject` runs stages 01-02 end to end: load a dataset, inject a
synthetic backdoor at a given poisoning rate, and save the result --
images, labels, and the ground-truth poison mask -- to disk, so stage 03
(training) can start from a fixed, reproducible artifact instead of
re-running injection every time.

`aegis-scan train` runs stage 03: train a classifier on that saved
artifact and save the resulting model checkpoint.

`aegis-scan extract-activations` runs stage 04: load a trained
checkpoint, run every sample back through it, and save one named layer's
activations to disk for stage 05's detectors to consume.

`aegis-scan detect` runs stage 05: score those activations with two
independent methods (spectral signature analysis, activation
clustering), neither of which is shown stage 02's ground-truth poison
mask -- that's held back for stage 07's evaluation, not used here.

Later stages (fuse, evaluate, report) will be added as further
subcommands here as they're built.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .activations.extract import extract_activations
from .datasets.loaders import load_benchmark_dataset, load_healthcare_dataset, load_synthetic_dataset
from .detect.clustering import activation_clustering_flags
from .detect.spectral import spectral_signature_scores
from .poison.inject import SquareTrigger, inject_poison
from .train import TrainConfig, load_checkpoint, save_checkpoint, train_classifier

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


def cmd_train(args: argparse.Namespace) -> None:
    print(f"[03] loading poisoned dataset from {args.data}...")
    with np.load(args.data) as npz:
        images, labels = npz["images"], npz["labels"]
    print(f"     {len(images)} images, shape {images.shape[1:]}, {int(labels.max()) + 1} classes")

    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    print(
        f"[03] training SmallResNet: epochs={config.epochs}, batch_size={config.batch_size}, "
        f"lr={config.lr}, seed={config.seed}..."
    )
    result = train_classifier(images, labels, config=config)
    last = result.history[-1]
    print(f"     final epoch: train_loss={last['train_loss']:.4f}, train_acc={last['train_acc']:.3%}")

    out_path = Path(args.out)
    save_checkpoint(result, out_path)
    print(f"[--] saved checkpoint to {out_path}")


def cmd_extract_activations(args: argparse.Namespace) -> None:
    print(f"[04] loading checkpoint from {args.model}...")
    model = load_checkpoint(args.model)

    print(f"[04] loading dataset from {args.data}...")
    with np.load(args.data) as npz:
        images = npz["images"]

    print(f"[04] extracting layer '{args.layer}' activations for {len(images)} samples...")
    activations = extract_activations(model, images, layer_name=args.layer, batch_size=args.batch_size)
    print(f"     activations shape: {activations.shape}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, activations=activations, layer=args.layer)
    print(f"[--] saved to {out_path}")


def cmd_detect(args: argparse.Namespace) -> None:
    print(f"[05] loading labels from {args.data}...")
    with np.load(args.data) as npz:
        labels = npz["labels"]

    print(f"[05] loading activations from {args.activations}...")
    with np.load(args.activations) as npz:
        activations = npz["activations"]

    if len(activations) != len(labels):
        raise SystemExit(
            f"activations ({len(activations)}) and labels ({len(labels)}) don't match in length -- "
            "did --data and --activations come from the same inject run?"
        )

    print(f"[05a] spectral signature analysis on {len(activations)} samples, per class...")
    spectral_scores = spectral_signature_scores(activations, labels)
    print(f"      score range: [{spectral_scores.min():.4f}, {spectral_scores.max():.4f}]")

    print(f"[05b] activation clustering (k=2 per class, PCA to {args.pca_components}-D)...")
    clustering_flags = activation_clustering_flags(
        activations, labels, pca_components=args.pca_components, seed=args.seed
    )
    print(f"      flagged {clustering_flags.sum()} / {len(clustering_flags)} samples as minority-cluster")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        spectral_scores=spectral_scores,
        clustering_flags=clustering_flags,
        labels=labels,
    )
    print(f"[--] saved to {out_path}")
    print("     (both methods ran blind to stage 02's ground-truth poison mask -- see stage 07 for scoring)")


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

    p_train = sub.add_parser("train", help="Stage 03: train a classifier on an injected .npz dataset.")
    p_train.add_argument("--data", required=True, help="Input .npz path (from `aegis-scan inject`)")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--out", required=True, help="Output checkpoint (.pt) path")
    p_train.set_defaults(func=cmd_train)

    p_extract = sub.add_parser(
        "extract-activations", help="Stage 04: extract a trained model's intermediate-layer activations."
    )
    p_extract.add_argument("--model", required=True, help="Checkpoint (.pt) path (from `aegis-scan train`)")
    p_extract.add_argument("--data", required=True, help="Input .npz path to run through the model")
    p_extract.add_argument("--layer", default="layer3", help="Named submodule to hook, e.g. 'layer3' (default)")
    p_extract.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    p_extract.add_argument("--out", required=True, help="Output .npz path for the extracted activations")
    p_extract.set_defaults(func=cmd_extract_activations)

    p_detect = sub.add_parser(
        "detect", help="Stage 05: score activations with spectral signature analysis and activation clustering."
    )
    p_detect.add_argument("--data", required=True, help="Input .npz path (for labels; from `aegis-scan inject`)")
    p_detect.add_argument(
        "--activations", required=True, help="Activations .npz path (from `aegis-scan extract-activations`)"
    )
    p_detect.add_argument("--pca-components", type=int, default=10, dest="pca_components")
    p_detect.add_argument("--seed", type=int, default=42)
    p_detect.add_argument("--out", required=True, help="Output .npz path for detection results")
    p_detect.set_defaults(func=cmd_detect)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
