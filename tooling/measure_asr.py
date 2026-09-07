#!/usr/bin/env python3
"""
Computes what the aegis-scan pipeline (as of commit 7c9c063) does not:
attack success rate (ASR) and clean test accuracy, from a trained
checkpoint. This is NOT part of the aegis-scan package -- it imports
aegis_scan as a library and runs alongside the CLI's own outputs.

Why this is needed: `aegis-scan evaluate` only scores the four
detectors against the ground-truth poison mask. It never touches a
held-out test set, so nothing in the existing pipeline tells you
whether the backdoor was actually learned. Without ASR, a detection
AUROC number describes detecting *something* -- not necessarily the
attack the paper claims to test.

Definitions used here (standard in the backdoor literature):
  - clean accuracy: accuracy on an ordinary, untriggered test set.
  - attack success rate (ASR): of test images NOT already labeled the
    target class, the fraction the model now classifies AS the target
    class once the trigger is stamped on them. High ASR = the backdoor
    was successfully installed. Low ASR = detecting it would be
    detecting a backdoor that barely works, which is a different and
    weaker claim.

Usage:
    python3 measure_asr.py \
        --model runs/pneumoniamnist_r0.10_s0/model.pt \
        --dataset healthcare \
        --target-label 0 \
        --trigger-size 3 \
        --out runs/pneumoniamnist_r0.10_s0/asr.json

--dataset/--target-label/--trigger-size must match whatever you passed
to `aegis-scan inject` for this run, or the numbers are meaningless.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from aegis_scan.datasets.loaders import load_benchmark_dataset, load_healthcare_dataset
from aegis_scan.poison.inject import SquareTrigger
from aegis_scan.train import load_checkpoint

LOADERS = {"healthcare": load_healthcare_dataset, "benchmark": load_benchmark_dataset}


def _predict(model: torch.nn.Module, images: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = torch.from_numpy(images[i : i + batch_size])
            preds.append(model(batch).argmax(dim=1).numpy())
    return np.concatenate(preds)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Checkpoint .pt path from `aegis-scan train`")
    ap.add_argument("--dataset", required=True, choices=list(LOADERS))
    ap.add_argument("--target-label", type=int, required=True, dest="target_label",
                     help="Must match the --target-label used in `aegis-scan inject` for this run")
    ap.add_argument("--trigger-size", type=int, default=3, dest="trigger_size",
                     help="Must match the --trigger-size used in `aegis-scan inject` for this run")
    ap.add_argument("--out", required=True, help="Output .json path")
    args = ap.parse_args()

    print(f"[asr] loading '{args.dataset}' test split...")
    ds = LOADERS[args.dataset](split="test")
    print(f"      {len(ds)} test images")

    print(f"[asr] loading checkpoint from {args.model}...")
    model = load_checkpoint(args.model)

    print("[asr] measuring clean accuracy...")
    clean_preds = _predict(model, ds.images)
    clean_acc = float((clean_preds == ds.labels).mean())
    print(f"      clean accuracy: {clean_acc:.3%}")

    print(f"[asr] stamping trigger (size={args.trigger_size}) on all non-target-class test images...")
    trigger = SquareTrigger(size=args.trigger_size)
    eligible = ds.labels != args.target_label
    if eligible.sum() == 0:
        raise SystemExit("no eligible (non-target-class) test images -- check --target-label")
    triggered_images = np.stack([trigger.apply(img) for img in ds.images[eligible]])

    print(f"[asr] measuring attack success rate on {eligible.sum()} triggered images...")
    triggered_preds = _predict(model, triggered_images)
    asr = float((triggered_preds == args.target_label).mean())
    print(f"      attack success rate: {asr:.3%}")

    result = {
        "clean_accuracy": clean_acc,
        "attack_success_rate": asr,
        "n_test": len(ds),
        "n_asr_eval": int(eligible.sum()),
        "target_label": args.target_label,
        "trigger_size": args.trigger_size,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[--] saved to {out_path}")


if __name__ == "__main__":
    main()
