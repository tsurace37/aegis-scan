#!/usr/bin/env bash
# Phase 1 benchmark grid for aegis-scan.
# Run this from inside a checkout of https://github.com/tsurace37/aegis-scan
# (commit 7c9c063 or later) with measure_asr.py copied into the repo root.
#
# 2 datasets x 4 rates (0%, 1%, 5%, 10%) x 5 seeds = 40 runs.
# CIFAR-10 (50k train images) will take far longer per run than
# PneumoniaMNIST (~4.7k). If you don't have a GPU, run PneumoniaMNIST
# first -- it alone gives you a real, complete healthcare-benchmark
# result, which is the paper's primary claim.

set -euo pipefail

DATASETS=("healthcare" "benchmark")   # healthcare=PneumoniaMNIST, benchmark=CIFAR-10
RATES=(0.00 0.01 0.05 0.10)
SEEDS=(0 1 2 3 4)
TARGET_LABEL=0
TRIGGER_SIZE=3

mkdir -p runs

for ds in "${DATASETS[@]}"; do
  for rate in "${RATES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      tag="${ds}_r${rate}_s${seed}"
      out="runs/${tag}"
      mkdir -p "$out"
      echo "=== ${tag} ==="

      aegis-scan inject \
        --dataset "$ds" --rate "$rate" --seed "$seed" \
        --target-label "$TARGET_LABEL" --trigger-size "$TRIGGER_SIZE" \
        --out "$out/inject.npz"

      aegis-scan train \
        --data "$out/inject.npz" --seed "$seed" \
        --epochs 10 --batch-size 64 --lr 1e-3 \
        --out "$out/model.pt"

      aegis-scan extract-activations \
        --model "$out/model.pt" --data "$out/inject.npz" \
        --layer layer3 \
        --out "$out/acts.npz"

      aegis-scan detect \
        --data "$out/inject.npz" --activations "$out/acts.npz" \
        --pca-components 10 --seed "$seed" \
        --out "$out/detect.npz"

      aegis-scan fuse \
        --detect "$out/detect.npz" \
        --spectral-percentile-cutoff 0.9 \
        --out "$out/fuse.npz"

      aegis-scan evaluate \
        --inject "$out/inject.npz" --detect "$out/detect.npz" --fuse "$out/fuse.npz" \
        --out "$out/report.json"

      python3 tooling/measure_asr.py \
        --model "$out/model.pt" --dataset "$ds" \
        --target-label "$TARGET_LABEL" --trigger-size "$TRIGGER_SIZE" \
        --out "$out/asr.json"

      aegis-scan report \
        --evaluate "$out/report.json" --dataset "$ds" \
        --out "$out/assurance_report.md"
    done
  done
done

echo "done. record the commit SHA that produced these runs:"
git rev-parse HEAD
