#!/usr/bin/env bash
# CIFAR-10 only: 1 dataset x 4 rates x 5 seeds = 20 runs.
# Large dataset (50,000 train images) -- the slow half of the grid,
# hours on CPU. Safe to interrupt (Ctrl-C, closed terminal, sleep) and
# re-run later; already-finished configs are skipped, so this can be
# run in several sittings instead of one unbroken block of time.
set -euo pipefail
cd "$(dirname "$0")/.."
source tooling/_pipeline_common.sh

run_dataset_grid benchmark

echo "benchmark (CIFAR-10) grid done. commit SHA:"
git rev-parse HEAD
