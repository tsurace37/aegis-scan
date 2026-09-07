#!/usr/bin/env bash
# PneumoniaMNIST only: 1 dataset x 4 rates x 5 seeds = 20 runs.
# Small dataset (4,708 train images) -- the fast half of the grid.
# Safe to interrupt and re-run; already-finished configs are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root, regardless of where this is called from
source tooling/_pipeline_common.sh

run_dataset_grid healthcare

echo "healthcare (PneumoniaMNIST) grid done. commit SHA:"
git rev-parse HEAD
