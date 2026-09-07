# Shared per-configuration pipeline logic. Not meant to be run directly --
# sourced by run_grid_healthcare.sh and run_grid_cifar.sh.
#
# Resume-safe: if a configuration's report.json already exists, it's
# skipped rather than re-run. This means an interrupted run (laptop
# sleep, closed terminal, etc.) can just be restarted with the same
# command and it picks up where it left off instead of redoing
# already-finished (and possibly hours-long) work.

TARGET_LABEL=0
TRIGGER_SIZE=3

run_one_config() {
  local ds="$1" rate="$2" seed="$3"
  local tag="${ds}_r${rate}_s${seed}"
  local out="runs/${tag}"

  if [ -f "$out/report.json" ] && [ -f "$out/asr.json" ]; then
    echo "=== ${tag} : already complete, skipping ==="
    return 0
  fi

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

  python tooling/measure_asr.py \
    --model "$out/model.pt" --dataset "$ds" \
    --target-label "$TARGET_LABEL" --trigger-size "$TRIGGER_SIZE" \
    --out "$out/asr.json"

  aegis-scan report \
    --evaluate "$out/report.json" --dataset "$ds" \
    --out "$out/assurance_report.md"
}

run_dataset_grid() {
  local ds="$1"
  local rates=(0.00 0.01 0.05 0.10)
  local seeds=(0 1 2 3 4)
  mkdir -p runs
  for rate in "${rates[@]}"; do
    for seed in "${seeds[@]}"; do
      run_one_config "$ds" "$rate" "$seed"
    done
  done
}
