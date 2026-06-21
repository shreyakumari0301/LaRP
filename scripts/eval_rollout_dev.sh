#!/usr/bin/env bash
# Dev eval: greedy vs rollout on 4 episodes. Target: rollout gap <= greedy (1.435% on dev).
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mnlp
cd /mnt/c/Users/SK/LaRP/MnLP

WM="${1:-checkpoints/cvrp_world_model_v2.pt}"
MARGIN="${2:-1.0}"

echo "=== Greedy (4 ep) ==="
python scripts/evaluate.py --size 200 --planner greedy \
  --rrc 0 --device cpu --episodes 4 --batch-size 1 2>&1 | grep -E "gap=|overrides="

echo ""
echo "=== Rollout WM: $WM margin=$MARGIN (4 ep) ==="
python scripts/evaluate.py --size 200 --planner rollout_wm \
  --wm-checkpoint "$WM" --wm-top-k 3 --wm-margin "$MARGIN" \
  --rrc 0 --device cpu --episodes 4 --batch-size 1 2>&1 | grep -E "gap=|overrides="

echo ""
echo "Paper baseline to beat (128 ep): 3.206%"
