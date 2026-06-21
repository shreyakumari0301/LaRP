#!/usr/bin/env bash
# v2: policy-only WM, top-K ranking loss aligned with inference (top_k=3).
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mnlp
cd /mnt/c/Users/SK/LaRP/MnLP

mkdir -p logs
LOG="logs/wm_v2_$(date +%Y%m%d_%H%M%S).log"
PIDFILE="logs/wm_v2.pid"

nohup python CVRP/train_world_model.py \
  --device cpu \
  --epochs 15 \
  --steps-per-epoch 256 \
  --batch-size 16 \
  --episodes 128 \
  --lr 1e-4 \
  --trajectory-source policy \
  --policy-checkpoint checkpoints/cvrp_mnlp.pt \
  --rank-top-k 3 \
  --max-rank-pairs 4 \
  --ranking-weight 2.0 \
  --ranking-fraction 0.6 \
  --ranking-margin 0.1 \
  --value-weight 0.3 \
  --output checkpoints/cvrp_world_model_v2.pt \
  > "$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "Started WM v2 training."
echo "  PID:  $(cat "$PIDFILE")"
echo "  Log:  $LOG"
echo ""
echo "Monitor:  tail -f $LOG"
echo "Stop:     kill \$(cat $PIDFILE)"
