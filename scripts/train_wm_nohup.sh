#!/usr/bin/env bash
# Run WM training detached from the terminal (survives SSH/terminal close).
# Does NOT survive laptop sleep/shutdown — disable sleep or keep machine awake.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mnlp
cd /mnt/c/Users/SK/LaRP/MnLP

mkdir -p logs
LOG="logs/wm_train_$(date +%Y%m%d_%H%M%S).log"
PIDFILE="logs/wm_train.pid"

nohup python CVRP/train_world_model.py \
  --device cpu \
  --epochs 15 \
  --steps-per-epoch 256 \
  --batch-size 16 \
  --episodes 128 \
  --lr 3e-4 \
  --trajectory-source policy \
  --policy-checkpoint checkpoints/cvrp_mnlp.pt \
  --ranking-weight 0.5 \
  --ranking-fraction 0.25 \
  --output checkpoints/cvrp_world_model_policy.pt \
  > "$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "Started WM training in background."
echo "  PID:  $(cat "$PIDFILE")"
echo "  Log:  $LOG"
echo ""
echo "Monitor:  tail -f $LOG"
echo "Stop:     kill \$(cat $PIDFILE)"
