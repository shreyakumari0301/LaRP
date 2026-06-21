#!/usr/bin/env bash
# Fine-tune WM from policy checkpoint: mixed prefixes + stronger ranking loss.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mnlp
cd /mnt/c/Users/SK/LaRP/MnLP

mkdir -p logs
LOG="logs/wm_finetune_$(date +%Y%m%d_%H%M%S).log"
PIDFILE="logs/wm_finetune.pid"

INIT=checkpoints/cvrp_world_model_policy.pt
if [[ ! -f "$INIT" ]]; then
  INIT=checkpoints/cvrp_world_model.pt
fi

nohup python CVRP/train_world_model.py \
  --device cpu \
  --epochs 20 \
  --steps-per-epoch 256 \
  --batch-size 16 \
  --episodes 128 \
  --lr 1e-4 \
  --trajectory-source mixed \
  --policy-checkpoint checkpoints/cvrp_mnlp.pt \
  --init-checkpoint "$INIT" \
  --ranking-weight 1.0 \
  --ranking-fraction 0.4 \
  --ranking-margin 0.1 \
  --output checkpoints/cvrp_world_model_policy_ft.pt \
  > "$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "Started WM fine-tuning in background."
echo "  PID:  $(cat "$PIDFILE")"
echo "  Log:  $LOG"
echo "  Init: $INIT"
echo ""
echo "Monitor:  tail -f $LOG"
echo "Stop:     kill \$(cat $PIDFILE)"
