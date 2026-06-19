#!/usr/bin/env bash
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mnlp
cd /mnt/c/Users/SK/LaRP/MnLP

echo "=== Training world model on policy prefixes ==="
python CVRP/train_world_model.py \
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
  --output checkpoints/cvrp_world_model_policy.pt

echo ""
echo "=== Greedy baseline (4 episodes) ==="
python scripts/evaluate.py --size 200 --checkpoint checkpoints/cvrp_mnlp.pt \
  --planner greedy --rrc 0 --device cpu --episodes 4 --batch-size 1

echo ""
echo "=== Rollout world model (4 episodes, margin gate) ==="
python scripts/evaluate.py --size 200 --checkpoint checkpoints/cvrp_mnlp.pt \
  --planner rollout_wm --wm-checkpoint checkpoints/cvrp_world_model_policy.pt \
  --wm-top-k 3 --wm-margin 1.0 --rrc 0 --device cpu --episodes 4 --batch-size 1
