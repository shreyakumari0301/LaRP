#!/usr/bin/env bash
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mnlp
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
cd /mnt/c/Users/SK/LaRP/MnLP
python scripts/evaluate.py \
  --size 200 \
  --checkpoint checkpoints/cvrp_mnlp.pt \
  --rrc 0 \
  --episodes 4 \
  --batch-size 4 \
  --device cpu
