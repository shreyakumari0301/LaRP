# Multi-node Lookahead Prediction for Vehicle Routing Problems (IJCAI'26)

This repository contains the code for Multi-node Lookahead Prediction
(MnLP) for CVRP, with rollout planning via a learned value world model.

MnLP is a training-time auxiliary objective. The released checkpoint runs with
the standard LEHD-style autoregressive decoder at inference time, so no MnLP
modules are used to add inference overhead beyond loading the trained model
weights.

## Repository Layout

```text
CVRP/                 CVRP model, environment, training, and testing code
utils/                Shared logging and utility helpers
scripts/evaluate.py   Greedy / rollout world-model evaluation entry point
checkpoints/          Released CVRP checkpoint
```

The paper CVRP checkpoint is included:

```text
checkpoints/cvrp_mnlp.pt
```

## Setup

The experiments were run with Python 3.8.6 and PyTorch 1.12.1.

```bash
pip install -r requirements.txt
```

CUDA is recommended for the full benchmark evaluations. CPU mode is supported
for small smoke tests by passing `--device cpu`.

## Data

Datasets are not committed to this repository. Place the benchmark files under
the original locations expected by the code:

```text
CVRP/data/vrp100_test_lkh.txt
CVRP/data/vrp200_test_lkh.txt
CVRP/data/vrp500_test_lkh.txt
CVRP/data/vrp1000_test_lkh.txt
```

Training data is also excluded because the original files are multi-GB.
The original training and test data can be found in the repository of LEHD (https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD).

## Evaluation

Run greedy CVRP200 with the released checkpoint:

```bash
python scripts/evaluate.py \
  --size 200 \
  --data CVRP/data/vrp200_test_lkh.txt \
  --checkpoint checkpoints/cvrp_mnlp.pt \
  --rrc 0
```

Rollout planning with a trained world model:

```bash
python scripts/evaluate.py \
  --size 200 \
  --planner rollout_wm \
  --wm-checkpoint checkpoints/cvrp_world_model_policy.pt \
  --wm-top-k 3 \
  --wm-margin 1.0 \
  --rrc 0 \
  --device cpu
```

## Reference

Please cite the paper if you used the code:

```bibtex
@inproceedings{jiang2026learning,
  title={Learning with Foresight: Enhancing Neural Routing Policy via Multi-Node Lookahead Prediction},
  author={Xia Jiang and Yaoxin Wu and Yew-Soon Ong and Yingqian Zhang},
  booktitle={International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2026}
}
```
