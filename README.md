# Multi-node Lookahead Prediction for Vehicle Routing Problems (IJCAI'26)

This repository contains the code for Multi-node Lookahead Prediction
(MnLP) for TSP and CVRP.

MnLP is a training-time auxiliary objective. The released checkpoints run with
the standard LEHD-style autoregressive decoder at inference time, so no MnLP
modules are used to add inference overhead beyond loading the trained model
weights.

## Repository Layout

```text
TSP/                  TSP model, environment, training, and testing code
CVRP/                 CVRP model, environment, training, and testing code
utils/                Shared logging and utility helpers
scripts/evaluate.py   Unified greedy/RRC evaluation entry point
checkpoints/          Released method checkpoints
```

The paper TSP and CVRP checkpoints are included:

```text
checkpoints/tsp_mnlp.pt
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
TSP/data/test_TSP100_n1w.txt
TSP/data/test_TSP200_n128.txt
TSP/data/test_TSP500_n128.txt
TSP/data/test_TSP1000_n128.txt
CVRP/data/vrp100_test_lkh.txt
CVRP/data/vrp200_test_lkh.txt
CVRP/data/vrp500_test_lkh.txt
CVRP/data/vrp1000_test_lkh.txt
```

Training data is also excluded because the original files are multi-GB.
The original training and test data can be found in the repository of LEHD (https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD).

## Evaluation

Run greedy TSP1000 with the released checkpoint:

```bash
python scripts/evaluate.py \
  --problem tsp \
  --size 1000 \
  --data TSP/data/test_TSP1000_n128.txt \
  --checkpoint checkpoints/tsp_mnlp.pt \
  --rrc 0
```

Run greedy CVRP1000 with the released checkpoint:

```bash
python scripts/evaluate.py \
  --problem cvrp \
  --size 1000 \
  --data CVRP/data/vrp1000_test_lkh.txt \
  --checkpoint checkpoints/cvrp_mnlp.pt \
  --rrc 0
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

