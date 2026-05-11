# LEHD-MnLP

This repository contains the release code for Multi-node Lookahead Prediction
(MnLP) on the LEHD backbone for TSP and CVRP.

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

Expected greedy paper results are documented in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Attribution

This implementation builds on the LEHD codebase:

```bibtex
@inproceedings{luo2023neural,
  title={Neural Combinatorial Optimization with Heavy Decoder: Toward Large Scale Generalization},
  author={Fu Luo and Xi Lin and Fei Liu and Qingfu Zhang and Zhenkun Wang},
  booktitle={NeurIPS},
  year={2023}
}
```

Please also cite the MnLP paper when using this release.

## License And Use

The upstream LEHD code states that it is for non-commercial use only. This
release preserves that notice. Contact the authors for business or commercial
use.
