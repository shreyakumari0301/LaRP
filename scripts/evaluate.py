#!/usr/bin/env python3
"""Unified evaluation entry point for released LEHD-MnLP checkpoints."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_DATA = {
    "tsp": {
        100: ("TSP/data/test_TSP100_n1w.txt", 10000, 10000),
        200: ("TSP/data/test_TSP200_n128.txt", 128, 128),
        500: ("TSP/data/test_TSP500_n128.txt", 128, 128),
        1000: ("TSP/data/test_TSP1000_n128.txt", 128, 128),
    },
    "cvrp": {
        100: ("CVRP/data/vrp100_test_lkh.txt", 10000, 10000),
        200: ("CVRP/data/vrp200_test_lkh.txt", 128, 128),
        500: ("CVRP/data/vrp500_test_lkh.txt", 128, 128),
        1000: ("CVRP/data/vrp1000_test_lkh.txt", 128, 128),
    },
}


DEFAULT_CHECKPOINT = {
    "tsp": "checkpoints/tsp_mnlp.pt",
    "cvrp": "checkpoints/cvrp_mnlp.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("tsp", "cvrp"), required=True)
    parser.add_argument("--size", type=int, choices=(100, 200, 500, 1000), required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--rrc", type=int, default=0, help="RRC budget; use 0 for greedy evaluation.")
    parser.add_argument("--episodes", type=int, help="Override number of test episodes.")
    parser.add_argument("--batch-size", type=int, help="Override evaluation batch size.")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Use 'cpu' or 'cuda:N'. Defaults to cuda:0.",
    )
    return parser.parse_args()


def resolve_path(path: Path | str) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def checkpoint_as_model_dir(checkpoint: Path) -> tempfile.TemporaryDirectory[str]:
    tmp = tempfile.TemporaryDirectory(prefix="lehd_mnlp_eval_")
    link = Path(tmp.name) / "checkpoint-0.pt"
    try:
        link.symlink_to(checkpoint)
    except OSError:
        import shutil

        shutil.copy2(checkpoint, link)
    return tmp


def configure_paths(problem: str) -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / problem.upper()))


def parse_device(device_arg: str) -> tuple[bool, int]:
    if device_arg == "cpu":
        return False, 0
    if device_arg.startswith("cuda"):
        if ":" in device_arg:
            return True, int(device_arg.split(":", 1)[1])
        return True, 0
    raise ValueError("--device must be 'cpu', 'cuda', or 'cuda:N'")


def run_tsp(args: argparse.Namespace, data_path: Path, checkpoint_dir: str) -> tuple[float, float, float]:
    configure_paths("tsp")
    from TSPTester import TSPTester

    use_cuda, cuda_device_num = parse_device(args.device)
    env_params = {
        "mode": "test",
        "data_path": str(data_path),
        "sub_path": False,
        "RRC_budget": args.rrc,
    }
    model_params = {
        "mode": "test",
        "embedding_dim": 128,
        "sqrt_embedding_dim": 128 ** (1 / 2),
        "decoder_layer_num": 6,
        "qkv_dim": 16,
        "head_num": 8,
        "ff_hidden_dim": 512,
        "mtp_depth": 4,
        "mtp_weight": 0.5,
    }
    tester_params = {
        "use_cuda": use_cuda,
        "cuda_device_num": cuda_device_num,
        "test_episodes": args.episodes,
        "test_batch_size": args.batch_size,
        "model_load": {"path": checkpoint_dir, "epoch": 0},
    }
    return TSPTester(env_params, model_params, tester_params).run()


def run_cvrp(args: argparse.Namespace, data_path: Path, checkpoint_dir: str) -> tuple[float, float, float]:
    configure_paths("cvrp")
    from VRPTester import VRPTester

    use_cuda, cuda_device_num = parse_device(args.device)
    env_params = {
        "mode": "test",
        "data_path": str(data_path),
        "sub_path": False,
        "RRC_budget": args.rrc,
    }
    model_params = {
        "mode": "test",
        "embedding_dim": 128,
        "sqrt_embedding_dim": 128 ** (1 / 2),
        "decoder_layer_num": 6,
        "qkv_dim": 16,
        "head_num": 8,
        "ff_hidden_dim": 512,
        "mtp_depth": 4,
        "mtp_weight": 0.3,
    }
    tester_params = {
        "use_cuda": use_cuda,
        "cuda_device_num": cuda_device_num,
        "test_episodes": args.episodes,
        "test_batch_size": args.batch_size,
        "model_load": {"path": checkpoint_dir, "epoch": 0},
    }
    return VRPTester(env_params, model_params, tester_params).run()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    default_data, default_episodes, default_batch_size = DEFAULT_DATA[args.problem][args.size]
    data_path = resolve_path(args.data or default_data)
    checkpoint_path = resolve_path(args.checkpoint or DEFAULT_CHECKPOINT[args.problem])

    args.episodes = args.episodes or default_episodes
    args.batch_size = args.batch_size or default_batch_size

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "For CVRP, only use the verified paper checkpoint."
        )
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if args.rrc < 0:
        raise ValueError("--rrc must be non-negative")

    os.chdir(ROOT)
    with checkpoint_as_model_dir(checkpoint_path) as checkpoint_dir:
        if args.problem == "tsp":
            optimal, student, gap = run_tsp(args, data_path, checkpoint_dir)
        else:
            optimal, student, gap = run_cvrp(args, data_path, checkpoint_dir)

    print(f"optimal={optimal:.6f}")
    print(f"student={student:.6f}")
    print(f"gap={gap:.6f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
