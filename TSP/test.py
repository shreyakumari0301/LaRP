"""Compatibility evaluation hook used by the TSP trainers.

For command-line evaluation of released checkpoints, prefer
`scripts/evaluate.py` from the repository root.
"""

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")
sys.path.insert(0, "../..")

from TSPTester import TSPTester as Tester


DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

TEST_FILES = {
    100: ("test_TSP100_n1w.txt", 10000, 10000),
    200: ("test_TSP200_n128.txt", 128, 128),
    500: ("test_TSP500_n128.txt", 128, 128),
    1000: ("test_TSP1000_n128.txt", 128, 128),
}


def main_test(epoch, path, use_RRC=None, cuda_device_num=None, problem_size=500):
    filename, episodes, batch_size = TEST_FILES[problem_size]
    rrc_budget = 0 if use_RRC is False or use_RRC is None else int(use_RRC)
    tester_params = {
        "use_cuda": USE_CUDA,
        "cuda_device_num": CUDA_DEVICE_NUM if cuda_device_num is None else cuda_device_num,
        "test_episodes": 100 if DEBUG_MODE else episodes,
        "test_batch_size": batch_size,
        "model_load": {"path": path, "epoch": epoch},
    }
    env_params = {
        "mode": "test",
        "data_path": os.path.abspath(f"data/{filename}"),
        "sub_path": False,
        "RRC_budget": rrc_budget,
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
    tester = Tester(env_params=env_params, model_params=model_params, tester_params=tester_params)
    return tester.run()
