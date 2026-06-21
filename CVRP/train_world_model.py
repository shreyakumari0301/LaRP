#!/usr/bin/env python3
"""Train the CVRP value world model from LKH and/or greedy policy trajectories."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "CVRP"))

from VRPModel import VRPModel
from VRPEnv import VRPEnv
from policy_rollout import (
    alternate_action_at_step,
    build_state_from_solution,
    complete_greedy_from_prefix,
    greedy_rollout,
    state_after_action,
    topk_ranking_pairs_at_step,
)
from simulator import SimState, apply_action
from world_model import CVRPWorldModel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "CVRP/data/vrp200_test_lkh.txt")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--val-episodes", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps-per-epoch", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--max-nodes", type=int, default=1001)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/cvrp_world_model.pt")
    parser.add_argument(
        "--trajectory-source",
        choices=("lkh", "policy", "mixed"),
        default="lkh",
        help="Train on LKH prefixes, greedy policy prefixes, or a 50/50 mix.",
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=Path,
        default=ROOT / "checkpoints/cvrp_mnlp.pt",
        help="MnLP checkpoint used to generate policy trajectories.",
    )
    parser.add_argument(
        "--ranking-weight",
        type=float,
        default=0.5,
        help="Weight for pairwise ranking hinge loss (policy/mixed only).",
    )
    parser.add_argument(
        "--ranking-fraction",
        type=float,
        default=0.25,
        help="Fraction of each batch devoted to ranking pairs.",
    )
    parser.add_argument(
        "--ranking-margin",
        type=float,
        default=0.05,
        help="Hinge margin in normalized cost units for ranking loss.",
    )
    parser.add_argument(
        "--rank-top-k",
        type=int,
        default=3,
        help="Top-K candidates used for ranking loss (should match inference top_k).",
    )
    parser.add_argument(
        "--max-rank-pairs",
        type=int,
        default=4,
        help="Max ranking pairs sampled per training step.",
    )
    parser.add_argument(
        "--value-weight",
        type=float,
        default=1.0,
        help="Weight for value regression loss relative to ranking loss scale.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional WM checkpoint to fine-tune from (loads model weights only).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_policy_model(checkpoint_path: Path, device: torch.device) -> VRPModel:
    model = VRPModel(
        mode="test",
        embedding_dim=128,
        sqrt_embedding_dim=128 ** (1 / 2),
        decoder_layer_num=6,
        qkv_dim=16,
        head_num=8,
        ff_hidden_dim=512,
        mtp_depth=4,
        mtp_weight=0.3,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def compute_cost_scale(env: VRPEnv, episode_indices: list[int]) -> float:
    costs = []
    for idx in episode_indices:
        env.load_problems(idx, 1)
        total = env._get_travel_distance_2(env.problems, env.solution).item()
        costs.append(total)
    return max(sum(costs) / len(costs), 1.0)


def prefix_cal_length(problems: torch.Tensor, order_node: torch.Tensor, order_flag: torch.Tensor) -> torch.Tensor:
    """Travel distance for a partial solution (T steps, T <= V)."""
    order_node_ = order_node.clone()
    order_flag_ = order_flag.clone()

    index_small = torch.le(order_flag_, 0.5)
    index_bigger = torch.gt(order_flag_, 0.5)
    order_flag_[index_small] = order_node_[index_small]
    order_flag_[index_bigger] = 0

    roll_node = order_node_.roll(dims=1, shifts=1)
    seq_len = order_node_.shape[1]

    order_gathering_index = order_node_.unsqueeze(2).expand(-1, seq_len, 2)
    order_loc = problems.gather(dim=1, index=order_gathering_index)

    roll_gathering_index = roll_node.unsqueeze(2).expand(-1, seq_len, 2)
    roll_loc = problems.gather(dim=1, index=roll_gathering_index)

    flag_gathering_index = order_flag_.unsqueeze(2).expand(-1, seq_len, 2)
    flag_loc = problems.gather(dim=1, index=flag_gathering_index)

    order_lengths = ((order_loc - flag_loc) ** 2)

    order_flag_[:, 0] = 0
    flag_gathering_index = order_flag_.unsqueeze(2).expand(-1, seq_len, 2)
    flag_loc = problems.gather(dim=1, index=flag_gathering_index)

    roll_lengths = ((roll_loc - flag_loc) ** 2)
    return (order_lengths.sum(2).sqrt() + roll_lengths.sum(2).sqrt()).sum(1)


def prefix_cost(problems: torch.Tensor, solution: torch.Tensor, step: int) -> float:
    if step <= 0:
        return 0.0
    prefix_node = solution[:, :step, 0]
    prefix_flag = solution[:, :step, 1]
    return float(prefix_cal_length(problems[:, :, [0, 1]], prefix_node, prefix_flag).item())


def build_state(
    problems: torch.Tensor,
    solution: torch.Tensor,
    step: int,
    raw_capacity: float,
    prefix_cost_value: float,
    device: torch.device,
) -> SimState:
    sim = SimState(
        problems=problems,
        raw_capacity=raw_capacity,
        selected_node_list=torch.zeros(1, 0, dtype=torch.long, device=device),
        selected_student_list=torch.zeros(1, 0, dtype=torch.long, device=device),
        selected_student_flag=torch.zeros(1, 0, dtype=torch.long, device=device),
        selected_count=0,
        accumulated_cost=torch.tensor([prefix_cost_value], device=device),
    )
    for t in range(step):
        sim = apply_action(sim, solution[:, t, 0], solution[:, t, 1])
    sim.accumulated_cost = torch.tensor([prefix_cost_value], device=device)
    return sim


def sample_lkh_transition(env: VRPEnv, episode_idx: int, device: torch.device):
    env.load_problems(episode_idx, 1)
    problems = env.problems.clone()
    solution = env.solution.clone()
    raw_capacity = float(env.raw_data_capacity.ravel()[0].item())
    total_cost = float(env._get_travel_distance_2(problems, solution).item())
    step = random.randrange(solution.shape[1])
    prefix_cost_value = prefix_cost(problems, solution, step)
    remaining = max(total_cost - prefix_cost_value, 0.0)
    state = build_state(problems, solution, step, raw_capacity, prefix_cost_value, device)
    return state, remaining


class PolicyTrajectoryCache:
    def __init__(self, env: VRPEnv, policy_model: VRPModel, device: torch.device):
        self.env = env
        self.policy_model = policy_model
        self.device = device
        self._cache: dict[int, tuple] = {}

    def prewarm(self, episode_indices: list[int]) -> None:
        for idx in tqdm(episode_indices, desc="policy cache", ascii=True):
            self.get(idx)

    def get(self, episode_idx: int):
        if episode_idx not in self._cache:
            self._cache[episode_idx] = greedy_rollout(
                self.env, self.policy_model, episode_idx, self.device
            )
        return self._cache[episode_idx]


def sample_policy_transition(cache: PolicyTrajectoryCache, episode_idx: int, device: torch.device):
    problems, policy_solution, _, policy_total, raw_capacity = cache.get(episode_idx)
    num_steps = policy_solution.shape[1]
    step = random.randrange(num_steps)
    prefix_cost_value = prefix_cost(problems, policy_solution, step)
    remaining = max(policy_total - prefix_cost_value, 0.0)
    state = build_state_from_solution(
        problems, policy_solution, step, raw_capacity, prefix_cost_value, device
    )
    return state, remaining


def sample_policy_ranking_pair(
    env: VRPEnv,
    cache: PolicyTrajectoryCache,
    policy_model: VRPModel,
    episode_idx: int,
    device: torch.device,
):
    problems, policy_solution, _, policy_total, raw_capacity = cache.get(episode_idx)
    num_steps = policy_solution.shape[1]
    step = random.randrange(num_steps)

    alt = alternate_action_at_step(env, policy_model, episode_idx, step, device, top_k=5)
    if alt is None:
        return None

    alt_node, alt_flag = alt
    prefix_cost_value = prefix_cost(problems, policy_solution, step)

    greedy_prefix_cost = prefix_cost(problems, policy_solution, step + 1)
    remaining_greedy = max(policy_total - greedy_prefix_cost, 0.0)
    state_greedy = build_state_from_solution(
        problems, policy_solution, step + 1, raw_capacity, greedy_prefix_cost, device
    )

    prefix_nodes = policy_solution[:, :step, 0]
    prefix_flags = policy_solution[:, :step, 1]
    alt_nodes = torch.cat((prefix_nodes, alt_node.unsqueeze(1)), dim=1)
    alt_flags = torch.cat((prefix_flags, alt_flag.unsqueeze(1)), dim=1)
    _, alt_total = complete_greedy_from_prefix(
        env, policy_model, episode_idx, alt_nodes, alt_flags, device
    )
    alt_prefix_cost = prefix_cost(
        problems,
        torch.cat((alt_nodes.unsqueeze(2), alt_flags.unsqueeze(2)), dim=2),
        step + 1,
    )
    remaining_alt = max(alt_total - alt_prefix_cost, 0.0)
    state_alt = state_after_action(
        problems,
        policy_solution,
        step,
        raw_capacity,
        prefix_cost_value,
        alt_node,
        alt_flag,
        device,
    )

    if remaining_greedy <= remaining_alt:
        return state_greedy, state_alt, remaining_greedy, remaining_alt
    return state_alt, state_greedy, remaining_alt, remaining_greedy


def sample_policy_topk_ranking_pairs(
    env: VRPEnv,
    cache: PolicyTrajectoryCache,
    policy_model: VRPModel,
    episode_idx: int,
    device: torch.device,
    top_k: int,
    max_pairs: int,
) -> list[tuple[SimState, SimState]]:
    problems, policy_solution, _, _, raw_capacity = cache.get(episode_idx)
    step = random.randrange(policy_solution.shape[1])
    pairs = topk_ranking_pairs_at_step(
        env,
        policy_model,
        episode_idx,
        step,
        problems,
        policy_solution,
        raw_capacity,
        device,
        top_k=top_k,
    )
    if not pairs:
        return []
    random.shuffle(pairs)
    return pairs[:max_pairs]


def collate_batch(states: list[SimState], targets: list[float], cost_scale: float, device: torch.device):
    batch_size = len(states)
    problems = torch.cat([s.problems for s in states], dim=0)
    capacities = torch.tensor([s.raw_capacity for s in states], device=device, dtype=problems.dtype)
    prefix_lens = torch.tensor(
        [s.selected_student_list.shape[1] for s in states],
        dtype=torch.long,
        device=device,
    )
    accumulated = torch.tensor(
        [float(s.accumulated_cost.item()) for s in states],
        device=device,
        dtype=problems.dtype,
    )

    max_len = int(prefix_lens.max().item()) if batch_size > 0 else 0
    if max_len > 0:
        selected_student_list = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        selected_student_flag = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        selected_node_list = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        for i, state in enumerate(states):
            plen = state.selected_count
            if plen > 0:
                selected_student_list[i, :plen] = state.selected_student_list[0, :plen]
                selected_student_flag[i, :plen] = state.selected_student_flag[0, :plen]
                selected_node_list[i, :plen] = state.selected_node_list[0, :plen]
    else:
        selected_student_list = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        selected_student_flag = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        selected_node_list = torch.zeros(batch_size, 0, dtype=torch.long, device=device)

    batched = SimState(
        problems=problems,
        raw_capacity=float(capacities.mean().item()),
        selected_node_list=selected_node_list,
        selected_student_list=selected_student_list,
        selected_student_flag=selected_student_flag,
        selected_count=int(prefix_lens.max().item()),
        accumulated_cost=accumulated,
        prefix_lens=prefix_lens,
        capacities=capacities,
    )

    target_tensor = torch.tensor(
        [t / cost_scale for t in targets],
        device=device,
        dtype=problems.dtype,
    )
    return batched, target_tensor, capacities


def ranking_loss(
    model: CVRPWorldModel,
    better_states: list[SimState],
    worse_states: list[SimState],
    margin: float,
    cost_scale: float,
    device: torch.device,
) -> torch.Tensor:
    if not better_states:
        return torch.tensor(0.0, device=device)

    better_batch, _, _ = collate_batch(better_states, [0.0] * len(better_states), cost_scale, device)
    worse_batch, _, _ = collate_batch(worse_states, [0.0] * len(worse_states), cost_scale, device)
    pred_better = model(better_batch)
    pred_worse = model(worse_batch)
    return F.relu(pred_worse - pred_better + margin).mean()


def evaluate(
    model: CVRPWorldModel,
    env: VRPEnv,
    episode_indices: list[int],
    cost_scale: float,
    device: torch.device,
    trajectory_source: str,
    policy_cache: PolicyTrajectoryCache | None,
):
    model.eval()
    abs_errors = []
    with torch.no_grad():
        for idx in episode_indices:
            if trajectory_source == "lkh":
                state, remaining = sample_lkh_transition(env, idx, device)
            else:
                state, remaining = sample_policy_transition(policy_cache, idx, device)
            pred = model(state)
            abs_errors.append(abs(float(pred.item()) * cost_scale - remaining))
    model.train()
    return sum(abs_errors) / max(len(abs_errors), 1)


def train():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    os.chdir(ROOT)

    env = VRPEnv(mode="test", data_path=str(args.data), sub_path=False, RRC_budget=0)
    env.load_raw_data(args.episodes)

    policy_model = None
    policy_cache = None
    if args.trajectory_source in ("policy", "mixed"):
        if not args.policy_checkpoint.exists():
            raise FileNotFoundError(f"Policy checkpoint not found: {args.policy_checkpoint}")
        policy_model = load_policy_model(args.policy_checkpoint, device)
        policy_cache = PolicyTrajectoryCache(env, policy_model, device)

    all_indices = list(range(args.episodes))
    random.shuffle(all_indices)
    val_count = min(args.val_episodes, max(1, args.episodes // 5))
    val_indices = all_indices[:val_count]
    train_indices = all_indices[val_count:]

    if policy_cache is not None:
        policy_cache.prewarm(train_indices)

    cost_scale = compute_cost_scale(env, train_indices)
    print(
        f"trajectory_source={args.trajectory_source} cost_scale={cost_scale:.4f} "
        f"train={len(train_indices)} val={len(val_indices)}"
    )

    model = CVRPWorldModel(
        max_nodes=args.max_nodes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        cost_scale=cost_scale,
    ).to(device)

    if args.init_checkpoint is not None:
        if not args.init_checkpoint.exists():
            raise FileNotFoundError(f"Init checkpoint not found: {args.init_checkpoint}")
        init_ckpt = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(init_ckpt["model_state_dict"])
        if "cost_scale" in init_ckpt:
            model.cost_scale = float(init_ckpt["cost_scale"])
            cost_scale = model.cost_scale
        print(f"loaded init weights from {args.init_checkpoint}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    use_ranking = args.ranking_weight > 0 and args.trajectory_source in ("policy", "mixed")
    val_source = "policy" if args.trajectory_source == "mixed" else args.trajectory_source

    best_val_mae = float("inf")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_rank_loss = 0.0
        epoch_steps = 0
        max_pred = 0.0
        max_target = 0.0

        for _ in tqdm(range(args.steps_per_epoch), desc=f"epoch {epoch + 1}", ascii=True):
            ranking_count = 0
            if use_ranking:
                ranking_count = max(1, int(args.batch_size * args.ranking_fraction))
            value_count = args.batch_size - ranking_count

            batch_states = []
            batch_targets = []
            better_states = []
            worse_states = []

            for _ in range(value_count):
                episode_idx = random.choice(train_indices)
                if args.trajectory_source == "lkh":
                    state, remaining = sample_lkh_transition(env, episode_idx, device)
                elif args.trajectory_source == "policy":
                    state, remaining = sample_policy_transition(policy_cache, episode_idx, device)
                else:
                    if random.random() < 0.5:
                        state, remaining = sample_lkh_transition(env, episode_idx, device)
                    else:
                        state, remaining = sample_policy_transition(policy_cache, episode_idx, device)
                batch_states.append(state)
                batch_targets.append(remaining)

            for _ in range(ranking_count):
                episode_idx = random.choice(train_indices)
                pairs = sample_policy_topk_ranking_pairs(
                    env,
                    policy_cache,
                    policy_model,
                    episode_idx,
                    device,
                    top_k=args.rank_top_k,
                    max_pairs=args.max_rank_pairs,
                )
                if not pairs:
                    pair = sample_policy_ranking_pair(
                        env, policy_cache, policy_model, episode_idx, device
                    )
                    if pair is None:
                        state, remaining = sample_policy_transition(policy_cache, episode_idx, device)
                        batch_states.append(state)
                        batch_targets.append(remaining)
                    else:
                        better, worse, _, _ = pair
                        better_states.append(better)
                        worse_states.append(worse)
                else:
                    for better, worse in pairs:
                        better_states.append(better)
                        worse_states.append(worse)

            if batch_states:
                batched, target_tensor, _ = collate_batch(
                    batch_states, batch_targets, cost_scale, device
                )
                pred = model(batched)
                value_loss = F.smooth_l1_loss(pred, target_tensor)
                max_pred = max(max_pred, float(pred.detach().max().item()))
                max_target = max(max_target, float(target_tensor.max().item()))
            else:
                value_loss = torch.tensor(0.0, device=device)

            rank_loss = ranking_loss(
                model,
                better_states,
                worse_states,
                args.ranking_margin,
                cost_scale,
                device,
            )
            loss = args.value_weight * value_loss + args.ranking_weight * rank_loss

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += float(value_loss.item())
            epoch_rank_loss += float(rank_loss.item())
            epoch_steps += 1

        train_loss = epoch_loss / max(epoch_steps, 1)
        rank_loss_avg = epoch_rank_loss / max(epoch_steps, 1)
        val_mae = evaluate(
            model, env, val_indices, cost_scale, device, val_source, policy_cache
        )
        print(
            f"epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.6f} rank_loss={rank_loss_avg:.6f} val_mae={val_mae:.4f} "
            f"max_pred={max_pred:.4f} max_target={max_target:.4f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "arch": "transformer_v2",
                    "model_state_dict": model.state_dict(),
                    "max_nodes": args.max_nodes,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "num_heads": args.num_heads,
                    "cost_scale": cost_scale,
                    "data_path": str(args.data),
                    "episodes": args.episodes,
                    "trajectory_source": args.trajectory_source,
                    "val_mae": val_mae,
                },
                args.output,
            )

    print(f"saved world model to {args.output} (best val_mae={best_val_mae:.4f})")


if __name__ == "__main__":
    train()
