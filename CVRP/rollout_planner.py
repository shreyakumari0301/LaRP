"""Rollout planner using top-K policy candidates and a learned value world model."""

from __future__ import annotations

import torch

from simulator import SimState, apply_action, snapshot_env, restore_env
from world_model import CVRPWorldModel


def decode_action(idx: int, split_line: int) -> tuple[int, int]:
    """Map joint decoder index to (customer_node, depot_flag)."""
    if idx >= split_line:
        return idx - split_line + 1, 1
    return idx + 1, 0


def visited_customer_set(visited_nodes: torch.Tensor, batch_idx: int) -> set[int]:
    visited = set()
    if visited_nodes.shape[1] == 0:
        return visited
    for node in visited_nodes[batch_idx]:
        value = int(node.item())
        if value > 0:
            visited.add(value)
    return visited


def filter_feasible_candidates(
    probs: torch.Tensor,
    split_line: int,
    visited_nodes: torch.Tensor,
    top_k: int,
    prob_eps: float = 1e-6,
):
    """Keep top-K decoder-valid actions that do not revisit customers."""
    batch_size = probs.shape[0]
    nodes = torch.zeros(batch_size, top_k, dtype=torch.long, device=probs.device)
    flags = torch.zeros(batch_size, top_k, dtype=torch.long, device=probs.device)
    valid = torch.zeros(batch_size, top_k, dtype=torch.bool, device=probs.device)

    for b in range(batch_size):
        visited = visited_customer_set(visited_nodes, b)
        ranked = torch.argsort(probs[b], descending=True)
        filled = 0
        for idx in ranked:
            idx = int(idx.item())
            if probs[b, idx].item() <= prob_eps:
                break
            node, flag = decode_action(idx, split_line)
            if flag == 0 and node in visited:
                continue
            if flag == 1 and node in visited:
                continue
            nodes[b, filled] = node
            flags[b, filled] = flag
            valid[b, filled] = True
            filled += 1
            if filled >= top_k:
                break
    return nodes, flags, valid


def policy_action_from_probs(probs: torch.Tensor, split_line: int):
    idx = probs.argmax(dim=1)
    batch_size = probs.shape[0]
    nodes = torch.zeros(batch_size, dtype=torch.long, device=probs.device)
    flags = torch.zeros(batch_size, dtype=torch.long, device=probs.device)
    for b in range(batch_size):
        node, flag = decode_action(int(idx[b].item()), split_line)
        nodes[b] = node
        flags[b] = flag
    return nodes, flags


class RolloutWMPlanner:
    def __init__(
        self,
        model,
        world_model: CVRPWorldModel,
        top_k: int = 5,
        horizon: int = 5,
        score_margin: float = 1.0,
        device: torch.device | None = None,
    ):
        self.model = model
        self.world_model = world_model
        self.top_k = top_k
        self.horizon = horizon
        self.score_margin = score_margin
        self.device = device or torch.device("cpu")

    def _sim_from_snap(self, snap: dict, raw_data_capacity) -> SimState:
        return SimState(
            problems=snap["problems"].clone(),
            raw_capacity=float(raw_data_capacity.ravel()[0].item()),
            selected_node_list=snap["selected_node_list"].clone(),
            selected_student_list=snap["selected_student_list"].clone(),
            selected_student_flag=snap["selected_student_flag"].clone(),
            selected_count=snap["selected_count"],
            accumulated_cost=torch.zeros(snap["problems"].shape[0], device=snap["problems"].device),
        )

    def _rollout_score(
        self,
        env,
        snap: dict,
        first_node: torch.Tensor,
        first_flag: torch.Tensor,
        start_step: int,
        raw_data_capacity,
        origin_problem: torch.Tensor,
    ) -> torch.Tensor:
        sim = self._sim_from_snap(snap, raw_data_capacity)
        node, flag = int(first_node.item()), int(first_flag.item())
        if start_step == 0:
            flag = 1

        visited = visited_customer_set(sim.selected_student_list, 0)
        if flag == 0 and node in visited:
            return torch.tensor([float("inf")], device=sim.problems.device)
        if flag == 1 and node in visited:
            return torch.tensor([float("inf")], device=sim.problems.device)

        sim = apply_action(
            sim,
            torch.tensor([node], dtype=torch.long, device=sim.problems.device),
            torch.tensor([flag], dtype=torch.long, device=sim.problems.device),
        )

        if sim.done:
            partial = torch.cat(
                (sim.selected_student_list.unsqueeze(2), sim.selected_student_flag.unsqueeze(2)),
                dim=2,
            )
            return env._get_travel_distance_2(origin_problem, partial)

        remaining = self.world_model.predict_remaining_cost(sim)
        return sim.accumulated_cost + remaining

    def select_action(
        self,
        env,
        state,
        current_step: int,
        raw_data_capacity,
        origin_problem: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs, split_line = self.model.get_action_probs(
            state, env.selected_node_list, raw_data_capacity, current_step
        )
        greedy_node, greedy_flag = policy_action_from_probs(probs, split_line)
        cand_nodes, cand_flags, cand_valid = filter_feasible_candidates(
            probs, split_line, env.selected_node_list, self.top_k
        )

        snap = snapshot_env(env)
        batch_size = env.batch_size
        best_nodes = greedy_node.clone()
        best_flags = greedy_flag.clone()
        best_scores = self._rollout_score(
            env, snap, greedy_node, greedy_flag, current_step, raw_data_capacity, origin_problem
        )

        for k in range(cand_nodes.shape[1]):
            if not cand_valid[:, k].all():
                continue
            scores = self._rollout_score(
                env,
                snap,
                cand_nodes[:, k],
                cand_flags[:, k],
                current_step,
                raw_data_capacity,
                origin_problem,
            )
            better = scores < (best_scores - self.score_margin)
            best_scores = torch.where(better, scores, best_scores)
            best_nodes = torch.where(better, cand_nodes[:, k], best_nodes)
            best_flags = torch.where(better, cand_flags[:, k], best_flags)

        restore_env(env, snap)
        return best_nodes, best_flags
