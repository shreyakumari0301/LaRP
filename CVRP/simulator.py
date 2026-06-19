"""Exact CVRP transition model for rollout planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from VRPEnv import Step_State


@dataclass
class SimState:
    problems: torch.Tensor
    raw_capacity: float
    selected_node_list: torch.Tensor
    selected_student_list: torch.Tensor
    selected_student_flag: torch.Tensor
    selected_count: int
    accumulated_cost: torch.Tensor
    prefix_lens: Optional[torch.Tensor] = None
    capacities: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        return self.problems.shape[0]

    @property
    def problem_size(self) -> int:
        return self.problems.shape[1] - 1

    @property
    def done(self) -> bool:
        return self.selected_count >= self.problem_size

    def step_state(self) -> Step_State:
        return Step_State(problems=self.problems)

    def clone(self) -> "SimState":
        return SimState(
            problems=self.problems.clone(),
            raw_capacity=self.raw_capacity,
            selected_node_list=self.selected_node_list.clone(),
            selected_student_list=self.selected_student_list.clone(),
            selected_student_flag=self.selected_student_flag.clone(),
            selected_count=self.selected_count,
            accumulated_cost=self.accumulated_cost.clone(),
        )


def _node_coords(problems: torch.Tensor, node_index: torch.Tensor) -> torch.Tensor:
    gather_index = node_index[:, None, None].expand(-1, 1, 2)
    return problems.gather(dim=1, index=gather_index).squeeze(1)


def _edge_cost(problems: torch.Tensor, from_node: torch.Tensor, to_node: torch.Tensor) -> torch.Tensor:
    from_xy = _node_coords(problems, from_node)
    to_xy = _node_coords(problems, to_node)
    return torch.sqrt(((from_xy - to_xy) ** 2).sum(dim=1) + 1e-12)


def state_from_env(env) -> SimState:
    return SimState(
        problems=env.problems.clone(),
        raw_capacity=float(env.raw_data_capacity.ravel()[0].item()),
        selected_node_list=env.selected_node_list.clone(),
        selected_student_list=env.selected_student_list.clone(),
        selected_student_flag=env.selected_student_flag.clone(),
        selected_count=int(env.selected_count),
        accumulated_cost=torch.zeros(env.batch_size, device=env.problems.device),
    )


def apply_action(state: SimState, node: torch.Tensor, flag: torch.Tensor) -> SimState:
    """Apply one routing action and return a new simulator state."""
    state = state.clone()
    batch_size = state.batch_size

    if state.selected_student_list.shape[1] == 0:
        prev_node = torch.zeros(batch_size, dtype=torch.long, device=state.problems.device)
    else:
        prev_node = state.selected_student_list[:, -1]

    if torch.any(flag == 1):
        depot_cost = _edge_cost(state.problems, prev_node, torch.zeros_like(prev_node))
        customer_cost = _edge_cost(state.problems, torch.zeros_like(node), node)
        step_cost = depot_cost + customer_cost
    else:
        step_cost = _edge_cost(state.problems, prev_node, node)

    state.accumulated_cost = state.accumulated_cost + step_cost

    gather_index = node[:, None, None].expand(batch_size, 1, 4)
    current_node = state.problems.gather(index=gather_index, dim=1).squeeze(1)
    demands = current_node[:, 2]

    is_depot = flag == 1
    state.problems[is_depot, :, 3] = state.raw_capacity

    smaller = state.problems[:, 0, 3] < demands
    flag = flag.clone()
    flag[smaller] = 1
    state.problems[smaller, :, 3] = state.raw_capacity

    state.problems[:, :, 3] = state.problems[:, :, 3] - demands[:, None]

    state.selected_node_list = torch.cat((state.selected_node_list, node[:, None]), dim=1)
    state.selected_student_list = torch.cat((state.selected_student_list, node[:, None]), dim=1)
    state.selected_student_flag = torch.cat((state.selected_student_flag, flag[:, None]), dim=1)
    state.selected_count += 1
    return state


def solution_tensor(state: SimState) -> torch.Tensor:
    return torch.cat(
        (
            state.selected_student_list.unsqueeze(2),
            state.selected_student_flag.unsqueeze(2),
        ),
        dim=2,
    )


def snapshot_env(env) -> dict:
    return {
        "problems": env.problems.clone(),
        "selected_count": env.selected_count,
        "selected_node_list": env.selected_node_list.clone(),
        "selected_teacher_flag": env.selected_teacher_flag.clone(),
        "selected_student_list": env.selected_student_list.clone(),
        "selected_student_flag": env.selected_student_flag.clone(),
    }


def restore_env(env, snap: dict) -> None:
    env.problems = snap["problems"].clone()
    env.selected_count = snap["selected_count"]
    env.selected_node_list = snap["selected_node_list"].clone()
    env.selected_teacher_flag = snap["selected_teacher_flag"].clone()
    env.selected_student_list = snap["selected_student_list"].clone()
    env.selected_student_flag = snap["selected_student_flag"].clone()
    from VRPEnv import Step_State

    env.step_state = Step_State(problems=env.problems)
