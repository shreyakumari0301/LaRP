"""Collect greedy MnLP policy trajectories for world-model training."""

from __future__ import annotations

import torch

from rollout_planner import decode_action, filter_feasible_candidates, policy_action_from_probs
from simulator import SimState, apply_action


def _step_env(env, state, selected_student, selected_flag, current_step, device):
    if current_step == 0:
        selected_flag = torch.ones(1, dtype=torch.long, device=device)
    return env.step(selected_student, selected_student, selected_flag, selected_flag)


def greedy_rollout(env, model, episode_idx: int, device: torch.device):
    """Run greedy MnLP decoding and return the policy solution tensor."""
    env.load_problems(episode_idx, 1)
    problems = env.problems.clone()
    lkh_solution = env.solution.clone()
    raw_capacity = float(env.raw_data_capacity.ravel()[0].item())
    total_lkh = float(env._get_travel_distance_2(problems, lkh_solution).item())

    env.reset("test")
    state, _, _, done = env.pre_step()
    current_step = 0
    model.eval()

    with torch.no_grad():
        while not done:
            _, _, selected_student, _, selected_flag = model(
                state,
                env.selected_node_list,
                lkh_solution,
                current_step,
                raw_data_capacity=env.raw_data_capacity,
            )
            if current_step == 0:
                selected_flag = torch.ones(1, dtype=torch.long, device=device)

            state, _, _, done = _step_env(
                env, state, selected_student, selected_flag, current_step, device
            )
            current_step += 1

    policy_solution = torch.cat(
        (
            env.selected_student_list.reshape(1, -1, 1),
            env.selected_student_flag.reshape(1, -1, 1),
        ),
        dim=2,
    )
    return problems, policy_solution, lkh_solution, total_lkh, raw_capacity


def replay_env_to_step(env, model, episode_idx: int, target_step: int, device: torch.device):
    """Replay greedy policy in env until target_step actions are applied."""
    env.load_problems(episode_idx, 1)
    lkh_solution = env.solution.clone()
    env.reset("test")
    state, _, _, done = env.pre_step()
    current_step = 0
    model.eval()

    with torch.no_grad():
        while current_step < target_step and not done:
            _, _, selected_student, _, selected_flag = model(
                state,
                env.selected_node_list,
                lkh_solution,
                current_step,
                raw_data_capacity=env.raw_data_capacity,
            )
            if current_step == 0:
                selected_flag = torch.ones(1, dtype=torch.long, device=device)
            state, _, _, done = _step_env(
                env, state, selected_student, selected_flag, current_step, device
            )
            current_step += 1
    return state, lkh_solution, done


def complete_greedy_from_prefix(
    env,
    model,
    episode_idx: int,
    prefix_nodes: torch.Tensor,
    prefix_flags: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Replay a prefix, finish with greedy MnLP, return (solution, total_cost)."""
    env.load_problems(episode_idx, 1)
    problems = env.problems.clone()
    lkh_solution = env.solution.clone()
    env.reset("test")
    state, _, _, done = env.pre_step()
    current_step = 0
    model.eval()

    with torch.no_grad():
        prefix_len = int(prefix_nodes.shape[1])
        while current_step < prefix_len and not done:
            node = prefix_nodes[:, current_step]
            flag = prefix_flags[:, current_step]
            state, _, _, done = _step_env(env, state, node, flag, current_step, device)
            current_step += 1

        while not done:
            _, _, selected_student, _, selected_flag = model(
                state,
                env.selected_node_list,
                lkh_solution,
                current_step,
                raw_data_capacity=env.raw_data_capacity,
            )
            state, _, _, done = _step_env(
                env, state, selected_student, selected_flag, current_step, device
            )
            current_step += 1

    solution = torch.cat(
        (
            env.selected_student_list.reshape(1, -1, 1),
            env.selected_student_flag.reshape(1, -1, 1),
        ),
        dim=2,
    )
    total_cost = float(env._get_travel_distance_2(problems, solution).item())
    return solution, total_cost


def topk_ranking_pairs_at_step(
    env,
    model,
    episode_idx: int,
    step: int,
    problems: torch.Tensor,
    policy_solution: torch.Tensor,
    raw_capacity: float,
    device: torch.device,
    top_k: int = 3,
) -> list[tuple[SimState, SimState]]:
    """All pairwise ranking pairs among top-K actions ranked by completed tour cost."""
    state, _, done = replay_env_to_step(env, model, episode_idx, step, device)
    if done:
        return []

    probs, split_line = model.get_action_probs(
        state, env.selected_node_list, env.raw_data_capacity, step
    )
    cand_nodes, cand_flags, cand_valid = filter_feasible_candidates(
        probs, split_line, env.selected_node_list, top_k
    )

    prefix_nodes = policy_solution[:, :step, 0]
    prefix_flags = policy_solution[:, :step, 1]
    if step > 0:
        prefix_cost_value = float(
            prefix_cal_length(
                problems[:, :, [0, 1]],
                prefix_nodes,
                prefix_flags,
            ).item()
        )
    else:
        prefix_cost_value = 0.0

    scored: list[tuple[float, SimState]] = []
    for k in range(cand_nodes.shape[1]):
        if not cand_valid[0, k]:
            continue
        node = cand_nodes[0, k : k + 1]
        flag = cand_flags[0, k : k + 1]
        ext_nodes = torch.cat((prefix_nodes, node.unsqueeze(1)), dim=1)
        ext_flags = torch.cat((prefix_flags, flag.unsqueeze(1)), dim=1)
        _, total_cost = complete_greedy_from_prefix(
            env, model, episode_idx, ext_nodes, ext_flags, device
        )
        post_state = state_after_action(
            problems,
            policy_solution,
            step,
            raw_capacity,
            prefix_cost_value,
            node,
            flag,
            device,
        )
        scored.append((total_cost, post_state))

    if len(scored) < 2:
        return []

    scored.sort(key=lambda item: item[0])
    best_state = scored[0][1]
    pairs: list[tuple[SimState, SimState]] = []
    for total_cost, worse_state in scored[1:]:
        if total_cost > scored[0][0] + 1e-6:
            pairs.append((best_state, worse_state))
    return pairs


def prefix_cal_length(problems: torch.Tensor, order_node: torch.Tensor, order_flag: torch.Tensor) -> torch.Tensor:
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
    order_lengths = ((order_loc - roll_loc) ** 2)
    order_flag_[:, 0] = 0
    flag_gathering_index = order_flag_.unsqueeze(2).expand(-1, seq_len, 2)
    flag_loc = problems.gather(dim=1, index=flag_gathering_index)
    roll_lengths = ((roll_loc - flag_loc) ** 2)
    return (order_lengths.sum(2).sqrt() + roll_lengths.sum(2).sqrt()).sum(1)


def alternate_action_at_step(
    env,
    model,
    episode_idx: int,
    step: int,
    device: torch.device,
    top_k: int = 5,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return a feasible non-greedy top-K action at policy state `step`, or None."""
    state, _, done = replay_env_to_step(env, model, episode_idx, step, device)
    if done:
        return None

    probs, split_line = model.get_action_probs(
        state, env.selected_node_list, env.raw_data_capacity, step
    )
    greedy_node, greedy_flag = policy_action_from_probs(probs, split_line)
    cand_nodes, cand_flags, cand_valid = filter_feasible_candidates(
        probs, split_line, env.selected_node_list, top_k
    )

    for k in range(cand_nodes.shape[1]):
        if not cand_valid[0, k]:
            continue
        node, flag = cand_nodes[0, k], cand_flags[0, k]
        if int(node.item()) == int(greedy_node[0].item()) and int(flag.item()) == int(greedy_flag[0].item()):
            continue
        return node.unsqueeze(0), flag.unsqueeze(0)
    return None


def build_state_from_solution(
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


def state_after_action(
    problems: torch.Tensor,
    solution: torch.Tensor,
    step: int,
    raw_capacity: float,
    prefix_cost_value: float,
    node: torch.Tensor,
    flag: torch.Tensor,
    device: torch.device,
) -> SimState:
    sim = build_state_from_solution(
        problems, solution, step, raw_capacity, prefix_cost_value, device
    )
    return apply_action(sim, node, flag)
