"""Learned value world model for rollout planning."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from simulator import SimState


class CVRPWorldModel(nn.Module):
    """Predicts normalized remaining tour cost from a partial CVRP state."""

    def __init__(
        self,
        max_nodes: int = 1001,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        cost_scale: float = 20.0,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim
        self.cost_scale = cost_scale

        node_in = 6
        context_in = 6

        self.node_proj = nn.Linear(node_in, hidden_dim)
        self.context_proj = nn.Linear(context_in, hidden_dim)
        self.pool_query = nn.Linear(hidden_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _capacities(self, state: SimState, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if state.capacities is not None:
            return state.capacities.view(batch_size).to(device=device, dtype=dtype)
        return torch.full(
            (batch_size,),
            float(state.raw_capacity),
            device=device,
            dtype=dtype,
        )

    def _prefix_lens(self, state: SimState, batch_size: int, device: torch.device) -> torch.Tensor:
        if state.prefix_lens is not None:
            return state.prefix_lens
        return torch.full(
            (batch_size,),
            state.selected_count,
            dtype=torch.long,
            device=device,
        )

    def _current_nodes(self, state: SimState, prefix_lens: torch.Tensor, device: torch.device) -> torch.Tensor:
        batch_size = state.problems.shape[0]
        current_node = torch.zeros(batch_size, dtype=torch.long, device=device)
        has_prefix = prefix_lens > 0
        if has_prefix.any():
            rows = torch.arange(batch_size, device=device)[has_prefix]
            cols = prefix_lens[has_prefix] - 1
            current_node[has_prefix] = state.selected_student_list[rows, cols]
        return current_node

    def _visited_mask(self, state: SimState, prefix_lens: torch.Tensor, num_nodes: int) -> torch.Tensor:
        batch_size = state.problems.shape[0]
        device = state.problems.device
        visited = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=device)
        for b in range(batch_size):
            plen = int(prefix_lens[b].item())
            if plen > 0:
                nodes = state.selected_student_list[b, :plen].long()
                visited[b, nodes] = True
        return visited

    def encode_state(self, state: SimState) -> torch.Tensor:
        problems = state.problems
        batch_size, num_nodes, _ = problems.shape
        device = problems.device
        prefix_lens = self._prefix_lens(state, batch_size, device)
        visited = self._visited_mask(state, prefix_lens, num_nodes)
        current_node = self._current_nodes(state, prefix_lens, device)
        cap_scale = self._capacities(state, batch_size, device, problems.dtype).view(batch_size, 1).clamp(min=1e-6)

        demand_scale = problems[:, 0:1, 2:3].amax(dim=1, keepdim=True).clamp(min=1e-6)

        current_xy = problems.gather(
            dim=1,
            index=current_node[:, None, None].expand(-1, 1, 2),
        ).squeeze(1)
        dist_to_current = torch.sqrt(
            ((problems[:, :, :2] - current_xy[:, None, :]) ** 2).sum(dim=-1) + 1e-12
        )

        node_feats = torch.cat(
            (
                problems[:, :, :2],
                problems[:, :, 2:3] / demand_scale,
                visited.float().unsqueeze(-1),
                problems[:, :, 3:4] / cap_scale.unsqueeze(1),
                dist_to_current.unsqueeze(-1),
            ),
            dim=-1,
        )
        node_emb = self.node_proj(node_feats)
        node_emb = self.encoder(node_emb)

        remaining_cap = problems[:, 0, 3:4] / cap_scale
        progress = prefix_lens.float().unsqueeze(-1) / max(state.problem_size, 1)
        prefix_cost = state.accumulated_cost.unsqueeze(-1) / self.cost_scale
        unvisited_frac = (~visited[:, 1:]).float().mean(dim=1, keepdim=True)

        context = torch.cat(
            (current_xy, remaining_cap, progress, prefix_cost, unvisited_frac),
            dim=-1,
        )
        context_emb = self.context_proj(context)

        pool_q = self.pool_query(context_emb).unsqueeze(1)
        pool_scores = (pool_q * node_emb).sum(dim=-1)
        pool_scores = pool_scores.masked_fill(visited, -1e9)
        pool_weights = F.softmax(pool_scores, dim=-1)
        graph_emb = torch.bmm(pool_weights.unsqueeze(1), node_emb).squeeze(1)

        return torch.cat((graph_emb, context_emb), dim=-1)

    def forward(self, state: SimState) -> torch.Tensor:
        return self.head(self.encode_state(state)).squeeze(-1)

    @torch.no_grad()
    def predict_remaining_cost(self, state: SimState) -> torch.Tensor:
        self.eval()
        pred = self.forward(state)
        return torch.clamp(pred, min=0.0) * self.cost_scale


class CVRPWorldModelLegacy(nn.Module):
    """Original MLP value model kept for old checkpoints."""

    def __init__(self, max_nodes: int = 1001, hidden_dim: int = 256, cost_scale: float = 20.0):
        super().__init__()
        self.max_nodes = max_nodes
        self.cost_scale = cost_scale
        self.node_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_state(self, state: SimState) -> torch.Tensor:
        problems = state.problems
        batch_size, num_nodes, _ = problems.shape

        visited = torch.zeros(batch_size, num_nodes, device=problems.device)
        if state.selected_student_list.shape[1] > 0:
            plen = state.prefix_lens if state.prefix_lens is not None else None
            if plen is not None:
                for b in range(batch_size):
                    length = int(plen[b].item())
                    if length > 0:
                        visited[b, state.selected_student_list[b, :length].long()] = 1.0
            else:
                visited.scatter_(1, state.selected_student_list.long(), 1.0)

        node_feats = torch.cat(
            (
                problems[:, :, :2],
                problems[:, :, 2:3] / (problems[:, 0:1, 2:3].amax(dim=1, keepdim=True) + 1e-6),
                visited.unsqueeze(-1),
            ),
            dim=-1,
        )
        node_emb = self.node_encoder(node_feats)
        graph_emb = node_emb.mean(dim=1)

        if state.selected_student_list.shape[1] == 0:
            current_node = torch.zeros(batch_size, dtype=torch.long, device=problems.device)
        else:
            if state.prefix_lens is not None:
                rows = torch.arange(batch_size, device=problems.device)
                cols = state.prefix_lens.clamp(min=1) - 1
                current_node = state.selected_student_list[rows, cols]
                current_node = torch.where(state.prefix_lens > 0, current_node, torch.zeros_like(current_node))
            else:
                current_node = state.selected_student_list[:, -1]

        cap = float(state.raw_capacity)
        remaining_cap = problems[:, 0, 3:4] / (cap + 1e-6)
        progress = torch.full(
            (batch_size, 1),
            state.selected_count / max(state.problem_size, 1),
            device=problems.device,
            dtype=problems.dtype,
        )
        current_xy = problems.gather(
            dim=1,
            index=current_node[:, None, None].expand(-1, 1, 2),
        ).squeeze(1)
        context = torch.cat((current_xy, remaining_cap, progress), dim=1)
        context_emb = self.context_encoder(context)
        return torch.cat((graph_emb, context_emb), dim=1)

    def forward(self, state: SimState) -> torch.Tensor:
        return self.head(self.encode_state(state)).squeeze(-1)

    @torch.no_grad()
    def predict_remaining_cost(self, state: SimState) -> torch.Tensor:
        self.eval()
        pred = self.forward(state)
        scale = self.cost_scale if self.cost_scale != 20.0 else 20.0
        return torch.clamp(pred, min=0.0) * scale


def detect_world_model_arch(state_dict: dict) -> str:
    if "node_proj.weight" in state_dict:
        return "transformer_v2"
    if "node_encoder.0.weight" in state_dict:
        return "mlp_v1"
    raise ValueError("Unrecognized world model checkpoint format")


def load_world_model(wm_ckpt: dict, device: torch.device) -> nn.Module:
    state_dict = wm_ckpt["model_state_dict"]
    arch = wm_ckpt.get("arch") or detect_world_model_arch(state_dict)
    max_nodes = wm_ckpt.get("max_nodes", 1001)
    cost_scale = wm_ckpt.get("cost_scale", 20.0)

    if arch == "mlp_v1":
        model = CVRPWorldModelLegacy(max_nodes=max_nodes, cost_scale=cost_scale)
    else:
        model = CVRPWorldModel(
            max_nodes=max_nodes,
            hidden_dim=wm_ckpt.get("hidden_dim", 256),
            num_layers=wm_ckpt.get("num_layers", 2),
            num_heads=wm_ckpt.get("num_heads", 4),
            cost_scale=cost_scale,
        )
    model.load_state_dict(state_dict)
    return model.to(device).eval()
