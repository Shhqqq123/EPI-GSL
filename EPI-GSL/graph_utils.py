from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _as_dense_adj(adj: Union[Tensor, np.ndarray]) -> Tensor:
    if isinstance(adj, np.ndarray):
        adj = torch.from_numpy(adj)
    if not torch.is_tensor(adj):
        raise TypeError("adj must be a torch.Tensor or numpy.ndarray")
    if adj.is_sparse:
        adj = adj.to_dense()
    return adj.float()


def build_dense_from_edge_index(
    num_nodes: int,
    edge_index: Tensor,
    edge_weight: Optional[Tensor] = None,
    symmetric: bool = True,
    include_self: bool = False,
) -> Tensor:
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=edge_index.device)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device, dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = edge_weight.float()
    if symmetric:
        adj[edge_index[1], edge_index[0]] = edge_weight.float()
    if include_self:
        adj.fill_diagonal_(1.0)
    return adj


def dense_to_edge_index(adj: Tensor, threshold: float = 0.0) -> Tuple[Tensor, Tensor]:
    if adj.is_sparse:
        adj = adj.to_dense()
    mask = adj > threshold
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()
    edge_weight = adj[mask]
    return edge_index, edge_weight


def normalize_adj(adj: Tensor, add_self_loops: bool = True, eps: float = 1e-12) -> Tensor:
    adj = adj.clone()
    if add_self_loops:
        adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
    deg = adj.sum(dim=-1)
    deg_inv_sqrt = torch.pow(deg.clamp_min(eps), -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    return deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)


class NodeEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GraphLearner(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.proj(x)
        z = F.normalize(z, p=2, dim=-1)
        return torch.matmul(z, z.transpose(-1, -2))


class NodeRegressionHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


class DenseGraphRefiner(nn.Module):
    def __init__(self, alpha: float = 0.5, add_self_loops: bool = True, topk_edges: int = 20):
        super().__init__()
        self.alpha = alpha
        self.add_self_loops = add_self_loops
        self.topk_edges = topk_edges

    def forward(self, init_adj: Tensor, learned_adj: Tensor) -> Tensor:
        init_adj = init_adj.float()
        learned_adj = learned_adj.float()
        if self.add_self_loops:
            eye = torch.eye(init_adj.size(0), device=init_adj.device, dtype=init_adj.dtype)
            init_adj = torch.maximum(init_adj, eye)
        learned_adj = torch.softmax(learned_adj, dim=-1)
        refined = self.alpha * init_adj + (1.0 - self.alpha) * learned_adj
        refined = 0.5 * (refined + refined.transpose(0, 1))
        if self.topk_edges is not None and self.topk_edges > 0:
            n = refined.size(0)
            k = min(self.topk_edges, n)
            _, topk_ind = torch.topk(refined, k=k, dim=-1)
            sparse_mask = torch.zeros_like(refined, dtype=torch.bool)
            sparse_mask.scatter_(-1, topk_ind, True)
            sparse_mask = sparse_mask | sparse_mask.transpose(0, 1)
            refined = refined * sparse_mask.float()
            if self.add_self_loops:
                refined.fill_diagonal_(1.0)
        return refined








