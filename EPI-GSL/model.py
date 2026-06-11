from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv

from graph_utils import DenseGraphRefiner, GraphLearner, NodeEncoder, NodeRegressionHead, _as_dense_adj, dense_to_edge_index


@dataclass
class PeakIDGLOutput:
    optimized_adj: Tensor
    node_pred: Tensor
    node_emb: Tensor
    learned_adj_logits: Tensor


class PeakLevelIDGLPyG(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        graph_alpha: float = 0.5,
        add_self_loops: bool = True,
        topk_edges: int = 20,
        graph_iters: int = 1,
        return_dense_adj: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.graph_iters = graph_iters
        self.return_dense_adj = return_dense_adj

        self.encoder = NodeEncoder(num_features, hidden_dim, dropout)
        self.graph_learner = GraphLearner(hidden_dim, dropout)
        self.refiner = DenseGraphRefiner(alpha=graph_alpha, add_self_loops=add_self_loops, topk_edges=topk_edges)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=False) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.reg_head = NodeRegressionHead(hidden_dim, dropout)

    def _prepare_adj(self, adj: Union[Tensor, np.ndarray]) -> Tensor:
        return _as_dense_adj(adj)

    def _message_pass(self, h: Tensor, optimized_adj: Tensor) -> Tensor:
        edge_index, edge_weight = dense_to_edge_index(optimized_adj, threshold=0.0)
        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index, edge_weight)
            h = self.norms[i](h + F.dropout(F.relu(h_new), p=self.dropout, training=self.training))
        return h

    def forward(
        self,
        adj: Union[Tensor, np.ndarray],
        node_features: Tensor,
        node_labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        adj = self._prepare_adj(adj).to(node_features.device)
        h = self.encoder(node_features.float())
        optimized_adj = adj

        num_iters = max(1, self.graph_iters)
        for _ in range(num_iters):
            learned_logits = self.graph_learner(h)
            optimized_adj = self.refiner(adj, learned_logits)
            h = self._message_pass(h, optimized_adj)

        node_pred = self.reg_head(h)
        if not self.return_dense_adj:
            optimized_adj = optimized_adj.to_sparse()
        return optimized_adj, node_pred




