from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PeakLevelIDGLLoss(nn.Module):
    def __init__(
        self,
        recon_weight: float = 1.0,
        sparsity_weight: float = 1e-3,
        smooth_weight: float = 1e-2,
        stability_weight: float = 1e-2,
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.sparsity_weight = sparsity_weight
        self.smooth_weight = smooth_weight
        self.stability_weight = stability_weight

    def forward(
        self,
        optimized_adj: Tensor,
        node_pred: Tensor,
        node_labels: Optional[Tensor],
        init_adj: Tensor,
    ) -> Dict[str, Tensor]:
        losses: Dict[str, Tensor] = {}
        total = optimized_adj.new_tensor(0.0)

        if node_labels is not None:
            target = node_labels.float().view(-1)
            recon = F.mse_loss(node_pred.view(-1), target)
            losses["node_mse"] = recon.detach()
            total = total + self.recon_weight * recon
        else:
            losses["node_mse"] = optimized_adj.new_tensor(0.0)

        opt_dense = optimized_adj.to_dense() if optimized_adj.is_sparse else optimized_adj
        init_dense = init_adj.to_dense() if init_adj.is_sparse else init_adj.float()

        sparsity = opt_dense.mean()
        stability = F.mse_loss(opt_dense, init_dense)
        smooth = self._graph_smoothness(node_pred, opt_dense)

        losses["sparsity"] = sparsity.detach()
        losses["stability"] = stability.detach()
        losses["smoothness"] = smooth.detach()

        total = total + self.sparsity_weight * sparsity
        total = total + self.stability_weight * stability
        total = total + self.smooth_weight * smooth

        losses["loss"] = total
        return losses

    @staticmethod
    def _graph_smoothness(node_pred: Tensor, adj: Tensor) -> Tensor:
        pred = node_pred.view(-1, 1)
        diff = pred - pred.transpose(0, 1)
        return (adj * diff.pow(2)).mean()

