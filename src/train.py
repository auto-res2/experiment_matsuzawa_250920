import os
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

# -----------------------------
# Mask–Quant components (taken from the monolithic script)
# -----------------------------
class MaskQuantSolver:
    """Bi-criteria knapsack solver that allocates (mask, bit-width) per edge.
    A lightweight CPU implementation is offered for smoke-tests.
    In production the DP kernel can be replaced by the CUDA version from the paper.
    """

    def __init__(self, flop_budget: float, q_err_budget: float):
        if not 0 < flop_budget <= 1:
            raise ValueError("flop_budget must be in (0,1]")
        self.C = flop_budget
        self.E = q_err_budget

    @torch.no_grad()
    def solve(self, phi: torch.Tensor, qerr: torch.Tensor) -> torch.Tensor:
        """Simple heuristic: value = phi/(qerr+ɛ); take top-k where k ≈ C·|E|.
        Returns a bit-width vector (0: drop, 4 or 8: keep).
        """
        if phi.numel() == 0:
            return torch.empty_like(phi, dtype=torch.int8)
        score = phi / (qerr + 1e-5)
        k = int(self.C * phi.numel())
        keep = torch.topk(score, k).indices
        bw = torch.zeros_like(phi, dtype=torch.int8)  # dropped (0)
        bw[keep] = torch.where(qerr[keep] < 0.02, torch.tensor(8, dtype=torch.int8), torch.tensor(4, dtype=torch.int8))
        return bw  # 0=dropped,4=INT4,8=INT8


class TiledScatter(torch.autograd.Function):
    """Placeholder for the fused scatter kernel.
    For this refactor the function falls back to a vanilla scatter_add which is
    sufficient for correctness (not for performance)."""

    @staticmethod
    def forward(ctx, grad_out: torch.Tensor, edge_idx: torch.Tensor, tile_perm: torch.Tensor, bitwidth: torch.Tensor):
        from torch_scatter import scatter_add  # local import to keep dependency localised

        # grad_out (E,F) – reorder by permutation generated during TCSR pre-processing
        grad_out = grad_out[tile_perm]
        scale = (bitwidth > 0).float() * (8 / bitwidth.clamp(min=1).float())
        scaled = grad_out * scale.unsqueeze(-1)
        dst = edge_idx[1]
        out = scatter_add(scaled, dst, dim=0)
        return out


# -----------------------------
# GNN layer & model wrapper
# -----------------------------
class ECOGATConv(GATConv):
    """GAT-v2 layer extended with Mask–Quant logic (JMQ) and fake-quant error stats."""

    def __init__(self, in_channels: int, out_channels: int, heads: int, solver: MaskQuantSolver):
        super().__init__(in_channels, out_channels // heads, heads=heads, concat=True)
        self.fake_quant = torch.quantization.FakeQuantize(observer=torch.quantization.MovingAverageMinMaxObserver,
                                                          quant_min=0, quant_max=255, dtype=torch.quint8)
        self.solver = solver

    def forward(self, x, edge_index, size=None):  # type: ignore[override]
        out, (idx, attn) = super().forward(x, edge_index, size, return_attention_weights=True)
        if self.training:
            # Debug shapes
            # print(f"attn shape: {attn.shape}, idx shape: {idx.shape}, edge_index[0] shape: {edge_index[0].shape}")
            # Use idx (the actual edge indices used for attention) instead of edge_index[0]
            node_norms = x[idx[0]].norm(dim=1)  # (num_actual_edges,)
            if len(attn.shape) > 1:
                attn_mean = attn.detach().mean(dim=1)  # average over heads: (num_actual_edges,)
            else:
                attn_mean = attn.detach()  # already single head: (num_actual_edges,)
            phi = attn_mean * node_norms  # gradient-norm proxy
            qerr = self.fake_quant(x[idx[0]].detach())  # reuse quant tensor to estimate error
            qerr = (x[idx[0]] - qerr).abs().mean(dim=1)
            bw = self.solver.solve(phi, qerr)
            # Persist bit-width decisions for later analysis
            self.register_buffer('bw', bw, persistent=False)
        return out


class ECOGAT(nn.Module):
    """Two-layer ECO-GAT network ready for Cora/Reddit scale graphs."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int,
                 flop_budget: float, q_err_budget: float):
        super().__init__()
        solver = MaskQuantSolver(flop_budget, q_err_budget)
        self.conv1 = ECOGATConv(in_channels, hidden_channels, heads=heads, solver=solver)
        self.conv2 = ECOGATConv(hidden_channels, out_channels, heads=1, solver=solver)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# -----------------------------
# Train / Validate helpers
# -----------------------------

def train(model: ECOGAT, data: Data, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    optimizer.zero_grad()
    out = model(data.to(device))
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())


def test(model: ECOGAT, data: Data, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    out = model(data.to(device))
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        if mask.sum() == 0:
            accs.append(0.0)
        else:
            acc = (pred[mask] == data.y[mask]).float().mean().item()
            accs.append(acc)
    return tuple(accs)  # train, val, test
