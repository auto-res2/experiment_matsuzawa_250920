import os
import time
from typing import List

import torch
import torch.nn as nn
import timm

__all__ = [
    'UniCORN',
    'Bus',
    'wrap_bn_with_unicorn',
    'load_backbone',
]


class UniCORN(nn.Module):
    """Universal Cross-layer Elastic Alignment normalisation layer.

    The layer can wrap BatchNorm, LayerNorm or GroupNorm transparently by
    specifying the reduction dims *red_dims* that correspond to the feature
    dimension(s) of the wrapped tensor.  All computations are gradient-free and
    therefore suitable for Test-Time Adaptation in streaming inference.
    """

    def __init__(self, num_feat: int, red_dims: tuple, shared_bus: "Bus"):
        super().__init__()
        self.C = num_feat
        self.red = red_dims
        self.bus = shared_bus

        # fast track statistics
        self.register_buffer("mu_s", torch.zeros(num_feat))
        self.register_buffer("var_s", torch.ones(num_feat))
        # slow track statistics
        self.register_buffer("mu_l", torch.zeros(num_feat))
        self.register_buffer("var_l", torch.ones(num_feat))

        # Tiny MLP that predicts (alpha, beta) correction terms.
        self.W = nn.Parameter(torch.randn(22, 8) * 0.1)
        self.U = nn.Parameter(torch.randn(8, num_feat) * 0.1)
        self.eps = 1e-5

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------
    def _moments(self, x: torch.Tensor):
        mu = x.mean(self.red)
        var = x.var(self.red, unbiased=False)
        # third-order moment (skewness) used as input feature for MLP
        centred = x - mu.reshape([*[-1 if d == 0 else 1 for d in x.shape]])
        skew = (centred ** 3).mean(self.red)
        return mu, var, skew

    @torch.no_grad()
    def _update_tracks(self, mu: torch.Tensor, var: torch.Tensor):
        # exponential moving averages – fast (0.001) and slow (0.01)
        self.mu_s.mul_(0.999).add_(0.001 * mu)
        self.var_s.mul_(0.999).add_(0.001 * var)
        self.mu_l.mul_(0.99).add_(0.01 * mu)
        self.var_l.mul_(0.99).add_(0.01 * var)

    # ---------------------------------------------------------------------
    # forward pass
    # ---------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        entropy: torch.Tensor | float = 0.0,
        *,
        power_ok: bool = True,
        latency_ok: bool = True,
    ) -> torch.Tensor:
        mu, var, sk = self._moments(x)
        self._update_tracks(mu, var)

        # Normalised deltas w.r.t. fast track
        dmu = (mu - self.mu_s) / (self.mu_s.abs() + 1e-6)
        dvar = (var - self.var_s) / (self.var_s.abs() + 1e-6)

        # Noisier moments -> rely more on slow track
        gate = torch.sigmoid(0.5 * (dvar.abs() + dmu.abs()))
        mu_hat = gate * mu + (1 - gate) * self.mu_l
        var_hat = gate * var + (1 - gate) * self.var_l

        if not (power_ok and latency_ok):
            alpha = beta = torch.zeros_like(mu)
        else:
            # shared 32-D cross-layer moment bus feature
            bus_feat = self.bus()
            feat = torch.cat(
                [
                    dmu.mean().view(1),
                    dvar.mean().view(1),
                    sk.mean().view(1),
                    torch.as_tensor(entropy).view(1),
                    bus_feat,
                ]
            )
            h = torch.relu(feat @ self.W)
            v = self.U.T @ h
            alpha = torch.sigmoid(v)
            beta = 0.1 * torch.tanh(v)

        mix_mu = alpha * mu_hat + (1 - alpha) * self.mu_s + beta
        mix_var = (
            alpha * var_hat
            + (1 - alpha) * self.var_s
            + alpha * (1 - alpha) * (mu_hat - self.mu_s) ** 2
        )
        xhat = (x - mix_mu.reshape([*[-1 if d == 0 else 1 for d in x.shape]])) / (
            torch.sqrt(mix_var.reshape([*[-1 if d == 0 else 1 for d in x.shape]]) + self.eps)
        )
        return xhat


class Bus(nn.Module):
    """Cross-layer moment bus – gathers first-layer statistics and broadcasts."""

    def __init__(self, first_layers: List[UniCORN]):
        super().__init__()
        self.layers = nn.ModuleList(first_layers)
        self.proj = nn.Parameter(
            torch.randn(sum(l.C for l in first_layers), 32) / 8, requires_grad=False
        )

    @torch.no_grad()
    def forward(self):
        vec = []
        for l in self.layers:
            vec.append(torch.cat([l.mu_s, l.var_s]))
        vec = torch.cat(vec)
        return (vec @ self.proj).detach()


# -------------------------------------------------------------------------
# utilities
# -------------------------------------------------------------------------

def _is_norm(layer):
    return isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm))


def wrap_bn_with_unicorn(model: nn.Module, shared_bus: bool = True) -> nn.Module:
    """Replace all normalisation layers with UniCORN adaptors."""

    first_layers: List[UniCORN] = []

    def _replace(child: nn.Module):
        for name, module in list(child.named_children()):
            if _is_norm(module):
                # Determine reduction dims based on tensor layout
                if isinstance(module, nn.BatchNorm2d):
                    red = (0, 2, 3)
                    C = module.num_features
                elif isinstance(module, nn.GroupNorm):
                    red = (0, 2, 3)
                    C = module.num_channels
                else:  # LayerNorm
                    red = (-1,)
                    C = module.normalized_shape[0]
                unicorn = UniCORN(C, red_dims=red, shared_bus=None)  # bus later
                child.add_module(name, unicorn)
                if len(first_layers) < 3:
                    first_layers.append(unicorn)
            else:
                _replace(module)

    _replace(model)

    if shared_bus:
        bus = Bus(first_layers)
        for l in first_layers:
            l.bus = bus
    return model


def load_backbone(backbone_name: str, *, pretrained: bool = True, device: str = "cuda") -> nn.Module:
    """Utility to create a backbone and wrap with UniCORN."""
    model = timm.create_model(backbone_name, pretrained=pretrained)
    model.eval().to(device)
    return wrap_bn_with_unicorn(model)
