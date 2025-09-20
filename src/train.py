import math
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

################################################################################
#                               CASP MODULES                                   #
################################################################################
class QSketch(nn.Module):
    """INT8 Greenwald-Khanna style quantile sketch.

    Keeps ``k`` bins per channel and exposes the robust six-moment vector
    ⟨q10,q25,µ,q75,q90,skew⟩.
    """

    def __init__(self, channels: int, k: int = 64, momentum: float = 0.9):
        super().__init__()
        self.k = k
        self.momentum = momentum
        self.register_buffer("bins", torch.zeros(channels, k, dtype=torch.int8))

    @torch.no_grad()
    def update(self, feats: torch.Tensor):  # pylint: disable=arguments-differ
        # ``feats`` shape: (N, C, …) – flatten spatial / temporal dims
        f = feats.detach().float()

        # Flatten spatial dimensions: (N, C, H, W) -> (N*H*W, C)
        if f.dim() > 2:
            f = f.permute(1, 0, *range(2, f.dim())).contiguous()  # (C, N, H, W, ...)
            f = f.flatten(1)  # (C, N*H*W*...)
            f = f.transpose(0, 1)  # (N*H*W*..., C)

        q10, q25, q50, q75, q90 = torch.quantile(
            f,
            torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], device=f.device),
            dim=0,
        )
        mean = f.mean(dim=0)
        skew = torch.mean(((f - mean) ** 3), dim=0) / (torch.std(f, dim=0) ** 3 + 1e-6)
        moment_vec = torch.stack([q10, q25, mean, q75, q90, skew], dim=1)  # (C,6)

        reps = math.ceil(self.k / 6)
        m64 = moment_vec.repeat_interleave(reps, dim=1)[:, : self.k]
        upd = (
            self.momentum * self.bins.float() + (1.0 - self.momentum) * m64
        ).round().clamp_(-128, 127).to(torch.int8)
        self.bins.copy_(upd)

    def robust_moments(self) -> torch.Tensor:
        """Return channel-wise mean over INT8 bins (C,)."""

        return self.bins.float().mean(dim=1)


class PropLayer(nn.Module):
    """Learnable interpolation between parent / child moment vectors."""

    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))  # α=0 ⇒ equal weighting

    def forward(self, m_parent: torch.Tensor, m_child: torch.Tensor) -> torch.Tensor:  # noqa: D401,E501
        w = torch.tanh(self.alpha) * 0.5 + 0.5
        return w * m_parent + (1.0 - w) * m_child


class HyperNet(nn.Module):
    """Lightweight Bayesian linear-attention transformer predicting Δγ,Δβ,logσ²."""

    def __init__(self, emb: int = 64):
        super().__init__()
        self.fc_in = nn.Linear(1, emb)
        enc_layer = nn.TransformerEncoderLayer(d_model=emb, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = nn.Linear(emb, 3)  # → Δγ, Δβ, logσ²

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (L,1) → (L,3)
        h = self.fc_in(x)
        h = self.encoder(h)
        return self.head(h)


################################################################################
class CASPAdapter(nn.Module):
    """Wrap backbone with CASP: sketch → propagation → Bayesian update."""

    def __init__(
        self,
        model: nn.Module,
        k: int = 64,
        alpha0: float = 0.15,
        prior_var: float = 0.05,
        tau: float = 3.0,
        device: torch.device | str = "cuda",
    ):
        super().__init__()
        self.base = model.to(device)
        self.device = torch.device(device)
        self.prior_var = prior_var
        self.tau = tau

        # ------------------------------------------------------------------
        # discover all normalisation layers & attach sketches / hooks
        # ------------------------------------------------------------------
        self.norm_layers: List[nn.Module] = []
        self.sketches: List[QSketch] = []
        for m in self.base.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                self.norm_layers.append(m)
                self.sketches.append(QSketch(m.weight.numel(), k).to(device))
        if not self.norm_layers:
            print("[CASP] Warning: No normalisation layers found in backbone. Creating dummy sketch.")
            # Create a single dummy sketch for models without norm layers (e.g., quantized models)
            self.sketches = nn.ModuleList([QSketch(1, k).to(device)])
            self.norm_layers = [nn.Identity()]

        self.prop_layers = nn.ModuleList(PropLayer() for _ in self.norm_layers).to(device)
        self.sketches = nn.ModuleList(self.sketches)  # Make sure sketches are registered as submodules
        self.hypernet = HyperNet().to(device)

        self._cached_moments = [torch.zeros_like(sk.robust_moments()) for sk in self.sketches]
        # Only register hooks for real norm layers, not dummy Identity layers
        self._hooks = []
        for sk, layer in zip(self.sketches, self.norm_layers):
            if not isinstance(layer, nn.Identity):
                self._hooks.append(layer.register_forward_hook(lambda _m, _i, o, sk=sk: sk.update(o)))

    # ------------------------------------------------------------------
    def forward(self, x):  # noqa: D401
        return self.base(x)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _propagate_moments(self):
        """Single iteration of bidirectional message passing."""

        L = len(self._cached_moments)
        moments = [sk.robust_moments().to(self.device) for sk in self.sketches]
        for idx in range(L):
            if 0 < idx < L - 1:
                m_parent, m_child = moments[idx - 1], moments[idx + 1]
                # Only propagate if tensors have compatible shapes
                if m_parent.shape == m_child.shape == moments[idx].shape:
                    moments[idx] = self.prop_layers[idx](m_parent, m_child)
        self._cached_moments = moments

    # ------------------------------------------------------------------
    @torch.no_grad()
    def adapt(self) -> Dict[str, float | bool]:
        """Run CASP update once per batch."""

        self._propagate_moments()
        layer_in = torch.stack(
            [m.mean().unsqueeze(0) for m in self._cached_moments], dim=0
        )  # (L,1)
        delta = self.hypernet(layer_in)  # (L,3)
        d_gamma, d_beta, log_sigma2 = delta[:, 0], delta[:, 1], delta[:, 2]
        sigma2 = log_sigma2.exp() + 1e-6

        maha = torch.norm(layer_in.squeeze() / torch.sqrt(sigma2))
        if maha > self.tau:
            return {"skipped": True, "maha": float(maha)}

        for layer, dg, db, s2 in zip(self.norm_layers, d_gamma, d_beta, sigma2):
            coeff = self.prior_var / (self.prior_var + s2)
            # Skip updates for dummy layers (Identity layers have no weight/bias)
            if hasattr(layer, 'weight') and layer.weight is not None:
                layer.weight.data.add_(coeff * dg)
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data.add_(coeff * db)
        return {"skipped": False, "maha": float(maha)}


################################################################################
#                              PROFILING UTILS                                 #
################################################################################
class EnergyMeter:
    """NVML-based GPU energy meter (records J)."""

    def __init__(self):
        import pynvml

        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._start = None
        self.energy = 0.0

    def __enter__(self):
        import pynvml

        self._start = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
        return self

    def __exit__(self, *_exc):
        import pynvml

        end = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
        self.energy = (end - self._start) / 1000.0  # mJ → J


################################################################################
#                               MODEL FACTORY                                   #
################################################################################

def _quantise_int8(model: nn.Module, sample: torch.Tensor) -> nn.Module:
    """Static INT8 quantisation using the default fbgemm/qnnpack backend."""

    backend = (
        "fbgemm" if "fbgemm" in torch.backends.quantized.supported_engines else "qnnpack"
    )
    model.eval()
    # Move model to CPU for quantization
    model = model.cpu()
    model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
    torch.ao.quantization.prepare(model, inplace=True)
    with torch.no_grad():
        model(sample.cpu())  # calibration pass
    torch.ao.quantization.convert(model, inplace=True)
    return model


def build_model(
    name: str,
    quantized: bool,
    device: torch.device,
    batch_size_for_calib: int = 4,
) -> CASPAdapter:
    """Instantiate backbone, optional INT8 path and wrap with ``CASPAdapter``."""

    model = timm.create_model(name, pretrained=True).to(device)
    if quantized:
        dummy = torch.randint(0, 255, (batch_size_for_calib, 3, 224, 224), dtype=torch.uint8)
        dummy = dummy.float().div_(255).to(device)
        model = _quantise_int8(model, dummy)
        # Quantized models need to stay on CPU for inference
        device = torch.device("cpu")

    return CASPAdapter(model, device=device)


################################################################################
#                         ONE-STREAM EXECUTION HELPER                           #
################################################################################

def run_one_stream(model: CASPAdapter, loader: DataLoader, device: torch.device) -> Dict:
    """Evaluate + adapt over one complete corruption stream."""

    top1, n = 0.0, 0
    # Use the device that the model is actually on
    model_device = next(model.parameters()).device

    if model_device.type == 'cuda':
        torch.cuda.synchronize()
        start_evt, end_evt = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        energy_meter = EnergyMeter()
        with energy_meter:
            start_evt.record()
            for images, labels in loader:
                images = images.to(model_device, non_blocking=True)
                labels = labels.to(model_device, non_blocking=True)
                with autocast(dtype=torch.bfloat16, enabled=False):
                    logits = model(images)
                pred = logits.argmax(1)
                top1 += (pred == labels).sum().item()
                n += labels.numel()
                model.adapt()
            end_evt.record()
            torch.cuda.synchronize()
        latency_ms = start_evt.elapsed_time(end_evt)
        energy_J = energy_meter.energy
    else:
        # CPU execution
        import time
        start_time = time.time()
        for images, labels in loader:
            images = images.to(model_device)
            labels = labels.to(model_device)
            with autocast(dtype=torch.bfloat16, enabled=False):
                logits = model(images)
            pred = logits.argmax(1)
            top1 += (pred == labels).sum().item()
            n += labels.numel()
            model.adapt()
        latency_ms = (time.time() - start_time) * 1000
        energy_J = 0.0  # No energy measurement for CPU

    return {
        "top1_error": round(100.0 * (1.0 - top1 / n), 3),
        "latency_ms": round(latency_ms, 2),
        "energy_J": round(energy_J, 3),
        "num_samples": n,
    }
