import math
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn


def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(
            torch.randn(out_channels, in_channels, *kernel)
        )

    def forward(
        self,
        x,
        gain=1,
    ):
        w = self.weight.to(torch.float32)
        # if self.training:
        # with torch.no_grad():
        #     self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w)  # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel()))  # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1] // 2,))


class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)


def broadcast_to_shape(tensor, shape):
    return tensor.view(-1, *((1,) * (len(shape) - 1)))


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def v(self, s, t, x, t_cond, x_cond, class_labels=None, **kwargs):
        """Should return the velocity. Must be implemented by subclass."""
        pass

    def X(self, s, t, x, v):
        s = broadcast_to_shape(s, x.shape)
        t = broadcast_to_shape(t, x.shape)
        return x + (t - s) * v

    def X_and_v(self, s, t, x, t_cond, x_cond, class_labels=None, **kwargs):
        v = self.forward(s, t, x, t_cond, x_cond, class_labels=class_labels, **kwargs)
        return self.X(s, t, x, v), v

    def forward(self, s, t, x, t_cond, x_cond, class_labels=None, **kwargs):
        """Forward pass that computes the map."""
        v = self.v(s, t, x, t_cond, x_cond, class_labels=class_labels, **kwargs)
        return self.X(s, t, x, v)


class LossWeightingNetwork(nn.Module):
    """
    A network to compute loss weighting based on timesteps.
    """

    def __init__(self, channels=128, clamp_min=-10.0, clamp_max=10.0):
        super().__init__()
        self.linear = MPConv(channels, 1, kernel=[])
        self.emb_fourier = MPFourier(channels)
        self.emb_noise_t = MPConv(channels, channels, kernel=[])
        self.emb_noise_t_cond = MPConv(channels, channels, kernel=[])
        self._weighting_stats = []
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, t, t_cond):
        t_emb = self.emb_noise_t(self.emb_fourier(t))
        t_cond_emb = self.emb_noise_t_cond(self.emb_fourier(t_cond))
        combined = (t_emb + t_cond_emb) / math.sqrt(2.0)
        weighting = self.linear(combined)
        self._record_weighting_stats(weighting)
        weighting = weighting.reshape(-1, 1, 1, 1)
        if self.clamp_min and self.clamp_max:
            weighting = torch.clamp(weighting, self.clamp_min, self.clamp_max)
        return weighting
