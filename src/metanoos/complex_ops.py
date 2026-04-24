"""Complex-valued building blocks used by the composed-state primitive."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def real_dtype_for(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex128:
        return torch.float64
    if dtype == torch.complex64:
        return torch.float32
    complex32 = getattr(torch, "complex32", None)
    if complex32 is not None and dtype == complex32:
        return torch.float16
    return dtype


def positive_gate(x: Tensor) -> Tensor:
    """Phase-blind nonnegative relevance gate g(x) = elu(|x|) + 1."""

    return F.elu(x.abs()) + 1.0


def phase_feature(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Phase-bearing feature phi(x) = g(x) x / max(|x|, eps)."""

    return positive_gate(x) * x / x.abs().clamp_min(eps)


class ComplexLinear(nn.Module):
    """A minimal dense complex affine map."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.cfloat))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.cfloat))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        scale = 1.0 / math.sqrt(2.0)
        with torch.no_grad():
            self.weight.real.uniform_(-bound * scale, bound * scale)
            self.weight.imag.uniform_(-bound * scale, bound * scale)
            if self.bias is not None:
                self.bias.real.uniform_(-bound * scale, bound * scale)
                self.bias.imag.uniform_(-bound * scale, bound * scale)

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.weight.transpose(0, 1)
        if self.bias is not None:
            y = y + self.bias
        return y


class ComplexEmbedding(nn.Module):
    """Complex token carrier table."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=torch.cfloat))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.embedding_dim)
        scale = 1.0 / math.sqrt(2.0)
        with torch.no_grad():
            self.weight.real.normal_(0.0, std * scale)
            self.weight.imag.normal_(0.0, std * scale)

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.weight[input_ids]


class ComplexRMSNorm(nn.Module):
    """RMS normalization over complex magnitude with phase preserved."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.abs().square().mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight.to(dtype=real_dtype_for(x.dtype), device=x.device)


class ModReLU(nn.Module):
    """Phase-preserving modReLU nonlinearity."""

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        magnitude = x.abs()
        scale = F.relu(magnitude + self.bias.to(device=x.device, dtype=magnitude.dtype))
        return scale * x / magnitude.clamp_min(self.eps)
