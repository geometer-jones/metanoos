"""Complex GLA layers built around an associative scan state."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from metanoos.complex_ops import (
    ComplexLinear,
    ComplexRMSNorm,
    ModReLU,
    phase_feature,
    positive_gate,
    real_dtype_for,
)
from metanoos.state import AssociativeState, compose, identity_state, local_state, measure, parallel_prefix_scan


TRANSPORT_MODES = {"rotary_decay", "decay", "none"}
FEATURE_MODES = {"complex", "magnitude"}


def normalize_transport_mode(transport: bool | str) -> str:
    if transport is True:
        return "rotary_decay"
    if transport is False:
        return "none"
    if transport not in TRANSPORT_MODES:
        available = ", ".join(sorted(TRANSPORT_MODES))
        raise ValueError(f"transport must be a bool or one of: {available}")
    return transport


class TemporalTransport(nn.Module):
    """Per-head attenuation and phase transport for local generators."""

    def __init__(self, num_heads: int, *, mode: bool | str = "rotary_decay", init_log_decay: float = -4.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.mode = normalize_transport_mode(mode)
        if self.mode != "none":
            self.log_decay_s = nn.Parameter(torch.full((num_heads,), init_log_decay))
            self.log_decay_z = nn.Parameter(torch.full((num_heads,), init_log_decay))
            if self.mode == "rotary_decay":
                self.phase = nn.Parameter(torch.zeros(num_heads))
            else:
                self.register_parameter("phase", None)
        else:
            self.register_parameter("log_decay_s", None)
            self.register_parameter("log_decay_z", None)
            self.register_parameter("phase", None)

    def forward(
        self,
        batch_shape: tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        real_dtype = real_dtype_for(dtype)
        if self.mode == "none":
            alpha_s = torch.ones(*batch_shape, self.num_heads, device=device, dtype=dtype)
            alpha_z = torch.ones(*batch_shape, self.num_heads, device=device, dtype=real_dtype)
            return alpha_s, alpha_z

        decay_s = F.softplus(self.log_decay_s).to(device=device, dtype=real_dtype)
        decay_z = F.softplus(self.log_decay_z).to(device=device, dtype=real_dtype)
        radius_s = torch.exp(-decay_s)
        alpha_z_head = torch.exp(-decay_z)
        if self.mode == "rotary_decay":
            phase = self.phase.to(device=device, dtype=real_dtype)
            alpha_s_head = torch.polar(radius_s, phase).to(dtype=dtype)
        else:
            alpha_s_head = radius_s.to(dtype=dtype)

        expand_shape = (*batch_shape, self.num_heads)
        alpha_s = alpha_s_head.reshape(*((1,) * len(batch_shape)), self.num_heads).expand(expand_shape)
        alpha_z = alpha_z_head.reshape(*((1,) * len(batch_shape)), self.num_heads).expand(expand_shape)
        return alpha_s, alpha_z


class ComposedStateMixing(nn.Module):
    """Causal sequence mixer using associative complex relational state."""

    def __init__(
        self,
        d_model: int,
        *,
        num_heads: int = 1,
        head_dim: int | None = None,
        key_dim: int | None = None,
        value_dim: int | None = None,
        transport: bool | str = "rotary_decay",
        feature_mode: str = "complex",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if head_dim is not None:
            if key_dim is not None and key_dim != head_dim:
                raise ValueError("key_dim must match head_dim when both are provided")
            if value_dim is not None and value_dim != head_dim:
                raise ValueError("value_dim must match head_dim when both are provided")
            key_dim = head_dim
            value_dim = head_dim

        if key_dim is None or value_dim is None:
            if d_model % num_heads != 0:
                raise ValueError(
                    "d_model must be divisible by num_heads when key_dim or value_dim is not provided"
                )
            default_head_dim = d_model // num_heads
            if key_dim is None:
                key_dim = default_head_dim
            if value_dim is None:
                value_dim = default_head_dim
        if key_dim <= 0 or value_dim <= 0:
            raise ValueError("key_dim and value_dim must be positive")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = key_dim if key_dim == value_dim else None
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.key_inner_dim = num_heads * key_dim
        self.value_inner_dim = num_heads * value_dim
        self.inner_dim = self.value_inner_dim
        self.eps = eps
        if feature_mode not in FEATURE_MODES:
            available = ", ".join(sorted(FEATURE_MODES))
            raise ValueError(f"feature_mode must be one of: {available}")
        self.feature_mode = feature_mode

        self.q_proj = ComplexLinear(d_model, self.key_inner_dim)
        self.k_proj = ComplexLinear(d_model, self.key_inner_dim)
        self.v_proj = ComplexLinear(d_model, self.value_inner_dim)
        self.out_proj = ComplexLinear(self.value_inner_dim, d_model)
        self.transport = TemporalTransport(num_heads, mode=transport)

    def _split_heads(self, x: Tensor, head_dim: int) -> Tensor:
        return x.reshape(*x.shape[:-1], self.num_heads, head_dim)

    def _merge_value_heads(self, x: Tensor) -> Tensor:
        return x.reshape(*x.shape[:-2], self.value_inner_dim)

    def _project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        q = self._split_heads(self.q_proj(x), self.key_dim)
        k = self._split_heads(self.k_proj(x), self.key_dim)
        v = self._split_heads(self.v_proj(x), self.value_dim)
        q_gate = positive_gate(q)
        k_gate = positive_gate(k)
        if self.feature_mode == "complex":
            q_feature = phase_feature(q, self.eps)
            k_feature = phase_feature(k, self.eps)
        else:
            q_feature = q_gate.to(dtype=q.dtype)
            k_feature = k_gate.to(dtype=k.dtype)
        return q_gate, q_feature, k_gate, k_feature, v

    def initial_state(self, batch_shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> AssociativeState:
        return identity_state(
            batch_shape,
            self.num_heads,
            self.value_dim,
            self.key_dim,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: Tensor, *, return_state: bool = False) -> Tensor | tuple[Tensor, AssociativeState]:
        if x.ndim != 3:
            raise ValueError("forward expects shape [batch, seq, d_model]")

        q_gate, q_phase, k_gate, k_phase, v = self._project(x)
        alpha_s, alpha_z = self.transport(x.shape[:-1], device=x.device, dtype=x.dtype)
        local = local_state(k_phase, k_gate, v, alpha_s, alpha_z)
        prefix = parallel_prefix_scan(local, dim=1)
        y = measure(prefix, q_phase, q_gate, self.eps)
        out = self.out_proj(self._merge_value_heads(y))
        if return_state:
            return out, prefix
        return out

    def step(self, x: Tensor, memory: AssociativeState | None = None) -> tuple[Tensor, AssociativeState]:
        if x.ndim != 2:
            raise ValueError("step expects shape [batch, d_model]")

        q_gate, q_phase, k_gate, k_phase, v = self._project(x)
        alpha_s, alpha_z = self.transport(x.shape[:-1], device=x.device, dtype=x.dtype)
        current = local_state(k_phase, k_gate, v, alpha_s, alpha_z)
        if memory is None:
            memory = self.initial_state(x.shape[:-1], device=x.device, dtype=x.dtype)

        new_memory = compose(memory, current)
        y = measure(new_memory, q_phase, q_gate, self.eps)
        return self.out_proj(self._merge_value_heads(y)), new_memory


class ComplexMLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = ComplexLinear(d_model, hidden_dim)
        self.act = ModReLU(hidden_dim)
        self.fc2 = ComplexLinear(hidden_dim, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class ComposedStateBlock(nn.Module):
    """Residual block: complex composed-state mixer plus phase-preserving MLP."""

    def __init__(
        self,
        d_model: int,
        *,
        num_heads: int = 1,
        head_dim: int | None = None,
        key_dim: int | None = None,
        value_dim: int | None = None,
        mlp_ratio: int = 4,
        transport: bool | str = "rotary_decay",
        feature_mode: str = "complex",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm1 = ComplexRMSNorm(d_model, eps)
        self.mix = ComposedStateMixing(
            d_model,
            num_heads=num_heads,
            head_dim=head_dim,
            key_dim=key_dim,
            value_dim=value_dim,
            transport=transport,
            feature_mode=feature_mode,
        )
        self.norm2 = ComplexRMSNorm(d_model, eps)
        self.mlp = ComplexMLP(d_model, d_model * mlp_ratio)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mix(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def step(self, x: Tensor, memory: AssociativeState | None = None) -> tuple[Tensor, AssociativeState]:
        mixed, new_memory = self.mix.step(self.norm1(x), memory)
        x = x + mixed
        x = x + self.mlp(self.norm2(x))
        return x, new_memory
