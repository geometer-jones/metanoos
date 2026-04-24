"""Associative scan state for complex gated linear attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from metanoos.complex_ops import real_dtype_for


@dataclass(frozen=True)
class AssociativeState:
    """Chronological state A = (alpha_S, alpha_Z, S, Z).

    Shape convention:
      alpha_s: [..., heads] complex
      alpha_z: [..., heads] real
      S:       [..., heads, value_dim, key_dim] complex
      Z:       [..., heads, key_dim] real
    """

    alpha_s: Tensor
    alpha_z: Tensor
    S: Tensor
    Z: Tensor

    def to(self, *args: object, **kwargs: object) -> "AssociativeState":
        return AssociativeState(
            alpha_s=self.alpha_s.to(*args, **kwargs),
            alpha_z=self.alpha_z.to(*args, **kwargs),
            S=self.S.to(*args, **kwargs),
            Z=self.Z.to(*args, **kwargs),
        )


@dataclass(frozen=True)
class StateMemoryEstimate:
    """Byte estimate for materialized prefix state tensors."""

    alpha_s_bytes: int
    alpha_z_bytes: int
    S_bytes: int
    Z_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.alpha_s_bytes + self.alpha_z_bytes + self.S_bytes + self.Z_bytes

    @property
    def total_gib(self) -> float:
        return self.total_bytes / (1024**3)

    def as_dict(self) -> dict[str, int]:
        return {
            "alpha_s_bytes": self.alpha_s_bytes,
            "alpha_z_bytes": self.alpha_z_bytes,
            "S_bytes": self.S_bytes,
            "Z_bytes": self.Z_bytes,
            "total_bytes": self.total_bytes,
        }


def state_memory_estimate(
    *,
    seq_len: int,
    batch_size: int,
    num_heads: int,
    value_dim: int,
    key_dim: int,
    dtype: torch.dtype = torch.cfloat,
    num_layers: int = 1,
) -> StateMemoryEstimate:
    """Estimate materialized inclusive-prefix state memory.

    The estimate counts the tensors returned by `parallel_prefix_scan` for one
    or more layers: complex `alpha_s` and `S`, plus real `alpha_z` and `Z`.
    It does not include autograd saved tensors, temporary scan tensors, model
    parameters, optimizer state, or allocator fragmentation.
    """

    dimensions = {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "num_heads": num_heads,
        "value_dim": value_dim,
        "key_dim": key_dim,
        "num_layers": num_layers,
    }
    invalid = [name for name, value in dimensions.items() if value <= 0]
    if invalid:
        names = ", ".join(invalid)
        raise ValueError(f"state memory dimensions must be positive: {names}")

    complex_bytes = torch.empty((), dtype=dtype).element_size()
    real_bytes = torch.empty((), dtype=real_dtype_for(dtype)).element_size()
    head_steps = num_layers * batch_size * seq_len * num_heads
    return StateMemoryEstimate(
        alpha_s_bytes=head_steps * complex_bytes,
        alpha_z_bytes=head_steps * real_bytes,
        S_bytes=head_steps * value_dim * key_dim * complex_bytes,
        Z_bytes=head_steps * key_dim * real_bytes,
    )


def compose(left: AssociativeState, right: AssociativeState) -> AssociativeState:
    """Chronologically compose `left o right`.

    The operation is associative and defines both scan construction and
    recurrent stepping.
    """

    return AssociativeState(
        alpha_s=right.alpha_s * left.alpha_s,
        alpha_z=right.alpha_z * left.alpha_z,
        S=right.alpha_s[..., None, None] * left.S + right.S,
        Z=right.alpha_z[..., None] * left.Z + right.Z,
    )


def identity_like(template: AssociativeState) -> AssociativeState:
    """Return the identity state with the same non-sequence shape as `template`."""

    return AssociativeState(
        alpha_s=torch.ones_like(template.alpha_s),
        alpha_z=torch.ones_like(template.alpha_z),
        S=torch.zeros_like(template.S),
        Z=torch.zeros_like(template.Z),
    )


def identity_state(
    batch_shape: tuple[int, ...],
    heads: int,
    value_dim: int,
    key_dim: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.cfloat,
) -> AssociativeState:
    """Construct I = (1, 1, 0, 0) for recurrent inference."""

    real_dtype = real_dtype_for(dtype)
    return AssociativeState(
        alpha_s=torch.ones(*batch_shape, heads, device=device, dtype=dtype),
        alpha_z=torch.ones(*batch_shape, heads, device=device, dtype=real_dtype),
        S=torch.zeros(*batch_shape, heads, value_dim, key_dim, device=device, dtype=dtype),
        Z=torch.zeros(*batch_shape, heads, key_dim, device=device, dtype=real_dtype),
    )


def local_state(k_phase: Tensor, k_gate: Tensor, v: Tensor, alpha_s: Tensor, alpha_z: Tensor) -> AssociativeState:
    """Build local generator A_t from key features, value amplitudes, and transport."""

    S = torch.einsum("...hv,...hk->...hvk", v, k_phase.conj())
    return AssociativeState(alpha_s=alpha_s, alpha_z=alpha_z, S=S, Z=k_gate)


def _select_state(state: AssociativeState, dim: int, index: int) -> AssociativeState:
    return AssociativeState(
        alpha_s=state.alpha_s.select(dim, index),
        alpha_z=state.alpha_z.select(dim, index),
        S=state.S.select(dim, index),
        Z=state.Z.select(dim, index),
    )


def _slice_tensor(x: Tensor, dim: int, start: int, end: int) -> Tensor:
    slices = [slice(None)] * x.ndim
    slices[dim] = slice(start, end)
    return x[tuple(slices)]


def _slice_state(state: AssociativeState, dim: int, start: int, end: int) -> AssociativeState:
    return AssociativeState(
        alpha_s=_slice_tensor(state.alpha_s, dim, start, end),
        alpha_z=_slice_tensor(state.alpha_z, dim, start, end),
        S=_slice_tensor(state.S, dim, start, end),
        Z=_slice_tensor(state.Z, dim, start, end),
    )


def _stack_states(states: list[AssociativeState], dim: int) -> AssociativeState:
    return AssociativeState(
        alpha_s=torch.stack([state.alpha_s for state in states], dim=dim),
        alpha_z=torch.stack([state.alpha_z for state in states], dim=dim),
        S=torch.stack([state.S for state in states], dim=dim),
        Z=torch.stack([state.Z for state in states], dim=dim),
    )


def _cat_states(states: list[AssociativeState], dim: int) -> AssociativeState:
    return AssociativeState(
        alpha_s=torch.cat([state.alpha_s for state in states], dim=dim),
        alpha_z=torch.cat([state.alpha_z for state in states], dim=dim),
        S=torch.cat([state.S for state in states], dim=dim),
        Z=torch.cat([state.Z for state in states], dim=dim),
    )


def _normalize_scan_dim(local: AssociativeState, dim: int) -> int:
    normalized = dim % local.alpha_s.ndim
    length = local.alpha_s.shape[normalized]
    if (
        local.alpha_z.shape[normalized] != length
        or local.S.shape[normalized] != length
        or local.Z.shape[normalized] != length
    ):
        raise ValueError("scan dimension must refer to a shared state dimension")
    return normalized


def prefix_scan(local: AssociativeState, *, dim: int = 1) -> AssociativeState:
    """Inclusive causal scan over local generators.

    This implementation is intentionally direct and deterministic. Because it
    uses the same `compose` operation as recurrent stepping, it is the reference
    implementation for training/inference equivalence.
    """

    dim = _normalize_scan_dim(local, dim)
    if local.S.shape[dim] == 0:
        raise ValueError("prefix_scan requires a non-empty sequence dimension")

    steps: list[AssociativeState] = []
    acc = identity_like(_select_state(local, dim, 0))
    for index in range(local.S.shape[dim]):
        acc = compose(acc, _select_state(local, dim, index))
        steps.append(acc)
    return _stack_states(steps, dim)


def parallel_prefix_scan(local: AssociativeState, *, dim: int = 1) -> AssociativeState:
    """Inclusive causal scan using log-depth tensor operations.

    The serial `prefix_scan` is the numerical reference. This implementation
    uses the same associative `compose` law but groups terms in Kogge-Stone
    order, which exposes the scan to accelerator backends.
    """

    dim = _normalize_scan_dim(local, dim)
    length = local.S.shape[dim]
    if length == 0:
        raise ValueError("parallel_prefix_scan requires a non-empty sequence dimension")

    scanned = local
    offset = 1
    while offset < length:
        left = _slice_state(scanned, dim, 0, length - offset)
        right = _slice_state(scanned, dim, offset, length)
        composed = compose(left, right)
        scanned = _cat_states([_slice_state(scanned, dim, 0, offset), composed], dim)
        offset *= 2
    return scanned


def measure(prefix: AssociativeState, q_phase: Tensor, q_gate: Tensor, eps: float = 1e-8) -> Tensor:
    """Measure prefix state with query features.

    Returns complex readout y_t = (S_prefix,t q_phase_t) /
    (<Z_prefix,t, q_gate_t> + eps).
    """

    numerator = torch.einsum("...hvk,...hk->...hv", prefix.S, q_phase)
    denominator = (prefix.Z * q_gate).sum(dim=-1).add(eps)
    return numerator / denominator[..., None]
