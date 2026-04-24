import torch
import pytest

from metanoos.state import (
    AssociativeState,
    compose,
    identity_state,
    local_state,
    measure,
    parallel_prefix_scan,
    prefix_scan,
    state_memory_estimate,
)


def random_state(batch: int = 2, heads: int = 3, value_dim: int = 4, key_dim: int = 5) -> AssociativeState:
    alpha_s = torch.randn(batch, heads, dtype=torch.cfloat)
    alpha_z = torch.rand(batch, heads).add(0.1)
    S = torch.randn(batch, heads, value_dim, key_dim, dtype=torch.cfloat)
    Z = torch.rand(batch, heads, key_dim)
    return AssociativeState(alpha_s=alpha_s, alpha_z=alpha_z, S=S, Z=Z)


def assert_state_close(left: AssociativeState, right: AssociativeState) -> None:
    torch.testing.assert_close(left.alpha_s, right.alpha_s)
    torch.testing.assert_close(left.alpha_z, right.alpha_z)
    torch.testing.assert_close(left.S, right.S)
    torch.testing.assert_close(left.Z, right.Z)


def test_composition_is_associative() -> None:
    a = random_state()
    b = random_state()
    c = random_state()

    assert_state_close(compose(compose(a, b), c), compose(a, compose(b, c)))


def test_identity_is_neutral() -> None:
    state = random_state()
    identity = identity_state((2,), 3, 4, 5)

    assert_state_close(compose(identity, state), state)
    assert_state_close(compose(state, identity), state)


def test_prefix_scan_matches_recurrent_composition() -> None:
    batch, seq_len, heads, value_dim, key_dim = 2, 6, 2, 3, 4
    k_phase = torch.randn(batch, seq_len, heads, key_dim, dtype=torch.cfloat)
    k_gate = torch.rand(batch, seq_len, heads, key_dim).add(0.1)
    v = torch.randn(batch, seq_len, heads, value_dim, dtype=torch.cfloat)
    alpha_s = torch.randn(batch, seq_len, heads, dtype=torch.cfloat)
    alpha_z = torch.rand(batch, seq_len, heads).add(0.1)
    local = local_state(k_phase, k_gate, v, alpha_s, alpha_z)

    scanned = prefix_scan(local)
    recurrent = []
    acc = identity_state((batch,), heads, value_dim, key_dim)
    for index in range(seq_len):
        step = AssociativeState(
            alpha_s=local.alpha_s[:, index],
            alpha_z=local.alpha_z[:, index],
            S=local.S[:, index],
            Z=local.Z[:, index],
        )
        acc = compose(acc, step)
        recurrent.append(acc)

    expected = AssociativeState(
        alpha_s=torch.stack([state.alpha_s for state in recurrent], dim=1),
        alpha_z=torch.stack([state.alpha_z for state in recurrent], dim=1),
        S=torch.stack([state.S for state in recurrent], dim=1),
        Z=torch.stack([state.Z for state in recurrent], dim=1),
    )
    assert_state_close(scanned, expected)


def test_parallel_prefix_scan_matches_serial_reference() -> None:
    torch.manual_seed(0)
    batch, seq_len, heads, value_dim, key_dim = 2, 7, 2, 3, 4
    k_phase = torch.randn(batch, seq_len, heads, key_dim, dtype=torch.cfloat)
    k_gate = torch.rand(batch, seq_len, heads, key_dim).add(0.1)
    v = torch.randn(batch, seq_len, heads, value_dim, dtype=torch.cfloat)
    alpha_s = torch.randn(batch, seq_len, heads, dtype=torch.cfloat)
    alpha_z = torch.rand(batch, seq_len, heads).add(0.1)
    local = local_state(k_phase, k_gate, v, alpha_s, alpha_z)

    serial = prefix_scan(local)
    parallel = parallel_prefix_scan(local)

    torch.testing.assert_close(parallel.alpha_s, serial.alpha_s, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(parallel.alpha_z, serial.alpha_z, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(parallel.S, serial.S, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(parallel.Z, serial.Z, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_parallel_prefix_scan_cuda_backward() -> None:
    device = torch.device("cuda")
    batch, seq_len, heads, value_dim, key_dim = 2, 8, 2, 3, 4
    k_phase = torch.randn(batch, seq_len, heads, key_dim, device=device, dtype=torch.cfloat, requires_grad=True)
    k_gate = torch.rand(batch, seq_len, heads, key_dim, device=device).add(0.1)
    v = torch.randn(batch, seq_len, heads, value_dim, device=device, dtype=torch.cfloat, requires_grad=True)
    alpha_s = torch.randn(batch, seq_len, heads, device=device, dtype=torch.cfloat, requires_grad=True)
    alpha_z = torch.rand(batch, seq_len, heads, device=device).add(0.1)
    local = local_state(k_phase, k_gate, v, alpha_s, alpha_z)

    scanned = parallel_prefix_scan(local)
    loss = scanned.S.abs().square().mean() + scanned.alpha_s.abs().square().mean()
    loss.backward()

    assert k_phase.grad is not None
    assert v.grad is not None
    assert alpha_s.grad is not None


def test_measurement_uses_real_positive_support() -> None:
    prefix = random_state(batch=2, heads=2, value_dim=3, key_dim=4)
    q_phase = torch.randn(2, 2, 4, dtype=torch.cfloat)
    q_gate = torch.rand(2, 2, 4).add(0.1)

    y = measure(prefix, q_phase, q_gate)

    assert y.shape == (2, 2, 3)
    assert y.is_complex()


def test_state_memory_estimate_counts_materialized_prefix_tensors() -> None:
    estimate = state_memory_estimate(
        seq_len=2048,
        batch_size=4,
        num_heads=8,
        value_dim=64,
        key_dim=64,
        dtype=torch.cfloat,
    )

    assert estimate.S_bytes == 2 * 1024**3
    assert estimate.Z_bytes == 4 * 2048 * 4 * 8 * 64
    assert estimate.total_bytes > estimate.S_bytes
    assert estimate.as_dict()["total_bytes"] == estimate.total_bytes
