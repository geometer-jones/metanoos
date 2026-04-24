import pytest
import torch

from metanoos import (
    ComposedStateLanguageModel,
    ComposedStateMixing,
    ablation_model_kwargs,
    available_ablations,
    phase_feature,
    positive_gate,
)


def test_gate_is_real_positive_and_phase_feature_is_complex() -> None:
    x = torch.randn(2, 4, dtype=torch.cfloat)

    gate = positive_gate(x)
    feature = phase_feature(x)

    assert not gate.is_complex()
    assert torch.all(gate > 0)
    assert feature.is_complex()


def test_mixing_scan_matches_step() -> None:
    torch.manual_seed(0)
    layer = ComposedStateMixing(d_model=8, num_heads=2)
    x = torch.randn(2, 5, 8, dtype=torch.cfloat)

    full = layer(x)
    memory = None
    stepped = []
    for index in range(x.shape[1]):
        y, memory = layer.step(x[:, index], memory)
        stepped.append(y)
    recurrent = torch.stack(stepped, dim=1)

    torch.testing.assert_close(full, recurrent, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("transport", ["rotary_decay", "decay", "none"])
@pytest.mark.parametrize("feature_mode", ["complex", "magnitude"])
def test_mixing_ablation_modes_scan_match_step(transport: str, feature_mode: str) -> None:
    torch.manual_seed(0)
    layer = ComposedStateMixing(d_model=8, num_heads=2, transport=transport, feature_mode=feature_mode)
    x = torch.randn(2, 5, 8, dtype=torch.cfloat)

    full = layer(x)
    memory = None
    stepped = []
    for index in range(x.shape[1]):
        y, memory = layer.step(x[:, index], memory)
        stepped.append(y)
    recurrent = torch.stack(stepped, dim=1)

    torch.testing.assert_close(full, recurrent, rtol=1e-5, atol=1e-5)


def test_language_model_outputs_real_logits_and_loss() -> None:
    torch.manual_seed(0)
    model = ComposedStateLanguageModel(vocab_size=17, d_model=8, num_layers=2, num_heads=2)
    input_ids = torch.randint(0, 17, (3, 4))

    out = model(input_ids, labels=input_ids)

    assert out.logits.shape == (3, 4, 17)
    assert not out.logits.is_complex()
    assert out.loss is not None
    assert torch.isfinite(out.loss)


def test_language_model_scan_matches_step() -> None:
    torch.manual_seed(0)
    model = ComposedStateLanguageModel(vocab_size=13, d_model=8, num_layers=2, num_heads=2)
    input_ids = torch.randint(0, 13, (2, 5))

    full = model(input_ids).logits
    memories = None
    stepped = []
    for index in range(input_ids.shape[1]):
        logits, memories = model.step(input_ids[:, index], memories)
        stepped.append(logits)
    recurrent = torch.stack(stepped, dim=1)

    torch.testing.assert_close(full, recurrent, rtol=1e-5, atol=1e-5)


def test_named_ablation_presets_instantiate() -> None:
    names = [spec.name for spec in available_ablations()]
    assert names == [
        "complex_decay_gla",
        "complex_linear_attention",
        "complex_rotary_gla",
        "magnitude_feature_gla",
        "real_readout_gla",
    ]

    input_ids = torch.randint(0, 13, (2, 4))
    for spec in available_ablations():
        torch.manual_seed(0)
        model = ComposedStateLanguageModel(
            vocab_size=13,
            d_model=8,
            num_layers=1,
            num_heads=2,
            **spec.kwargs(),
        )
        logits = model(input_ids).logits
        assert logits.shape == (2, 4, 13)
        assert not logits.is_complex()


def test_ablation_kwargs_allow_overrides() -> None:
    kwargs = ablation_model_kwargs("complex_rotary_gla", readout="real")

    assert kwargs["transport"] == "rotary_decay"
    assert kwargs["feature_mode"] == "complex"
    assert kwargs["readout"] == "real"


def test_born_readout_is_global_phase_invariant() -> None:
    torch.manual_seed(0)
    model = ComposedStateLanguageModel(vocab_size=11, d_model=6, num_layers=1, num_heads=2, readout="born")
    hidden = torch.randn(2, 3, 6, dtype=torch.cfloat)
    theta = torch.tensor(1.7)

    logits = model.logits_from_hidden(hidden)
    rotated = model.logits_from_hidden(torch.exp(1j * theta) * hidden)

    torch.testing.assert_close(logits, rotated, rtol=1e-5, atol=1e-5)


def test_language_model_adamw_step_handles_complex_readout_gradients() -> None:
    torch.manual_seed(0)
    model = ComposedStateLanguageModel(vocab_size=17, d_model=8, num_layers=1, num_heads=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, 17, (2, 5))
    labels = torch.randint(0, 17, (2, 5))

    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids, labels=labels).loss
    assert loss is not None
    loss.backward()

    assert model.readout_carrier.weight.grad is not None
    assert not model.readout_carrier.weight.grad.is_conj()

    optimizer.step()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_language_model_cuda_forward_backward() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    model = ComposedStateLanguageModel(vocab_size=17, d_model=8, num_layers=1, num_heads=2).to(device)
    input_ids = torch.randint(0, 17, (2, 5), device=device)
    labels = torch.randint(0, 17, (2, 5), device=device)

    loss = model(input_ids, labels=labels).loss
    assert loss is not None
    loss.backward()

    assert model.readout_carrier.weight.grad is not None
