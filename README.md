# Metanoos

PyTorch research testbed for complex-valued gated linear attention with learned
per-head rotational decay.

This is in the GLA/linear-attention/SSM family. The core object is an
associative recurrent state:

```text
A = (alpha_S, alpha_Z, S, Z)

A_left o A_right =
(
  alpha_S,right alpha_S,left,
  alpha_Z,right alpha_Z,left,
  alpha_S,right S_left + S_right,
  alpha_Z,right Z_left + Z_right
)
```

where `S` is complex value-key content, `Z` is real positive normalizer mass,
`alpha_S` is optional complex content decay, and `alpha_Z` is real normalizer
decay.

The package keeps the primitive separate from model blocks:

- `metanoos.state`: associative state, composition, causal scan, measurement.
- `metanoos.complex_ops`: complex gates, phase features, normalization, layers.
- `metanoos.layers`: composed-state sequence mixer and residual block.
- `metanoos.model`: language-model wrapper with Born-style vocabulary readout.
- `metanoos.ablations`: named immediate-neighbor ablation presets.

## Ablations

Implemented presets:

- `complex_rotary_gla`: default complex GLA with per-head complex decay.
- `complex_decay_gla`: removes transport phase rotation.
- `complex_linear_attention`: removes temporal decay.
- `complex_rope_decay_gla`: fixed RoPE q/k phase with real content decay.
- `complex_rope_rotary_gla`: fixed RoPE q/k phase plus learned rotary decay.
- `magnitude_feature_gla`: removes q/k phase features.
- `real_readout_gla`: replaces Born logits with real-part logits.

```python
from metanoos import ComposedStateLanguageModel, ablation_model_kwargs

model = ComposedStateLanguageModel(
    vocab_size=128,
    d_model=64,
    num_layers=2,
    num_heads=4,
    **ablation_model_kwargs("complex_decay_gla"),
)
```

For quick order-sensitivity probes, the RoPE/transport matrix can be run on
synthetic tasks:

```bash
python3 scripts/run_order_sweep.py --task permutation
```

External comparisons should include real-valued GLA and diagonal real/complex
SSM baselines with matched parameter and compute budgets.

## Experiment Sizing

```python
from metanoos import ComposedStateLanguageModel, state_memory_estimate

model = ComposedStateLanguageModel(
    vocab_size=8000,
    d_model=512,
    num_layers=6,
    num_heads=8,
    key_dim=32,              # per-head key dimension
    value_dim=64,            # per-head value dimension
    tie_readout_carrier=True,
)

print(model.real_param_count())

estimate = state_memory_estimate(
    seq_len=2048,
    batch_size=4,
    num_heads=8,
    key_dim=32,
    value_dim=64,
)
print(estimate.S_bytes, estimate.total_bytes)
```

## Quick Start

```python
import torch

from metanoos import ComposedStateLanguageModel

model = ComposedStateLanguageModel(
    vocab_size=128,
    d_model=64,
    num_layers=2,
    num_heads=4,
)

tokens = torch.randint(0, 128, (2, 16))
out = model(tokens, labels=tokens)

print(out.logits.shape)
print(out.loss)
```

## Verification

```bash
python3 -m pytest
```

The sequence mixer uses `parallel_prefix_scan` for full-sequence training. The
serial `prefix_scan` remains the reference implementation for equivalence
checks and recurrent inference parity. CUDA forward/backward tests run
automatically when a CUDA device is available; Apple MPS does not currently
support the full complex backward path used by this model.
