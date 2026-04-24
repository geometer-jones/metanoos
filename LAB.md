# Lab Notebook

This file is the running experimentation notebook for Metanoos. Every entry is
numbered and dated. Keep entries factual: hypothesis, setup, command or code
path, expected evidence, and result/status.

TODO experiments are kept at the top. Recorded experiments are reverse
chronological by experiment number: newest at top, oldest at bottom.

## TODO - Experiments To Run

### Experiment 008 - 2026-04-24 - 50M Layout Sweep

**Purpose:** Compare depth, width, key/value state size, and MLP expansion at
roughly 50M real-valued parameters.

**Status:** TODO. Requires implementation support for tied input/output carrier
weights and separate `d_k` / `d_v` dimensions before the parameter counts below
are exact.

**Budget convention:**

- Count each complex parameter as two real-valued parameters.
- Target approximately 50M real-valued parameters.
- Assume tied input embedding and output unembedding for the Born readout.
- Initial vocabulary assumption: 8k subwords. Recompute if using 16k or larger.

**Current implementation gaps to close first:**

- Add optional tying between `embed.weight` and the Born readout carrier.
- Split the current single `head_dim` into explicit `key_dim` and `value_dim`.
- Add a parameter-count utility that reports real-valued parameter count.
- Add a memory-estimate utility for prefix state size:
  `batch * heads * value_dim * key_dim * seq_len` complex elements for `S`.

**Layout candidates:**

| Layout | `d_model` | Layers | `d_k` | `d_v` | MLP ratio | Emb. params at V=8k | Block params | Total approx. | Hypothesis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A - Balanced baseline | 512 | 6 | 512 | 512 | 4 | 8.4M | 39.3M | 47.7M | Standard deep-ish starting point |
| B - Wide and shallow | 768 | 3 | 768 | 768 | 4 | 12.3M | 31.7M | 44.0M | Tests single-step richness vs depth |
| C - Deep and narrow | 384 | 12 | 384 | 384 | 4 | 6.1M | 44.2M | 50.3M | Tests whether rotational memory benefits from many layers |
| D - Bottleneck state | 512 | 6 | 256 | 512 | 4 | 8.4M | 30.1M | 38.5M | Tests low-rank relational state and longer-context efficiency |
| E - Extra-wide MLP | 512 | 5 | 512 | 512 | 6 | 8.4M | 40.9M | 49.3M | Tests whether local per-token compute is the bottleneck |

**Recommended order:**

1. Run Layout A first to validate the CUDA training pipeline.
2. Compare Layout B vs Layout C for depth/width tradeoff.
3. Run Layout D if sequence length or batch size is constrained by `S` memory.
4. Run Layout E if losses suggest the scan is not the bottleneck and more local
   processing may help.

**Primary comparisons:**

- A vs B: depth vs width at similar scale.
- B vs C: single-step capacity vs many transport/composition layers.
- A vs D: full key state vs bottleneck key state.
- A vs E: associative scan capacity vs local MLP capacity.

**Metrics:**

- validation loss and bpc;
- train loss curve and stability;
- tokens per second;
- peak GPU memory;
- parameter count in real-valued terms;
- sample quality at fixed prompts.

**Acceptance rule:** Layout comparisons only make sense after the ablation arms
from Experiment 006 are runnable on CUDA. Do not mix conclusions from different
vocabularies or tokenizers without recomputing parameter budgets.

### Experiment 007 - 2026-04-24 - External Baselines

**Purpose:** Keep the comparison set honest by adding neighbors outside this
implementation.

**Status:** TODO.

**Baselines to add or import:**

- real-valued GLA with matched parameter count;
- real-valued causal linear attention without decay;
- diagonal real SSM with matched state size;
- diagonal complex SSM or S4D-style model if available.

**Primary requirement:** Baselines must use the same data split, tokenizer,
optimizer budget, logging schema, and evaluation metrics as the internal
ablation sweep.

### Experiment 006 - 2026-04-24 - CUDA Ablation Sweep

**Purpose:** Compare the default complex rotational GLA against its immediate
neighbors under matched training conditions.

**Status:** TODO. Run on a CUDA machine.

**Fixed settings:**

- corpus: `greek_classics` initially, then repeat on other bundled corpora;
- tokenization: UTF-8 bytes;
- seed: same seed across all runs;
- model scale: same `d_model`, `num_layers`, `num_heads`, `mlp_ratio`;
- optimizer: AdamW with same LR, betas, and weight decay;
- batch shape: same `batch_size` and `seq_len`;
- validation: same held-out split and `val_batches`;
- logging: write `metadata.json`, `metrics.jsonl`, `summary.json`.

**Run matrix:**

| Experiment arm | Preset | Primary comparison |
| --- | --- | --- |
| A | `complex_rotary_gla` | Default |
| B | `complex_decay_gla` | A vs B isolates phase rotation |
| C | `complex_linear_attention` | A/B vs C isolates temporal decay |
| D | `magnitude_feature_gla` | A vs D isolates q/k phase features |
| E | `real_readout_gla` | A vs E isolates Born readout |

**Primary metrics:**

- validation loss;
- validation bits per byte/character where applicable;
- train loss curve;
- tokens per second;
- gradient norm stability;
- sample quality snapshots at fixed prompts.

**Acceptance rule:** No architectural claim should be made from one run. Treat
single-seed results as smoke evidence. Require repeated seeds before claiming an
ablation effect.

## Recorded Experiments

### Experiment 005 - 2026-04-24 - Device Suitability For Ablation Training

**Purpose:** Determine which accelerator backend is valid for real ablation
training.

**Probe:**

```bash
python3 - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
print("mps_available", torch.backends.mps.is_available())
PY
```

**Observed on 2026-04-24 local machine:**

```text
torch 2.6.0
cuda_available False
cuda_device_count 0
mps_available True
```

**MPS backward probe result:** MPS forward completed, but backward failed in
PyTorch's complex tensor path. Therefore MPS is not a valid training backend for
these ablations.

**Conclusion:** Real ablation runs should be on CUDA. CPU is acceptable for
smoke tests and correctness tests only.

**Status:** Local machine blocked for GPU ablation training because CUDA is not
available.

### Experiment 004 - 2026-04-24 - Readout And Optimizer Safety

**Purpose:** Verify that the vocabulary readout is usable with PyTorch
autograd/optimizers and that the Born readout keeps the intended global phase
invariance.

**Files:**

- `src/metanoos/model.py`
- `tests/test_model.py`

**Tests:**

- `test_language_model_outputs_real_logits_and_loss`
- `test_born_readout_is_global_phase_invariant`
- `test_language_model_adamw_step_handles_complex_readout_gradients`
- `test_language_model_cuda_forward_backward`

**Command:**

```bash
python3 -m pytest tests/test_model.py -k "readout or adamw or cuda or outputs"
```

**Expected evidence:**

- Language model logits are real-valued.
- Loss is finite.
- Born logits are invariant under global hidden-state phase rotation.
- Complex readout gradients are physical tensors accepted by AdamW.
- CUDA forward/backward works when CUDA is available.

**Status:** Implemented. CUDA-specific test is skipped on machines without
CUDA.

### Experiment 003 - 2026-04-24 - Immediate-Neighbor Ablation Registry

**Purpose:** Make the nearest GLA-family ablations explicit and reproducible
from names rather than ad hoc keyword arguments.

**Files:**

- `src/metanoos/ablations.py`
- `tests/test_model.py`

**Ablations:**

| Name | Transport | q/k features | Readout | Question |
| --- | --- | --- | --- | --- |
| `complex_rotary_gla` | complex decay | complex phase | Born | Default model |
| `complex_decay_gla` | real decay | complex phase | Born | Does rotation help? |
| `complex_linear_attention` | none | complex phase | Born | Does decay help? |
| `magnitude_feature_gla` | complex decay | magnitude-only | Born | Do q/k phases help? |
| `real_readout_gla` | complex decay | complex phase | real part | Does Born readout help? |

**Tests:**

- `test_named_ablation_presets_instantiate`
- `test_ablation_kwargs_allow_overrides`

**Command:**

```bash
python3 -m pytest tests/test_model.py -k ablation
```

**Expected evidence:**

- Every named preset instantiates a language model.
- Every preset produces real logits with the expected shape.
- Preset kwargs can be overridden by experiment code.

**Status:** Implemented.

### Experiment 002 - 2026-04-24 - Model-Level Scan/Step Equivalence

**Purpose:** Verify that full-sequence training and one-token inference evaluate
the same recurrence at the layer and language-model levels.

**Files:**

- `src/metanoos/layers.py`
- `src/metanoos/model.py`
- `tests/test_model.py`

**Tests:**

- `test_mixing_scan_matches_step`
- `test_language_model_scan_matches_step`
- `test_mixing_ablation_modes_scan_match_step`

**Command:**

```bash
python3 -m pytest tests/test_model.py
```

**Expected evidence:**

- `ComposedStateMixing.forward(...)` matches repeated `step(...)`.
- `ComposedStateLanguageModel.forward(...)` logits match repeated
  `step(...)` logits.
- Equivalence holds for all implemented transport and q/k feature ablation
  modes.

**Status:** Implemented.

### Experiment 001 - 2026-04-24 - Core Associative State Tests

**Purpose:** Verify that the complex GLA state algebra is well formed before
training comparisons are trusted.

**Files:**

- `src/metanoos/state.py`
- `tests/test_state.py`

**Tests:**

- `test_composition_is_associative`
- `test_identity_is_neutral`
- `test_prefix_scan_matches_recurrent_composition`
- `test_parallel_prefix_scan_matches_serial_reference`
- `test_measurement_uses_real_positive_support`
- `test_parallel_prefix_scan_cuda_backward`

**Command:**

```bash
python3 -m pytest tests/test_state.py
```

**Expected evidence:**

- Associative grouping produces the same state.
- Identity state is neutral on both sides.
- Serial prefix scan matches recurrent stepping.
- Parallel scan matches the serial reference.
- Measurement returns complex values with real positive support.
- CUDA backward works when CUDA is available.

**Status:** Implemented. CUDA-specific test is skipped on machines without
CUDA.
