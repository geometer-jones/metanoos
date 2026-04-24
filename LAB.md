# Lab Notebook

Chronological experiment log for Metanoos. Lower numbers are older and appear
earlier. Keep entries to the information needed to reproduce or interpret
results.

## Recorded

### Experiment 001 - 2026-04-24 - Core State Algebra

**Status:** Implemented.

**Purpose:** Verify the associative state algebra before trusting training
comparisons.

**Scope:** `src/metanoos/state.py`, `tests/test_state.py`

**Command:** `python3 -m pytest tests/test_state.py`

**Checks:** associativity, identity, serial/recurrent prefix equivalence,
parallel scan equivalence, measurement shape/type, CUDA backward when available.

### Experiment 002 - 2026-04-24 - Scan/Step Equivalence

**Status:** Implemented.

**Purpose:** Verify full-sequence training and one-token inference use the same
recurrence.

**Scope:** `src/metanoos/layers.py`, `src/metanoos/model.py`,
`tests/test_model.py`

**Command:** `python3 -m pytest tests/test_model.py`

**Checks:** mixer scan matches repeated `step`, language-model logits match
repeated `step`, and ablation transport/feature modes preserve equivalence.

### Experiment 003 - 2026-04-24 - Ablation Registry

**Status:** Implemented.

**Purpose:** Make immediate-neighbor GLA ablations reproducible by name.

**Scope:** `src/metanoos/ablations.py`, `tests/test_model.py`

| Preset | Isolation |
| --- | --- |
| `complex_rotary_gla` | default |
| `complex_decay_gla` | phase rotation |
| `complex_linear_attention` | temporal decay |
| `magnitude_feature_gla` | q/k phase features |
| `real_readout_gla` | Born readout |

**Command:** `python3 -m pytest tests/test_model.py -k ablation`

### Experiment 004 - 2026-04-24 - Readout And Optimizer Safety

**Status:** Implemented.

**Purpose:** Verify real logits/loss, Born global-phase invariance, AdamW
compatibility with complex gradients, and CUDA forward/backward when available.

**Scope:** `src/metanoos/model.py`, `tests/test_model.py`

**Command:**
`python3 -m pytest tests/test_model.py -k "readout or adamw or cuda or outputs"`

### Experiment 005 - 2026-04-24 - Training Device Suitability

**Status:** Recorded.

**Purpose:** Decide which accelerator backend is valid for real ablation
training.

**Probe:** `torch.cuda.is_available()`, `torch.cuda.device_count()`,
`torch.backends.mps.is_available()`, plus complex backward on MPS.

**Result:** Prior local machine had MPS but no CUDA. MPS forward completed, but
backward failed in PyTorch's complex tensor path.

**Conclusion:** Real ablation runs require CUDA. CPU is for smoke/correctness
only.

### Experiment 006 - 2026-04-24 - Byte-Level Runner Smoke

**Status:** Implemented smoke check. Not architectural evidence.

**Purpose:** Verify `scripts/run_ablation.py` can train, evaluate, log metrics,
write summaries, generate samples, and report CUDA memory.

**Command:**

```bash
python3 scripts/run_ablation.py \
  --device cuda \
  --require-cuda \
  --corpus greek_classics \
  --preset complex_rotary_gla \
  --seed 0 \
  --d-model 16 \
  --num-layers 1 \
  --num-heads 2 \
  --seq-len 16 \
  --batch-size 2 \
  --steps 2 \
  --val-batches 1 \
  --eval-interval 1 \
  --log-interval 1 \
  --max-tokens 2000 \
  --sample-tokens 8 \
  --output-dir /tmp/metanoos-runs
```

**Result:** `best_val_loss=6.867833614349365`,
`best_val_bpb=9.908189497072355`, `final_train_loss=6.860815525054932`,
`peak_cuda_memory_bytes=17954304`, `real_parameter_count=23190`.

## In Progress

### Experiment 007 - 2026-04-24 - CUDA Ablation Sweep

**Status:** Seed-0 500-step sweep complete. Longer runs and repeated seeds are
pending before making architectural claims.

**Purpose:** Compare default complex rotational GLA against immediate neighbors
under matched training conditions.

**Fixed settings:** `greek_classics` first, UTF-8 bytes, shared seed/model
scale/optimizer/batch shape/validation budget/logging schema.

| Arm | Preset | Comparison |
| --- | --- | --- |
| A | `complex_rotary_gla` | default |
| B | `complex_decay_gla` | A vs B isolates phase rotation |
| C | `complex_linear_attention` | A/B vs C isolates temporal decay |
| D | `magnitude_feature_gla` | A vs D isolates q/k phase features |
| E | `real_readout_gla` | A vs E isolates Born readout |

**Seed-0 500-step result:**

| Preset | Best val loss | Best val bpb | Final train loss | Peak CUDA memory |
| --- | ---: | ---: | ---: | ---: |
| `real_readout_gla` | 2.2738 | 3.2804 | 2.2835 | 12.83 GB |
| `magnitude_feature_gla` | 2.3810 | 3.4351 | 2.4030 | 12.63 GB |
| `complex_rotary_gla` | 2.3907 | 3.4490 | 2.4364 | 12.83 GB |
| `complex_decay_gla` | 2.4165 | 3.4863 | 2.4520 | 12.83 GB |
| `complex_linear_attention` | 2.4805 | 3.5787 | 2.5226 | 3.41 GB |

**Run dirs:** `runs/20260424T192218Z-greek_classics-complex_rotary_gla-seed0`,
`runs/20260424T193451Z-greek_classics-complex_decay_gla-seed0`,
`runs/20260424T194728Z-greek_classics-complex_linear_attention-seed0`,
`runs/20260424T195701Z-greek_classics-magnitude_feature_gla-seed0`,
`runs/20260424T201027Z-greek_classics-real_readout_gla-seed0`.

**Metrics:** validation loss/bpb, train loss curve, tokens/sec, gradient norm,
peak CUDA memory, fixed-prompt samples.

**Rule:** Treat one seed as smoke evidence only; require repeated seeds before
claiming ablation effects.

### Experiment 008 - 2026-04-24 - External Baselines

**Status:** TODO.

**Purpose:** Add comparison models outside this implementation.

**Baselines:** matched real-valued GLA, real-valued causal linear attention
without decay, diagonal real SSM, and diagonal complex SSM or S4D-style model if
available.

**Rule:** Use the same split, tokenizer, optimizer budget, logging schema, and
metrics as Experiment 007.

### Experiment 009 - 2026-04-24 - 50M Layout Sweep

**Status:** TODO. Requires Experiment 007 first.

**Purpose:** Compare depth, width, key/value state size, and MLP expansion near
50M real-valued parameters.

**Budget:** complex params count as two real params; assume tied Born
embedding/readout; initial vocabulary assumption is 8k subwords; `key_dim` and
`value_dim` are per-head constructor values.

| Layout | `d_model` | Layers | `d_k` | `d_v` | MLP | Total approx. | Hypothesis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 512 | 6 | 512 | 512 | 4 | 47.7M | balanced baseline |
| B | 768 | 3 | 768 | 768 | 4 | 44.0M | width vs depth |
| C | 384 | 12 | 384 | 384 | 4 | 50.3M | many rotational layers |
| D | 512 | 6 | 256 | 512 | 4 | 38.5M | bottleneck key state |
| E | 512 | 5 | 512 | 512 | 6 | 49.3M | wider local MLP |

**Order:** A first, then B vs C, then D if memory-constrained, then E if local
compute looks limiting.

**Metrics:** validation loss/bpc, train stability, tokens/sec, peak CUDA memory,
real parameter count, fixed-prompt samples.

**Rule:** Do not compare layouts across different vocabularies/tokenizers
without recomputing budgets.
