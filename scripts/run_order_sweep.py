#!/usr/bin/env python3
"""Run small synthetic order-sensitivity ablations."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from metanoos import ComposedStateLanguageModel, ablation_model_kwargs  # noqa: E402


DEFAULT_PRESETS = [
    "complex_linear_attention",
    "complex_rotary_gla",
    "complex_rope_decay_gla",
    "complex_rope_rotary_gla",
]

PAD = 2
QUERY = 3
A_TOKEN = 4
B_TOKEN = 5
MIN_VOCAB_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["permutation", "span_copy"], default="permutation")
    parser.add_argument("--preset", action="append", choices=DEFAULT_PRESETS, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=100)

    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--key-dim", type=int, default=None)
    parser.add_argument("--value-dim", type=int, default=None)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    for field in ("vocab_size", "seq_len", "steps", "batch_size", "log_interval", "d_model", "num_layers", "num_heads"):
        if getattr(args, field) <= 0:
            raise SystemExit(f"--{field.replace('_', '-')} must be positive")
    if args.vocab_size < MIN_VOCAB_SIZE:
        raise SystemExit(f"--vocab-size must be at least {MIN_VOCAB_SIZE}")
    if args.seq_len < 8:
        raise SystemExit("--seq-len must be at least 8")
    if args.eval_batches < 0:
        raise SystemExit("--eval-batches must be nonnegative")
    if args.learning_rate <= 0.0:
        raise SystemExit("--learning-rate must be positive")
    if args.weight_decay < 0.0:
        raise SystemExit("--weight-decay must be nonnegative")
    if args.grad_clip < 0.0:
        raise SystemExit("--grad-clip must be nonnegative")
    for field in ("key_dim", "value_dim"):
        value = getattr(args, field)
        if value is not None and value <= 0:
            raise SystemExit(f"--{field.replace('_', '-')} must be positive when provided")


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def permutation_batch(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randint(6, vocab_size, (batch_size, seq_len), generator=generator, device=device)
    inputs[:, -1] = QUERY
    labels = torch.randint(0, 2, (batch_size,), generator=generator, device=device)
    first = torch.randint(0, seq_len - 2, (batch_size,), generator=generator, device=device)
    gap = torch.randint(1, seq_len - 1, (batch_size,), generator=generator, device=device)
    second = (first + gap).clamp_max(seq_len - 2)
    first = torch.minimum(first, second - 1)

    rows = torch.arange(batch_size, device=device)
    inputs[rows, first] = torch.where(labels == 0, A_TOKEN, B_TOKEN)
    inputs[rows, second] = torch.where(labels == 0, B_TOKEN, A_TOKEN)
    return inputs, labels


def span_copy_batch(
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.full((batch_size, seq_len), PAD, device=device, dtype=torch.long)
    inputs[:, -1] = QUERY
    cue_left = torch.randint(0, max(1, seq_len // 3), (batch_size,), generator=generator, device=device)
    cue_right = torch.randint(seq_len // 2, seq_len - 3, (batch_size,), generator=generator, device=device)
    inside = cue_left + 1 + torch.randint(0, seq_len, (batch_size,), generator=generator, device=device) % (cue_right - cue_left - 1)
    outside = cue_right + 1 + torch.randint(0, seq_len, (batch_size,), generator=generator, device=device) % (seq_len - cue_right - 2)

    left_candidate = torch.randint(6, vocab_size, (batch_size,), generator=generator, device=device)
    offset = torch.randint(1, vocab_size - 6, (batch_size,), generator=generator, device=device)
    right_candidate = 6 + ((left_candidate - 6 + offset) % (vocab_size - 6))
    choose_left = torch.randint(0, 2, (batch_size,), generator=generator, device=device).bool()
    labels = torch.where(choose_left, left_candidate, right_candidate)
    distractor = torch.where(choose_left, right_candidate, left_candidate)

    rows = torch.arange(batch_size, device=device)
    inputs[rows, cue_left] = A_TOKEN
    inputs[rows, cue_right] = A_TOKEN
    inputs[rows, inside] = labels
    inputs[rows, outside] = distractor
    return inputs, labels


def sample_batch(args: argparse.Namespace, device: torch.device, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    if args.task == "permutation":
        return permutation_batch(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            device=device,
            generator=generator,
        )
    return span_copy_batch(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        device=device,
        generator=generator,
    )


def build_model(args: argparse.Namespace, preset: str, device: torch.device) -> ComposedStateLanguageModel:
    return ComposedStateLanguageModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        mlp_ratio=args.mlp_ratio,
        **ablation_model_kwargs(preset),
    ).to(device)


def transport_snapshot(model: ComposedStateLanguageModel) -> list[dict[str, Any]]:
    snapshots = []
    for layer_index, block in enumerate(model.blocks):
        transport = block.mix.transport
        if transport.log_decay_s is None or transport.log_decay_z is None:
            snapshots.append({"layer": layer_index, "phase": None, "radius_s": None, "alpha_z": None})
            continue
        radius_s = torch.exp(-F.softplus(transport.log_decay_s.detach())).cpu()
        alpha_z = torch.exp(-F.softplus(transport.log_decay_z.detach())).cpu()
        phase = None if transport.phase is None else transport.phase.detach().cpu()
        snapshots.append(
            {
                "layer": layer_index,
                "phase": None if phase is None else phase.tolist(),
                "radius_s": radius_s.tolist(),
                "alpha_z": alpha_z.tolist(),
            }
        )
    return snapshots


@torch.no_grad()
def evaluate(
    model: ComposedStateLanguageModel,
    args: argparse.Namespace,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[float | None, float | None]:
    if args.eval_batches == 0:
        return None, None
    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for _ in range(args.eval_batches):
        inputs, labels = sample_batch(args, device, generator)
        logits = model(inputs).logits[:, -1]
        loss = F.cross_entropy(logits, labels)
        total_loss += float(loss.item())
        correct += int((logits.argmax(dim=-1) == labels).sum().item())
        total += labels.numel()
    if was_training:
        model.train()
    return total_loss / args.eval_batches, correct / total


def train_preset(args: argparse.Namespace, preset: str, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = build_model(args, preset, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_generator = torch.Generator(device=device).manual_seed(args.seed)
    eval_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    logs = []

    model.train()
    for step in range(1, args.steps + 1):
        inputs, labels = sample_batch(args, device, train_generator)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs).logits[:, -1]
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{preset} produced non-finite loss at step {step}: {float(loss.item())}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) if args.grad_clip > 0 else None
        optimizer.step()

        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            eval_loss, eval_accuracy = evaluate(model, args, device=device, generator=eval_generator)
            entry = {
                "preset": preset,
                "step": step,
                "train_loss": float(loss.item()),
                "eval_loss": eval_loss,
                "eval_accuracy": eval_accuracy,
                "grad_norm": None if grad_norm is None else float(grad_norm.item()),
                "transport": transport_snapshot(model),
            }
            logs.append(entry)
            accuracy_text = "n/a" if eval_accuracy is None else f"{eval_accuracy:.3f}"
            print(f"{preset:26s} step={step:5d} loss={float(loss.item()):.4f} eval_acc={accuracy_text}")

    return {
        "preset": preset,
        "ablation_kwargs": ablation_model_kwargs(preset),
        "logs": logs,
        "final": logs[-1] if logs else None,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    presets = args.preset if args.preset is not None else DEFAULT_PRESETS
    results = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": str(device),
        "presets": [],
    }
    for preset in presets:
        results["presets"].append(train_preset(args, preset, device))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        summary = [{"preset": item["preset"], "final": item["final"]} for item in results["presets"]]
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
