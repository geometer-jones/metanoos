#!/usr/bin/env python3
"""Run byte-level language-model ablation experiments."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from metanoos import (  # noqa: E402
    ReciprocationLanguageModel,
    ablation_model_kwargs,
    available_corpora,
    read_corpus_text,
    state_memory_estimate,
)


ABLATION_ORDER = [
    "complex_rotary_gla",
    "complex_decay_gla",
    "complex_linear_attention",
    "complex_rope_decay_gla",
    "complex_rope_rotary_gla",
    "magnitude_feature_gla",
    "real_readout_gla",
]
DEFAULT_PROMPTS = ["The ", "Socrates ", "(define "]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=[corpus.name for corpus in available_corpora()], default="greek_classics")
    parser.add_argument("--preset", choices=ABLATION_ORDER, default="complex_rotary_gla")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None, help="Checkpoint file or run directory to resume.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--key-dim", type=int, default=None, help="Per-head key dimension.")
    parser.add_argument("--value-dim", type=int, default=None, help="Per-head value dimension.")
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--tie-readout-carrier", action="store_true")

    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Save periodic checkpoints; 0 disables periodic saves.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Debug option: truncate corpus bytes before split.")

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Disable clipping with 0.")
    parser.add_argument("--no-tf32", action="store_true", help="Disable TF32 matmul on CUDA.")

    parser.add_argument("--sample-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--prompt", action="append", default=None)

    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    positive_int_fields = [
        "d_model",
        "num_layers",
        "num_heads",
        "steps",
        "batch_size",
        "seq_len",
        "log_interval",
        "eval_interval",
        "mlp_ratio",
    ]
    for field in positive_int_fields:
        if getattr(args, field) <= 0:
            raise SystemExit(f"--{field.replace('_', '-')} must be positive")
    for field in ("head_dim", "key_dim", "value_dim", "max_tokens"):
        value = getattr(args, field)
        if value is not None and value <= 0:
            raise SystemExit(f"--{field.replace('_', '-')} must be positive when provided")
    if args.val_batches < 0:
        raise SystemExit("--val-batches must be nonnegative")
    if args.checkpoint_interval < 0:
        raise SystemExit("--checkpoint-interval must be nonnegative")
    if not 0.0 < args.val_fraction < 1.0:
        raise SystemExit("--val-fraction must be between 0 and 1")
    if args.learning_rate <= 0.0:
        raise SystemExit("--learning-rate must be positive")
    if not 0.0 <= args.beta1 < 1.0 or not 0.0 <= args.beta2 < 1.0:
        raise SystemExit("--beta1 and --beta2 must be in [0, 1)")
    if args.weight_decay < 0.0:
        raise SystemExit("--weight-decay must be nonnegative")
    if args.grad_clip < 0.0:
        raise SystemExit("--grad-clip must be nonnegative")
    if args.temperature <= 0.0:
        raise SystemExit("--temperature must be positive")
    if args.sample_tokens < 0:
        raise SystemExit("--sample-tokens must be nonnegative")


def resolve_device(args: argparse.Namespace) -> torch.device:
    cuda_available = torch.cuda.is_available()
    if args.require_cuda and not cuda_available:
        raise SystemExit("CUDA was required, but torch.cuda.is_available() is false.")
    if args.device == "cuda":
        if not cuda_available:
            raise SystemExit("CUDA device was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if args.device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if cuda_available else "cpu")


def encode_corpus_bytes(corpus: str, max_tokens: int | None) -> torch.Tensor:
    payload = read_corpus_text(corpus).encode("utf-8")
    if max_tokens is not None:
        payload = payload[:max_tokens]
    tokens = torch.frombuffer(bytearray(payload), dtype=torch.uint8).long()
    if tokens.numel() < 4:
        raise SystemExit(f"Corpus {corpus!r} is too small after byte encoding.")
    return tokens


def split_tokens(tokens: torch.Tensor, val_fraction: float, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    split_index = int(tokens.numel() * (1.0 - val_fraction))
    split_index = max(split_index, seq_len + 2)
    split_index = min(split_index, tokens.numel() - seq_len - 2)
    train = tokens[:split_index].contiguous()
    val = tokens[split_index:].contiguous()
    if train.numel() <= seq_len + 1 or val.numel() <= seq_len + 1:
        raise SystemExit(
            "Corpus split is too small for the requested sequence length; "
            "lower --seq-len, lower --val-fraction, or increase --max-tokens."
        )
    return train, val


def sample_batch(
    data: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, data.numel() - seq_len - 1, (batch_size,), generator=generator)
    offsets = starts[:, None] + torch.arange(seq_len + 1)
    batch = data[offsets].to(device=device, non_blocking=True)
    return batch[:, :-1], batch[:, 1:]


def parameter_grad_norm(model: torch.nn.Module) -> float:
    total = None
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().norm(2).square()
        total = value if total is None else total + value
    if total is None:
        return 0.0
    return float(total.sqrt().item())


@torch.no_grad()
def evaluate(
    model: ReciprocationLanguageModel,
    data: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    val_batches: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[float | None, float | None]:
    if val_batches == 0:
        return None, None
    was_training = model.training
    model.eval()
    total_loss = 0.0
    for _ in range(val_batches):
        inputs, labels = sample_batch(data, batch_size=batch_size, seq_len=seq_len, device=device, generator=generator)
        loss = model(inputs, labels=labels).loss
        if loss is None:
            raise RuntimeError("model returned no validation loss")
        total_loss += float(loss.item())
    if was_training:
        model.train()
    val_loss = total_loss / val_batches
    return val_loss, val_loss / math.log(2.0)


@torch.no_grad()
def generate_sample(
    model: ReciprocationLanguageModel,
    *,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> str:
    model.eval()
    prompt_bytes = list(prompt.encode("utf-8")) or [10]
    memories = None
    logits = None
    position = 0
    for token in prompt_bytes:
        token_tensor = torch.tensor([token], device=device)
        logits, memories = model.step(token_tensor, memories, position=position)
        position += 1
    generated = list(prompt_bytes)
    assert logits is not None

    for _ in range(max_new_tokens):
        next_logits = torch.nan_to_num(logits[0] / temperature, nan=0.0, posinf=1e4, neginf=-1e4)
        probs = torch.softmax(next_logits, dim=-1)
        next_token = int(torch.multinomial(probs, 1).item())
        generated.append(next_token)
        logits, memories = model.step(torch.tensor([next_token], device=device), memories, position=position)
        position += 1

    return bytes(generated).decode("utf-8", errors="replace")


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(handle: Any, data: Any) -> None:
    handle.write(json.dumps(data, sort_keys=True) + "\n")
    handle.flush()


def make_run_dir(args: argparse.Namespace) -> Path:
    if args.resume_from is not None:
        if args.run_dir is not None:
            return args.run_dir
        resume_path = args.resume_from
        if resume_path.is_dir():
            return resume_path
        return resume_path.parent.parent if resume_path.parent.name == "checkpoints" else resume_path.parent

    if args.run_dir is not None:
        run_dir = args.run_dir
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = args.output_dir / f"{stamp}-{args.corpus}-{args.preset}-seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_model(args: argparse.Namespace, device: torch.device) -> ReciprocationLanguageModel:
    kwargs = ablation_model_kwargs(args.preset)
    model = ReciprocationLanguageModel(
        vocab_size=256,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        mlp_ratio=args.mlp_ratio,
        tie_readout_carrier=args.tie_readout_carrier,
        **kwargs,
    )
    return model.to(device)


def checkpoint_path_from_resume(path: Path) -> Path:
    if path.is_dir():
        return path / "checkpoints" / "last.pt"
    return path


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint_path = checkpoint_path_from_resume(path)
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint does not exist: {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def save_checkpoint(
    run_dir: Path,
    *,
    step: int,
    args: argparse.Namespace,
    model: ReciprocationLanguageModel,
    optimizer: torch.optim.Optimizer,
    train_generator: torch.Generator,
    val_generator: torch.Generator,
    best_val_loss: float | None,
    last_train_loss: float | None,
    last_grad_norm: float | None,
    device: torch.device,
    final: bool = False,
) -> Path:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_generator_state": train_generator.get_state(),
        "val_generator_state": val_generator.get_state(),
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        "best_val_loss": best_val_loss,
        "last_train_loss": last_train_loss,
        "last_grad_norm": last_grad_norm,
    }

    step_path = checkpoint_dir / f"step_{step:08d}.pt"
    torch.save(checkpoint, step_path)
    torch.save(checkpoint, checkpoint_dir / "last.pt")
    if final:
        torch.save(checkpoint, checkpoint_dir / "final.pt")
    return step_path


def last_metric_elapsed_seconds(metrics_path: Path) -> float:
    if not metrics_path.is_file():
        return 0.0
    last_line = ""
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last_line = line
    if not last_line:
        return 0.0
    try:
        metric = json.loads(last_line)
    except json.JSONDecodeError:
        return 0.0
    elapsed = metric.get("elapsed_seconds", 0.0)
    return float(elapsed) if isinstance(elapsed, (int, float)) else 0.0


def metadata_for_run(
    args: argparse.Namespace,
    *,
    model: ReciprocationLanguageModel,
    train_tokens: int,
    val_tokens: int,
    device: torch.device,
    run_dir: Path,
) -> dict[str, Any]:
    first_mixer = model.blocks[0].mix
    estimate = state_memory_estimate(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_heads=first_mixer.num_heads,
        value_dim=first_mixer.value_dim,
        key_dim=first_mixer.key_dim,
        num_layers=args.num_layers,
    )
    cuda_info = None
    if device.type == "cuda":
        cuda_info = {
            "device_name": torch.cuda.get_device_name(device),
            "device_count": torch.cuda.device_count(),
            "torch_cuda_version": torch.version.cuda,
        }

    return {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "ablation_kwargs": ablation_model_kwargs(args.preset),
        "ablation_order": ABLATION_ORDER,
        "corpus": {
            "name": args.corpus,
            "tokenization": "utf-8 bytes",
            "vocab_size": 256,
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
        },
        "device": str(device),
        "cuda": cuda_info,
        "model": {
            "real_parameter_count": model.real_param_count(),
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": first_mixer.num_heads,
            "key_dim_per_head": first_mixer.key_dim,
            "value_dim_per_head": first_mixer.value_dim,
            "position_encoding": first_mixer.position_encoding,
            "rotary_base": first_mixer.rotary_base,
            "mlp_ratio": args.mlp_ratio,
            "tie_readout_carrier": args.tie_readout_carrier,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "betas": [args.beta1, args.beta2],
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
        },
        "run_dir": str(run_dir),
        "state_memory_estimate": {
            **estimate.as_dict(),
            "total_gib": estimate.total_gib,
            "note": "Prefix tensors only; excludes autograd saves, temporaries, params, optimizer state, and fragmentation.",
        },
        "torch": {
            "version": torch.__version__,
            "tf32_allowed": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else None,
        },
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32

    device = resolve_device(args)
    tokens = encode_corpus_bytes(args.corpus, args.max_tokens)
    train_data, val_data = split_tokens(tokens, args.val_fraction, args.seq_len)
    train_generator = torch.Generator().manual_seed(args.seed)
    val_generator = torch.Generator().manual_seed(args.seed + 1)

    run_dir = make_run_dir(args)
    model = build_model(args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    start_step = 0
    best_val_loss = None
    last_train_loss = None
    last_grad_norm = None
    if args.resume_from is not None:
        checkpoint = load_checkpoint(args.resume_from, device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        train_generator.set_state(checkpoint["train_generator_state"])
        val_generator.set_state(checkpoint["val_generator_state"])
        random.setstate(checkpoint["python_random_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        start_step = int(checkpoint["step"])
        best_val_loss = checkpoint.get("best_val_loss")
        last_train_loss = checkpoint.get("last_train_loss")
        last_grad_norm = checkpoint.get("last_grad_norm")
        if args.steps <= start_step:
            raise SystemExit(f"--steps must be greater than resumed step {start_step}")

    metadata = metadata_for_run(
        args,
        model=model,
        train_tokens=train_data.numel(),
        val_tokens=val_data.numel(),
        device=device,
        run_dir=run_dir,
    )
    if args.resume_from is None:
        write_json(run_dir / "metadata.json", metadata)
    else:
        write_json(run_dir / f"resume_from_step_{start_step:08d}.json", metadata)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    last_log_time = time.perf_counter()
    last_log_step = start_step
    metrics_path = run_dir / "metrics.jsonl"
    started = time.perf_counter() - last_metric_elapsed_seconds(metrics_path)

    model.train()
    metrics_mode = "a" if args.resume_from is not None else "w"
    with metrics_path.open(metrics_mode, encoding="utf-8") as metrics_handle:
        for step in range(start_step + 1, args.steps + 1):
            inputs, labels = sample_batch(
                train_data,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                device=device,
                generator=train_generator,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model(inputs, labels=labels).loss
            if loss is None:
                raise RuntimeError("model returned no training loss")
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {float(loss.item())}")
            loss.backward()
            grad_norm = parameter_grad_norm(model)
            if not math.isfinite(grad_norm):
                raise RuntimeError(f"non-finite gradient norm at step {step}: {grad_norm}")
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            last_train_loss = float(loss.item())
            last_grad_norm = grad_norm

            should_eval = args.val_batches > 0 and (step == 1 or step % args.eval_interval == 0 or step == args.steps)
            should_log = step == 1 or step % args.log_interval == 0 or step == args.steps or should_eval
            val_loss = None
            val_bpb = None
            if should_eval:
                val_loss, val_bpb = evaluate(
                    model,
                    val_data,
                    batch_size=args.batch_size,
                    seq_len=args.seq_len,
                    val_batches=args.val_batches,
                    device=device,
                    generator=val_generator,
                )
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss

            if should_log:
                now = time.perf_counter()
                interval_tokens = (step - last_log_step) * args.batch_size * args.seq_len
                interval_seconds = max(now - last_log_time, 1e-12)
                peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                write_jsonl(
                    metrics_handle,
                    {
                        "step": step,
                        "elapsed_seconds": now - started,
                        "train_loss": last_train_loss,
                        "train_bpb": last_train_loss / math.log(2.0),
                        "val_loss": val_loss,
                        "val_bpb": val_bpb,
                        "best_val_loss": best_val_loss,
                        "grad_norm": last_grad_norm,
                        "tokens_per_second": interval_tokens / interval_seconds,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "peak_cuda_memory_bytes": peak_memory,
                    },
                )
                last_log_time = now
                last_log_step = step
            if args.checkpoint_interval > 0 and step % args.checkpoint_interval == 0:
                save_checkpoint(
                    run_dir,
                    step=step,
                    args=args,
                    model=model,
                    optimizer=optimizer,
                    train_generator=train_generator,
                    val_generator=val_generator,
                    best_val_loss=best_val_loss,
                    last_train_loss=last_train_loss,
                    last_grad_norm=last_grad_norm,
                    device=device,
                )

    save_checkpoint(
        run_dir,
        step=args.steps,
        args=args,
        model=model,
        optimizer=optimizer,
        train_generator=train_generator,
        val_generator=val_generator,
        best_val_loss=best_val_loss,
        last_train_loss=last_train_loss,
        last_grad_norm=last_grad_norm,
        device=device,
        final=True,
    )

    prompts = args.prompt if args.prompt is not None else DEFAULT_PROMPTS
    samples = []
    if args.sample_tokens > 0:
        for prompt in prompts:
            samples.append(
                {
                    "prompt": prompt,
                    "text": generate_sample(
                        model,
                        prompt=prompt,
                        max_new_tokens=args.sample_tokens,
                        temperature=args.temperature,
                        device=device,
                    ),
                }
            )
        write_json(run_dir / "samples.json", samples)

    summary = {
        "run_dir": str(run_dir),
        "preset": args.preset,
        "corpus": args.corpus,
        "seed": args.seed,
        "steps": args.steps,
        "final_train_loss": last_train_loss,
        "final_train_bpb": None if last_train_loss is None else last_train_loss / math.log(2.0),
        "best_val_loss": best_val_loss,
        "best_val_bpb": None if best_val_loss is None else best_val_loss / math.log(2.0),
        "final_grad_norm": last_grad_norm,
        "real_parameter_count": model.real_param_count(),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "samples_written": bool(samples),
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
