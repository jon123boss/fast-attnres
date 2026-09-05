"""Run a tiny CUDA BF16 training smoke check for standard or sliced AttnRes."""

from __future__ import annotations

import argparse

import torch

from benchmarks.model import TrainingConfig, make_model, training_step


def _require_cuda(device: str | torch.device) -> torch.device:
    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this example requires an available CUDA device")
    return selected


def run_training(
    config: TrainingConfig | None = None,
    *,
    steps: int = 2,
    device: str | torch.device = "cuda",
    accumulation: int = 1,
) -> tuple[torch.nn.Module, list[float]]:
    """Train on random next-token data using the CUDA BF16 operator path."""

    if steps < 1:
        raise ValueError("steps must be positive")
    selected = _require_cuda(device)
    config = TrainingConfig() if config is None else config
    model = make_model(config, backend="kernel").to(
        device=selected,
        dtype=torch.bfloat16,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses: list[float] = []
    for _ in range(steps):
        tokens = torch.randint(
            config.vocab,
            (config.batch, config.sequence),
            device=selected,
        )
        targets = torch.roll(tokens, shifts=-1, dims=-1)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = training_step(
                model,
                optimizer,
                tokens,
                targets,
                accumulation=accumulation,
            )
        losses.append(float(loss))
    return model, losses


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn", type=int, default=384)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=64)
    parser.add_argument("--vocab", type=int, default=512)
    parser.add_argument("--block-count", type=int, default=2)
    parser.add_argument("--variant", choices=("standard", "sliced"), default="standard")
    parser.add_argument("--mode", choices=("full", "block"), default="full")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--source-layout", choices=("packed", "list"), default="packed")
    parser.add_argument("--accumulation", type=int, default=1)
    args = parser.parse_args()
    if args.variant == "sliced" and args.rank is None:
        parser.error("--rank is required when --variant=sliced")
    return args


def main() -> None:
    args = _parse_args()
    config = TrainingConfig(
        layers=args.layers,
        width=args.width,
        heads=args.heads,
        ffn=args.ffn,
        batch=args.batch,
        sequence=args.sequence,
        vocab=args.vocab,
        block_count=args.block_count,
        variant=args.variant,
        mode=args.mode,
        rank=args.rank,
        source_layout=args.source_layout,
    )
    _, losses = run_training(
        config,
        steps=args.steps,
        device=args.device,
        accumulation=args.accumulation,
    )
    print(" ".join(f"step={i + 1} loss={loss:.6f}" for i, loss in enumerate(losses)))


if __name__ == "__main__":
    main()
