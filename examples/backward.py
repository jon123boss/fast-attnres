"""Check first-order value and query gradients for Fast-AttnRes."""

from __future__ import annotations

import argparse

import torch

from attnres import attnres


def _check_finite(name: str, gradient: torch.Tensor | None) -> None:
    if gradient is None or not torch.isfinite(gradient).all():
        raise AssertionError(f"missing or non-finite {name} gradient")


def run(device: str | torch.device | None = None) -> None:
    """Run standard and sliced backward passes through the public operator."""

    torch.manual_seed(3)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    source_count, batch, width = 4, 2, 24

    standard_sources = [
        torch.randn(batch, width, device=device, dtype=dtype, requires_grad=True)
        for _ in range(source_count)
    ]
    standard_query = torch.randn(width, device=device, dtype=torch.float32, requires_grad=True)
    standard_loss = attnres(standard_sources, standard_query).float().square().mean()
    standard_gradients = torch.autograd.grad(standard_loss, (*standard_sources, standard_query))
    for index, gradient in enumerate(standard_gradients[:-1]):
        _check_finite(f"standard source {index}", gradient)
    _check_finite("standard query", standard_gradients[-1])

    sliced_values = torch.randn(
        source_count,
        batch,
        width,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    rank = 6
    sliced_query = torch.randn(rank, device=device, dtype=torch.float32, requires_grad=True)
    sliced_loss = attnres(sliced_values, sliced_query).float().square().mean()
    sliced_values_gradient, sliced_query_gradient = torch.autograd.grad(
        sliced_loss,
        (sliced_values, sliced_query),
    )
    _check_finite("sliced values", sliced_values_gradient)
    _check_finite("sliced query", sliced_query_gradient)

    print(
        "backward: "
        f"standard_query_norm={standard_gradients[-1].norm().item():.4f} "
        f"sliced_values_norm={sliced_values_gradient.norm().item():.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(args.device)


if __name__ == "__main__":
    main()
