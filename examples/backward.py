"""Check first-order CUDA BF16 value and query gradients."""

from __future__ import annotations

import argparse

import torch

from attnres import attnres


def _check_finite(name: str, gradient: torch.Tensor | None) -> None:
    if (
        gradient is None
        or gradient.dtype != torch.bfloat16
        or not torch.isfinite(gradient).all()
    ):
        raise AssertionError(f"missing, non-BF16, or non-finite {name} gradient")


def run(device: str | torch.device | None = None) -> None:
    """Run standard and sliced backward passes through the public operator."""

    torch.manual_seed(3)
    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this example requires an available CUDA device")
    dtype = torch.bfloat16
    source_count, batch, width = 4, 2, 24

    standard_sources = [
        torch.randn(batch, width, device=device, dtype=dtype, requires_grad=True)
        for _ in range(source_count)
    ]
    standard_query = torch.randn(width, device=device, dtype=dtype, requires_grad=True)
    standard_output = attnres(standard_sources, standard_query)
    standard_loss = standard_output.square().mean()
    if standard_output.dtype != dtype:
        raise AssertionError("standard output must be BF16")
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
    sliced_query = torch.randn(rank, device=device, dtype=dtype, requires_grad=True)
    sliced_output = attnres(sliced_values, sliced_query)
    sliced_loss = sliced_output.square().mean()
    if sliced_output.dtype != dtype:
        raise AssertionError("sliced output must be BF16")
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
