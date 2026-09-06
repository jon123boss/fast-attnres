"""Run a sliced low-rank Attention Residuals call on CUDA BF16."""

from __future__ import annotations

import argparse

import torch

from attnres import LearnedQuery, attnres


def run(device: str | torch.device | None = None) -> torch.Tensor:
    """Return a sliced LR-AttnRes output with a BF16 static learned query."""

    torch.manual_seed(1)
    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this example requires an available CUDA device")
    dtype = torch.bfloat16
    source_count, batch, value_width, rank = 4, 2, 32, 8
    sources = tuple(
        torch.randn(batch, value_width, device=device, dtype=dtype)
        for _ in range(source_count)
    )
    learned_query = LearnedQuery(rank).to(device=device, dtype=dtype)
    output = attnres(sources, learned_query())
    expected_shape = (batch, value_width)
    if output.shape != expected_shape:
        raise AssertionError(f"unexpected output shape: {tuple(output.shape)}")
    if learned_query.query.shape != (rank,):
        raise AssertionError("LearnedQuery has an unexpected shape")
    if output.dtype != dtype or learned_query.query.dtype != dtype:
        raise AssertionError("values, query, and output must be BF16")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    output = run(args.device)
    print(f"sliced LR-AttnRes: shape={tuple(output.shape)} dtype={output.dtype}")


if __name__ == "__main__":
    main()
