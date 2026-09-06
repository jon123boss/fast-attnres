"""Build Full and Block source sets through one CUDA BF16 primitive."""

from __future__ import annotations

import argparse

import torch

from attnres import attnres


def run(device: str | torch.device | None = None) -> None:
    """Check ordinary embedding, writer, block, and partial source assembly."""

    torch.manual_seed(5)
    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this example requires an available CUDA device")
    dtype = torch.bfloat16
    writer_count, batch, width = 4, 2, 16
    embedding = torch.randn(batch, width, device=device, dtype=dtype)
    writers = tuple(
        torch.randn(batch, width, device=device, dtype=dtype)
        for _ in range(writer_count)
    )
    query = torch.randn(width, device=device, dtype=dtype)

    # Full keeps the embedding and every writer as ordinary sources.
    full_sources = (embedding, *writers)
    full_output = attnres(full_sources, query)

    # Block sums are assembled by the caller. Each read supplies the embedding,
    # completed block sums, and the current partial block as ordinary sources.
    block_size = 2
    completed = [embedding]
    partial_writers: list[torch.Tensor] = []
    block_outputs: list[torch.Tensor] = []
    for writer in writers:
        partial_writers.append(writer)
        partial = torch.stack(tuple(partial_writers), dim=0).sum(dim=0)
        read_sources = tuple(completed) + (partial,)
        block_outputs.append(attnres(read_sources, query))
        if len(partial_writers) == block_size:
            completed.append(partial)
            partial_writers.clear()

    block_output = block_outputs[-1]
    if block_output.dtype != dtype or full_output.dtype != dtype:
        raise AssertionError("AttnRes must return BF16 output")
    if len(completed) != 1 + writer_count // block_size:
        raise AssertionError("unexpected completed block count")

    print(
        "Fast-AttnRes Block schedules: "
        f"full_shape={tuple(full_output.shape)} block_shape={tuple(block_output.shape)} "
        f"reads={len(block_outputs)} completed_sources={len(completed)} "
        f"dtype={full_output.dtype}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(args.device)


if __name__ == "__main__":
    main()
