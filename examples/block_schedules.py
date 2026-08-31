"""Compare Full and per-read Block schedules through one public primitive."""

from __future__ import annotations

import argparse

import torch

from attnres import attnres


def run(device: str | torch.device | None = None) -> None:
    """Check the same source schedule in Full and per-read forms."""

    torch.manual_seed(5)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    source_count, batch, width = 5, 2, 16
    sources = tuple(
        torch.randn(batch, width, device=device, dtype=dtype)
        for _ in range(source_count)
    )
    query = torch.randn(width, device=device, dtype=torch.float32)

    # Full evaluates the complete source set at the final read.
    full_output = attnres(sources, query)

    # A per-read Block schedule grows the completed source set one source at a
    # time. A real model may make each ``partial`` by summing sublayer outputs;
    # the public operator sees the same ordinary source container either way.
    checked_reads = 0
    for read in range(1, source_count):
        completed = sources[:read]
        partial = sources[read]
        read_sources = completed + (partial,)
        per_read_output = attnres(read_sources, query)
        if read == source_count - 1:
            tolerance = {"rtol": 0.05, "atol": 0.05} if dtype == torch.bfloat16 else {
                "rtol": 0.001,
                "atol": 0.0001,
            }
            torch.testing.assert_close(per_read_output, full_output, **tolerance)
        checked_reads += 1

    print(
        "Fast-AttnRes Block schedules: "
        f"full_shape={tuple(full_output.shape)} checked_reads={checked_reads} "
        f"dtype={full_output.dtype}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(args.device)


if __name__ == "__main__":
    main()
