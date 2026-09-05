"""Run a CUDA BF16 standard Full Attention Residuals call."""

from __future__ import annotations

import argparse

import torch

from attnres import attnres


def run(device: str | torch.device | None = None) -> torch.Tensor:
    """Compare packed and source-list standard AttnRes calls on CUDA BF16."""

    torch.manual_seed(0)
    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this example requires an available CUDA device")
    dtype = torch.bfloat16
    source_count, batch, width = 4, 2, 32
    values = torch.randn(source_count, batch, width, device=device, dtype=dtype)
    query = torch.randn(width, device=device, dtype=dtype)

    packed_output = attnres(values, query)
    source_output = attnres(tuple(values.unbind(0)), query)
    torch.testing.assert_close(source_output, packed_output, rtol=0.05, atol=0.05)
    expected_shape = (batch, width)
    if packed_output.shape != expected_shape:
        raise AssertionError(f"unexpected output shape: {tuple(packed_output.shape)}")
    if packed_output.dtype != dtype or source_output.dtype != dtype:
        raise AssertionError("AttnRes must return BF16 output")
    return packed_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    output = run(args.device)
    print(f"Fast-AttnRes standard AttnRes: shape={tuple(output.shape)} dtype={output.dtype}")


if __name__ == "__main__":
    main()
