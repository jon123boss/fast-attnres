"""Run standard Full Attention Residuals from an installed Fast-AttnRes package."""

from __future__ import annotations

import argparse

import torch

from attnres import attnres


def run(device: str | torch.device | None = None) -> torch.Tensor:
    """Compare packed and source-list standard AttnRes calls."""

    torch.manual_seed(0)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    source_count, batch, width = 4, 2, 32
    values = torch.randn(source_count, batch, width, device=device, dtype=dtype)
    query = torch.randn(width, device=device, dtype=torch.float32)

    packed_output = attnres(values, query)
    source_output = attnres(tuple(values.unbind(0)), query)
    tolerance = {"rtol": 0.05, "atol": 0.05} if dtype == torch.bfloat16 else {
        "rtol": 0.001,
        "atol": 0.0001,
    }
    torch.testing.assert_close(source_output, packed_output, **tolerance)
    expected_shape = (batch, width)
    if packed_output.shape != expected_shape:
        raise AssertionError(f"unexpected output shape: {tuple(packed_output.shape)}")
    return packed_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    output = run(args.device)
    print(f"Fast-AttnRes standard AttnRes: shape={tuple(output.shape)} dtype={output.dtype}")


if __name__ == "__main__":
    main()
