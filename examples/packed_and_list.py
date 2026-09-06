"""Exercise packed, list, and tuple CUDA BF16 source containers."""

from __future__ import annotations

import argparse

import torch

from attnres import attnres


def _assert_same(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)


def run(device: str | torch.device | None = None) -> None:
    """Check equivalent standard and sliced CUDA BF16 calls."""

    torch.manual_seed(2)
    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this example requires an available CUDA device")
    dtype = torch.bfloat16
    source_count, batch, width = 5, 2, 24
    packed_values = torch.randn(source_count, batch, width, device=device, dtype=dtype)
    value_list = list(packed_values.unbind(0))
    value_tuple = tuple(value_list)

    standard_query = torch.randn(width, device=device, dtype=dtype)
    standard_packed = attnres(packed_values, standard_query)
    _assert_same(attnres(value_list, standard_query), standard_packed)
    _assert_same(attnres(value_tuple, standard_query), standard_packed)

    rank = 6
    sliced_query = torch.randn(rank, device=device, dtype=dtype)
    sliced_packed = attnres(packed_values, sliced_query)
    _assert_same(attnres(value_list, sliced_query), sliced_packed)
    _assert_same(attnres(value_tuple, sliced_query), sliced_packed)
    if standard_packed.dtype != dtype or sliced_packed.dtype != dtype:
        raise AssertionError("AttnRes must return BF16 output")

    print(
        "source containers: "
        f"standard_shape={tuple(standard_packed.shape)} "
        f"sliced_shape={tuple(sliced_packed.shape)} dtype={dtype}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(args.device)


if __name__ == "__main__":
    main()
