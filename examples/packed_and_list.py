"""Exercise packed, list, and tuple source containers."""

from __future__ import annotations

import argparse

import torch

from attnres import attnres


def _assert_same(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> None:
    tolerance = {"rtol": 0.05, "atol": 0.05} if dtype == torch.bfloat16 else {
        "rtol": 0.001,
        "atol": 0.0001,
    }
    torch.testing.assert_close(actual, expected, **tolerance)


def run(device: str | torch.device | None = None) -> None:
    """Check equivalent standard and sliced calls for each container form."""

    torch.manual_seed(2)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    source_count, batch, width = 5, 2, 24
    packed_values = torch.randn(source_count, batch, width, device=device, dtype=dtype)
    value_list = list(packed_values.unbind(0))
    value_tuple = tuple(value_list)

    standard_query = torch.randn(width, device=device, dtype=torch.float32)
    standard_packed = attnres(packed_values, standard_query)
    _assert_same(attnres(value_list, standard_query), standard_packed, dtype)
    _assert_same(attnres(value_tuple, standard_query), standard_packed, dtype)

    rank = 6
    sliced_query = torch.randn(rank, device=device, dtype=torch.float32)
    sliced_packed = attnres(packed_values, sliced_query)
    _assert_same(attnres(value_list, sliced_query), sliced_packed, dtype)
    _assert_same(attnres(value_tuple, sliced_query), sliced_packed, dtype)

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
