"""Smoke-test ``torch.compile`` around the CUDA BF16 Fast-AttnRes operator."""

from __future__ import annotations

import argparse

import torch

from attnres import attnres


def run(
    device: str | torch.device | None = None,
    *,
    backend: str = "eager",
) -> torch.Tensor:
    """Compile a packed standard AttnRes function on CUDA BF16."""

    if not hasattr(torch, "compile"):
        raise RuntimeError("this example requires PyTorch with torch.compile")
    torch.manual_seed(4)
    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this example requires an available CUDA device")
    dtype = torch.bfloat16
    values = torch.randn(4, 2, 16, device=device, dtype=dtype)
    query = torch.randn(16, device=device, dtype=dtype)

    def residual(source_values: torch.Tensor, source_query: torch.Tensor) -> torch.Tensor:
        return attnres(source_values, source_query)

    eager_output = residual(values, query)
    compiled_residual = torch.compile(
        residual,
        backend=backend,
        fullgraph=False,
        dynamic=False,
    )
    compiled_output = compiled_residual(values, query)
    torch.testing.assert_close(compiled_output, eager_output, rtol=0.05, atol=0.05)
    if compiled_output.dtype != dtype:
        raise AssertionError("compiled output must be BF16")
    return compiled_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--backend",
        default="eager",
        help="torch.compile backend; eager is a quick smoke test, inductor generates kernels",
    )
    args = parser.parse_args()
    output = run(args.device, backend=args.backend)
    print(
        f"Fast-AttnRes torch.compile ({args.backend}): "
        f"shape={tuple(output.shape)} dtype={output.dtype}"
    )


if __name__ == "__main__":
    main()
