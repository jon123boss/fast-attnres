"""Smoke-test ``torch.compile`` around the installed Fast-AttnRes operator."""

from __future__ import annotations

import argparse

import torch

from attnres import attnres


def run(
    device: str | torch.device | None = None,
    *,
    backend: str = "eager",
) -> torch.Tensor:
    """Compile a packed standard AttnRes function and compare its output."""

    if not hasattr(torch, "compile"):
        raise RuntimeError("this example requires PyTorch with torch.compile")
    torch.manual_seed(4)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    values = torch.randn(4, 2, 16, device=device, dtype=dtype)
    query = torch.randn(16, device=device, dtype=torch.float32)

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
    tolerance = {"rtol": 0.05, "atol": 0.05} if dtype == torch.bfloat16 else {
        "rtol": 0.001,
        "atol": 0.0001,
    }
    torch.testing.assert_close(compiled_output, eager_output, **tolerance)
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
