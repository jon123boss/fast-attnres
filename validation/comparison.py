"""Finite-value and BF16-tolerance checks with routing error metrics."""
import torch


def compare(actual, expected):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError("nonfinite output or gradient")
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.05)
    difference = (actual.detach().float() - expected.detach().float()).abs()
    return {"max_abs": float(difference.max()),
            "relative_l2": float(torch.linalg.vector_norm(difference) /
                                 torch.linalg.vector_norm(expected.detach().float()).clamp_min(1e-20))}
