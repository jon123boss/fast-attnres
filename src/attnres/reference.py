"""Readable equation reference and explicit CPU implementation."""
import torch

EPS = 2**-23


def reference_attnres(values, query, *, eps=EPS, scale=1.0,
                     compute_dtype=torch.float32):
    """Mix ``[S,...,D]`` values with normalized tail keys and query ``[R]``."""
    v = values.to(compute_dtype)
    k = values[..., -query.numel():].to(compute_dtype)
    q = query.to(compute_dtype)
    scores = (k * torch.rsqrt(k.square().mean(-1, keepdim=True) + eps) * q).sum(-1)
    weights = torch.softmax(scores * scale, dim=0)
    return (weights.unsqueeze(-1) * v).sum(0).to(values.dtype)
