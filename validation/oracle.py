"""Independent evaluator: explicit source scores and accumulation."""
import torch


def oracle(values, query, *, keys=None, eps=2**-23, scale=1.0,
           compute_dtype=torch.float32):
    v = values.to(compute_dtype)
    q = query.to(compute_dtype)
    k = (values[..., -query.numel():] if keys is None else keys).to(compute_dtype)
    score = []
    for s in range(values.shape[0]):
        inv_rms = torch.rsqrt(torch.sum(k[s] * k[s], dim=-1) / q.numel() + eps)
        score.append(torch.sum(k[s] * q, dim=-1) * inv_rms * scale)
    probabilities = torch.stack(score).softmax(0)
    result = torch.zeros_like(v[0])
    for s in range(values.shape[0]):
        result = result + probabilities[s].unsqueeze(-1) * v[s]
    return result.to(values.dtype)
