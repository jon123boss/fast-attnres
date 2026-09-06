"""Independent BF16 PyTorch reference for validation and benchmark controls."""
import torch


def oracle(values, query, *, keys=None, eps=2**-23, scale=1.0):
    sequence = isinstance(values, (tuple, list))
    tensors = (*values, query) if sequence else (values, query)
    if keys is not None:
        tensors = (*tensors, keys)
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise TypeError("the validation reference requires BF16 tensors")
    v = torch.stack(tuple(values)) if sequence else values
    k = v[..., -query.numel():] if keys is None else keys
    inv_rms = torch.rsqrt(k.square().mean(-1) + eps)
    scores = (k * query).sum(-1) * inv_rms * scale
    probabilities = scores.softmax(0)
    return (probabilities.unsqueeze(-1) * v).sum(0)
