"""Independent BF16 reference with FP32 internal accumulation."""
import torch


def oracle(values, query, *, keys=None, eps=2**-23, scale=1.0):
    sequence = isinstance(values, (tuple, list))
    tensors = (*values, query) if sequence else (values, query)
    if keys is not None:
        tensors = (*tensors, keys)
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise TypeError("the validation reference requires BF16 tensors")
    with torch.autocast(device_type=query.device.type, enabled=False):
        v = (torch.stack(tuple(values)) if sequence else values).float()
        k = v[..., -query.numel():] if keys is None else keys.float()
        inv_rms = torch.rsqrt(k.square().mean(-1, keepdim=True) + eps)
        scores = (k * inv_rms * query.float()).sum(-1) * scale
        probabilities = scores.softmax(0)
        return (probabilities.unsqueeze(-1) * v).sum(0).to(torch.bfloat16)
