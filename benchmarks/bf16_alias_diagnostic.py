"""Forensic BF16 rounding comparisons; these records never qualify a kernel."""
from __future__ import annotations

import torch

from benchmarks.baseline import load_baseline
# Retain the original two-cast graph so this diagnostic remains reproducible.
# Qualification uses validation.oracle, never this historical cast topology.
def _two_cast_oracle(values, query, *, keys=None, eps=2**-23, scale=1.0,
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



def _single_boundary(values, query, *, packed=False):
    # A source gradient crosses BF16 once per operator input. Casting each
    # source before assembly also preserves duplicate-input accumulation.
    v = torch.stack(values).float() if packed else torch.stack([x.float() for x in values])
    k, q = v[..., -query.numel():], query.float()
    inverse = (k.square().mean(-1) + 2**-23).rsqrt()
    weights = ((k * q).sum(-1) * inverse).softmax(0)
    return (weights[..., None] * v).sum(0).to(query.dtype)


def _difference(a, b):
    a, b = a.detach(), b.detach()
    difference = (a.float() - b.float()).abs()
    mask = difference > (.05 + .05 * b.float().abs())
    indices = mask.nonzero()[:8]
    return {"mismatches": int(mask.sum()), "elements": a.numel(),
            "max_abs": float(difference.max()),
            "examples": [{"index": x.tolist(), "actual": float(a[tuple(x)]),
                          "expected": float(b[tuple(x)])} for x in indices]}


def alias_case(shape, backends, *, device="cuda", seed=20260827):
    s, n, d, r = shape
    torch.manual_seed(seed)
    count = s - 1
    leaves = tuple(torch.randn(*((n, 2*d), (d, n), (n, d+7))[i % 3],
                               device=device, dtype=torch.bfloat16, requires_grad=True)
                   for i in range(count))
    query = torch.randn(3, 2*r, device=device, dtype=torch.bfloat16, requires_grad=True)
    partial = torch.randn(n, d, device=device, dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn(5, d, n, device=device, dtype=torch.bfloat16).transpose(1, 2)
    params = (*leaves, query, partial)

    def evaluate(name, read, *, single=False):
        values = tuple(a[..., ::2] if i % 3 == 0 else a.T if i % 3 == 1 else a[..., :d]
                       for i, a in enumerate(leaves))
        values = (*values, values[0])
        q = query[..., ::2]
        if name == "frozen_reused_stack":
            completed = torch.stack(values)
            combined = torch.cat((completed, partial.unsqueeze(0)))
        else:
            completed, combined = values, (*values, partial)
        outputs = [read(combined, q[index]) for index in ((0,) if single else (0, 1, 1, 2))]
        if not single:
            outputs.append(read(completed, q[0]))
        loss = sum((o.float() * w.float()).sum() for o, w in zip(outputs, upstream))
        return tuple(outputs), torch.autograd.grad(loss, params)

    methods = {"frozen_reused_stack": _two_cast_oracle,
               "per_read_stack": lambda v, q: _two_cast_oracle(torch.stack(v), q),
               "one_cast_packed": lambda v, q: _single_boundary(v, q, packed=True),
               "one_cast_sources": _single_boundary, **backends}
    records = {}
    for single in (True, False):
        results = {name: evaluate(name, read, single=single) for name, read in methods.items()}
        comparisons = {}
        for expected in ("frozen_reused_stack", "per_read_stack", "one_cast_sources"):
            out, grad = results[expected]
            comparisons[expected] = {
                name: {"outputs": [_difference(a, b) for a, b in zip(actual[0], out)],
                       "gradients": [_difference(a, b) for a, b in zip(actual[1], grad)]}
                for name, actual in results.items() if name != expected}
        records["single_read" if single else "five_reads"] = comparisons
    return {"shape": shape, "seed": seed, "comparisons": records}


def run_diagnostic(config, checkpoint):
    sources = {name: load_baseline(path) for name, path in config.get("sources", {}).items()}
    report = {"kind": "alias_rounding_diagnostic", "status": "running",
              "qualification": False, "tolerance": {"rtol": .05, "atol": .05},
              "identities": {name: source.metadata for name, source in sources.items()},
              "results": []}
    for shape in config.get("shapes", [[5, 7, 513, 257], [9, 17, 1536, 96]]):
        report["results"].append(alias_case(shape, sources, device=config.get("device", "cuda")))
        checkpoint(report)
    report["status"] = "complete"
    checkpoint(report)
    return report
