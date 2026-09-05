"""Campaign adapters; upstream kernels are frozen and never edited in place."""
from __future__ import annotations

import importlib
import hashlib
import re
import tempfile
import sys
from pathlib import Path
from typing import List

import torch
from torch import Tensor


class Ineligible(ValueError):
    """The upstream implementation does not expose this equation or shape."""


def load_fla(root):
    """Expose native FLA checkpoint 0/1 through a fullgraph-compatible boundary.

    Only the Python boundary is adapted. FLA's original forward, backward,
    autotuning, normalization weight gradient, and scratch allocations remain.
    """
    from benchmarks.gluon_compat import install_gluon_barrier_compatibility
    compatibility = install_gluon_barrier_compatibility()
    sys.path.insert(0, str(root))
    native = importlib.import_module("fla.ops.attnres.fused")
    if not Path(native.__file__).resolve().is_relative_to(Path(root).resolve()):
        raise RuntimeError("FLA import resolved outside the frozen checkout")

    def table(values):
        return native._build_ptr_table(values)

    @torch.library.custom_op("campaign_fla::forward", mutates_args=())
    def forward(values: List[Tensor], query: Tensor, eps: float,
                scale: float, level: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        sources = [v.contiguous() for v in values]
        weight = torch.ones_like(query, dtype=torch.float32)
        out, pre, rms, logits, lse = native.fused_attnres_fwd(
            query.contiguous(), sources, table(sources), weight, None, eps, scale, level)
        if pre is None:
            pre = query.new_empty(0)
        return out, pre, rms, logits, lse

    @forward.register_fake
    def fake_forward(values, query, eps, scale, level):
        x = values[0]
        stats = (len(values), *x.shape[:-1])
        return (torch.empty_like(x), torch.empty_like(x) if level == 0 else query.new_empty(0),
                x.new_empty(stats, dtype=torch.float32), x.new_empty(stats, dtype=torch.float32),
                x.new_empty(x.shape[:-1], dtype=torch.float32))

    @torch.library.custom_op("campaign_fla::backward", mutates_args=())
    def backward(values: List[Tensor], query: Tensor, upstream: Tensor,
                 pre: Tensor, rms: Tensor, logits: Tensor, lse: Tensor,
                 eps: float, scale: float, level: int) -> tuple[List[Tensor], Tensor]:
        sources = [v.contiguous() for v in values]
        weight = torch.ones_like(query, dtype=torch.float32)
        dvs, dq, _, _ = native.fused_attnres_bwd(
            upstream.contiguous(), query.contiguous(), sources, table(sources), weight,
            None, pre if level == 0 else None, rms, logits, lse, eps, scale, level)
        return dvs, dq

    @backward.register_fake
    def fake_backward(values, query, upstream, pre, rms, logits, lse, eps, scale, level):
        return [torch.empty_like(v) for v in values], torch.empty_like(query)

    def setup(ctx, inputs, output):
        values, query, ctx.eps, ctx.scale, ctx.level = inputs
        ctx.save_for_backward(query, *output[1:], *values)
        ctx.mark_non_differentiable(*output[1:])

    def autograd_backward(ctx, upstream, *_):
        query, pre, rms, logits, lse, *values = ctx.saved_tensors
        dvs, dq = backward(values, query, upstream, pre, rms, logits, lse,
                           ctx.eps, ctx.scale, ctx.level)
        return dvs, dq, None, None, None

    forward.register_autograd(autograd_backward, setup_context=setup)

    def backend(level):
        def call(values, query, *, eps=2**-23, scale=1.0):
            if query.numel() != values[0].shape[-1]:
                raise Ineligible("FLA implements full-width keys only")
            values = list(values.unbind(0)) if isinstance(values, Tensor) else list(values)
            return forward(values, query, float(eps), float(scale), level)[0]
        return call
    return {"fla_checkpoint0": backend(0), "fla_checkpoint1": backend(1)}, compatibility


def load_liger(root):
    sys.path.insert(0, str(Path(root) / "src"))
    native = importlib.import_module("liger_kernel.ops.attn_res")
    if not Path(native.__file__).resolve().is_relative_to(Path(root).resolve()):
        raise RuntimeError("Liger import resolved outside the frozen checkout")

    def call(values, query, *, eps=2**-23, scale=1.0):
        if query.numel() != values[0].shape[-1] or len(values) > 32 or scale != 1:
            raise Ineligible("Liger requires full-width keys, S<=32 and scale=1")
        packed = torch.stack(values) if not isinstance(values, Tensor) else values
        return native.LigerAttnResFunction.apply(packed, query,
                                                 torch.ones_like(query), eps)
    return {"liger": call}, {"adapter": "native autograd; source stack included"}



def load_legacy(root):
    """Uncached research reads with the campaign epsilon, unchanged GPU kernels.

    The historical wrappers hard-code BF16 machine epsilon. Only those scalar
    launch arguments and library namespaces are adapted in a temporary module;
    their original and adapted hashes are retained in the report.
    """
    source = (Path(root) / "attnres_ops.py").read_text()
    adapted, replacements = re.subn(r"torch\.finfo\(values(?:\[0\])?\.dtype\)\.eps",
                                     "(2**-23)", source)
    adapted = adapted.replace('"attnres::', '"campaign_legacy_native::')
    temporary = tempfile.TemporaryDirectory(prefix="attnres-legacy-")
    path = Path(temporary.name) / "campaign_legacy.py"
    path.write_text(adapted)
    spec = importlib.util.spec_from_file_location("campaign_legacy", path)
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)

    @torch.library.custom_op("campaign_legacy::forward", mutates_args=())
    def forward(values: List[Tensor], query: Tensor, scale: float) -> Tensor:
        if query.numel() == values[0].shape[-1]:
            if len(values) <= 16:
                return native._attnres_read_list_triton(values, query, True)
            return native._attnres_read_triton(torch.stack(values), query, True)
        keys = [v[..., -query.numel():] for v in values]
        return native._lrid_read_list_triton(values, keys, query.view(1, -1), 1, scale, True)

    @forward.register_fake
    def fake_forward(values, query, scale):
        return torch.empty_like(values[0])

    @torch.library.custom_op("campaign_legacy::backward", mutates_args=())
    def backward(values: List[Tensor], query: Tensor, upstream: Tensor,
                 scale: float) -> tuple[List[Tensor], Tensor]:
        if query.numel() == values[0].shape[-1]:
            if len(values) <= 16:
                return native._attnres_read_list_backward_triton(values, query, upstream, True)
            dvs, dq = native._attnres_read_backward_triton(torch.stack(values), query, upstream, True)
            return list(dvs.unbind(0)), dq
        rank = query.numel()
        keys = [v[..., -rank:] for v in values]
        dvs, dks, dq = native._lrid_read_list_backward_triton(
            values, keys, query.view(1, -1), upstream, 1, scale, True)
        for dv, dk in zip(dvs, dks):
            dv[..., -rank:].add_(dk)
        return dvs, dq.reshape_as(query)

    @backward.register_fake
    def fake_backward(values, query, upstream, scale):
        return [torch.empty_like(v) for v in values], torch.empty_like(query)

    def setup(ctx, inputs, output):
        values, query, ctx.scale = inputs
        ctx.save_for_backward(query, *values)

    def autograd_backward(ctx, upstream):
        query, *values = ctx.saved_tensors
        dvs, dq = backward(values, query, upstream, ctx.scale)
        return dvs, dq, None

    forward.register_autograd(autograd_backward, setup_context=setup)

    def call(values, query, *, eps=2**-23, scale=1.0):
        if eps != 2**-23:
            raise Ineligible("legacy bridge is frozen at campaign epsilon 2**-23")
        values = list(values.unbind(0)) if isinstance(values, Tensor) else list(values)
        shape = values[0].shape
        width, rank = shape[-1], query.numel()
        if width > 4096 or (rank == width and scale != 1):
            raise Ineligible("legacy standard requires width<=4096 and scale=1")
        if rank != width and (rank > 256 or len(values) > 16):
            raise Ineligible("legacy uncached LR requires rank<=256 and sources<=16")
        # Contiguous copies and restoring source shape are part of the boundary.
        sources = [v.reshape(1, -1, width).contiguous() for v in values]
        return forward(sources, query.contiguous(), float(scale)).reshape(shape)

    call._temporary = temporary
    return {"legacy_uncached": call}, {
        "adapter": "uncached native reads; epsilon scalar adaptation, source copies, key gradient accumulation included",
        "epsilon": 2**-23, "epsilon_replacements": replacements,
        "original_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "adapted_sha256": hashlib.sha256(adapted.encode()).hexdigest()}

def load_all(roots):
    from benchmarks.bf16_device import source_digest
    backends, identities, failures = {}, {}, {}
    for name, root in roots.items():
        identities[name] = source_digest(root)
        try:
            ops, adapter = {"fla": load_fla, "liger": load_liger, "legacy": load_legacy}[name](root)
            backends.update(ops)
            identities[name]["adapter"] = adapter
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
    return backends, identities, failures
