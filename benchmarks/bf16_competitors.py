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

    gluon = importlib.import_module("fla.ops.attnres.backends.gluon")
    def engine(level):
        if level >= 2:
            return gluon._fused_attnres_fwd, gluon._fused_attnres_bwd, gluon._check_sources
        return native.fused_attnres_fwd, native.fused_attnres_bwd, native._build_ptr_table

    @torch.library.custom_op("campaign_fla::forward", mutates_args=())
    def forward(values: List[Tensor], query: Tensor, eps: float,
                scale: float, level: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        sources = [v.contiguous() for v in values]
        weight = torch.ones_like(query, dtype=torch.float32)
        fwd, _, table = engine(level)
        out, pre, rms, logits, lse = fwd(
            query.contiguous(), sources, table(sources), weight, None, eps, scale, level % 2)
        if pre is None:
            pre = query.new_empty(0)
        return out, pre, rms, logits, lse

    @forward.register_fake
    def fake_forward(values, query, eps, scale, level):
        x = values[0]
        stats = (len(values), *x.shape[:-1])
        return (torch.empty_like(x), torch.empty_like(x) if level % 2 == 0 else query.new_empty(0),
                x.new_empty(stats, dtype=torch.float32), x.new_empty(stats, dtype=torch.float32),
                x.new_empty(x.shape[:-1], dtype=torch.float32))

    @torch.library.custom_op("campaign_fla::backward", mutates_args=())
    def backward(values: List[Tensor], query: Tensor, upstream: Tensor,
                 pre: Tensor, rms: Tensor, logits: Tensor, lse: Tensor,
                 eps: float, scale: float, level: int) -> tuple[List[Tensor], Tensor]:
        sources = [v.contiguous() for v in values]
        weight = torch.ones_like(query, dtype=torch.float32)
        _, bwd, table = engine(level)
        dvs, dq, _, _ = bwd(
            upstream.contiguous(), query.contiguous(), sources, table(sources), weight,
            None, pre if level % 2 == 0 else None, rms, logits, lse, eps, scale, level % 2)
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
            if level >= 2 and query.numel() < 64:
                raise Ineligible("FLA Gluon requires BF16 width>=64")
            values = list(values.unbind(0)) if isinstance(values, Tensor) else list(values)
            return forward(values, query, float(eps), float(scale), level)[0]
        return call
    return {"fla_checkpoint0": backend(0), "fla_checkpoint1": backend(1),
            "fla_gluon_checkpoint0": backend(2), "fla_gluon_checkpoint1": backend(3)}, compatibility


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

    @torch.library.custom_op("campaign_legacy::packed_backward", mutates_args=())
    def packed_backward(values: List[Tensor], query: Tensor, upstream: Tensor) -> tuple[Tensor, Tensor]:
        return native._attnres_read_backward_triton(torch.stack(values), query, upstream, True)

    @packed_backward.register_fake
    def fake_packed_backward(values, query, upstream):
        return values[0].new_empty((len(values), *values[0].shape)), torch.empty_like(query)

    def setup(ctx, inputs, output):
        values, query, ctx.scale = inputs
        ctx.save_for_backward(query, *values)

    def autograd_backward(ctx, upstream):
        query, *values = ctx.saved_tensors
        if query.numel() == values[0].shape[-1] and len(values) > 16:
            packed, dq = packed_backward(values, query, upstream)
            dvs = list(packed.unbind(0))
        else:
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


def load_catswe(root):
    """Call frozen native phase 1 afresh, with packing inside its boundary."""
    from benchmarks import catswe
    root = Path(root).resolve()
    for relative, expected in catswe._VENDOR_SHA256.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Catswe source mismatch: {relative}")
    sys.path.insert(0, str(root / "src"))
    native = importlib.import_module("flash_attn_res.ops.phase_1")
    if Path(native.__file__).resolve() != root / "src/flash_attn_res/ops/phase_1.py":
        raise RuntimeError("Catswe import resolved outside the frozen checkout")
    backend = catswe.CatsweBackend(native.phase_1_batched_attention_triton_op, root)
    def call(values, query, *, eps=2**-23, scale=1.0):
        if query.dtype != torch.bfloat16:
            raise Ineligible("campaign queries use BF16 storage")
        return backend(values, query, eps=eps, scale=scale)
    return {"catswe_phase1": call}, {
        "revision": catswe.PINNED_REVISION,
        "adapter": "native single-query phase1 per read; source stack/copies and all gradients included"}


def load_hydra(root):
    """Run native Hydra phase 1+2 afresh for one full-width query."""
    root = Path(root).resolve()
    sys.path.insert(0, str(root / 'src'))
    module = importlib.import_module('attnres_kernel.triton_impl')
    if Path(module.__file__).resolve() != root / 'src/attnres_kernel/triton_impl.py':
        raise RuntimeError('Hydra import resolved outside the frozen checkout')
    native = module._TritonAttnRes

    class Context:
        def save_for_backward(self, *tensors):
            self.saved_tensors = tensors

    @torch.library.custom_op('campaign_hydra::forward', mutates_args=())
    def forward(blocks: Tensor, query: Tensor, warps: int, eps: float) -> tuple[Tensor, Tensor, Tensor]:
        ctx = Context()
        partial = torch.zeros_like(blocks[:1])
        enabled = torch.zeros(1, device=query.device, dtype=torch.bool)
        width = query.shape[-1]
        out = native.forward(ctx, query, blocks, partial, enabled,
                             1 << (width - 1).bit_length(), warps, eps)
        return out, ctx.saved_tensors[-2], ctx.saved_tensors[-1]

    @forward.register_fake
    def fake_forward(blocks, query, warps, eps):
        stats = blocks.new_empty((1, blocks.shape[1]), dtype=torch.float32)
        return torch.empty_like(blocks[:1]), stats, torch.empty_like(stats)

    @torch.library.custom_op('campaign_hydra::backward', mutates_args=())
    def backward(blocks: Tensor, query: Tensor, output: Tensor, maximum: Tensor,
                 denominator: Tensor, upstream: Tensor, warps: int, eps: float) -> tuple[Tensor, Tensor]:
        ctx = Context()
        partial = torch.zeros_like(blocks[:1])
        enabled = torch.zeros(1, device=query.device, dtype=torch.bool)
        ctx.saved_tensors = (query, blocks, partial, enabled, output, maximum, denominator)
        ctx.meta = (blocks.shape, partial.shape, 1 << (query.shape[-1] - 1).bit_length(), warps, eps)
        dq, dv, _, *_ = native.backward(ctx, upstream.contiguous())
        return dv, dq

    @backward.register_fake
    def fake_backward(blocks, query, output, maximum, denominator, upstream, warps, eps):
        return torch.empty_like(blocks), torch.empty_like(query)

    def setup(ctx, inputs, output):
        blocks, query, ctx.warps, ctx.eps = inputs
        ctx.save_for_backward(blocks, query, *output)
        ctx.mark_non_differentiable(*output[1:])

    def autograd_backward(ctx, upstream, *_):
        dv, dq = backward(*ctx.saved_tensors, upstream, ctx.warps, ctx.eps)
        return dv, dq, None, None

    forward.register_autograd(autograd_backward, setup_context=setup)

    def backend(warps):
        def call(values, query, *, eps=2**-23, scale=1.0):
            width = values[0].shape[-1]
            if query.numel() != width or scale != 1:
                raise Ineligible('Hydra implements full-width keys and scale=1')
            if width > 8192 or len(values) > 129:
                raise Ineligible('Hydra exceeds the frozen width/source envelope')
            if query.dtype != torch.bfloat16 or values[0].dtype != torch.bfloat16:
                raise Ineligible('Hydra campaign storage is BF16')
            shape = values[0].shape
            packed = values if isinstance(values, Tensor) else torch.stack(tuple(values))
            blocks = packed.reshape(len(values), -1, width).contiguous()
            return forward(blocks, query.reshape(1, width).contiguous(), warps, float(eps))[0].reshape(shape)
        return call
    return {'hydra_2p': backend(4), 'hydra_2p8': backend(8)}, {
        'revision': 'ea1f63eda8e31b0f10456b3b49cacd8fb66091dc',
        'adapter': 'native single-query phase1+phase2 per read; source preparation and disabled partial included',
        'block_d': 'nextpow2(D)', 'num_warps': {'hydra_2p': 4, 'hydra_2p8': 8}}


def model_ineligibility(name, model):
    """Static public-comparator limits for an entire uncached model schedule."""
    width, rank = model.width, model.rank
    sources = 2 * model.layers + 1 if model.mode == "full" else min(2 * model.layers, model.block_count) + 1
    if name.startswith("fla_") and rank != width:
        return "FLA implements full-width routing keys only"
    if name.startswith("fla_gluon_") and width < 64:
        return "FLA Gluon requires BF16 width>=64"
    if name == "liger" and (rank != width or sources > 32 or model.attnres_scale != 1):
        return "Liger requires full-width keys, sources<=32, scale=1"
    if name == "legacy_uncached":
        if model.attnres_eps != 2**-23 or width > 4096:
            return "legacy bridge requires campaign epsilon and width<=4096"
        if rank == width and model.attnres_scale != 1:
            return "legacy standard requires scale=1"
        if rank != width and (rank > 256 or sources > 16):
            return "legacy uncached LR requires rank<=256 and sources<=16"
    if name.startswith("hydra_2p") and (rank != width or model.attnres_scale != 1 or width > 8192 or sources > 129):
        return "Hydra requires R=D, scale=1, width<=8192 and sources<=129"
    if name == "catswe_phase1":
        if rank != width or width & (width - 1):
            return "Catswe phase1 requires R=D and power-of-two D"
        if model.attnres_eps != 2**-23 or model.attnres_scale != 1:
            return "Catswe phase1 requires campaign epsilon and scale=1"
        if width > 8192 or sources > 129 or (1 << (sources - 1).bit_length()) * width > 1048576:
            return "Catswe phase1 exceeds its source/width envelope"
    return None

def load_all(roots):
    from benchmarks.bf16_device import source_digest
    backends, identities, failures = {}, {}, {}
    for name, root in roots.items():
        identities[name] = source_digest(root)
        try:
            ops, adapter = {"fla": load_fla, "liger": load_liger,
                            "legacy": load_legacy, "catswe": load_catswe, "hydra": load_hydra}[name](root)
            backends.update(ops)
            identities[name]["adapter"] = adapter
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
    return backends, identities, failures
