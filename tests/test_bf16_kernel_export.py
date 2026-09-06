"""Binary export must preserve payloads and restore launch hooks after failures."""
from collections import namedtuple
from types import SimpleNamespace

import pytest

from benchmarks.bf16_kernel_export import export_selected


def test_selected_export_preserves_binary_and_specialization(tmp_path):
    metadata = namedtuple('Metadata', 'shared')(128)
    kernel = SimpleNamespace(hash='abc', name='route', n_regs=32, n_spills=0,
                             metadata=metadata, asm={'ptx': 'instructions', 'cubin': b'\x00\xff'},
                             src=SimpleNamespace(constants={(1,): 640}))
    def launch():
        return kernel
    target = SimpleNamespace(device_caches={}, run=launch)
    module = SimpleNamespace(_routing_forward=target)
    report = export_selected(module, lambda: target.run(), tmp_path)['_routing_forward']
    assert (tmp_path/'abc/kernel.cubin').read_bytes() == b'\x00\xff'
    assert report['registers'] == 32 and report['spills'] == 0
    assert report['constants'] == [{'path': [1], 'value': '640'}]
    assert target.run is launch


def test_export_restores_autotuned_launch_after_operator_failure(tmp_path):
    def launch():
        return None
    target = SimpleNamespace(device_caches={}, run=launch)
    module = SimpleNamespace(_fla_standard_forward_kernel=SimpleNamespace(fn=target))
    def fail():
        target.run()
        raise ValueError('operator failed')
    with pytest.raises(ValueError, match='operator failed'):
        export_selected(module, fail, tmp_path)
    assert target.run is launch
    assert not list(tmp_path.iterdir())


def test_export_retains_both_query_reductions_and_restores_aliased_hooks(tmp_path):
    metadata = namedtuple('Metadata', 'shared')(0)
    def launch(index, **kwargs):
        return SimpleNamespace(hash=str(index), name='query_reduce', n_regs=32, n_spills=0,
                               metadata=metadata, asm={'cubin': bytes([index])},
                               src=SimpleNamespace(constants={}, signature={}, attrs={}))
    target = SimpleNamespace(device_caches={}, run=launch)
    module = SimpleNamespace(_fla_query_reduce_kernel=target,
                             _fla_standard_query_reduce_kernel=target)
    def call():
        target.run(1, grid=(4, 32))
        target.run(2, grid=(4,))
    result = export_selected(module, call, tmp_path)
    assert result['_fla_query_reduce_kernel[0]']['hash'] == '1'
    assert result['_fla_query_reduce_kernel[1]']['hash'] == '2'
    assert result['_fla_query_reduce_kernel[1]']['launch']['grid'] == (4,)
    assert target.run is launch


def test_export_rejects_cold_autotuning_trials(tmp_path):
    def launch():
        return object()
    target = SimpleNamespace(device_caches={}, run=launch)
    module = SimpleNamespace(_routing_forward=target)
    with pytest.raises(RuntimeError, match='extra launches'):
        export_selected(module, lambda: [target.run() for _ in range(3)], tmp_path)
    assert target.run is launch
    assert not list(tmp_path.iterdir())


def test_backward_identity_includes_referenced_helpers_and_scalar_globals():
    from benchmarks.bf16_kernel_export import _jit_identity
    def kernel(helper):
        return SimpleNamespace(src='def backward(x):\n    return helper(x) * SCALE\n',
                               fn=SimpleNamespace(__globals__={'helper': helper, 'SCALE': 2}))
    helper = SimpleNamespace(src='def helper(x):\n    return x + 1\n')
    first = _jit_identity(kernel(helper))
    helper.src = helper.src.replace('+ 1', '+ 2')
    assert _jit_identity(kernel(helper)) != first
    second = kernel(helper)
    before = _jit_identity(second)
    second.fn.__globals__['SCALE'] = 3
    assert _jit_identity(second) != before


@pytest.mark.parametrize('different', [False, True])
def test_shared_backward_requires_identical_equations(monkeypatch, different):
    import importlib
    from benchmarks.bf16_kernel_export import share_identical_backward
    def original(values, query):
        pass
    def candidate(values, query):
        pass
    original.__module__ = 'original.api'
    candidate.__module__ = 'candidate.api'
    source = 'def backward(x):\n    return x * 2\n'
    a = SimpleNamespace(fn=SimpleNamespace(src=source))
    b = SimpleNamespace(fn=SimpleNamespace(src=source.replace('2', '3') if different else source))
    modules = {'original._kernels.fla_full_sources': SimpleNamespace(_fla_standard_backward_kernel=a),
               'candidate._kernels.fla_full_sources': SimpleNamespace(_fla_standard_backward_kernel=b)}
    monkeypatch.setattr(importlib, 'import_module', modules.__getitem__)
    backends = {'original': original, 'candidate': candidate}
    if different:
        with pytest.raises(ValueError, match='different backward equations'):
            share_identical_backward(backends, {'candidate': 'original'})
        assert modules['candidate._kernels.fla_full_sources']._fla_standard_backward_kernel is b
    else:
        result = share_identical_backward(backends, {'candidate': 'original'})
        assert result['candidate']['incumbent'] == 'original'
        assert modules['candidate._kernels.fla_full_sources']._fla_standard_backward_kernel is a
