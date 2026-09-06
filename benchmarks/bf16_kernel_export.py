"""Retain actual selected Triton binaries outside capture, profiling, and timing."""
from pathlib import Path
import hashlib,json


def _tensor_layouts(arguments):
    result = []
    def visit(value, path):
        if isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(child, [*path, index])
        elif hasattr(value, 'data_ptr'):
            pointer = value.data_ptr()
            result.append({'argument': path, 'shape': list(value.shape),
                           'strides': list(value.stride()), 'dtype': str(value.dtype),
                           'storage_offset': value.storage_offset(),
                           'alignment_bytes': min(pointer & -pointer, 256)})
    visit(arguments, [])
    return result


def export_selected(module, call, directory):
    selected, originals = {}, []
    names = ('_fla_standard_forward_kernel', '_fla_standard_backward_kernel',
             '_routing_forward', '_weighted_value_forward', '_routing_probabilities',
             '_fla_query_reduce_kernel', '_fla_standard_query_reduce_kernel',
             '_copy_gradient_kernel')
    seen = set()
    try:
        for name in names:
            target = getattr(module, name, None)
            if target is None:
                continue
            if not hasattr(target, 'device_caches'):
                target = target.fn
            if id(target) in seen:
                continue
            seen.add(id(target))
            original = target.run
            originals.append((target, original))
            def record(*args, _run=original, _name=name, **kwargs):
                kernel = _run(*args, **kwargs)
                if kernel is not None:
                    selected.setdefault(_name, []).append((kernel, {
                        'grid': kwargs.get('grid'), 'inputs': _tensor_layouts(args)}))
                return kernel
            target.run = record
        call()
    finally:
        for target, original in originals:
            target.run = original
    result = {}
    launches = []
    for name, calls in selected.items():
        if len(calls) > (2 if name == '_fla_query_reduce_kernel' else 1):
            raise RuntimeError('export observed extra launches; warm the exact call before export')
        for index, (kernel, launch) in enumerate(calls):
            label = name if len(calls) == 1 else f'{name}[{index}]'
            launches.append((label, kernel, launch))
    for name, kernel, launch in launches:
        folder = Path(directory) / kernel.hash
        folder.mkdir(parents=True, exist_ok=True)
        files = {}
        for extension in ('ttir', 'ttgir', 'llir', 'ptx', 'cubin'):
            content = kernel.asm.get(extension)
            if content is None:
                continue
            data = content if isinstance(content, bytes) else content.encode()
            path = folder / ('kernel.' + extension)
            path.write_bytes(data)
            files[extension] = {'sha256': hashlib.sha256(data).hexdigest(),
                                'bytes': len(data), 'file': str(path)}
        result[name] = {'hash': kernel.hash, 'name': kernel.name,
                        'registers': kernel.n_regs, 'spills': kernel.n_spills,
                        'shared_bytes': kernel.metadata.shared,
                        'launch': launch,
                        'metadata': kernel.metadata._asdict(),
                        'signature': {str(k): str(v) for k,v in
                                      getattr(kernel.src, 'signature', {}).items()},
                        'attributes': str(getattr(kernel.src, 'attrs', None)),
                        'constants': [{'path': list(k), 'value': str(v)}
                                      for k,v in kernel.src.constants.items()],
                        'files': files}
        (folder / 'metadata.json').write_text(json.dumps(result[name], indent=2, default=str)+'\n')
    return result


def _jit_identity(jit, seen=None):
    """Include referenced JIT bodies and scalar globals in the equation check."""
    import ast
    seen = set() if seen is None else seen
    if id(jit) in seen:
        return None
    seen.add(id(jit))
    tree = ast.parse(jit.src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    globals_ = getattr(getattr(jit, 'fn', None), '__globals__', {})
    dependencies = {}
    for name in sorted({node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}):
        value = globals_.get(name)
        if hasattr(value, 'src'):
            dependencies[name] = _jit_identity(value, seen)
        elif isinstance(value, (str, int, float, bool)):
            dependencies[name] = value
    return {'ast': ast.dump(tree), 'dependencies': dependencies}


def share_identical_backward(backends, mapping):
    """Reuse incumbent tuning only for byte-equivalent backward function ASTs."""
    import importlib
    result = {}
    for name, incumbent in mapping.items():
        modules = [importlib.import_module(backends[key].__module__.rsplit('.', 1)[0]
                                          + '._kernels.fla_full_sources')
                   for key in (name, incumbent)]
        kernels = [module._fla_standard_backward_kernel for module in modules]
        bodies = [json.dumps(_jit_identity(kernel.fn), sort_keys=True) for kernel in kernels]
        if bodies[0] != bodies[1]:
            raise ValueError('cannot share tuning across different backward equations')
        modules[0]._fla_standard_backward_kernel = kernels[1]
        result[name] = {'incumbent': incumbent,
                        'backward_ast_sha256': hashlib.sha256(bodies[0].encode()).hexdigest(),
                        'purpose': 'forward attribution with identical incumbent backward tuning'}
    return result
