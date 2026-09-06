"""Retain actual selected Triton binaries outside capture, profiling, and timing."""
from pathlib import Path
import hashlib,json


def export_selected(module, call, directory):
    selected, originals = {}, []
    names = ('_fla_standard_forward_kernel', '_fla_standard_backward_kernel',
             '_routing_forward', '_weighted_value_forward')
    try:
        for name in names:
            target = getattr(module, name, None)
            if target is None:
                continue
            if not hasattr(target, 'device_caches'):
                target = target.fn
            original = target.run
            originals.append((target, original))
            def record(*args, _run=original, _name=name, **kwargs):
                kernel = _run(*args, **kwargs)
                if kernel is not None:
                    selected[_name] = kernel
                return kernel
            target.run = record
        call()
    finally:
        for target, original in originals:
            target.run = original
    result = {}
    for name, kernel in selected.items():
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
                        'metadata': kernel.metadata._asdict(),
                        'constants': [{'path': list(k), 'value': str(v)}
                                      for k,v in kernel.src.constants.items()],
                        'files': files}
        (folder / 'metadata.json').write_text(json.dumps(result[name], indent=2, default=str)+'\n')
    return result


def share_identical_backward(backends, mapping):
    """Reuse incumbent tuning only for byte-equivalent backward function ASTs."""
    import ast
    import importlib
    result = {}
    for name, incumbent in mapping.items():
        modules = [importlib.import_module(backends[key].__module__.rsplit('.', 1)[0]
                                          + '._kernels.fla_full_sources')
                   for key in (name, incumbent)]
        kernels = [module._fla_standard_backward_kernel for module in modules]
        bodies = [ast.dump(ast.parse(kernel.fn.src)) for kernel in kernels]
        if bodies[0] != bodies[1]:
            raise ValueError('cannot share tuning across different backward equations')
        modules[0]._fla_standard_backward_kernel = kernels[1]
        result[name] = {'incumbent': incumbent,
                        'backward_ast_sha256': hashlib.sha256(bodies[0].encode()).hexdigest(),
                        'purpose': 'forward attribution with identical incumbent backward tuning'}
    return result
