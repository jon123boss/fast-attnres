"""Independent CUDA checks. This file is evaluator-owned, not candidate-owned."""
import gc
import json
from pathlib import Path
import traceback
import torch
from .oracle import oracle

PROTOCOL = json.loads(Path(__file__).with_name("protocol.json").read_text())


def _metrics(actual, expected):
    a, e = actual.detach().float(), expected.detach().float()
    return {"max_abs": (a-e).abs().max().item(),
            "rel_l2": ((a-e).norm() / e.norm().clamp_min(1e-12)).item()}


def _compare(actual, expected, dtype):
    tol = PROTOCOL["bf16" if dtype == torch.bfloat16 else "fp32"]
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError("non-finite output or gradient")
    torch.testing.assert_close(actual, expected, **tol)
    return _metrics(actual, expected)


def _single(shape, dtype, *, strided=False, scale=1., zero_query=False):
    from attnres import attnres
    s,n,d,r = shape
    width = d+(7 if strided else 0)
    producer = torch.randn(s,n,width,device="cuda",dtype=dtype,requires_grad=True)
    q = (torch.zeros(r,device="cuda") if zero_query else
         torch.randn(r,device="cuda") * .25).requires_grad_()
    v = producer[..., :d]
    actual = attnres(v,q,scale=scale)
    expected = oracle(v,q,scale=scale)
    # A transposed upstream catches implementations that silently assume contiguous dY.
    upstream = torch.randn(d,n,device="cuda",dtype=dtype).transpose(0,1)
    ga = torch.autograd.grad(actual,(producer,q),upstream)
    ge = torch.autograd.grad(expected,(producer,q),upstream)
    return {"output":_compare(actual,expected,dtype),
            "producer_grad":_compare(ga[0],ge[0],dtype),
            "query_grad":_compare(ga[1],ge[1],dtype)}


def _block(dtype, rank):
    from attnres import attnres
    s,n,d,qcount = 3,11,128,3
    width = d
    source = torch.randn(s,n,width,device="cuda",dtype=dtype,requires_grad=True)
    partial = torch.randn(n,width,device="cuda",dtype=dtype,requires_grad=True)
    q = (torch.randn(qcount,rank,device="cuda")*.25).requires_grad_()
    values=source[...,:d]
    indexes=[0,1,1,2]  # A duplicate consumer must accumulate rather than overwrite.
    combined=torch.cat((values,partial[None,...,:d]),dim=0)
    outputs=[attnres(combined,q[i]) for i in indexes]
    outputs.append(attnres(values,q[0]))
    # One common FP32 source equation: accumulation is equation-level, not staged BF16.
    sf,pf=source.float(),partial.float()
    expected=[]
    for i in indexes:
        packed=torch.cat([sf[...,:d],pf[None,...,:d]],dim=0)
        expected.append(oracle(packed,q[i]).to(dtype))
    expected.append(oracle(sf[...,:d],q[0]).to(dtype))
    weights=[torch.randn(d,n,device="cuda",dtype=dtype).T for _ in outputs]
    actual_loss=sum((o.float()*w.float()).sum() for o,w in zip(outputs,weights))
    expected_loss=sum((o.float()*w.float()).sum() for o,w in zip(expected,weights))
    params=(source,partial,q)
    ga=torch.autograd.grad(actual_loss,params,retain_graph=True)
    ge=torch.autograd.grad(expected_loss,params)
    repeat=torch.autograd.grad(actual_loss,params)
    result={"outputs":[_compare(a,e,dtype) for a,e in zip(outputs,expected)],
            "grads":[_compare(a,e,dtype) for a,e in zip(ga,ge)]}
    for a,b in zip(ga,repeat):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    return result


def _compiled_and_graph(*, dtype=torch.bfloat16, shape=(9,65,256,32)):
    from attnres import attnres
    s,n,d,r=shape
    width=d
    source=torch.randn(s,n,width,device="cuda",dtype=dtype,requires_grad=True)
    query=torch.randn(r,device="cuda",requires_grad=True)
    def forward(v,q):
        return attnres(v[...,:d],q)
    compiled=torch.compile(forward,fullgraph=True,dynamic=False)
    upstream=torch.randn(n,d,device="cuda",dtype=dtype)
    for _ in range(3):
        output=compiled(source,query)
        torch.autograd.grad(output,(source,query),upstream)
    output=compiled(source,query)
    expected=oracle(source[...,:d],query)
    go=torch.autograd.grad(output,(source,query),upstream)
    ge=torch.autograd.grad(expected,(source,query),upstream)
    _compare(output,expected,dtype)
    for a,b in zip(go,ge): _compare(a,b,dtype)
    # Capture eager registered operations; compile's own CUDAGraph policy is separate.
    # Use fresh static leaves and warm up on the capture stream. Reusing
    # autograd leaves warmed on the default stream can add a forbidden
    # legacy-stream dependency during capture (PyTorch CUDA graph contract).
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        source=source.detach().clone().requires_grad_()
        query=query.detach().clone().requires_grad_()
        upstream=upstream.clone()
        for _ in range(3):
            temp=forward(source,query)
            torch.autograd.grad(temp,(source,query),upstream)
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        captured=forward(source,query)
        captured_grads=torch.autograd.grad(captured,(source,query),upstream)
    with torch.no_grad():
        source.copy_(torch.randn_like(source))
        query.copy_(torch.randn_like(query))
        upstream.copy_(torch.randn_like(upstream))
    graph.replay()
    torch.cuda.synchronize()
    expected=oracle(source[...,:d],query)
    ge=torch.autograd.grad(expected,(source,query),upstream)
    _compare(captured,expected,dtype)
    for a,b in zip(captured_grads,ge): _compare(a,b,dtype)
    return {"compiled_forward_backward":True,"changed_input_graph":True}


def run_checks(config):
    kind=config.get("kind","all")
    dtype=torch.bfloat16 if config.get("dtype","bf16")=="bf16" else torch.float32
    torch.manual_seed(config.get("seed",PROTOCOL["seeds"][0]))
    cases=[]
    if kind in ("all", "full"):
        shapes=config.get("cases",PROTOCOL["operator_smoke"])
        for shape in shapes:
            cases.append((f"full_{shape}",
                          lambda sh=shape: _single(sh,dtype,strided=True)))
        cases.append(("zero_query",lambda:
                      _single([2,11,128,16],dtype,scale=-.5,zero_query=True)))
        if config.get("compile",True):
            cases.append(("compiled_graph",lambda: _compiled_and_graph(dtype=dtype)))
    if kind in ("all","block"):
        for rank in (16,128):
            cases.append((f"block_{rank}",lambda r=rank: _block(dtype,r)))
    result={"passed":0,"failed":0,"cases":[],"config":config}
    for name,fn in cases:
        try:
            item={"name":name,"status":"passed","metrics":fn()}
            result["passed"]+=1
        except Exception as exc:
            item={"name":name,"status":"failed","error":f"{type(exc).__name__}: {exc}",
                  "traceback":traceback.format_exc()}
            result["failed"]+=1
        result["cases"].append(item)
        gc.collect()
        torch.cuda.empty_cache()
        if item["status"]=="failed" and ("CompilationError" in item.get("error","") or
                                         "illegal memory" in item.get("error","").lower()):
            result["not_run"]=len(cases)-len(result["cases"])
            break
    return result
