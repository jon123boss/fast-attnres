"""Diagnose rank-order effects with qualified, simultaneously resident graphs."""
from __future__ import annotations

import subprocess

import torch

from benchmarks.baseline import load_baseline
from benchmarks.bf16_device import _inputs, bf16_torch, compare, metadata


def device_state():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}
    return {"fields": "temperature_C,sm_MHz,memory_MHz,power_W",
            "value": result.stdout.strip(), "error": result.stderr.strip()}


def run_diagnostic(config, checkpoint):
    if "primary_contract_sha256" in config:
        raise ValueError("rank-order observations cannot replace primary measurements")
    source = load_baseline(config["sources"]["candidate"])
    report = {"kind": "rank_order_diagnostic", "status": "running", "primary_eligible": False,
              "config": config, "runtime": metadata(), "identity": source.metadata,
              "qualifications": [], "results": []}
    ranks, seed = config["ranks"], config["seed"]
    s, n, d = config["shape"]
    if not ranks or len(set(ranks)) != len(ranks) or not all(1 <= rank <= d for rank in ranks):
        raise ValueError("invalid diagnostic rank set")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        values, params, master_query, upstream = _inputs({"shape": [s, n, d, d]}, seed)
        leaves, graphs = params[:-1], {}
        for rank in ranks:
            query = master_query[-rank:].detach().clone().requires_grad_()
            parameters = (*leaves, query)

            def call():
                output = source.attnres(values, query)
                return output, torch.autograd.grad(output, parameters, upstream)

            expected = bf16_torch(values, query)
            expected_grad = torch.autograd.grad(expected, parameters, upstream)
            output, gradients = call()
            errors = [compare(output, expected), *[compare(a, b) for a, b in zip(gradients, expected_grad)]]
            if any(x.dtype != torch.bfloat16 for x in (output, *gradients)):
                raise AssertionError("operator output and gradients must remain BF16")
            for _ in range(config["warmups"]):
                call()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = call()
            graphs[rank] = {"graph": graph, "result": captured, "query": query,
                            "parameters": parameters}
            report["qualifications"].append({"rank": rank, "errors": errors})
            checkpoint(report)

        for replay in range(config["replays"]):
            torch.manual_seed(seed + 1000 + replay)
            with torch.no_grad():
                for value in leaves:
                    value.copy_(torch.randn_like(value))
                master_query.copy_(torch.randn_like(master_query) * .05)
                for rank, arm in graphs.items():
                    arm["query"].copy_(master_query[-rank:])
            for arm in graphs.values():
                expected = bf16_torch(values, arm["query"])
                expected_grad = torch.autograd.grad(expected, arm["parameters"], upstream)
                arm["graph"].replay()
                torch.cuda.synchronize()
                output, gradients = arm["result"]
                compare(output, expected)
                for actual, reference in zip(gradients, expected_grad):
                    compare(actual, reference)

        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        rounds = config["rounds"]
        orders = {"grouped_forward": [rank for rank in ranks for _ in range(rounds)],
                  "grouped_reverse": [rank for rank in reversed(ranks) for _ in range(rounds)],
                  "alternating": [rank for iteration in range(rounds)
                                  for rank in (ranks if iteration % 2 == 0 else list(reversed(ranks)))]}
        for name, order in orders.items():
            samples = {rank: [] for rank in ranks}
            before = device_state()
            for rank in order:
                begin.record()
                for _ in range(10):
                    graphs[rank]["graph"].replay()
                end.record()
                end.synchronize()
                samples[rank].append(begin.elapsed_time(end) / 10)
            report["results"].append({"order": name, "samples_ms": samples,
                                      "before": before, "after": device_state()})
            checkpoint(report)
    torch.cuda.current_stream().wait_stream(stream)
    report["status"] = "complete"
    checkpoint(report)
    return report
