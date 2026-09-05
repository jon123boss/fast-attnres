"""Budgeted, resumable Modal transport for immutable BF16 campaign snapshots.

Only this local entry point admits paid jobs. A reservation covers the full
timeout, startup allowance, requested CPU/memory, and GPU multiplicity.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
import traceback

import modal

PROJECT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("ATTNRES_CAMPAIGN_WORK", PROJECT.parent / "kernel-work"))
SNAPSHOT = Path(os.environ.get("ATTNRES_JOB_SNAPSHOT", WORK / "snapshots" / "none"))
VOLUME_NAME = "bf16-attnres-20260905"
_SNAPSHOT_JOB = json.loads((SNAPSHOT / "job.json").read_text()) if (SNAPSHOT / "job.json").exists() else {}
CPU_CORES = _SNAPSHOT_JOB.get("cpu_cores", 8)
MEMORY_MIB = _SNAPSHOT_JOB.get("memory_mib", 65536)
TIMEOUT_S = _SNAPSHOT_JOB.get("timeout_s", 2400)
app = modal.App("bf16-attnres-optimization")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .uv_pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cu130")
         .uv_pip_install("numpy==2.2.6", "einops==0.8.1", "pytest==8.4.2")
         .env({"PYTHONPATH": "/job/runner:/job/runner/src",
               "TORCHINDUCTOR_CACHE_DIR": "/evidence/cache/inductor",
               "TRITON_CACHE_DIR": "/evidence/cache/triton"}))
if SNAPSHOT.is_dir():
    image = image.add_local_dir(str(SNAPSHOT), "/job", copy=True)


def _remote(job):
    faulthandler.enable()
    os.environ["TRITON_CACHE_DIR"] = "/tmp/attnres-triton"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/attnres-inductor"
    sys.path[:0] = ["/job/runner", "/job/runner/src"]
    from benchmarks.bf16_device import run_operator, source_digest
    root = Path("/evidence") / job["id"]
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "started.json"
    if marker.exists():
        previous = root / "report.json"
        report = json.loads(previous.read_text()) if previous.exists() else {}
        report.update(status="failed", error="Container restarted; automatic experiment rerun rejected")
        (root / "restart-rejected.json").write_text(json.dumps(report, indent=2) + "\n")
        volume.commit()
        return report
    marker.write_text(json.dumps({"started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}) + "\n")
    volume.commit()
    started = time.monotonic()
    sequence = 0
    cache_root = Path("/evidence/compiler-cache") / job["config"]["gpu"] / "torch2.13-triton3.7.1-cu130"
    cache_enabled = job["config"].get("reuse_compiler_cache", False)
    cache_loaded = False
    cache_archive = cache_root / "artifacts.tar.gz"
    cached_results = 0
    def save_compiler_cache():
        if not cache_enabled:
            return
        try:
            local_archive = Path("/tmp/compiler-artifacts.tar.gz")
            def completed_file(info):
                parts = Path(info.name).parts
                if any(part in ("locks", "__pycache__") or part.endswith((".lock", ".tmp"))
                       or part.startswith("tmp.") for part in parts):
                    return None
                return info if info.isfile() or info.isdir() else None
            with tarfile.open(local_archive, "w:gz", compresslevel=1) as archive:
                for name in ("triton", "inductor"):
                    source = Path("/tmp") / f"attnres-{name}"
                    if source.exists():
                        archive.add(source, arcname=source.name, filter=completed_file)
            cache_root.mkdir(parents=True, exist_ok=True)
            pending = cache_root / "artifacts.pending"
            shutil.copyfile(local_archive, pending)
            pending.replace(cache_archive)
        except Exception:
            (root / "compiler-cache-error.txt").write_text(traceback.format_exc())

    def checkpoint(report):
        nonlocal sequence, cached_results
        sequence += 1
        report["elapsed_s"] = time.monotonic() - started
        report["compiler_cache"] = {"reuse_enabled": cache_enabled, "loaded": cache_loaded}
        path = root / "report.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, default=str) + "\n")
        temporary.replace(path)
        history = root / "history"
        history.mkdir(exist_ok=True)
        shutil.copy2(path, history / f"{sequence:06d}.json")
        completed = len(report.get("results", []))
        if completed > cached_results:
            save_compiler_cache()
            cached_results = completed
        volume.commit()
    try:
        checkpoint({"status": "running", "phase": "load_compiler_cache", "config": job["config"]})
        if cache_enabled and cache_archive.exists():
            with tarfile.open(cache_archive, "r:gz") as archive:
                archive.extractall("/tmp", filter="data")
            cache_loaded = True
        for label, expected in job["hashes"].items():
            actual = source_digest(Path("/job") / label)["sha256"]
            if actual != expected:
                raise RuntimeError(f"snapshot hash mismatch: {label}")
        config = dict(job["config"])
        config["sources"] = {k: "/job/" + v for k, v in config["sources"].items()}
        config["competitors"] = {k: "/job/" + v for k, v in config.get("competitors", {}).items()}
        from benchmarks.bf16_device import metadata
        actual = metadata()
        if actual["capability"] != {"H100": [9, 0], "B200": [10, 0]}[config["gpu"]]:
            raise RuntimeError(f"GPU substitution: {actual}")
        if actual["torch"] != "2.13.0+cu130" or actual["triton"] != "3.7.1":
            raise RuntimeError(f"runtime substitution: {actual}")
        if config.get("kind") in ("qualification", "distributed"):
            runner_source = source_digest("/job/runner/src/attnres")
            candidate_source = source_digest(config["sources"]["candidate"] + "/src/attnres")
            if runner_source["sha256"] != candidate_source["sha256"]:
                raise RuntimeError("qualification runner differs from the selected candidate")
            if config["kind"] == "qualification":
                from benchmarks.bf16_qualification import run_qualification
                report = run_qualification(config, checkpoint)
                report["identity"] = candidate_source
                checkpoint(report)
                return json.loads(json.dumps(report, default=str))
            report = _distributed(config, job, root, checkpoint)
            report["identity"] = candidate_source
            checkpoint(report)
            return report
        if config.get("kind", "operator") == "operator":
            return json.loads(json.dumps(run_operator(config, checkpoint), default=str))
        if config.get("kind") == "alias_diagnostic":
            from benchmarks.bf16_alias_diagnostic import run_diagnostic
            return json.loads(json.dumps(run_diagnostic(config, checkpoint), default=str))
        if config.get("optimizer_source"):
            config["optimizer_source"] = "/job/optimizer"
        return _training(config, root, checkpoint)
    except Exception as exc:
        failure = {"status": "failed", "job": job,
                   "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        checkpoint(failure)
        return failure
    finally:
        # Preserve compilation work even when a later measurement fails.
        save_compiler_cache()
        volume.commit()


def _training(config, root, checkpoint):
    """Isolate compiler state per cell while retaining paired arms together."""
    import subprocess
    report = {"kind": "training", "status": "running", "config": config,
              "results": [], "import_failures": {}, "process_failures": []}
    for case_index, case in enumerate(config["cases"]):
        for seed in config["seeds"]:
            cell = root / f"cell-{case_index}-{seed}"
            cell.mkdir(exist_ok=True)
            specification = {**config, "cases": [case], "seeds": [seed]}
            config_file, output = cell / "config.json", cell / "report.json"
            config_file.write_text(json.dumps(specification) + "\n")
            report["in_progress"] = {"case": case, "seed": seed}
            checkpoint(report)
            command = [sys.executable, "-X", "faulthandler", "-m", "benchmarks.bf16_training",
                       "--config", str(config_file), "--output", str(output)]
            with (cell / "process.log").open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                modified = None
                while process.poll() is None:
                    if output.exists() and output.stat().st_mtime_ns != modified:
                        modified = output.stat().st_mtime_ns
                        partial = json.loads(output.read_text())
                        report["in_progress"] = partial.get("in_progress", {"case": case, "seed": seed})
                        checkpoint(report)
                    volume.commit()
                    time.sleep(10)
            result = json.loads(output.read_text()) if output.exists() else {}
            for field in ("identities", "runtime", "dynamo"):
                if field in result:
                    if field == "identities" and field in report and report[field] != result[field]:
                        raise RuntimeError("source identities changed between isolated cells")
                    report[field] = result[field]
            report["import_failures"].update(result.get("import_failures", {}))
            if process.returncode or result.get("status") != "complete":
                failure = {"case": case, "seed": seed, "exit_code": process.returncode,
                           "error": result.get("error", "isolated training process failed"),
                           "log": str(cell.relative_to(root) / "process.log")}
                report["process_failures"].append(failure)
                report["results"].append({"case": case, "seed": seed, "arms": {},
                                          "requested_backends": case.get("backends", []),
                                          "error": failure})
            else:
                report["results"].extend(result["results"])
            report.pop("in_progress", None)
            checkpoint(report)
    report["status"] = "complete"
    checkpoint(report)
    return report


def _distributed(config, job, root, checkpoint):
    import subprocess
    import torch
    if job["gpu_count"] != 8 or torch.cuda.device_count() != 8:
        raise RuntimeError("distributed qualification requires exactly eight GPUs")
    config = dict(config)
    if config.get("optimizer_source"):
        config["optimizer_source"] = "/job/optimizer"
    config_file = root / "distributed-config.json"
    config_file.write_text(json.dumps(config, indent=2) + "\n")
    output = root / "distributed-report.json"
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               "--nproc-per-node=8", "-m", "benchmarks.bf16_qualification_distributed",
               "--config", str(config_file), "--output", str(output)]
    log = root / "distributed.log"
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        while process.poll() is None:
            if output.exists():
                checkpoint(json.loads(output.read_text()))
            volume.commit()
            time.sleep(15)
    report = json.loads(output.read_text()) if output.exists() else {"status": "failed"}
    report["exit_code"] = process.returncode
    if process.returncode:
        report["status"] = "failed"
        report["log_tail"] = log.read_text()[-12000:]
    checkpoint(report)
    return report


@app.function(image=image, gpu="H100!", cpu=(CPU_CORES, CPU_CORES),
              memory=(MEMORY_MIB, MEMORY_MIB), timeout=TIMEOUT_S,
              max_containers=1, min_containers=0, buffer_containers=0, retries=0,
              volumes={"/evidence": volume})
def h100(job):
    return _remote(job)


@app.function(image=image, gpu="B200", cpu=(CPU_CORES, CPU_CORES),
              memory=(MEMORY_MIB, MEMORY_MIB), timeout=TIMEOUT_S,
              max_containers=1, min_containers=0, buffer_containers=0, retries=0,
              volumes={"/evidence": volume})
def b200(job):
    return _remote(job)


@app.function(image=image, gpu="H100!:8", cpu=(CPU_CORES, CPU_CORES),
              memory=(MEMORY_MIB, MEMORY_MIB), timeout=TIMEOUT_S,
              max_containers=1, min_containers=0, buffer_containers=0, retries=0,
              volumes={"/evidence": volume})
def h100_distributed(job):
    return _remote(job)


@app.function(image=image, gpu="B200:8", cpu=(CPU_CORES, CPU_CORES),
              memory=(MEMORY_MIB, MEMORY_MIB), timeout=TIMEOUT_S,
              max_containers=1, min_containers=0, buffer_containers=0, retries=0,
              volumes={"/evidence": volume})
def b200_distributed(job):
    return _remote(job)


def _digest(root):
    paths = sorted(Path(root).rglob("*.py"))
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths if "__pycache__" not in p.parts}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def _copy_tree(source, destination):
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", ".pytest_cache", ".venv", "results", "build", "dist", "*.egg-info"))


def prepare(args):
    config = json.loads(Path(args.config).read_text())
    if args.gpus == 8 and config.get("kind") != "distributed":
        raise ValueError("eight GPUs are reserved for exclusive distributed qualification")
    if args.gpus != 8 and config.get("kind") == "distributed":
        raise ValueError("distributed qualification requires --gpus 8")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job_id = f"{stamp}-{args.gpu.lower()}-{args.name}"
    snapshot = WORK / "snapshots" / job_id
    snapshot.mkdir(parents=True)
    _copy_tree(PROJECT, snapshot / "runner")
    sources = {}
    for value in args.source:
        name, path = value.split("=", 1)
        if not name.replace("_", "").isalnum():
            raise ValueError("source names must be alphanumeric")
        _copy_tree(Path(path).resolve(), snapshot / "sources" / name)
        sources[name] = f"sources/{name}"
    competitors = {}
    for value in args.competitor:
        name, path = value.split("=", 1)
        if name not in ("fla", "liger", "legacy", "catswe", "hydra"):
            raise ValueError("unsupported competitor")
        _copy_tree(Path(path).resolve(), snapshot / "competitors" / name)
        competitors[name] = f"competitors/{name}"
    config.update(gpu=args.gpu, sources=sources, competitors=competitors)
    hashes = {"runner": _digest(snapshot / "runner")}
    hashes.update({p: _digest(snapshot / p) for p in sources.values()})
    hashes.update({p: _digest(snapshot / p) for p in competitors.values()})
    if args.optimizer_source:
        optimizer_root = Path(args.optimizer_source).resolve()
        (snapshot / "optimizer").mkdir()
        shutil.copy2(optimizer_root / "optimizer.py", snapshot / "optimizer" / "optimizer.py")
        _copy_tree(optimizer_root / "muon", snapshot / "optimizer" / "muon")
        config["optimizer_source"] = "optimizer"
        hashes["optimizer"] = _digest(snapshot / "optimizer")
    job = {"id": job_id, "config": config, "hashes": hashes,
           "stage": args.stage, "timeout_s": args.timeout, "gpu_count": args.gpus,
           "cpu_cores": 32 if args.gpus == 8 else 8,
           "memory_mib": 262144 if args.gpus == 8 else 65536}
    (snapshot / "job.json").write_text(json.dumps(job, indent=2) + "\n")
    print(snapshot)
    return snapshot


def reserve(job):
    ledger_path = WORK / "ledger.json"
    # 300 seconds startup allowance in addition to the full execution timeout.
    rate = {"H100": .001097, "B200": .001736}[job["config"]["gpu"]] * job["gpu_count"]
    rate += job["cpu_cores"] * .0000131 + (job["memory_mib"] / 1024) * .00000222
    bound = (job["timeout_s"] + 300) * rate * 1.1
    with ledger_path.open("r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        data = json.load(stream)
        if any(x["id"] == job["id"] for x in data["jobs"]):
            raise RuntimeError("job already admitted; retrieve durable evidence instead of rerunning")
        active = [x for x in data["jobs"] if x["status"] in ("reserved", "running")]
        if any(x.get("gpu_count", 1) > 1 or job["gpu_count"] > 1 or
               x["gpu"] == job["config"]["gpu"] for x in active):
            raise RuntimeError("GPU concurrency limit: reconcile the active job before admission")
        committed = sum(x["reserved_usd"] for x in data["jobs"])
        if committed + bound > data["cap_usd"]:
            raise RuntimeError("campaign spending cap would be exceeded")
        stages = {"baseline": 80, "experiments": 220, "confirmation": 140, "reserve": 60}
        spent = sum(x["reserved_usd"] for x in data["jobs"] if x["stage"] == job["stage"])
        if spent + bound > stages[job["stage"]]:
            raise RuntimeError("stage reservation cap would be exceeded")
        # Reservations are never automatically refunded, including failed jobs.
        data["jobs"].append({"id": job["id"], "stage": job["stage"],
            "gpu": job["config"]["gpu"], "reserved_usd": bound,
            "gpu_count": job["gpu_count"],
            "status": "reserved", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        stream.seek(0); json.dump(data, stream, indent=2); stream.write("\n"); stream.truncate()
    return bound


def update_job(job_id, **fields):
    with (WORK / "ledger.json").open("r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        data = json.load(stream)
        entry = next(x for x in data["jobs"] if x["id"] == job_id)
        entry.update(fields)
        stream.seek(0); json.dump(data, stream, indent=2); stream.write("\n"); stream.truncate()


def run(snapshot):
    job = json.loads((snapshot / "job.json").read_text())
    if (job.get("cpu_cores"), job.get("memory_mib")) != (CPU_CORES, MEMORY_MIB):
        raise RuntimeError("snapshot resources differ from the bounded launcher; prepare a new job")
    reserve(job)
    result_dir = WORK / "results" / job["id"]
    result_dir.mkdir(parents=True)
    shutil.copy2(snapshot / "job.json", result_dir / "job.json")
    with app.run(detach=True):
        fn = {("H100", 1): h100, ("B200", 1): b200,
              ("H100", 8): h100_distributed, ("B200", 8): b200_distributed}[(job["config"]["gpu"], job["gpu_count"])]
        call = fn.spawn(job)
        update_job(job["id"], status="running", call_id=call.object_id, app_id=app.app_id)
        (result_dir / "call.json").write_text(json.dumps({"call_id": call.object_id,
                                                         "app_id": app.app_id}) + "\n")
        print(json.dumps({"job": job["id"], "call_id": call.object_id,
                          "app_id": app.app_id}), flush=True)
        report = call.get()
        (result_dir / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        update_job(job["id"], status=report["status"], elapsed_s=report.get("elapsed_s"),
                   completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        print(json.dumps({"job": job["id"], "status": report["status"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("init", help="create a fresh capped ledger without renting GPUs")
    p.add_argument("--cap", type=float, default=500)
    p = sub.add_parser("prepare")
    p.add_argument("--config", required=True)
    p.add_argument("--source", action="append", required=True)
    p.add_argument("--competitor", action="append", default=[])
    p.add_argument("--optimizer-source")
    p.add_argument("--gpus", type=int, choices=[1, 8], default=1)
    p.add_argument("--timeout", type=int, choices=range(600, 3601), default=2400)
    p.add_argument("--gpu", choices=["H100", "B200"], required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--stage", choices=["baseline", "experiments", "confirmation", "reserve"], required=True)
    p = sub.add_parser("run"); p.add_argument("snapshot")
    args = parser.parse_args()
    if args.action == "init":
        if not 0 < args.cap <= 500:
            raise ValueError("cap must be in (0, 500] USD")
        WORK.mkdir(parents=True, exist_ok=True)
        with (WORK / "ledger.json").open("x") as stream:
            json.dump({"cap_usd": args.cap, "jobs": []}, stream, indent=2)
            stream.write("\n")
        print(WORK / "ledger.json")
    elif args.action == "prepare":
        prepare(args)
    else:
        path = Path(args.snapshot).resolve()
        frozen_launcher = path / "runner" / "benchmarks" / "bf16_modal.py"
        if SNAPSHOT.resolve() != path or Path(__file__).resolve() != frozen_launcher:
            os.environ["ATTNRES_JOB_SNAPSHOT"] = str(path)
            os.environ["ATTNRES_CAMPAIGN_WORK"] = str(WORK.resolve())
            os.execv(sys.executable, [sys.executable, str(frozen_launcher), "run", str(path)])
        run(path)


if __name__ == "__main__":
    main()
