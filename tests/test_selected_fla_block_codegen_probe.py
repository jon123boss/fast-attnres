"""CPU/static coverage for the external selected FLA Block codegen probe."""

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import selected_fla_block_codegen_probe as probe


ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = (
    ROOT / "benchmarks" / "selected_fla_block_codegen_probe.py"
)


class FakeTensor:
    def __init__(self, shape, dtype="torch.float32"):
        self.shape = shape
        self.dtype = dtype


def test_probe_has_no_import_time_gpu_or_timing_dependencies():
    tree = ast.parse(PROBE_PATH.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    )
    assert not imported.intersection({"torch", "triton", "modal", "time"})
    source = PROBE_PATH.read_text(encoding="utf-8")
    assert "--allow-gpu" in source
    assert "torch.cuda.Event" not in source
    assert "do_bench" not in source
    assert "attnres import" not in source


def test_exact_candidate_scope_and_production_matrix_are_frozen():
    assert probe.BASE_COMMIT == (
        "134d9d3a206b185a83a6e4d5a5765790ee675201"
    )
    assert probe.IMPLEMENTATION_COMMIT == (
        "25a85a9b99985ac90d69ce636d6b42b5f636a129"
    )
    assert probe.IMPLEMENTATION_TREE == (
        "74f0b86eac24c2ff85ad01d7a77039dcaf84044c"
    )
    assert probe.SOURCE_SHA256 == (
        "2cd7ac89b15faeb13640bff4a7948e437453b69446bfc8c7922511e341843e10"
    )
    assert probe.SOURCE_COUNTS == (2, 9)
    assert probe.RANKS == (128, 512, 1024)
    assert probe.VALUE_WIDTH == 1024
    assert probe.BATCH_SIZE == 2 and probe.TOKEN_COUNT == 2048
    assert probe.AUTOTUNER_KEY_FIELDS == ("L2", "D", "R")
    assert set(probe.HARDWARE) == {"H100", "B200"}


@pytest.mark.parametrize(
    "direction,suffixes",
    [
        ("forward", probe.FORWARD_DTYPE_SUFFIXES),
        ("backward", probe.BACKWARD_DTYPE_SUFFIXES),
    ],
)
@pytest.mark.parametrize("source_count,rank,l2", [(2, 128, 8), (9, 512, 16)])
def test_expected_keys_contain_exact_dtype_suffixes(
    direction, suffixes, source_count, rank, l2
):
    key = probe._expected_tuning_key(direction, source_count, rank)
    assert key[:3] == (l2, 1024, rank)
    assert key[3:] == suffixes
    constants = probe._expected_launch_constants(
        direction, source_count, rank, 4, 0
    )
    assert constants["L2"] == l2
    if direction == "forward":
        assert constants["ROW_STRIDES"] == (1024,) * l2
    else:
        assert constants["BLOCK_PREFIX"] == probe._next_power_of_two(
            max(1, probe.VALUE_WIDTH - rank)
        )
        assert constants["VALUE_ROW_STRIDES"] == (1024,) * l2
        assert constants["GRAD_VALUE_ROW_STRIDES"] == (1024,) * l2


def test_autotuner_binding_reconstructs_forward_and_backward_keys():
    forward_target = SimpleNamespace(
        arg_names=[
            "values",
            "query",
            "output",
            "saved_mixed",
            "saved_inv_rms",
            "saved_logit",
            "saved_lse",
            "count",
            "sources",
            "eps",
            "scale",
            "D",
            "R",
            "BLOCK_D",
            "BLOCK_R",
            "BL",
            "LAYOUT_FAMILY",
            "QUERY_STRIDE",
            "OUTPUT_ROW_STRIDE",
            "OUTPUT_D_STRIDE",
            "L2",
            "ROW_STRIDES",
            "FEATURE_STRIDES",
        ],
        keys=("L2", "D", "R"),
    )
    forward_args = (
        (FakeTensor((2, 2048, 1024), "torch.bfloat16"),),
        FakeTensor((128,)),
        FakeTensor((2, 2048, 1024), "torch.bfloat16"),
        FakeTensor((4096, 1024)),
        FakeTensor((2, 4096)),
        FakeTensor((2, 4096)),
        FakeTensor((4096,)),
        4096,
        2,
        2**-23,
        1.0,
    )
    key, binding = probe._autotuner_call_binding(
        forward_target,
        forward_args,
        {
            "L2": 8,
            "D": 1024,
            "R": 128,
            "BLOCK_D": 1024,
            "BLOCK_R": 128,
            "BL": 4,
            "LAYOUT_FAMILY": 0,
        },
    )
    assert key == probe._expected_tuning_key("forward", 2, 128)
    assert binding["dtype_suffixes"] == list(probe.FORWARD_DTYPE_SUFFIXES)
    assert binding["values"]["values"]["length"] == 1

    backward_target = SimpleNamespace(
        arg_names=[
            "values",
            "query",
            "saved_mixed",
            "grad_output",
            "saved_inv_rms",
            "saved_logit",
            "saved_lse",
            "grad_values",
            "grad_query_partial",
            "count",
            "sources",
            "scale",
            "D",
            "R",
            "BLOCK_D",
            "BLOCK_R",
            "BL",
            "LAYOUT_FAMILY",
            "QUERY_STRIDE",
            "GRAD_OUTPUT_ROW_STRIDE",
            "GRAD_OUTPUT_D_STRIDE",
            "L2",
            "VALUE_ROW_STRIDES",
            "VALUE_FEATURE_STRIDES",
            "GRAD_VALUE_ROW_STRIDES",
            "GRAD_VALUE_FEATURE_STRIDES",
        ],
        keys=("L2", "D", "R"),
    )
    backward_args = (
        (FakeTensor((2, 2048, 1024), "torch.bfloat16"),),
        FakeTensor((128,)),
        FakeTensor((4096, 1024)),
        FakeTensor((2, 2048, 1024), "torch.bfloat16"),
        FakeTensor((2, 4096)),
        FakeTensor((2, 4096)),
        FakeTensor((4096,)),
        (FakeTensor((2, 2048, 1024), "torch.bfloat16"),),
        FakeTensor((4096, 128)),
        4096,
        2,
        1.0,
    )
    backward_key, backward_binding = probe._autotuner_call_binding(
        backward_target,
        backward_args,
        {
            "L2": 8,
            "D": 1024,
            "R": 128,
            "BLOCK_D": 1024,
            "BLOCK_R": 128,
            "BL": 4,
            "LAYOUT_FAMILY": 0,
        },
    )
    assert backward_key == probe._expected_tuning_key("backward", 2, 128)
    assert backward_binding["dtype_suffixes"] == list(
        probe.BACKWARD_DTYPE_SUFFIXES
    )


def test_observer_does_not_read_artifacts_or_wrap_internal_bench(monkeypatch):
    class FakeAutotuner:
        def __init__(self):
            self.arg_names = ["L2", "D", "R"]
            self.keys = ("L2", "D", "R")

        def run(self, *args, **kwargs):
            return object()

    target = FakeAutotuner()
    observations = {"calls": []}
    restore = probe._install_autotuner_observer(
        target, observations, "backward"
    )
    try:
        monkeypatch.setattr(
            probe,
            "_record_codegen",
            lambda *args: (_ for _ in ()).throw(
                AssertionError("artifact I/O in observer")
            ),
        )
        result = target.run(L2=8, D=1024, R=128)
    finally:
        restore()
    assert result is not None
    assert len(observations["calls"]) == 1
    assert observations["calls"][0]["direction"] == "backward"
    assert observations["calls"][0]["status"] == "returned"
    assert type(target).run.__name__ == "run"


def test_selected_config_matches_exact_key_and_config_identity():
    class Config:
        kwargs = {"BL": 2, "LAYOUT_FAMILY": 0}
        num_warps = 4
        num_stages = 2
        num_ctas = 1
        maxnreg = None
        pre_hook = None
        ir_override = None

        def all_kwargs(self):
            return {
                **self.kwargs,
                "num_warps": self.num_warps,
                "num_stages": self.num_stages,
                "num_ctas": self.num_ctas,
            }

    config = Config()
    config1 = Config()
    config1.kwargs = {"BL": 2, "LAYOUT_FAMILY": 1}
    key = probe._expected_tuning_key("forward", 2, 128)
    target = SimpleNamespace(
        cache={key: config},
        best_config=config,
        configs=[config, config1],
        configs_timings={key: [1.0]},
    )
    selected = probe._selected_config(target, key, "forward")
    assert selected["key"] == list(key)
    assert selected["block"] == 2
    assert selected["layout_family"] == 0
    assert selected["candidate_count"] == 2


def test_backward_selected_config_requires_and_records_three_family_inventory():
    class Config:
        def __init__(self, family):
            self.kwargs = {"BL": 2, "LAYOUT_FAMILY": family}
            self.num_warps = 4
            self.num_stages = 2
            self.num_ctas = 1
            self.maxnreg = None
            self.pre_hook = None
            self.ir_override = None

        def all_kwargs(self):
            return {
                **self.kwargs,
                "num_warps": self.num_warps,
                "num_stages": self.num_stages,
                "num_ctas": self.num_ctas,
            }

    configs = [Config(family) for family in (0, 1, 2)]
    key = probe._expected_tuning_key("backward", 9, 512)
    target = SimpleNamespace(
        cache={key: configs[1]},
        best_config=configs[1],
        configs=configs,
        configs_timings={key: [1.0]},
    )
    selected = probe._selected_config(target, key, "backward")
    assert selected["layout_family"] == 1
    assert selected["candidate_count"] == 3
    assert selected["allowed_layout_families"] == [0, 1, 2]
    assert selected["candidate_family_counts"] == {"0": 1, "1": 1, "2": 1}


def test_direct_cache_mapping_rejects_ambiguous_hash_binding():
    compiled = SimpleNamespace(hash="h")
    function = SimpleNamespace(
        device_caches={
            0: (
                {"jit": compiled},
                {
                    (("constexpr", 1024), "options"): "jit",
                    (("constexpr", 1024), "other-options"): "jit",
                },
                None,
                None,
                None,
            )
        }
    )
    with pytest.raises(probe.ProbeError, match="ambiguous specialization mapping"):
        probe._cache_entries(function, 0, "forward")


def test_artifact_record_and_analysis_keep_all_requested_evidence(tmp_path):
    text = (
        "module {\n"
        "  %x = ttg.convert_layout %a\n"
        "  %p = load ptr addrspace(1)\n"
        "  %q = load ptr addrspace(3)\n"
        "  llvm.nvvm.barrier0()\n"
        "  llvm.nvvm.shfl.sync()\n"
        "}\n"
    )
    artifact = tmp_path / "kernel.ttgir"
    artifact.write_text(text, encoding="utf-8")
    compiled = SimpleNamespace(
        asm={"ttir": text, "ttgir": text, "llir": text, "ptx": text},
        metadata_group={"kernel.ttgir": str(artifact)},
    )
    record = probe._artifact_record(compiled, "ttgir", text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    assert record["sha256"] == digest
    assert record["path_sha256"] == digest
    assert record["path_matches_asm"] is True
    analysis = probe._artifact_analysis(text, "ttgir")
    assert analysis["flags"] == {
        "convert_layout": True,
        "shared": True,
        "barrier": True,
        "shuffle": True,
        "local_loads": False,
        "global_loads": True,
    }
    report = probe._record_codegen(compiled)
    assert set(report["artifacts"]) == {"ttir", "ttgir", "llir", "ptx"}
    assert all(
        item["sha256"] == digest for item in report["artifacts"].values()
    )


def test_hardware_scope_requires_matching_name_and_capability():
    class Cuda:
        def __init__(self, name, capability):
            self.name = name
            self.capability = capability

        def get_device_capability(self):
            return self.capability

        def get_device_name(self):
            return self.name

    assert probe._hardware_scope(
        SimpleNamespace(cuda=Cuda("NVIDIA H100", (9, 0))), "H100"
    )["sm"] == "sm90"
    assert probe._hardware_scope(
        SimpleNamespace(cuda=Cuda("NVIDIA B200", (10, 0))), "B200"
    )["sm"] == "sm100"
    with pytest.raises(probe.ProbeError, match="maps to H100"):
        probe._hardware_scope(
            SimpleNamespace(cuda=Cuda("NVIDIA B200", (9, 0))), None
        )


def test_graph_capture_and_directional_artifact_record_are_post_observer():
    source = PROBE_PATH.read_text(encoding="utf-8")
    assert "CUDAGraph" in source
    assert "post_capture_codegen" in source
    assert source.index("post_capture_codegen") > source.index(
        "_install_autotuner_observer"
    )
    assert "forward" in source and "backward" in source


def test_cli_refuses_gpu_without_explicit_consent(tmp_path):
    args = [
        "--source-count",
        "2",
        "--rank",
        "128",
        "--cache-dir",
        str(tmp_path),
    ]
    with pytest.raises(SystemExit, match="allow-gpu"):
        probe.main(args)
