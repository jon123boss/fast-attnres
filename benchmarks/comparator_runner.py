"""Capability-scoped dispatch for the optional matched comparators.

The checked-in protocol describes more comparator cells than any one vendor
can execute.  This module is the executable seam between that declaration and
the CUDA runner.  It deliberately keeps discovery lazy and routes every call
through the adapter module's public ``invoke_comparator`` hook.

There are three independent gates before a timing can enter a result:

* the dependency-free registry says that the cell is applicable;
* the pinned adapter is discovered from the requested source root; and
* the adapter output, every source-value gradient, and the query gradient
  agree with ``validation.oracle.oracle``.

Matched operator rows apply the same gate to the public ``attnres`` candidate,
then time both arms on the same input leaves using the protocol's balanced
ABBA order.  Their ratio is recorded as candidate-over-baseline; no timing
number is materialized when either arm fails qualification.

Unsupported cells are materialized as ``not_applicable`` rows.  They never
allocate inputs or invoke an adapter.  Missing vendors and failed gates remain
visible, but are never counted in an eligible denominator. Block uses the
project's public per-read model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib
import math
from pathlib import Path
from typing import Any

from .comparator_registry import (
    canonical_name,
    capability_for,
    competitor_eligibility,
)


_DISCOVERY_SPECS: dict[str, tuple[str, str, str]] = {
    # canonical name: (adapter module, discovery function, returned name)
    "native_fla_triton_checkpoint1": (
        "benchmarks.competitors",
        "discover_comparators",
        "fla_triton_checkpoint1",
    ),
    "native_fla_gluon": (
        "benchmarks.competitors",
        "discover_comparators",
        "fla_gluon",
    ),
    "native_fla_triton_checkpoint0": (
        "benchmarks.competitors",
        "discover_comparators",
        "fla_triton_checkpoint0",
    ),
    "liger": ("benchmarks.liger", "discover_comparator", "liger"),
    "catswe_phase1": ("benchmarks.catswe", "discover_comparator", "catswe_phase1"),
    "manish_hydra_2p": ("benchmarks.hydra", "discover_comparator", "hydra_2p"),
}

_TIMING_MODES = frozenset({"forward", "forward_backward"})
MATCHED_TIMING_EXCLUDED_WORK = (
    "one independent output/value-gradient/query-gradient qualification before timing",
    "pre-invocation source and query gradient-buffer clearing",
    "tensor-content hashing, device-to-host input copies, and per-round mutation audits are disabled",
    "raw-row validation and paired statistics",
)
_RAW_TIMING_FIELDS = (
    "seed",
    "gpu",
    "round_index",
    "order_index",
    "input_hash",
    "arm",
    "status",
    "latency_ms",
    "failure_phase",
    "failure_reason",
    "failure_at_round",
    "failure_at_order",
)


def _jsonable(value: Any) -> Any:
    """Detach common values without importing Torch on module import."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass
    return str(value)


def _exception(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _inside(path: str | Path, root: Path) -> bool:
    try:
        Path(path).expanduser().resolve().relative_to(root.resolve())
    except (OSError, TypeError, ValueError):
        return False
    return True


def _metadata(comparator: Any) -> dict[str, Any]:
    describe = getattr(comparator, "describe", None)
    if callable(describe):
        try:
            result = describe()
            if isinstance(result, Mapping):
                return _jsonable(dict(result))
        except Exception as exc:
            return {"describe_error": _exception(exc)}
    result: dict[str, Any] = {
        "name": getattr(comparator, "name", None),
        "status": getattr(comparator, "status", None),
        "reason": getattr(comparator, "reason", None),
    }
    vendor_root = getattr(comparator, "vendor_root", None)
    if vendor_root is not None:
        result["vendor_root"] = str(vendor_root)
    vendor_revision = getattr(comparator, "vendor_revision", None)
    if vendor_revision is not None:
        result["vendor_revision"] = str(vendor_revision)
    return _jsonable(result)


def _provenance_error(
    comparator: Any,
    requested_root: str | Path | None,
    metadata: Mapping[str, Any],
) -> str | None:
    """Reject a discovered object that escaped an explicit source root."""

    if requested_root is None:
        return None
    root = Path(requested_root).expanduser().resolve()
    observed = getattr(comparator, "vendor_root", None)
    if observed is None:
        observed = metadata.get("vendor_root", metadata.get("path"))
    if observed is None:
        return f"explicit comparator root {root} produced no observed vendor root"
    if Path(observed).expanduser().resolve() != root:
        return (
            f"comparator resolved outside explicit root {root}: "
            f"observed {Path(observed).expanduser().resolve()}"
        )

    # Adapter identity helpers already perform this check.  Repeating it at
    # the dispatch boundary prevents a future adapter from returning a stale
    # module object while still reporting the requested checkout.
    origins = metadata.get("module_origins")
    if isinstance(origins, Mapping):
        for module_origins in origins.values():
            values = module_origins if isinstance(module_origins, Sequence) and not isinstance(module_origins, (str, bytes)) else (module_origins,)
            for origin in values:
                if origin and not _inside(origin, root):
                    return (
                        f"loaded comparator module origin {origin!r} is outside "
                        f"explicit root {root}"
                    )
    return None


def _root_for(
    name: str,
    vendor_roots: Mapping[str, str | Path] | str | Path | None,
) -> str | Path | None:
    if vendor_roots is None:
        return None
    if isinstance(vendor_roots, (str, Path)):
        capability = capability_for(name)
        return vendor_roots if capability["family"] == "fla" else None
    if not isinstance(vendor_roots, Mapping):
        raise TypeError("vendor_roots must be a path or a mapping")
    canonical = canonical_name(name)
    capability = capability_for(canonical)
    for key in (
        canonical,
        name,
        capability["family"],
        "fla" if capability["family"] == "fla" else "",
        "vendor_root" if capability["family"] == "fla" else "",
        "fla_root" if capability["family"] == "fla" else "",
    ):
        if key and key in vendor_roots:
            return vendor_roots[key]
    return None


@dataclass
class ComparatorRoute:
    """One capability record plus its lazily discovered adapter object."""

    name: str
    adapter_module: str
    adapter_name: str
    capability: dict[str, Any]
    comparator: Any | None = None
    status: str = "missing"
    reason: str | None = None
    metadata: dict[str, Any] | None = None
    requested_root: str | None = None
    invoke_function: Callable[[Any, Any], Any] | None = None

    @property
    def available(self) -> bool:
        available = getattr(self.comparator, "available", True)
        return (
            self.status == "available"
            and self.comparator is not None
            # Adapter metadata is a capability gate, not a truthy hint.  A
            # string such as ``"false"`` must never make an unavailable
            # vendor route executable.
            and type(available) is bool
            and available
        )

    def describe(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "adapter_module": self.adapter_module,
            "adapter_name": self.adapter_name,
            "status": self.status,
            "reason": self.reason,
            "requested_root": self.requested_root,
            "capability": deepcopy(self.capability),
            "discovery": deepcopy(self.metadata or {}),
        }
        return _jsonable(result)

    def applicable(self, values: Any, query: Any) -> tuple[bool, str | None]:
        if not self.available:
            return False, self.reason or f"{self.name} is {self.status}"
        checker = getattr(self.comparator, "applicable", None)
        if not callable(checker):
            return True, None
        try:
            result = checker(values, query)
        except Exception as exc:
            return False, str(exc)
        if not isinstance(result, tuple) or len(result) != 2:
            return False, "adapter applicable() must return (bool, reason)"
        if type(result[0]) is not bool:
            return False, "adapter applicable() must return an actual boolean decision"
        if result[1] is not None and not isinstance(result[1], str):
            return False, "adapter applicable() reason must be a string or None"
        return result[0], result[1]

    def invoke(self, values: Any, query: Any) -> Any:
        if not self.available:
            raise RuntimeError(self.reason or f"{self.name} is {self.status}")
        if self.invoke_function is not None:
            return self.invoke_function(values, query)
        module = importlib.import_module(self.adapter_module)
        invoke = getattr(module, "invoke_comparator", None)
        if callable(invoke):
            return invoke(self.comparator, values, query)
        call = getattr(self.comparator, "call", None)
        if not callable(call):
            raise TypeError(f"{self.name} adapter has no invoke_comparator or callable call")
        return call(values, query)


def _missing_route(
    name: str,
    reason: str,
    *,
    requested_root: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ComparatorRoute:
    module, _discover, adapter_name = _DISCOVERY_SPECS[name]
    return ComparatorRoute(
        name=name,
        adapter_module=module,
        adapter_name=adapter_name,
        capability=capability_for(name),
        status="missing",
        reason=reason,
        metadata=dict(metadata or {}),
        requested_root=str(Path(requested_root).expanduser().resolve()) if requested_root is not None else None,
    )


def discover_registered_comparators(
    project_root: str | Path | None = None,
    vendor_roots: Mapping[str, str | Path] | str | Path | None = None,
    names: Sequence[str] | None = None,
) -> dict[str, ComparatorRoute]:
    """Discover registry adapters without importing unrequested vendors.

    ``vendor_roots`` is authoritative when supplied.  A root is checked again
    against the adapter's observed vendor path and loaded module origins before
    an available route is returned.  The returned mapping contains every
    requested canonical name, including explicit ``missing`` routes.
    """

    selected = tuple(canonical_name(name) for name in (names or _DISCOVERY_SPECS))
    if len(set(selected)) != len(selected):
        raise ValueError("names must contain distinct comparator names")
    root = str(Path(project_root).expanduser().resolve()) if project_root is not None else None
    module_results: dict[tuple[str, str], Mapping[str, Any]] = {}
    routes: dict[str, ComparatorRoute] = {}
    for name in selected:
        module_name, discover_name, adapter_name = _DISCOVERY_SPECS[name]
        requested_root = _root_for(name, vendor_roots)
        if vendor_roots is not None and requested_root is None:
            # Once a caller supplies a root map, it is an explicit provenance
            # contract.  Do not let an omitted family fall back to an ambient
            # checkout discovered through environment variables or defaults.
            routes[name] = _missing_route(
                name,
                f"no explicit vendor root supplied for {capability_for(name)['family']}",
            )
            continue
        key = (module_name, str(requested_root) if requested_root is not None else "")
        if key not in module_results:
            try:
                module = importlib.import_module(module_name)
                discover = getattr(module, discover_name)
                if not callable(discover):
                    raise TypeError(f"{module_name}.{discover_name} is not callable")
                discovered = discover(project_root=root, vendor_root=requested_root)
                if discover_name == "discover_comparators":
                    if not isinstance(discovered, Mapping):
                        raise TypeError("adapter discovery must return a mapping")
                    module_results[key] = discovered
                else:
                    module_results[key] = {adapter_name: discovered}
            except Exception as exc:
                module_results[key] = {"__error__": _exception(exc)}
        discovered = module_results[key]
        if "__error__" in discovered:
            routes[name] = _missing_route(
                name,
                f"comparator discovery failed: {discovered['__error__']['type']}: {discovered['__error__']['message']}",
                requested_root=requested_root,
            )
            continue
        # The protocol identity and the adapter's returned identity are
        # deliberately separate.  Resolve only the explicit adapter name;
        # falling back to a protocol alias could silently run a different
        # implementation while the result still claimed the canonical arm.
        comparator = discovered.get(adapter_name)
        if comparator is None:
            routes[name] = _missing_route(
                name,
                f"adapter did not return comparator {adapter_name!r}",
                requested_root=requested_root,
            )
            continue
        metadata = _metadata(comparator)
        status = str(getattr(comparator, "status", metadata.get("status", "missing")))
        reason = getattr(comparator, "reason", None) or metadata.get("reason")
        provenance_error = _provenance_error(comparator, requested_root, metadata)
        if provenance_error is not None:
            status, reason = "missing", provenance_error
        routes[name] = ComparatorRoute(
            name=name,
            adapter_module=module_name,
            adapter_name=adapter_name,
            capability=capability_for(name),
            comparator=comparator,
            status=status,
            reason=str(reason) if reason else None,
            metadata=metadata,
            requested_root=(
                str(Path(requested_root).expanduser().resolve())
                if requested_root is not None
                else None
            ),
        )
    return routes


def _route_for(
    name: str,
    route: ComparatorRoute | None,
    *,
    project_root: str | Path | None = None,
    vendor_roots: Mapping[str, str | Path] | str | Path | None = None,
) -> ComparatorRoute:
    canonical = canonical_name(name)
    if route is not None:
        if canonical_name(route.name) != canonical:
            raise ValueError(f"route {route.name!r} does not match {name!r}")
        return route
    return discover_registered_comparators(
        project_root=project_root,
        vendor_roots=vendor_roots,
        names=(canonical,),
    )[canonical]


def _eligibility(name: str, cell: Mapping[str, Any]) -> dict[str, Any]:
    return competitor_eligibility(name, cell)


def _dtype_key(dtype: Any) -> str:
    value = str(dtype).lower().replace("torch.", "")
    return "bf16" if value in {"bf16", "bfloat16"} else "fp32"


def _tolerance(protocol: Mapping[str, Any], dtype: Any) -> dict[str, float]:
    key = _dtype_key(dtype)
    values = protocol.get(key)
    if not isinstance(values, Mapping):
        oracle = protocol.get("oracle", {})
        values = oracle.get("tolerances", {}).get(key, {}) if isinstance(oracle, Mapping) else {}
    if not isinstance(values, Mapping):
        raise ValueError(f"protocol has no {key} tolerance")
    return {"rtol": float(values["rtol"]), "atol": float(values["atol"])}


def _source_inputs(values: Any) -> tuple[Any, ...]:
    import torch

    if isinstance(values, torch.Tensor):
        return (values,)
    if isinstance(values, (list, tuple)) and values and all(isinstance(item, torch.Tensor) for item in values):
        return tuple(values)
    raise TypeError("values must be a tensor or a nonempty tensor sequence")


def _clone_container(values: Any, sources: Sequence[Any]) -> Any:
    """Preserve the caller's packed/list container while cloning its leaves."""

    if not isinstance(values, (list, tuple)):
        return sources[0]
    if isinstance(values, list):
        return list(sources)
    if isinstance(values, tuple):
        return tuple(sources)
    # ``_source_inputs`` rejects all other containers, so this branch is only
    # defensive for tensor-like test doubles.
    return tuple(sources)


def _clone_for_qualification(values: Any, query: Any) -> tuple[Any, Any, Any, Any, tuple[Any, ...], tuple[Any, ...]]:
    import torch

    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a tensor")
    sources = _source_inputs(values)
    actual_sources = tuple(source.detach().clone().requires_grad_(True) for source in sources)
    expected_sources = tuple(source.detach().clone().requires_grad_(True) for source in sources)
    packed_input = isinstance(values, torch.Tensor)
    actual_values = _clone_container(values, actual_sources)
    expected_values = expected_sources[0] if packed_input else torch.stack(expected_sources, dim=0)
    actual_query = query.detach().clone().requires_grad_(True)
    expected_query = query.detach().clone().requires_grad_(True)
    return (
        actual_values,
        actual_query,
        expected_values,
        expected_query,
        actual_sources,
        expected_sources,
    )


def _finite(tensor: Any, label: str) -> None:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{label} must be a tensor")
    if not bool(torch.isfinite(tensor).all().item()):
        raise FloatingPointError(f"{label} contains non-finite values")


def _max_abs(actual: Any, expected: Any) -> float:
    return float((actual.detach().float() - expected.detach().float()).abs().max().item())


def _oracle_input(values: Any) -> Any:
    """Return the packed representation consumed by the independent oracle."""

    import torch

    sources = _source_inputs(values)
    if isinstance(values, torch.Tensor):
        return sources[0]
    # The public operator receives an ordered source list.  Packing is used
    # only for the evaluator and never for the timed candidate/comparator
    # invocation, so the adapter's list preparation remains measurable.
    return torch.stack(tuple(sources), dim=0)


def _oracle_upstream(values: Any, query: Any) -> Any:
    """Build one deterministic upstream with the oracle's exact output shape."""

    import torch
    from validation.oracle import oracle

    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a tensor")
    with torch.no_grad():
        expected = oracle(_oracle_input(values).detach(), query.detach())
    _finite(expected, "oracle output used for timing upstream")
    return torch.ones_like(expected)


def _qualify_callable(
    label: str,
    invoke: Callable[[Any, Any], Any],
    values: Any,
    query: Any,
    protocol: Mapping[str, Any],
    *,
    discovery: Mapping[str, Any] | None = None,
    eligibility: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Qualify one callable against independent output and gradient oracles."""

    import torch
    from validation.oracle import oracle

    (
        actual_values,
        actual_query,
        expected_values,
        expected_query,
        actual_sources,
        expected_sources,
    ) = _clone_for_qualification(values, query)
    actual_output = invoke(actual_values, actual_query)
    expected_output = oracle(expected_values, expected_query)
    if not isinstance(actual_output, torch.Tensor):
        raise TypeError(f"{label} returned {type(actual_output).__name__}; expected a tensor")
    _finite(actual_output, f"{label} output")
    _finite(expected_output, "oracle output")
    tolerance = _tolerance(protocol, actual_output.dtype)
    torch.testing.assert_close(actual_output, expected_output, **tolerance)
    # Keep the implementation and oracle gradient seeds independent.  A
    # custom autograd implementation is allowed to read ``grad_output`` but
    # must never be able to mutate the tensor subsequently consumed by the
    # independent oracle.
    actual_upstream = torch.ones_like(expected_output)
    expected_upstream = actual_upstream.clone()
    actual_gradients = torch.autograd.grad(
        actual_output,
        (*actual_sources, actual_query),
        actual_upstream,
        allow_unused=False,
    )
    expected_gradients = torch.autograd.grad(
        expected_output,
        (*expected_sources, expected_query),
        expected_upstream,
        allow_unused=False,
    )
    value_errors: list[float] = []
    for index, (actual_grad, expected_grad) in enumerate(
        zip(actual_gradients[:-1], expected_gradients[:-1])
    ):
        _finite(actual_grad, f"{label} value gradient {index}")
        _finite(expected_grad, f"oracle value gradient {index}")
        torch.testing.assert_close(actual_grad, expected_grad, **tolerance)
        value_errors.append(_max_abs(actual_grad, expected_grad))
    actual_query_grad, expected_query_grad = actual_gradients[-1], expected_gradients[-1]
    _finite(actual_query_grad, f"{label} query gradient")
    _finite(expected_query_grad, "oracle query gradient")
    torch.testing.assert_close(actual_query_grad, expected_query_grad, **tolerance)
    query_error = _max_abs(actual_query_grad, expected_query_grad)
    result: dict[str, Any] = {
        "status": "qualified",
        "eligible": True,
        "eligible_denominator": bool(
            eligibility is None or eligibility.get("eligible_denominator", True)
        ),
        "checks": {
            "output": {"status": "passed", "max_abs": _max_abs(actual_output, expected_output)},
            "values_gradient": {
                "status": "passed",
                "source_count": len(value_errors),
                "max_abs": value_errors,
            },
            "query_gradient": {"status": "passed", "max_abs": query_error},
        },
        "output_max_abs": _max_abs(actual_output, expected_output),
        "gradient_max_abs": [*value_errors, query_error],
        "tolerance": tolerance,
    }
    if discovery is not None:
        result["discovery"] = _jsonable(dict(discovery))
    if eligibility is not None:
        result["eligibility"] = _jsonable(dict(eligibility))
    return result


def qualify_comparator(
    name: str,
    cell: Mapping[str, Any],
    values: Any,
    query: Any,
    protocol: Mapping[str, Any],
    *,
    route: ComparatorRoute | None = None,
    project_root: str | Path | None = None,
    vendor_roots: Mapping[str, str | Path] | str | Path | None = None,
) -> dict[str, Any]:
    """Run the independent output/value-gradient/query-gradient gate."""

    canonical = canonical_name(name)
    eligibility = _eligibility(canonical, cell)
    if not eligibility["eligible"]:
        return {
            "status": "not_applicable",
            "eligible": False,
            "eligible_denominator": False,
            "eligibility": eligibility,
            "reason": eligibility["reason"],
        }
    selected = _route_for(
        canonical,
        route,
        project_root=project_root,
        vendor_roots=vendor_roots,
    )
    if not selected.available:
        return {
            "status": "missing",
            "eligible": True,
            "eligible_denominator": False,
            "eligibility": eligibility,
            "discovery": selected.describe(),
            "reason": selected.reason or f"{canonical} is {selected.status}",
        }
    applicable, reason = selected.applicable(values, query)
    if not applicable:
        return {
            "status": "failed",
            "eligible": True,
            "eligible_denominator": False,
            "eligibility": eligibility,
            "discovery": selected.describe(),
            "failure_phase": "adapter_applicability",
            "reason": reason or f"{canonical} rejected the input tensors",
        }
    return _qualify_callable(
        canonical,
        selected.invoke,
        values,
        query,
        protocol,
        discovery=selected.describe(),
        eligibility=eligibility,
    )


def _public_candidate(values: Any, query: Any) -> Any:
    """Call the public per-read AttnRes operator without a cached route."""

    from attnres import attnres

    return attnres(values, query)


def qualify_candidate(
    cell: Mapping[str, Any],
    values: Any,
    query: Any,
    protocol: Mapping[str, Any],
    *,
    candidate: Callable[[Any, Any], Any] | None = None,
) -> dict[str, Any]:
    """Qualify the public candidate on the same values/query as a comparator."""

    candidate_eligibility = _eligibility("native_fla_triton_checkpoint1", cell)
    if not candidate_eligibility["eligible"]:
        return {
            "status": "not_applicable",
            "eligible": False,
            "eligible_denominator": False,
            "reason": "candidate is not a standard R=D cell",
            "eligibility": candidate_eligibility,
        }
    selected = candidate or _public_candidate
    return _qualify_callable(
        "attnres",
        selected,
        values,
        query,
        protocol,
        discovery={
            "name": "attnres",
            "status": "public_candidate",
            "model": "attnres",
            "schedule": "per_read",
        },
        eligibility=candidate_eligibility,
    )


def _validated_eligibility(
    name: str,
    cell: Mapping[str, Any],
    supplied: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return one typed capability decision, failing closed on bad metadata."""

    if supplied is None:
        result = _eligibility(name, cell)
    elif isinstance(supplied, Mapping):
        result = dict(supplied)
    else:
        raise TypeError("eligibility must be a mapping")
    for field in ("eligible", "eligible_denominator"):
        if field not in result or type(result[field]) is not bool:
            raise ValueError(f"eligibility.{field} must be a boolean")
    if result["eligible_denominator"] and not result["eligible"]:
        raise ValueError(
            "eligibility.eligible_denominator cannot be true for an inapplicable cell"
        )
    return result


def _timing_mode(cell: Mapping[str, Any]) -> str:
    """Resolve a timing mode without allowing a truthy/unknown fallback."""

    value = cell.get("timing_mode", cell.get("operator_mode", "forward"))
    if not isinstance(value, str) or value not in _TIMING_MODES:
        raise ValueError(
            f"timing_mode must be one of {sorted(_TIMING_MODES)!r}; got {value!r}"
        )
    return value


def _complete_timing_payload(
    timing: Mapping[str, Any] | None,
    *,
    cell: Mapping[str, Any] | None = None,
) -> bool:
    """Check the minimum sealed evidence before a row can enter a denominator."""

    if not isinstance(timing, Mapping) or timing.get("status") != "complete":
        return False
    if "timing_mode" not in timing:
        return False
    try:
        timing_mode = _timing_mode(timing)
    except (TypeError, ValueError):
        # Timing payloads carry their own mode.  Never infer it at the
        # denominator boundary, since that can turn an FWB plan into a forward
        # result.
        return False
    if cell is not None:
        try:
            expected_mode = _timing_mode(cell)
        except (TypeError, ValueError):
            return False
        if timing_mode != expected_mode:
            return False
        if cell.get("scope") == "operator" and expected_mode != "forward_backward":
            return False
    raw_rounds = timing.get("rounds")
    raw_warmup = timing.get("warmup")
    if type(raw_rounds) is not int or raw_rounds < 1:
        return False
    if type(raw_warmup) is not int or raw_warmup < 0:
        return False
    raw_arms = timing.get("arms")
    if raw_arms is None:
        # ``time_comparator`` is a single-arm API; its writer now records the
        # one arm explicitly.  Missing arms are therefore malformed.
        return False
    if (
        isinstance(raw_arms, (str, bytes))
        or not isinstance(raw_arms, Sequence)
        or not raw_arms
        or any(not isinstance(arm, str) or not arm for arm in raw_arms)
        or len(set(raw_arms)) != len(raw_arms)
    ):
        return False
    samples = timing.get("raw_samples")
    if (
        isinstance(samples, (str, bytes))
        or not isinstance(samples, Sequence)
        or len(samples) != raw_rounds * len(raw_arms)
    ):
        return False
    raw_validation = timing.get("raw_validation")
    if raw_validation is not None and (
        not isinstance(raw_validation, Mapping)
        or raw_validation.get("status") != "passed"
    ):
        return False
    for row in samples:
        if not isinstance(row, Mapping) or any(field not in row for field in _RAW_TIMING_FIELDS):
            return False
        if row.get("status") != "ok" or row.get("arm") not in raw_arms:
            return False
        latency = row.get("latency_ms")
        try:
            finite_latency = math.isfinite(float(latency))
        except (OverflowError, TypeError, ValueError):
            finite_latency = False
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not finite_latency
            or latency <= 0
        ):
            return False
        if type(row.get("seed")) is not int or row.get("gpu") not in {"H100!", "B200"}:
            return False
        if not isinstance(row.get("input_hash"), str) or not row["input_hash"].strip():
            return False
    # A complete-looking payload must also have the sealed order and pairing
    # structure.  The summary path performs this check again against the
    # complete plan, but materialization is itself a denominator boundary and
    # must not accept duplicated or permuted rows when called directly.
    try:
        from .competitor_protocol import validate_raw_samples

        first = samples[0]
        validate_raw_samples(
            samples,
            tuple(raw_arms),
            rounds=raw_rounds,
            seed=first["seed"],
            gpu=first["gpu"],
            planned_eligibility={arm: True for arm in raw_arms},
        )
    except Exception:
        return False
    return True


def materialize_comparison_result(
    cell: Mapping[str, Any],
    *,
    status: str | None = None,
    eligibility: Mapping[str, Any] | None = None,
    route: ComparatorRoute | None = None,
    qualification: Mapping[str, Any] | None = None,
    candidate_qualification: Mapping[str, Any] | None = None,
    timing: Mapping[str, Any] | None = None,
    reason: str | None = None,
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a JSON-safe result row without inventing a latency.

    The capability eligibility flag is authoritative for ``not_applicable``
    rows.  A row only enters ``eligible_denominator`` after qualification and a
    complete set of successful timing samples are both present.
    """

    if not isinstance(cell, Mapping):
        raise TypeError("comparison cell must be a mapping")
    raw_name = str(cell.get("competitor", ""))
    try:
        name = canonical_name(raw_name)
    except (KeyError, ValueError):
        name = raw_name
    eligibility = _validated_eligibility(name, cell, eligibility)
    applicable = eligibility["eligible"]
    if not applicable:
        final_status = "not_applicable"
    else:
        final_status = str(status or "not_run")
    qual = deepcopy(dict(qualification)) if qualification is not None else None
    candidate_qual = (
        deepcopy(dict(candidate_qualification))
        if candidate_qualification is not None
        else None
    )
    timing_report = deepcopy(dict(timing)) if timing is not None else None
    if not applicable:
        # An inapplicable cell was rejected before inputs, qualification, and
        # timing.  Do not let a caller accidentally attach a latency payload to
        # the audit row after that rejection.
        qual = None
        candidate_qual = None
        timing_report = None
    complete_timing = _complete_timing_payload(timing_report, cell=cell)
    qualified = isinstance(qual, Mapping) and qual.get("status") == "qualified"
    candidate_qualified = (
        candidate_qual is None or candidate_qual.get("status") == "qualified"
    )
    eligible_denominator = bool(
        applicable
        and eligibility["eligible_denominator"]
        and qualified
        and candidate_qualified
        and complete_timing
        and final_status in {"complete", "ok"}
    )
    if applicable and final_status in {"complete", "ok"} and not eligible_denominator:
        final_status = "incomplete"
    cell_copy = deepcopy(dict(cell))
    if name != raw_name:
        cell_copy["competitor"] = name
    result: dict[str, Any] = {
        "comparison_cell_id": cell.get("comparison_cell_id", cell.get("cell_id")),
        "cell": cell_copy,
        "competitor": name,
        "status": final_status,
        "eligible": applicable,
        "eligible_denominator": eligible_denominator,
        "eligibility": _jsonable(eligibility),
        "eligibility_reason": str(eligibility.get("reason", "")),
    }
    if route is not None:
        result["discovery"] = route.describe()
    if reason is not None:
        result["reason"] = str(reason)
    if failure is not None:
        result["failure"] = _jsonable(dict(failure))
    if qual is not None:
        result["qualification"] = _jsonable(qual)
    if candidate_qual is not None:
        result["candidate_qualification"] = _jsonable(candidate_qual)
    if timing_report is not None:
        result["timing"] = _jsonable(timing_report)
    return _jsonable(result)


def materialize_comparison_plan(
    plan: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    status: str = "not_run",
    reason: str | None = None,
) -> dict[str, Any]:
    """Materialize every planned and inapplicable protocol cell for a report."""

    planned = list(plan.get("planned", ()))
    not_applicable = list(plan.get("not_applicable", ()))
    rows = [
        materialize_comparison_result(
            cell,
            status=status,
            eligibility=cell.get("eligibility"),
            reason=reason,
        )
        for cell in [*planned, *not_applicable]
    ]
    capability_eligible = sum(
        1
        for cell in [*planned, *not_applicable]
        if type(cell.get("eligible_denominator", False)) is bool
        and cell.get("eligible_denominator") is True
    )
    return {
        "status": "planned",
        "cells": rows,
        "planned": sum(1 for row in rows if row["status"] != "not_applicable"),
        "not_applicable": sum(1 for row in rows if row["status"] == "not_applicable"),
        # This is the predeclared capability denominator.  It intentionally
        # stays separate from rows that later qualify and time successfully.
        "eligible_denominator": capability_eligible,
        "qualified_denominator": sum(1 for row in rows if row["eligible_denominator"]),
        "reason": reason,
    }


def _default_timing_call(function: Callable[[], Any], device: Any = None) -> tuple[float, Any]:
    import torch

    if device is None:
        raise RuntimeError("CUDA timing requires an explicit device")
    if getattr(device, "type", None) != "cuda":
        raise RuntimeError("native comparator timing requires a CUDA device")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.device(device):
        start.record()
        output = function()
        end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end)), output


def _input_hash(values: Any, query: Any) -> str:
    digest = hashlib.sha256()
    # Include the container contract as well as tensor bytes.  A packed
    # [S,N,D] tensor and a source list of [N,D] leaves can represent the same
    # numerical data while exercising different adapter paths.
    digest.update(("packed" if _is_tensor(values) else "source_list").encode())
    tensors = (*_source_inputs(values), query)
    for tensor in tensors:
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        import torch

        digest.update(
            tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def _logical_input_hash(cell: Mapping[str, Any], *, seed: int, gpu: str | None) -> str:
    """Return a pairing ID without reading tensor contents or copying from CUDA.

    Operator inputs are deterministic functions of the sealed cell and seed.
    The report schema still calls this field ``input_hash`` for compatibility,
    but the speed-first runner uses it only to pair the two arms of a round.
    It is deliberately not tensor-integrity evidence.
    """

    fields = (
        cell.get("comparison_cell_id", cell.get("cell_id", "matched_operator")),
        seed,
        gpu,
        cell.get("S"),
        cell.get("N"),
        cell.get("D"),
        cell.get("R"),
        cell.get("dtype"),
        cell.get("timing_mode", cell.get("operator_mode")),
    )
    return hashlib.sha256(repr(fields).encode("utf-8")).hexdigest()


def _is_tensor(value: Any) -> bool:
    """Avoid importing Torch at module scope while recognizing a tensor."""

    import torch

    return isinstance(value, torch.Tensor)


def time_comparator(
    name: str,
    cell: Mapping[str, Any],
    values: Any,
    query: Any,
    qualification: Mapping[str, Any],
    *,
    route: ComparatorRoute | None = None,
    rounds: int = 1,
    warmup: int = 0,
    device: Any = None,
    timing_call: Callable[[Callable[[], Any]], tuple[float, Any]] | None = None,
    seed: int | None = None,
    gpu: str | None = None,
    input_hash: str | None = None,
    project_root: str | Path | None = None,
    vendor_roots: Mapping[str, str | Path] | str | Path | None = None,
) -> dict[str, Any]:
    """Time only after qualification, with the adapter boundary inside events."""

    canonical = canonical_name(name)
    eligibility = _eligibility(canonical, cell)
    if not eligibility["eligible"]:
        return materialize_comparison_result(
            cell, status="not_applicable", eligibility=eligibility, reason=eligibility["reason"]
        )
    if qualification.get("status") != "qualified":
        return materialize_comparison_result(
            cell,
            status="failed",
            eligibility=eligibility,
            qualification=qualification,
            reason="timing is blocked until output, value-gradient, and query-gradient qualification passes",
        )
    if type(rounds) is not int or rounds < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError("rounds must be positive and warmup must be nonnegative")
    timing_mode = _timing_mode(cell)
    if timing_mode != "forward":
        return materialize_comparison_result(
            cell,
            status="incomplete",
            eligibility=eligibility,
            qualification=qualification,
            timing={
                "status": "incomplete",
                "timing_mode": timing_mode,
                "warmup": warmup,
                "rounds": rounds,
                "arms": (canonical,),
                "raw_samples": [],
                "failure": {
                    "phase": "timing_mode",
                    "reason": (
                        "time_comparator only implements a forward-only boundary; "
                        "use time_matched_pair for forward_backward cells"
                    ),
                },
            },
            reason=(
                "time_comparator only implements a forward-only boundary; "
                "use time_matched_pair for forward_backward cells"
            ),
        )
    selected = _route_for(
        canonical,
        route,
        project_root=project_root,
        vendor_roots=vendor_roots,
    )
    if not selected.available:
        return materialize_comparison_result(
            cell,
            status="missing",
            eligibility=eligibility,
            route=selected,
            qualification=qualification,
            reason=selected.reason or f"{canonical} is {selected.status}",
        )
    applicable, applicable_reason = selected.applicable(values, query)
    if not applicable:
        return materialize_comparison_result(
            cell,
            status="failed",
            eligibility=eligibility,
            route=selected,
            qualification=qualification,
            reason=applicable_reason or "adapter rejected timing inputs",
        )
    timer = timing_call or (lambda function: _default_timing_call(function, device))
    hash_value = input_hash or _input_hash(values, query)
    samples: list[dict[str, Any]] = []
    failed_at: tuple[int, int] | None = None
    failure_reason: str | None = None
    for index in range(warmup):
        try:
            selected.invoke(values, query)
            if device is not None and getattr(device, "type", None) == "cuda":
                import torch

                torch.cuda.synchronize(device)
        except Exception as exc:
            timing = {
                "status": "failed",
                "timing_mode": timing_mode,
                "warmup": warmup,
                "rounds": rounds,
                "arms": (canonical,),
                "timing_boundary": "adapter invocation, including source stacking/contiguous preparation, is inside each timing event",
                "raw_samples": samples,
                "failure": {"phase": "warmup", "index": index, "error": _exception(exc)},
            }
            return materialize_comparison_result(
                cell,
                status="incomplete",
                eligibility=eligibility,
                route=selected,
                qualification=qualification,
                timing=timing,
                reason=str(exc),
            )
    for index in range(rounds):
        row: dict[str, Any] = {
            "seed": seed,
            "gpu": gpu,
            "round_index": index,
            "order_index": 0,
            "arm": canonical,
            "eligible": True,
            "status": "failed",
            "latency_ms": None,
            "input_hash": hash_value,
            "failure_phase": None,
            "failure_reason": None,
            "failure_at_round": None,
            "failure_at_order": None,
        }
        if failed_at is not None:
            row.update(
                status="skipped_due_to_failure",
                failure_phase="timing_skip",
                failure_reason=(
                    f"skipped after timing failure at round {failed_at[0]} "
                    f"order {failed_at[1]}: {failure_reason}"
                ),
                failure_at_round=failed_at[0],
                failure_at_order=failed_at[1],
            )
            samples.append(row)
            continue
        try:
            # ``selected.invoke`` is created inside the timer callback.  The
            # callback therefore includes adapter-owned stack/contiguous cost.
            elapsed, output = timer(lambda: selected.invoke(values, query))
            import torch

            if not isinstance(output, torch.Tensor):
                raise TypeError("timed comparator returned a non-tensor")
            _finite(output, f"{canonical} timed output")
            if (
                not isinstance(elapsed, (int, float))
                or isinstance(elapsed, bool)
                or not math.isfinite(float(elapsed))
                or elapsed <= 0
            ):
                raise ValueError("timing callback must return a positive latency in milliseconds")
            row.update(status="ok", latency_ms=float(elapsed))
        except Exception as exc:
            failed_at = (index, 0)
            failure_reason = str(exc) or type(exc).__name__
            row.update(
                failure_phase="timing",
                failure_reason=failure_reason,
                failure_at_round=index,
                failure_at_order=0,
                error=_exception(exc),
            )
        samples.append(row)
    raw_validation: dict[str, Any]
    if seed is None or gpu not in {"H100!", "B200"}:
        raw_validation = {
            "status": "failed",
            "reason": "standalone timing rows require an integer seed and protocol GPU selector",
        }
    else:
        try:
            from .competitor_protocol import validate_raw_samples

            validate_raw_samples(
                samples,
                (canonical,),
                rounds=rounds,
                seed=seed,
                gpu=gpu,
                planned_eligibility={canonical: True},
            )
        except Exception as exc:
            raw_validation = {"status": "failed", "error": _exception(exc)}
        else:
            raw_validation = {"status": "passed"}
    complete = (
        len(samples) == rounds
        and all(row["status"] == "ok" for row in samples)
        and raw_validation["status"] == "passed"
    )
    timing = {
        "status": "complete" if complete else "incomplete",
        "timing_mode": timing_mode,
        "warmup": warmup,
        "rounds": rounds,
        "arms": (canonical,),
        "timing_boundary": "adapter invocation, including source stacking/contiguous preparation, is inside each timing event",
        "adapter_stack_in_timing": True,
        "raw_samples": samples,
        "raw_validation": raw_validation,
    }
    return materialize_comparison_result(
        cell,
        status="complete" if complete else "incomplete",
        eligibility=eligibility,
        route=selected,
        qualification=qualification,
        timing=timing,
        reason=None if complete else "timing did not produce a complete successful sample set",
    )


def _timing_state(values: Any, query: Any, upstream: Any | None = None) -> tuple[Any, Any, Any]:
    """Create one private leaf pair and upstream for a timed arm.

    The caller's values and query are immutable templates.  Each arm gets
    detached leaves, while the upstream is cloned from one oracle-shaped
    tensor so packed and source-list inputs cannot accidentally use different
    backward shapes.
    """

    import torch

    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a tensor")
    sources = _source_inputs(values)
    cloned_sources = tuple(
        source.detach().clone().requires_grad_(True) for source in sources
    )
    timed_values = _clone_container(values, cloned_sources)
    timed_query = query.detach().clone().requires_grad_(True)
    if upstream is None:
        upstream = _oracle_upstream(values, query)
    if not isinstance(upstream, torch.Tensor):
        raise TypeError("timing upstream must be a tensor")
    return timed_values, timed_query, upstream.detach().clone()


def _timed_arm_call(
    invoke: Callable[[Any, Any], Any],
    values: Any,
    query: Any,
    upstream: Any,
    timing_mode: str,
) -> Any:
    """Run one arm's exact forward or forward+backward operator boundary."""

    output = invoke(values, query)
    if timing_mode == "forward_backward":
        import torch

        if not isinstance(output, torch.Tensor):
            raise TypeError("forward_backward comparator returned a non-tensor")
        output.backward(upstream)
    return output


def _clear_timing_gradients(values: Any, query: Any) -> None:
    """Clear leaf gradients before, rather than inside, a timed invocation."""

    for source in _source_inputs(values):
        source.grad = None
    query.grad = None


def _pair_report(
    result: dict[str, Any],
    *,
    canonical: str,
    candidate_qualification: Mapping[str, Any] | None = None,
    comparator_qualification: Mapping[str, Any] | None = None,
    timing: Mapping[str, Any] | None = None,
    ratio: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach canonical pair metadata to one materialized comparator row."""

    if candidate_qualification is not None and "candidate_qualification" not in result:
        result["candidate_qualification"] = _jsonable(dict(candidate_qualification))
    result_cell = result.get("cell")
    fallback_timing_mode = (
        str(result_cell.get("timing_mode", result_cell.get("operator_mode", "forward")))
        if isinstance(result_cell, Mapping)
        else "forward"
    )
    result["pair"] = {
        "candidate_arm": "attnres",
        "comparator_arm": canonical,
        "ratio_orientation": "candidate_over_baseline",
        "timing_mode": (
            timing.get("timing_mode") or fallback_timing_mode
            if isinstance(timing, Mapping)
            else fallback_timing_mode
        ),
        "same_inputs": True,
    }
    if isinstance(timing, Mapping):
        orders = timing.get("orders")
        if orders is not None:
            result["pair"]["orders"] = _jsonable(orders)
        if timing.get("input_hash") is not None:
            result["pair"]["input_hash"] = timing["input_hash"]
    if ratio is not None:
        result["pair"]["ratio"] = _jsonable(dict(ratio))
    return _jsonable(result)


def time_matched_pair(
    name: str,
    cell: Mapping[str, Any],
    values: Any,
    query: Any,
    candidate_qualification: Mapping[str, Any],
    comparator_qualification: Mapping[str, Any],
    *,
    route: ComparatorRoute | None = None,
    candidate: Callable[[Any, Any], Any] | None = None,
    rounds: int = 1,
    warmup: int = 0,
    device: Any = None,
    timing_call: Callable[[Callable[[], Any]], tuple[float, Any]] | None = None,
    seed: int = 0,
    gpu: str | None = None,
    protocol: Mapping[str, Any] | None = None,
    project_root: str | Path | None = None,
    vendor_roots: Mapping[str, str | Path] | str | Path | None = None,
) -> dict[str, Any]:
    """Time public AttnRes and a comparator on shared inputs in ABBA order."""

    canonical = canonical_name(name)
    eligibility = _eligibility(canonical, cell)
    if not eligibility["eligible"]:
        return _pair_report(
            materialize_comparison_result(
                cell,
                status="not_applicable",
                eligibility=eligibility,
                reason=eligibility["reason"],
            ),
            canonical=canonical,
        )
    if (
        candidate_qualification.get("status") != "qualified"
        or comparator_qualification.get("status") != "qualified"
    ):
        return _pair_report(
            materialize_comparison_result(
                cell,
                status="failed",
                eligibility=eligibility,
                route=route,
                qualification=comparator_qualification,
                candidate_qualification=candidate_qualification,
                reason="pair timing is blocked until both candidate and comparator pass qualification",
            ),
            canonical=canonical,
            candidate_qualification=candidate_qualification,
        )
    if type(rounds) is not int or rounds < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError("rounds must be positive and warmup must be nonnegative")
    selected = route
    if selected is None:
        selected = _route_for(
            canonical,
            None,
            project_root=project_root,
            vendor_roots=vendor_roots,
        )
    if not selected.available:
        return _pair_report(
            materialize_comparison_result(
                cell,
                status="missing",
                eligibility=eligibility,
                route=selected,
                qualification=comparator_qualification,
                candidate_qualification=candidate_qualification,
                reason=selected.reason or f"{canonical} is {selected.status}",
            ),
            canonical=canonical,
            candidate_qualification=candidate_qualification,
        )
    comparator_applicable, comparator_reason = selected.applicable(values, query)
    if not comparator_applicable:
        return _pair_report(
            materialize_comparison_result(
                cell,
                status="failed",
                eligibility=eligibility,
                route=selected,
                qualification=comparator_qualification,
                candidate_qualification=candidate_qualification,
                reason=comparator_reason or "comparator rejected timing inputs",
            ),
            canonical=canonical,
            candidate_qualification=candidate_qualification,
        )

    import torch
    from .competitor_protocol import paired_orders, validate_raw_samples

    candidate_invoke = candidate or _public_candidate
    timing_mode = str(cell.get("timing_mode", cell.get("operator_mode", "forward")))
    if timing_mode not in {"forward", "forward_backward"}:
        raise ValueError(f"unsupported matched timing mode {timing_mode!r}")
    if type(seed) is not int:
        raise ValueError("matched timing requires an integer order seed")
    timer = timing_call or (lambda function: _default_timing_call(function, device))
    input_hash = _logical_input_hash(cell, seed=seed, gpu=gpu)
    arm_names = ("attnres", canonical)
    orders = paired_orders(arm_names, rounds, seed=seed)

    # Derive one common upstream from an oracle output.  This keeps backward
    # work identical across arms and gives packed [S,N,D] inputs the correct
    # [N,D] upstream shape.  The upstream construction is outside timing.
    upstream = _oracle_upstream(values, query)
    states = {
        "attnres": _timing_state(values, query, upstream),
        canonical: _timing_state(values, query, upstream),
    }

    def prepare_arm(arm: str) -> None:
        state_values, state_query, _arm_upstream = states[arm]
        if timing_mode == "forward_backward":
            _clear_timing_gradients(state_values, state_query)

    def call_arm(arm: str) -> Any:
        state_values, state_query, arm_upstream = states[arm]
        invoke = candidate_invoke if arm == "attnres" else selected.invoke
        return _timed_arm_call(
            invoke,
            state_values,
            state_query,
            arm_upstream,
            timing_mode,
        )

    def _timing_payload(*, raw_samples: Sequence[Mapping[str, Any]], **extra: Any) -> dict[str, Any]:
        return {
            "status": "incomplete",
            "timing_mode": timing_mode,
            "warmup": warmup,
            "rounds": rounds,
            "arms": arm_names,
            "timing_boundary": (
                "forward+backward operator invocation, including adapter-owned source "
                "stacking/contiguous preparation, is inside each timing event"
                if timing_mode == "forward_backward"
                else "forward operator invocation, including adapter-owned source "
                "stacking/contiguous preparation, is inside each timing event"
            ),
            "adapter_stack_in_timing": True,
            "timing_excluded_work": list(MATCHED_TIMING_EXCLUDED_WORK),
            "same_upstream": True,
            "upstream_shape": list(upstream.shape),
            "input_layout": "packed" if isinstance(values, torch.Tensor) else "list",
            "input_hash": input_hash,
            "input_hash_kind": "logical_case_seed_id_no_tensor_readback",
            "integrity_mode": "speed_first_no_tensor_hashing",
            "orders": orders,
            "raw_samples": list(raw_samples),
            **extra,
        }

    # Warmups are excluded from measured rows, but use the same arm callbacks
    # so adapter-owned packing and contiguous preparation are exercised.
    for warmup_index in range(warmup):
        for arm in orders[warmup_index % len(orders)]:
            try:
                prepare_arm(arm)
                output = call_arm(arm)
                if not isinstance(output, torch.Tensor):
                    raise TypeError(f"{arm} warmup returned a non-tensor")
                if device is not None and getattr(device, "type", None) == "cuda":
                    torch.cuda.synchronize(device)
            except Exception as exc:
                # Preserve a complete failed/skipped raw matrix even when a
                # warmup fails.  The first timed slot retains the warmup
                # failure provenance; every later slot records why it was
                # skipped.  This keeps the raw artifact validator honest.
                warmup_reason = str(exc) or type(exc).__name__
                warmup_samples: list[dict[str, Any]] = []
                # The warmup schedule can start with either arm.  Attribute
                # the retained failure to that arm's first timed slot rather
                # than hard-coding (0, 0), which could mislabel a comparator
                # failure as an AttnRes failure.
                warmup_failed_at = next(
                    (
                        (failed_round, failed_order)
                        for failed_round, round_order in enumerate(orders)
                        for failed_order, failed_arm in enumerate(round_order)
                        if failed_arm == arm
                    ),
                    (0, 0),
                )
                for failed_round, failed_order in (
                    (round_index, order_index)
                    for round_index, round_order in enumerate(orders)
                    for order_index, _failed_arm in enumerate(round_order)
                ):
                    failed_row: dict[str, Any] = {
                        "seed": seed,
                        "gpu": gpu,
                        "round_index": failed_round,
                        "order_index": failed_order,
                        "input_hash": input_hash,
                        "arm": orders[failed_round][failed_order],
                        "eligible": True,
                        "status": "failed" if (failed_round, failed_order) == warmup_failed_at else "skipped_due_to_failure",
                        "latency_ms": None,
                        "failure_phase": "warmup" if (failed_round, failed_order) == warmup_failed_at else "timing_skip",
                        "failure_reason": warmup_reason if (failed_round, failed_order) == warmup_failed_at else (
                            f"skipped after warmup failure at round "
                            f"{warmup_failed_at[0]} order {warmup_failed_at[1]}: "
                            f"{warmup_reason}"
                        ),
                        "failure_at_round": warmup_failed_at[0],
                        "failure_at_order": warmup_failed_at[1],
                    }
                    if (failed_round, failed_order) == warmup_failed_at:
                        failed_row["error"] = _exception(exc)
                    warmup_samples.append(failed_row)
                raw_validation: dict[str, Any]
                if gpu not in {"H100!", "B200"}:
                    raw_validation = {
                        "status": "failed",
                        "reason": "matched raw rows require a protocol GPU selector (H100! or B200)",
                    }
                else:
                    try:
                        validate_raw_samples(
                            warmup_samples,
                            arm_names,
                            rounds=rounds,
                            seed=seed,
                            gpu=gpu,
                            planned_eligibility={arm_name: True for arm_name in arm_names},
                            require_eligible_ok=False,
                        )
                    except Exception as validation_exc:
                        raw_validation = {"status": "failed", "error": _exception(validation_exc)}
                    else:
                        raw_validation = {"status": "passed"}
                timing = _timing_payload(
                    raw_samples=warmup_samples,
                    raw_validation=raw_validation,
                    failure={
                        "phase": "warmup",
                        "index": warmup_index,
                        "arm": arm,
                        "error": _exception(exc),
                    },
                )
                result = materialize_comparison_result(
                    cell,
                    status="incomplete",
                    eligibility=eligibility,
                    route=selected,
                    qualification=comparator_qualification,
                    candidate_qualification=candidate_qualification,
                    timing=timing,
                    reason=str(exc),
                )
                return _pair_report(
                    result,
                    canonical=canonical,
                    candidate_qualification=candidate_qualification,
                    timing=timing,
                )

    samples: list[dict[str, Any]] = []
    failed_at: tuple[int, int] | None = None
    failure_reason: str | None = None
    for round_index, order in enumerate(orders):
        for order_index, arm in enumerate(order):
            row: dict[str, Any] = {
                "seed": seed,
                "gpu": gpu,
                "round_index": round_index,
                "order_index": order_index,
                "input_hash": input_hash,
                "arm": arm,
                "eligible": True,
                "status": "failed",
                "latency_ms": None,
                "failure_phase": None,
                "failure_reason": None,
                "failure_at_round": None,
                "failure_at_order": None,
            }
            if failed_at is not None:
                row.update(
                    status="skipped_due_to_failure",
                    failure_phase="timing_skip",
                    failure_reason=(
                        f"skipped after timing failure at round {failed_at[0]} "
                        f"order {failed_at[1]}: {failure_reason}"
                    ),
                    failure_at_round=failed_at[0],
                    failure_at_order=failed_at[1],
                )
                samples.append(row)
                continue
            try:
                prepare_arm(arm)
                elapsed, output = timer(lambda arm=arm: call_arm(arm))
                if not isinstance(output, torch.Tensor):
                    raise TypeError(f"{arm} timed comparator returned a non-tensor")
                if (
                    isinstance(elapsed, bool)
                    or not isinstance(elapsed, (int, float))
                    or not math.isfinite(float(elapsed))
                    or elapsed <= 0
                ):
                    raise ValueError("timing callback must return a positive latency in milliseconds")
                row.update(status="ok", latency_ms=float(elapsed))
            except Exception as exc:
                failed_at = (round_index, order_index)
                failure_reason = str(exc) or type(exc).__name__
                row.update(
                    failure_phase="timing",
                    failure_reason=failure_reason,
                    failure_at_round=round_index,
                    failure_at_order=order_index,
                    error=_exception(exc),
                )
            samples.append(row)

    raw_validation: dict[str, Any]
    if gpu not in {"H100!", "B200"}:
        raw_validation = {
            "status": "failed",
            "reason": "matched raw rows require a protocol GPU selector (H100! or B200)",
        }
    else:
        try:
            validate_raw_samples(
                samples,
                arm_names,
                rounds=rounds,
                seed=seed,
                gpu=gpu,
                planned_eligibility={arm: True for arm in arm_names},
                require_eligible_ok=False,
            )
        except Exception as exc:
            raw_validation = {"status": "failed", "error": _exception(exc)}
        else:
            raw_validation = {"status": "passed"}

    complete = (
        failed_at is None
        and len(samples) == 2 * rounds
        and all(row["status"] == "ok" for row in samples)
        and raw_validation["status"] == "passed"
    )
    candidate_samples = [
        float(row["latency_ms"])
        for row in samples
        if row["arm"] == "attnres" and row["status"] == "ok"
    ]
    comparator_samples = [
        float(row["latency_ms"])
        for row in samples
        if row["arm"] == canonical and row["status"] == "ok"
    ]
    ratio: dict[str, Any] | None = None
    if complete:
        settings = protocol or {}
        statistics = settings.get("statistics", {})
        if not isinstance(statistics, Mapping):
            statistics = {}
        from .statistics import paired_ratio_bootstrap

        ratio = paired_ratio_bootstrap(
            comparator_samples,
            candidate_samples,
            samples=int(statistics.get("bootstrap_samples", 20_000)),
            seed=int(statistics.get("seed", int(seed) + 17000)),
            confidence=float(statistics.get("confidence", 0.95)),
            margin=float(statistics.get("plateau_margin", 0.01)),
        )
        ratio.update(
            {
                "candidate_arm": "attnres",
                "comparator_arm": canonical,
                "orientation": "candidate_over_baseline",
            }
        )
    timing = {
        "status": "complete" if complete else "incomplete",
        "timing_mode": timing_mode,
        "warmup": warmup,
        "rounds": rounds,
        "arms": arm_names,
        "timing_boundary": (
            "forward+backward operator invocation, including adapter-owned source "
            "stacking/contiguous preparation, is inside each timing event"
            if timing_mode == "forward_backward"
            else "forward operator invocation, including adapter-owned source "
            "stacking/contiguous preparation, is inside each timing event"
        ),
        "adapter_stack_in_timing": True,
        "timing_excluded_work": list(MATCHED_TIMING_EXCLUDED_WORK),
        "same_upstream": True,
        "upstream_shape": list(upstream.shape),
        "input_layout": "packed" if isinstance(values, torch.Tensor) else "list",
        "input_hash": input_hash,
        "input_hash_kind": "logical_case_seed_id_no_tensor_readback",
        "integrity_mode": "speed_first_no_tensor_hashing",
        "orders": orders,
        "raw_samples": samples,
        "candidate_samples": candidate_samples,
        "comparator_samples": comparator_samples,
        "raw_validation": raw_validation,
    }
    if ratio is not None:
        timing["ratio"] = ratio
    result = materialize_comparison_result(
        cell,
        status="complete" if complete else "incomplete",
        eligibility=eligibility,
        route=selected,
        qualification=comparator_qualification,
        candidate_qualification=candidate_qualification,
        timing=timing,
        reason=None if complete else "paired timing did not produce a complete successful sample set",
    )
    return _pair_report(
        result,
        canonical=canonical,
        candidate_qualification=candidate_qualification,
        timing=timing,
        ratio=ratio,
    )


def run_matched_comparison(
    name: str,
    cell: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    values: Any = None,
    query: Any = None,
    input_factory: Callable[[Mapping[str, Any]], tuple[Any, Any]] | None = None,
    route: ComparatorRoute | None = None,
    candidate: Callable[[Any, Any], Any] | None = None,
    rounds: int = 1,
    warmup: int = 0,
    device: Any = None,
    timing_call: Callable[[Callable[[], Any]], tuple[float, Any]] | None = None,
    seed: int = 0,
    gpu: str | None = None,
    project_root: str | Path | None = None,
    vendor_roots: Mapping[str, str | Path] | str | Path | None = None,
) -> dict[str, Any]:
    """Qualify both arms and then time their matched operator pair."""

    canonical = canonical_name(name)
    eligibility = _eligibility(canonical, cell)
    if not eligibility["eligible"]:
        return _pair_report(
            materialize_comparison_result(
                cell,
                status="not_applicable",
                eligibility=eligibility,
                reason=eligibility["reason"],
            ),
            canonical=canonical,
        )
    selected = _route_for(
        canonical,
        route,
        project_root=project_root,
        vendor_roots=vendor_roots,
    )
    if not selected.available:
        return _pair_report(
            materialize_comparison_result(
                cell,
                status="missing",
                eligibility=eligibility,
                route=selected,
                reason=selected.reason or f"{canonical} is {selected.status}",
            ),
            canonical=canonical,
        )
    if values is None or query is None:
        if input_factory is None:
            return _pair_report(
                materialize_comparison_result(
                    cell,
                    status="not_run",
                    eligibility=eligibility,
                    route=selected,
                    reason="no comparator inputs were supplied",
                ),
                canonical=canonical,
            )
        try:
            values, query = input_factory(cell)
        except Exception as exc:
            return _pair_report(
                materialize_comparison_result(
                    cell,
                    status="failed",
                    eligibility=eligibility,
                    route=selected,
                    reason=str(exc),
                    failure={"phase": "input_factory", "error": _exception(exc)},
                ),
                canonical=canonical,
            )
    try:
        candidate_qualification = qualify_candidate(
            cell,
            values,
            query,
            protocol,
            candidate=candidate,
        )
    except Exception as exc:
        candidate_qualification = {
            "status": "failed",
            "failure_phase": "candidate_qualification",
            "error": _exception(exc),
        }
    try:
        comparator_qualification = qualify_comparator(
            canonical,
            cell,
            values,
            query,
            protocol,
            route=selected,
            project_root=project_root,
            vendor_roots=vendor_roots,
        )
    except Exception as exc:
        comparator_qualification = {
            "status": "failed",
            "failure_phase": "comparator_qualification",
            "error": _exception(exc),
        }
    if (
        candidate_qualification.get("status") != "qualified"
        or comparator_qualification.get("status") != "qualified"
    ):
        return _pair_report(
            materialize_comparison_result(
                cell,
                status="failed",
                eligibility=eligibility,
                route=selected,
                qualification=comparator_qualification,
                candidate_qualification=candidate_qualification,
                reason="both candidate and comparator must pass independent qualification before timing",
            ),
            canonical=canonical,
            candidate_qualification=candidate_qualification,
        )
    return time_matched_pair(
        canonical,
        cell,
        values,
        query,
        candidate_qualification,
        comparator_qualification,
        route=selected,
        candidate=candidate,
        rounds=rounds,
        warmup=warmup,
        device=device,
        timing_call=timing_call,
        seed=seed,
        gpu=gpu,
        protocol=protocol,
        project_root=project_root,
        vendor_roots=vendor_roots,
    )


def run_matched_comparison_cells(
    cells: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    routes: Mapping[str, ComparatorRoute] | None = None,
    input_factory: Callable[[Mapping[str, Any]], tuple[Any, Any]] | None = None,
    candidate: Callable[[Any, Any], Any] | None = None,
    rounds: int = 1,
    warmup: int = 0,
    device: Any = None,
    timing_call: Callable[[Callable[[], Any]], tuple[float, Any]] | None = None,
    seed: int | None = None,
    gpu: str | None = None,
    project_root: str | Path | None = None,
    vendor_roots: Mapping[str, str | Path] | str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run matched candidate/comparator pairs in deterministic cell order."""

    results = []
    for cell in cells:
        name = str(cell.get("competitor", ""))
        selected = None if routes is None else routes.get(canonical_name(name))
        cell_seed = cell.get("seed", seed)
        if type(cell_seed) is not int:
            raise ValueError("each matched comparison cell must carry an integer protocol seed")
        results.append(
            run_matched_comparison(
                name,
                cell,
                protocol,
                input_factory=input_factory,
                route=selected,
                candidate=candidate,
                rounds=rounds,
                warmup=warmup,
                device=device,
                timing_call=timing_call,
                seed=cell_seed,
                gpu=gpu if gpu is not None else cell.get("gpu"),
                project_root=project_root,
                vendor_roots=vendor_roots,
            )
        )
    return results


def run_registered_comparison(
    name: str,
    cell: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    values: Any = None,
    query: Any = None,
    input_factory: Callable[[Mapping[str, Any]], tuple[Any, Any]] | None = None,
    route: ComparatorRoute | None = None,
    rounds: int = 1,
    warmup: int = 0,
    device: Any = None,
    timing_call: Callable[[Callable[[], Any]], tuple[float, Any]] | None = None,
    seed: int | None = None,
    gpu: str | None = None,
    project_root: str | Path | None = None,
    vendor_roots: Mapping[str, str | Path] | str | Path | None = None,
) -> dict[str, Any]:
    """Dispatch one cell, qualify it, then optionally materialize timings."""

    canonical = canonical_name(name)
    eligibility = _eligibility(canonical, cell)
    if not eligibility["eligible"]:
        return materialize_comparison_result(
            cell, status="not_applicable", eligibility=eligibility, reason=eligibility["reason"]
        )
    selected = _route_for(
        canonical,
        route,
        project_root=project_root,
        vendor_roots=vendor_roots,
    )
    if not selected.available:
        return materialize_comparison_result(
            cell,
            status="missing",
            eligibility=eligibility,
            route=selected,
            reason=selected.reason or f"{canonical} is {selected.status}",
        )
    if values is None or query is None:
        if input_factory is None:
            return materialize_comparison_result(
                cell,
                status="not_run",
                eligibility=eligibility,
                route=selected,
                reason="no comparator inputs were supplied",
            )
        try:
            values, query = input_factory(cell)
        except Exception as exc:
            return materialize_comparison_result(
                cell,
                status="failed",
                eligibility=eligibility,
                route=selected,
                reason=str(exc),
                failure={"phase": "input_factory", "error": _exception(exc)},
            )
    try:
        qualification = qualify_comparator(
            canonical,
            cell,
            values,
            query,
            protocol,
            route=selected,
            project_root=project_root,
            vendor_roots=vendor_roots,
        )
    except Exception as exc:
        qualification = {
            "status": "failed",
            "failure_phase": "qualification",
            "error": _exception(exc),
        }
    if qualification.get("status") != "qualified":
        return materialize_comparison_result(
            cell,
            status="failed" if qualification.get("status") != "not_applicable" else "not_applicable",
            eligibility=eligibility,
            route=selected,
            qualification=qualification,
            reason=str(qualification.get("reason") or "independent comparator qualification failed"),
        )
    if rounds is None or rounds == 0:
        return materialize_comparison_result(
            cell,
            status="not_run",
            eligibility=eligibility,
            route=selected,
            qualification=qualification,
            reason="qualification passed; timing was not requested",
        )
    return time_comparator(
        canonical,
        cell,
        values,
        query,
        qualification,
        route=selected,
        rounds=rounds,
        warmup=warmup,
        device=device,
        timing_call=timing_call,
        seed=seed,
        gpu=gpu,
        project_root=project_root,
        vendor_roots=vendor_roots,
    )


def run_registered_comparison_cells(
    cells: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    routes: Mapping[str, ComparatorRoute] | None = None,
    input_factory: Callable[[Mapping[str, Any]], tuple[Any, Any]] | None = None,
    rounds: int = 0,
    warmup: int = 0,
    device: Any = None,
    timing_call: Callable[[Callable[[], Any]], tuple[float, Any]] | None = None,
    seed: int | None = None,
    gpu: str | None = None,
    project_root: str | Path | None = None,
    vendor_roots: Mapping[str, str | Path] | str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run a deterministic list of cells with route reuse."""

    results = []
    for cell in cells:
        name = str(cell.get("competitor", ""))
        selected = None if routes is None else routes.get(canonical_name(name))
        result = run_registered_comparison(
            name,
            cell,
            protocol,
            input_factory=input_factory,
            route=selected,
            rounds=rounds,
            warmup=warmup,
            device=device,
            timing_call=timing_call,
            seed=seed,
            gpu=gpu,
            project_root=project_root,
            vendor_roots=vendor_roots,
        )
        results.append(result)
    return results


def summarize_matched_statistics(
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute protocol-level simultaneous intervals for complete pair rows.

    A per-cell ordinary bootstrap is retained on the raw timing payload for
    descriptive use.  The sealed protocol's promotion statistic is the
    common-index, max-deviation interval over every complete comparison in a
    ``(comparison_family, gpu, seed)`` group.  Incomplete, missing, diagnostic,
    and explicitly inapplicable rows stay visible but do not enter a group.
    """

    from .competitor_protocol import comparison_plan, validate_raw_samples
    from .statistics import simultaneous_paired_ratio_bootstrap

    statistics = protocol.get("statistics", {}) if isinstance(protocol, Mapping) else {}
    if not isinstance(statistics, Mapping):
        statistics = {}
    samples = int(statistics.get("bootstrap_samples", 20_000))
    confidence = float(statistics.get("confidence", 0.95))
    margin = float(statistics.get("plateau_margin", 0.01))
    try:
        plan = comparison_plan(protocol)
    except Exception as exc:
        return {
            "status": "incomplete",
            "estimator": str(statistics.get("estimator", "simultaneous_paired_ratio_bootstrap")),
            "ratio": "candidate_over_baseline",
            "confidence": confidence,
            "bootstrap_samples": samples,
            "common_resample_indices": True,
            "familywise_scope": str(
                statistics.get(
                    "familywise_scope",
                    "all planned comparisons within each GPU_seed_predeclared_competitor_family_cell",
                )
            ),
            "plateau_margin": margin,
            "groups": [],
            "excluded": [{"reason": "cannot load the sealed comparison plan", "error": _exception(exc)}],
        }

    planned_by_id = {
        str(cell["comparison_cell_id"]): cell
        for cell in plan["planned"]
        if cell.get("eligible") is True and cell.get("eligible_denominator") is True
    }
    grouped: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    excluded: list[dict[str, Any]] = []
    for row in rows:
        timing = row.get("timing")
        cell = row.get("cell", {})
        raw_comparison_id = row.get("comparison_cell_id")
        if type(raw_comparison_id) is not str or not raw_comparison_id.strip():
            excluded.append(
                {
                    "comparison_cell_id": raw_comparison_id,
                    "reason": "complete row lacks a canonical comparison cell id",
                }
            )
            continue
        comparison_id = raw_comparison_id
        context = {"comparison_cell_id": comparison_id}
        if not isinstance(timing, Mapping) or not isinstance(cell, Mapping):
            excluded.append({**context, "reason": "complete row lacks cell/timing mappings"})
            continue
        cell_comparison_id = cell.get("comparison_cell_id")
        if type(cell_comparison_id) is not str or cell_comparison_id != comparison_id:
            excluded.append(
                {
                    **context,
                    "reason": "row and cell comparison cell ids disagree",
                }
            )
            continue
        if row.get("status") != "complete" or row.get("eligible_denominator") is not True:
            continue
        expected_cell = planned_by_id.get(comparison_id)
        if expected_cell is None:
            excluded.append(
                {**context, "reason": "row is not an eligible cell in the sealed comparison plan"}
            )
            continue
        # The row's identity is part of the sealed plan.  Do not let a caller
        # forge a complete row under a different shape, seed, family, or arm.
        identity_fields = (
            "competitor",
            "comparison_family",
            "scope",
            "operator_scope",
            "gpu",
            "seed",
            "dtype",
            "mode",
            "S",
            "N",
            "D",
            "R",
            "rank",
            "source_count",
            "read_source_count",
            "timing",
            "timing_mode",
        )
        if any(cell.get(field) != expected_cell.get(field) for field in identity_fields):
            excluded.append({**context, "reason": "row cell metadata differs from sealed plan"})
            continue
        if not _complete_timing_payload(timing):
            excluded.append({**context, "reason": "timing payload is not complete sealed evidence"})
            continue
        timing_mode = timing.get("timing_mode")
        try:
            expected_timing_mode = _timing_mode(expected_cell)
        except (TypeError, ValueError):
            excluded.append({**context, "reason": "sealed plan has no valid timing mode"})
            continue
        if timing_mode != expected_timing_mode:
            excluded.append({**context, "reason": "timing mode differs from sealed plan"})
            continue
        if timing.get("rounds") != protocol.get("rounds") or timing.get("warmup") != protocol.get("warmup"):
            excluded.append({**context, "reason": "timing rounds/warmup differ from sealed protocol"})
            continue
        try:
            canonical = canonical_name(str(expected_cell["competitor"]))
            arms = timing.get("arms")
            if tuple(arms) != ("attnres", canonical):
                raise ValueError("timing arms do not match candidate and canonical comparator")
            validate_raw_samples(
                timing["raw_samples"],
                ("attnres", canonical),
                rounds=int(timing["rounds"]),
                seed=int(expected_cell["seed"]),
                gpu=str(expected_cell["gpu"]),
                planned_eligibility={"attnres": True, canonical: True},
            )
        except Exception as exc:
            excluded.append({**context, "reason": "raw samples fail the sealed pairing validator", "error": _exception(exc)})
            continue
        candidate_samples = timing.get("candidate_samples")
        comparator_samples = timing.get("comparator_samples")
        raw_samples = timing["raw_samples"]
        raw_candidate = [float(item["latency_ms"]) for item in raw_samples if item["arm"] == "attnres"]
        raw_comparator = [float(item["latency_ms"]) for item in raw_samples if item["arm"] == canonical]
        if (
            isinstance(candidate_samples, (str, bytes))
            or not isinstance(candidate_samples, Sequence)
            or isinstance(comparator_samples, (str, bytes))
            or not isinstance(comparator_samples, Sequence)
            or list(candidate_samples) != raw_candidate
            or list(comparator_samples) != raw_comparator
        ):
            excluded.append(
                {**context, "reason": "paired summary arrays do not match validated raw samples"}
            )
            continue
        family = expected_cell["comparison_family"]
        gpu = expected_cell["gpu"]
        raw_seed = expected_cell["seed"]
        key = (str(family), str(gpu), int(raw_seed), str(timing_mode))
        group = grouped.setdefault(key, {})
        if comparison_id in group:
            excluded.append(
                {**context, "reason": "duplicate comparison cell id in simultaneous group"}
            )
            continue
        group[comparison_id] = {
            "baseline": list(comparator_samples),
            "candidate": list(candidate_samples),
            "candidate_arm": "attnres",
            "comparator_arm": canonical,
        }

    groups: list[dict[str, Any]] = []
    for (family, gpu, seed, timing_mode), comparisons in sorted(grouped.items()):
        result: dict[str, Any] = {
            "family": family,
            "gpu": gpu,
            "seed": seed,
            "timing_mode": timing_mode,
            "comparison_count": len(comparisons),
            "status": "incomplete",
            "comparisons": {},
        }
        # A deterministic per-group seed prevents iteration order or Python's
        # randomized hash seed from changing the report.
        digest = hashlib.sha256(f"{family}\0{gpu}\0{seed}".encode()).digest()
        group_seed = int.from_bytes(digest[:8], "big") % (2**32)
        result["bootstrap_seed"] = group_seed
        try:
            summaries = simultaneous_paired_ratio_bootstrap(
                comparisons,
                samples=samples,
                seed=group_seed,
                confidence=confidence,
                margin=margin,
            )
        except Exception as exc:
            result["error"] = _exception(exc)
        else:
            result["status"] = "complete"
            for comparison_id, summary in summaries.items():
                entry = dict(summary)
                entry.update(comparisons[comparison_id])
                entry["orientation"] = "candidate_over_baseline"
                result["comparisons"][comparison_id] = _jsonable(entry)
        groups.append(result)
    return {
        "status": "complete" if groups and all(group["status"] == "complete" for group in groups) else "incomplete",
        "estimator": str(statistics.get("estimator", "simultaneous_paired_ratio_bootstrap")),
        "ratio": "candidate_over_baseline",
        "confidence": confidence,
        "bootstrap_samples": samples,
        "common_resample_indices": True,
        "familywise_scope": str(
            statistics.get(
                "familywise_scope",
                "all planned comparisons within each GPU_seed_predeclared_competitor_family_cell",
            )
        ),
        "plateau_margin": margin,
        "groups": groups,
        "excluded": excluded,
    }


def _matched_operator_case(raw: Any, index: int) -> dict[str, int | str]:
    """Normalize one sealed matched-protocol operator case."""

    dtype: str | None = None
    if isinstance(raw, Mapping):
        source_count = raw.get("S", raw.get("sources", raw.get("source_count")))
        rows = raw.get("N", raw.get("rows", raw.get("tokens", raw.get("T", 1))))
        width = raw.get("D", raw.get("width", raw.get("hidden")))
        rank = raw.get("R", raw.get("rank", width))
        raw_dtype = raw.get("dtype", raw.get("value_dtype"))
        if raw_dtype is not None:
            dtype = str(raw_dtype)
    else:
        try:
            fields = list(raw)
        except TypeError as exc:
            raise ValueError(f"operator case {index} is not a sequence or mapping") from exc
        if len(fields) == 5 and isinstance(fields[-1], str):
            dtype = str(fields.pop())
        if len(fields) == 3:
            source_count, rows, width = fields
            rank = width
        elif len(fields) == 4:
            source_count, rows, width, rank = fields
        else:
            raise ValueError(f"operator case {index} needs three or four fields")
    values = {"S": source_count, "N": rows, "D": width, "R": rank}
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values.values()):
        raise ValueError(f"operator case {index} has noninteger dimensions: {values}")
    if any(value < 1 for value in values.values()) or int(rank) > int(width):
        raise ValueError(f"operator case {index} has invalid dimensions: {values}")
    result: dict[str, int | str] = {
        **{key: int(value) for key, value in values.items()},
        "id": f"operator_{index}",
    }
    if dtype is not None:
        result["dtype"] = dtype
    return result


def _validate_execution_gpu(
    config: Mapping[str, Any],
    gpu: str,
    device: Any,
    torch_module: Any,
) -> dict[str, Any]:
    """Validate the protocol selector against the CUDA device when possible."""

    hardware = config.get("hardware")
    if not isinstance(hardware, Mapping) or gpu not in hardware:
        raise ValueError(
            "GPU selector must be one of "
            f"{list(config.get('hardware_order', ()))!r}"
        )
    contract = hardware[gpu]
    expected = str(contract.get("compute_capability", "")) if isinstance(contract, Mapping) else ""
    if not expected.startswith("sm"):
        raise ValueError(f"sealed hardware contract has no compute capability for {gpu!r}")
    available = bool(torch_module.cuda.is_available())
    result: dict[str, Any] = {
        "selector": gpu,
        "expected_compute_capability": expected,
        "verified": False,
    }
    if not available:
        # CPU tests may inject a CUDA-shaped device while stubbing the actual
        # runner.  Preserve that seam, but mark the physical identity as
        # unverified; a real worker must have CUDA available before reporting
        # measurements.
        result["reason"] = "torch.cuda.is_available() is false; physical selector not verified"
        return result
    try:
        capability = torch_module.cuda.get_device_capability(device)
    except Exception as exc:
        raise RuntimeError(f"could not inspect CUDA device for selector {gpu!r}: {exc}") from exc
    if not isinstance(capability, (tuple, list)) or len(capability) != 2:
        raise RuntimeError(f"CUDA returned an invalid compute capability for selector {gpu!r}")
    actual = f"sm{int(capability[0])}{int(capability[1])}"
    result["actual_compute_capability"] = actual
    if actual != expected:
        raise RuntimeError(
            f"hardware mismatch for selector {gpu}: expected {expected}, got {actual}"
        )
    try:
        device_name = str(torch_module.cuda.get_device_name(device))
    except Exception as exc:
        raise RuntimeError(f"could not inspect CUDA device name for selector {gpu!r}: {exc}") from exc
    expected_name = str(contract.get("modal_name", gpu)).rstrip("!")
    if expected_name.upper() not in device_name.upper():
        raise RuntimeError(
            f"hardware mismatch for selector {gpu}: expected device name containing "
            f"{expected_name!r}, got {device_name!r}"
        )
    result["actual_name"] = device_name
    result["verified"] = True
    return result


def run_matched_registry(
    config: Mapping[str, Any] | None = None,
    *,
    project_root: str | Path | None = None,
    vendor_roots: Mapping[str, str | Path] | str | Path | None = None,
    device: Any = None,
    execute_operator: bool = False,
    scope: str = "smoke",
    names: Sequence[str] | None = None,
    rounds: int | None = None,
    warmup: int | None = None,
    seed: int | None = None,
    gpu: str | None = None,
    timing_call: Callable[[Callable[[], Any]], tuple[float, Any]] | None = None,
    routes: Mapping[str, ComparatorRoute] | None = None,
) -> dict[str, Any]:
    """Run the sealed registry surface used by the 2.13/3.7.1 worker.

    The plan is always materialized.  Operator execution is opt-in because a
    caller must explicitly select a CUDA worker; model cells remain planned
    until the existing per-read model runner supplies their complete-step
    inputs. Both schedules use that shared model runner.
    """

    from .competitor_protocol import comparison_plan, config_digest, load_config, validate_config

    cfg = load_config() if config is None else validate_config(config)
    if scope not in {"smoke", "primary", "heldout"}:
        raise ValueError(f"unsupported matched comparator scope {scope!r}")
    plan = comparison_plan(cfg)
    report = materialize_comparison_plan(
        plan,
        status="not_run",
        reason="capability plan materialized; operator execution is opt-in",
    )
    report["config_hash"] = config_digest(cfg)
    if not execute_operator:
        report["execution_status"] = "not_requested"
        return report
    if device is None or getattr(device, "type", None) != "cuda":
        report["execution_status"] = "not_run"
        report["execution_reason"] = "matched comparator execution requires a CUDA device"
        return report
    if type(gpu) is not str or gpu not in tuple(cfg["hardware_order"]):
        raise ValueError(
            "matched comparator execution requires exactly one GPU selector from "
            f"{list(cfg['hardware_order'])!r}"
        )
    import torch

    report["hardware"] = _validate_execution_gpu(cfg, gpu, device, torch)
    if report["hardware"].get("verified") is not True:
        report["execution_status"] = "not_run"
        report["execution_reason"] = report["hardware"].get(
            "reason", "physical GPU selector could not be verified"
        )
        return report

    protocol_seeds = tuple(int(value) for value in cfg.get("seeds", (0,)))
    if seed is not None:
        raise ValueError(
            "run_matched_registry cannot override sealed seeds; execute every protocol seed"
        )
    seeds = protocol_seeds
    if not seeds:
        raise ValueError("matched protocol must define at least one seed")
    if rounds is not None or warmup is not None:
        raise ValueError("matched protocol rounds and warmup are sealed; overrides are not allowed")
    operator_rows: list[dict[str, Any]] = []

    # Select the exact pre-expanded operator cells before adapter discovery.
    # Inapplicable rows therefore remain audit-only and cannot trigger vendor
    # imports, input allocation, or an adapter call.
    operator_cells = [
        dict(cell)
        for cell in (*plan["planned"], *plan["not_applicable"])
        if cell.get("scope") == "operator"
        and cell.get("operator_scope") == scope
        and (gpu is None or cell.get("gpu") == gpu)
    ]
    eligible_names = {
        canonical_name(str(cell["competitor"]))
        for cell in operator_cells
        if type(cell.get("eligible", False)) is bool and cell.get("eligible") is True
    }
    if names is None:
        requested_names = tuple(
            dict.fromkeys(
                canonical_name(str(cell["competitor"]))
                for cell in operator_cells
                if canonical_name(str(cell["competitor"])) in eligible_names
            )
        )
    else:
        selected_by_user = tuple(dict.fromkeys(canonical_name(name) for name in names))
        operator_cells = [
            cell
            for cell in operator_cells
            if canonical_name(str(cell["competitor"])) in selected_by_user
        ]
        requested_names = tuple(name for name in selected_by_user if name in eligible_names)
    if routes is None:
        selected_routes = discover_registered_comparators(
            project_root=project_root,
            vendor_roots=vendor_roots,
            names=requested_names,
        )
    else:
        selected_routes = {
            canonical_name(name): route
            for name, route in routes.items()
            if canonical_name(name) in requested_names
        }
    if routes is not None:
        # A caller may supply a route map for only a subset of the selected
        # names.  Materialize those cells as missing rather than rediscovering
        # an ambient checkout under a different provenance root.
        missing_names = [name for name in requested_names if name not in selected_routes]
        for name in missing_names:
            selected_routes[name] = _missing_route(
                name,
                "explicit route map omitted this comparator; ambient discovery is disabled",
                requested_root=_root_for(name, vendor_roots),
            )

    cell_seeds = {int(cell["seed"]) for cell in operator_cells}
    if cell_seeds and cell_seeds != set(seeds):
        raise ValueError(
            "operator plan must include every sealed protocol seed; "
            f"got {sorted(cell_seeds)!r}, expected {list(seeds)!r}"
        )

    def randn(shape: tuple[int, ...], dtype: Any, local_seed: int) -> Any:
        import torch

        generator = torch.Generator(device=device)
        generator.manual_seed(local_seed)
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    for cell in operator_cells:
        cell["timing_mode"] = "forward_backward"
        # Operator rows are measured, so preserve timing-only capability
        # limits when the runner revalidates the pre-expanded plan.  Without
        # this field a planned Hydra D>256 not-applicable row could be
        # reclassified as eligible immediately before execution.
        cell["timing"] = True

    def input_factory(cell: Mapping[str, Any]) -> tuple[Any, Any]:
        dtype_name = str(cell["dtype"]).lower().replace("torch.", "")
        if dtype_name in {"bf16", "bfloat16"}:
            dtype = torch.bfloat16
        elif dtype_name in {"fp32", "float32", "float"}:
            dtype = torch.float32
        else:
            raise ValueError(f"unsupported operator case dtype {cell['dtype']!r}")
        local_seed = int(cell["seed"]) + 1000 + int(cell.get("operator_case_index", 0)) * 100
        # The sealed standard operator uses source_layout=list.  Generate one
        # leaf per source so both public AttnRes and external adapters observe
        # the same ordered list contract; cloning for each arm happens later.
        values = [
            randn((int(cell["N"]), int(cell["D"])), dtype, local_seed + source_index)
            for source_index in range(int(cell["S"]))
        ]
        query = randn((int(cell["R"]),), dtype, local_seed + int(cell["S"]))
        return values, query

    operator_rows.extend(
        run_matched_comparison_cells(
            operator_cells,
            cfg,
            routes=selected_routes,
            input_factory=input_factory,
            rounds=int(cfg["rounds"]),
            warmup=int(cfg["warmup"]),
            device=device,
            timing_call=timing_call,
            seed=None,
            gpu=gpu,
            project_root=project_root,
            vendor_roots=vendor_roots,
        )
    )
    report["routes"] = {name: route.describe() for name, route in selected_routes.items()}
    if any(row["status"] == "failed" for row in operator_rows):
        operator_status = "failed"
    elif not operator_rows:
        operator_status = "incomplete"
    elif not any(row.get("eligible") is True for row in operator_rows):
        # A scope with no capability-eligible cells is an audit-only NA set;
        # it is never a successful execution merely because every row is NA.
        operator_status = "not_applicable"
    elif all(row["status"] in {"complete", "not_applicable"} for row in operator_rows):
        operator_status = "complete"
    else:
        operator_status = "incomplete"
    report["operator"] = {
        "status": operator_status,
        "scope": scope,
        "cells": operator_rows,
        "rounds": int(cfg["rounds"]),
        "warmup": int(cfg["warmup"]),
        "timing_mode": "forward_backward",
        "timing_boundary": (
            "forward+backward operator invocation, including adapter-owned source "
            "stacking/contiguous preparation, is inside each timing event"
        ),
        "adapter_stack_in_timing": True,
        "timing_excluded_work": list(MATCHED_TIMING_EXCLUDED_WORK),
        "eligible_denominator": sum(
            1 for row in operator_rows if row["eligible_denominator"]
        ),
        "qualified_denominator": sum(1 for row in operator_rows if row["eligible_denominator"]),
    }
    report["statistics"] = summarize_matched_statistics(operator_rows, cfg)
    report["execution_status"] = report["operator"]["status"]
    report["status"] = report["operator"]["status"]
    return report


# Short aliases make the dispatch seam discoverable to callers that use the
# protocol's terminology rather than the module's longer names.
dispatch_comparator = run_registered_comparison
dispatch_registered_comparator = run_registered_comparison
materialize_result = materialize_comparison_result


__all__ = [
    "ComparatorRoute",
    "discover_registered_comparators",
    "dispatch_comparator",
    "dispatch_registered_comparator",
    "materialize_comparison_plan",
    "materialize_comparison_result",
    "materialize_result",
    "qualify_comparator",
    "qualify_candidate",
    "run_registered_comparison",
    "run_registered_comparison_cells",
    "run_matched_comparison",
    "run_matched_comparison_cells",
    "run_matched_registry",
    "time_matched_pair",
    "time_comparator",
]
