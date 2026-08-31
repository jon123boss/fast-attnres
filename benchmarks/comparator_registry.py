"""CPU only registry for the optional matched competitor adapters.

The benchmark runner has several different kinds of optional backends.  A
native adapter describes how to call one implementation, while this module
describes *where that implementation is allowed to appear in the matched
protocol*.  Keeping the latter in a dependency free registry means that
protocol planning and report validation do not import Torch, Triton, or a
vendor checkout.

The registry is intentionally conservative.  ``eligible`` means that a cell
is inside the predeclared shape and schedule envelope; it does not mean that
the vendor checkout was found, that the oracle passed, or that a timing was
measured.  Callers must still run adapter discovery and the independent oracle
before retaining a result in a denominator.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


_FLA_REVISION = "5e02dd3a7651f5f2797eb8b12bbec401826031e1"
_FLA_TREE = "7e4199902fb291c78b3937f223b08ae7bca82bb1"
_FLA_ORIGIN = "https://github.com/fla-org/flash-linear-attention.git"
_FLA_PACKAGE_SHA256 = (
    "2cd59a9a50f34ecc4d9535ad51c9668cd4d8b67f519b8eb78b45ce2156288781"
)
_FLA_SOURCE_HASHES = {
    "fla/ops/attnres/fused.py": (
        "0e4683ab291086a9c3919d7352e2a998112973c94f5363e58f76ea7efea114f3"
    ),
    "fla/ops/attnres/backends/gluon.py": (
        "f8f163fb7ebb8d035236674aeb668483812fb4e9a29572ed2ae937c626990190"
    ),
}
_LIGER_REVISION = "000be60929938fd1358e03524c6ab398b6d421bd"
_LIGER_TREE = "746af1fc03014cf47cad895d01cf0d23fddf5e75"
_LIGER_ORIGIN = "https://github.com/linkedin/Liger-Kernel.git"
_LIGER_SOURCE_SHA256 = (
    "57da6fed98f794088b2a56223e6c7ef9fc920824f0c483cb0ef0b5a343dab0b1"
)
_LIGER_LICENSE_SHA256 = (
    "3a1ccb0c7274b68e1af2ca1d54b10b662085ca56753400182ecf87ae33f2d1a8"
)
_LIGER_NOTICE_SHA256 = (
    "9e3c27a0f64b87d00df12250cf1bc218b1e2fbc5fffc0bd64737ba8e8357218f"
)
_LIGER_PYPROJECT_SHA256 = (
    "f55effccdecc17ca87357ed8ecd4e73a58b1a56ee275367bfe5db2827dc9ac22"
)
_CATSWE_REVISION = "ff92865e4e1b18809da7a8f0c0c5252039cded7c"
_CATSWE_TREE = "f4f96a21dbe609044edef2fdbaf66a820c260fc0"
_CATSWE_ORIGIN = "https://github.com/catswe/flash-attention-residuals.git"
_CATSWE_LICENSE_SHA256 = (
    "299e72fdffa70bc47c4c6b7e60d71d698c9f0808b82275ba524538ed8233e08f"
)
_CATSWE_MAX_PROGRAM_ELEMENTS = 1_048_576
_CATSWE_SOURCE_HASHES = {
    "src/flash_attn_res/__init__.py": (
        "04d5c0eefd4d4a994f7521b26fb04d7fbe4fe29245c6f9ff64ac9a56fe224868"
    ),
    "src/flash_attn_res/kernels/configs.py": (
        "ebec80ad42fa781e54169e69b53d154c00c1b80ac2e1bb7a63a5ec817ef9bf85"
    ),
    "src/flash_attn_res/kernels/phase_1.py": (
        "bd45efeb8a69b6ff47f2caee66103a56d19494e5a0e50640b3bf4ba5b4048982"
    ),
    "src/flash_attn_res/ops/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "src/flash_attn_res/ops/phase_1.py": (
        "251610079e6a8847c860391641c8903f250e9dd628ec4522e1039ecadb4e7060"
    ),
}
_CATSWE_VENDOR_FILE_HASHES = {
    "LICENSE": _CATSWE_LICENSE_SHA256,
    "pyproject.toml": "081b3067bca515c24edb090678fad477e52ef759a9cfa571449962f0ff63f164",
    "src/flash_attn_res/__init__.py": _CATSWE_SOURCE_HASHES[
        "src/flash_attn_res/__init__.py"
    ],
    "src/flash_attn_res/kernels/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "src/flash_attn_res/kernels/configs.py": _CATSWE_SOURCE_HASHES[
        "src/flash_attn_res/kernels/configs.py"
    ],
    "src/flash_attn_res/kernels/phase_1.py": _CATSWE_SOURCE_HASHES[
        "src/flash_attn_res/kernels/phase_1.py"
    ],
    "src/flash_attn_res/kernels/phase_2.py": (
        "aa2e92e5979d3093d26bbf477bf28daad834cea98b0d35e4225dd1d8c621623d"
    ),
    "src/flash_attn_res/kernels/reduce.py": (
        "8dfdeb9a5031a0a4b25d9cb49e003ea39fd49157ee4a571fe91ff91e5b04ce0b"
    ),
    "src/flash_attn_res/ops/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "src/flash_attn_res/ops/phase_1.py": _CATSWE_SOURCE_HASHES[
        "src/flash_attn_res/ops/phase_1.py"
    ],
    "src/flash_attn_res/ops/phase_2.py": (
        "b56aecf3856be87b48c7891aabb249a7b34f6a59be0af7f89eacb9ff07cd9479"
    ),
}
_MANISH_REVISION = "ea1f63eda8e31b0f10456b3b49cacd8fb66091dc"
_MANISH_TREE = "b6ae55c737f9b17c9c7ea064b17bd0210510496a"
_MANISH_ORIGIN = "https://github.com/manishklach/attnres-kernel-lab.git"
_MANISH_LICENSE_SHA256 = (
    "6d7cc4b730aafd6e596d41c5cb2250c30a9ef8bcffd350fe5e3fe566936a6ebd"
)
_MANISH_SOURCE_HASHES = {
    "src/attnres_kernel/api.py": (
        "5067773602c08aaf7c44837dd15f9d95e839c788199a0385407ae9bb80d81b44"
    ),
    "src/attnres_kernel/triton_impl.py": (
        "520dbf69280f0b200e6b6b2f5ef9bf30863b20778795f7e1b1c23f99176b7e1c"
    ),
}

# Triton Gluon checkpoint 1 statically unrolls the source loop.  This is a
# transparent compile envelope for the pinned source, not a performance
# dispatch table: ``S`` is the source count visible to one operator call and
# ``BD`` is the padded feature width used by the vendor kernel.  Keeping the
# rule here lets the protocol, adapter, and report planner make the same
# decision before allocating or compiling a large case.
GLUON_COMPILE_ENVELOPE = {
    "padded_width_rule": "BD=next_power_of_two(D)",
    "max_padded_width": 4096,
    "source_width_product_rule": "S*BD",
    "max_source_width_product": 2**18,
    "checkpoint1_static_work_rule": "33*S*BD",
    "checkpoint1_static_work_multiplier": 33,
    "max_checkpoint1_static_work": 33 * (2**18),
}


def _capability(
    *,
    family: str,
    adapter: str,
    implementation: str,
    backend: str,
    role: str,
    denominator: bool,
    modes: tuple[str, ...],
    max_sources: int,
    max_width: int,
    timing_max_width: int | None = None,
    max_program_elements: int | None = None,
    requires_power_of_two_width: bool = False,
    dtypes: tuple[str, ...],
    schedules: Mapping[str, str],
    rank_scope: str = "R=D",
    external_route: bool = False,
    model_scope: str = "operator_and_model",
    supports_per_read_block: bool | None = None,
    block_scope: str | None = None,
    revision: str | None = None,
    tree: str | None = None,
    origin: str | None = None,
    source: str | None = None,
    source_hashes: Mapping[str, str] | None = None,
    package_sha256: str | None = None,
    license_name: str | None = None,
    license_sha256: str | None = None,
    notice_sha256: str | None = None,
    pyproject_sha256: str | None = None,
    vendor_file_hashes: Mapping[str, str] | None = None,
    compile_envelope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one JSON friendly capability record."""

    result: dict[str, Any] = {
        "family": family,
        "adapter": adapter,
        "implementation": implementation,
        "backend": backend,
        "role": role,
        "eligible_denominator": denominator,
        "rank_scope": rank_scope,
        "max_sources": max_sources,
        "max_width": max_width,
        "dtypes": list(dtypes),
        "modes": list(modes),
        "schedules": dict(schedules),
        "external_route": external_route,
        "model_scope": model_scope,
        "cuda_required": True,
        "oracle": "validation.oracle.oracle",
    }
    if timing_max_width is not None:
        result["timing_max_width"] = timing_max_width
    if max_program_elements is not None:
        result["max_program_elements"] = max_program_elements
    if requires_power_of_two_width:
        result["requires_power_of_two_width"] = True
    if revision is not None:
        result["revision"] = revision
    if tree is not None:
        result["tree"] = tree
    if origin is not None:
        result["origin"] = origin
    if supports_per_read_block is not None:
        result["supports_per_read_block"] = supports_per_read_block
    if block_scope is not None:
        result["block_scope"] = block_scope
    if source is not None:
        result["source"] = source
    if source_hashes is not None:
        result["source_hashes"] = dict(source_hashes)
    if package_sha256 is not None:
        result["package_sha256"] = package_sha256
    if license_name is not None:
        result["license"] = license_name
    if license_sha256 is not None:
        result["license_sha256"] = license_sha256
    if notice_sha256 is not None:
        result["notice_sha256"] = notice_sha256
    if pyproject_sha256 is not None:
        result["pyproject_sha256"] = pyproject_sha256
    if vendor_file_hashes is not None:
        result["vendor_file_sha256"] = dict(vendor_file_hashes)
    if compile_envelope is not None:
        result["compile_envelope"] = deepcopy(dict(compile_envelope))
    return result


# The public mapping is copied on return by ``capability_for``.  Keeping the
# source values here, rather than obtaining them from an adapter import, is
# what makes this module safe to use during CPU protocol validation.
COMPETITOR_CAPABILITIES: dict[str, dict[str, Any]] = {
    "native_fla_triton_checkpoint1": _capability(
        family="fla",
        adapter="benchmarks.competitors",
        implementation="fla_native",
        backend="triton",
        role="eligible",
        denominator=True,
        modes=("standard_operator", "full", "block_per_read"),
        max_sources=129,
        max_width=8192,
        dtypes=("bf16", "fp32"),
        schedules={"full": "per_read", "block_per_read": "per_read"},
        revision=_FLA_REVISION,
        tree=_FLA_TREE,
        origin=_FLA_ORIGIN,
        source="fla-org/flash-linear-attention",
        source_hashes=_FLA_SOURCE_HASHES,
        package_sha256=_FLA_PACKAGE_SHA256,
        license_name="MIT",
        license_sha256="41a83c8187efc1e3ccc21909e806a9e52338e69448554d9754706c3d1cd610e7",
    ),
    "native_fla_gluon": _capability(
        family="fla",
        adapter="benchmarks.competitors",
        implementation="fla_native",
        backend="gluon",
        role="conditional_eligible",
        denominator=True,
        modes=("standard_operator", "full", "block_per_read"),
        max_sources=129,
        max_width=8192,
        dtypes=("bf16", "fp32"),
        schedules={"full": "per_read", "block_per_read": "per_read"},
        revision=_FLA_REVISION,
        tree=_FLA_TREE,
        origin=_FLA_ORIGIN,
        source="fla-org/flash-linear-attention",
        source_hashes=_FLA_SOURCE_HASHES,
        package_sha256=_FLA_PACKAGE_SHA256,
        license_name="MIT",
        license_sha256="41a83c8187efc1e3ccc21909e806a9e52338e69448554d9754706c3d1cd610e7",
        compile_envelope=GLUON_COMPILE_ENVELOPE,
    ),
    "native_fla_triton_checkpoint0": _capability(
        family="fla",
        adapter="benchmarks.competitors",
        implementation="fla_native",
        backend="triton",
        role="diagnostic",
        denominator=False,
        modes=("standard_operator", "full", "block_per_read"),
        max_sources=129,
        max_width=8192,
        dtypes=("bf16", "fp32"),
        schedules={"full": "per_read", "block_per_read": "per_read"},
        revision=_FLA_REVISION,
        tree=_FLA_TREE,
        origin=_FLA_ORIGIN,
        source="fla-org/flash-linear-attention",
        source_hashes=_FLA_SOURCE_HASHES,
        package_sha256=_FLA_PACKAGE_SHA256,
        license_name="MIT",
        license_sha256="41a83c8187efc1e3ccc21909e806a9e52338e69448554d9754706c3d1cd610e7",
    ),
    "liger": _capability(
        family="liger",
        adapter="benchmarks.liger",
        implementation="liger_native",
        backend="triton",
        role="conditional_eligible",
        denominator=True,
        modes=("standard_operator", "full", "block_per_read"),
        max_sources=32,
        max_width=8192,
        dtypes=("bf16", "fp32"),
        schedules={"full": "per_read", "block_per_read": "per_read"},
        revision=_LIGER_REVISION,
        tree=_LIGER_TREE,
        origin=_LIGER_ORIGIN,
        source="linkedin/Liger-Kernel",
        source_hashes={"src/liger_kernel/ops/attn_res.py": _LIGER_SOURCE_SHA256},
        license_name="BSD-2-Clause",
        license_sha256=_LIGER_LICENSE_SHA256,
        notice_sha256=_LIGER_NOTICE_SHA256,
        pyproject_sha256=_LIGER_PYPROJECT_SHA256,
    ),
    "catswe_phase1": _capability(
        family="catswe",
        adapter="benchmarks.catswe",
        implementation="catswe_native",
        backend="triton",
        role="external_comparator",
        denominator=True,
        modes=("standard_operator",),
        max_sources=129,
        max_width=8192,
        max_program_elements=_CATSWE_MAX_PROGRAM_ELEMENTS,
        requires_power_of_two_width=True,
        dtypes=("bf16",),
        schedules={"standard_operator": "native_phase1_one_query"},
        external_route=False,
        model_scope="standard_operator_only",
        revision=_CATSWE_REVISION,
        tree=_CATSWE_TREE,
        origin=_CATSWE_ORIGIN,
        source="catswe/flash-attn-res",
        source_hashes=_CATSWE_SOURCE_HASHES,
        license_name="Apache-2.0",
        license_sha256=_CATSWE_LICENSE_SHA256,
    ),
    "manish_hydra_2p": _capability(
        family="manish",
        adapter="benchmarks.hydra",
        implementation="manish_hydra_2p",
        backend="triton",
        role="small_d_panel",
        denominator=True,
        modes=("standard_operator", "block_panel"),
        max_sources=129,
        max_width=8192,
        timing_max_width=256,
        dtypes=("bf16", "fp32"),
        schedules={"standard_operator": "native_operator", "block_panel": "native_block"},
        model_scope="small_d_operator_panel_only",
        supports_per_read_block=False,
        block_scope="external_block_panel",
        revision=_MANISH_REVISION,
        tree=_MANISH_TREE,
        origin=_MANISH_ORIGIN,
        source="manishklach/attnres-kernel-lab",
        source_hashes=_MANISH_SOURCE_HASHES,
        license_name="MIT",
        license_sha256=_MANISH_LICENSE_SHA256,
    ),
}


# Model routes are deliberately kept out of ``COMPETITOR_CAPABILITIES``.  The
# latter is the sealed operator protocol surface consumed by
# ``competitor_protocol.planned_comparison_cells``; adding a model-only row
# there would silently change its denominator.  ``eligibility_for`` selects
# this second, explicit scope only when the caller opts into a compiled model
# comparison.
MODEL_COMPETITOR_CAPABILITIES: dict[str, dict[str, Any]] = {
    "catswe_phase1": _capability(
        family="catswe",
        adapter="benchmarks.catswe.make_model_backend",
        implementation="catswe_native_phase1_per_read",
        backend="triton",
        role="external_model_comparator",
        denominator=True,
        modes=("full", "block_per_read"),
        max_sources=129,
        max_width=8192,
        max_program_elements=_CATSWE_MAX_PROGRAM_ELEMENTS,
        requires_power_of_two_width=True,
        dtypes=("bf16",),
        schedules={
            "full": "public_phase1_per_read",
            "block_per_read": "public_phase1_per_read",
        },
        external_route=False,
        model_scope="compiled_training_step",
        supports_per_read_block=True,
        block_scope="public_phase1_per_read",
        revision=_CATSWE_REVISION,
        tree=_CATSWE_TREE,
        origin=_CATSWE_ORIGIN,
        source="catswe/flash-attn-res",
        source_hashes=_CATSWE_SOURCE_HASHES,
        license_name="Apache-2.0",
        license_sha256=_CATSWE_LICENSE_SHA256,
        vendor_file_hashes=_CATSWE_VENDOR_FILE_HASHES,
    ),
}


# These aliases are accepted by API callers, but the protocol uses the
# ``native_*`` FLA names and ``manish_hydra_2p`` so every comparison cell has a
# stable identity.
COMPETITOR_ALIASES = {
    "fla_triton_checkpoint1": "native_fla_triton_checkpoint1",
    "fla_gluon": "native_fla_gluon",
    "fla_triton_checkpoint0": "native_fla_triton_checkpoint0",
    "hydra_2p": "manish_hydra_2p",
    "hydra": "manish_hydra_2p",
    "manish": "manish_hydra_2p",
    "catswe": "catswe_phase1",
}


def canonical_name(name: str) -> str:
    """Return a protocol name for one known adapter alias."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("competitor name must be a nonempty string")
    canonical = COMPETITOR_ALIASES.get(name.strip(), name.strip())
    if canonical not in COMPETITOR_CAPABILITIES:
        raise KeyError(f"unknown competitor {name!r}")
    return canonical


def _capabilities_for_scope(scope: str) -> dict[str, dict[str, Any]]:
    if scope == "operator":
        return COMPETITOR_CAPABILITIES
    if scope == "model":
        return MODEL_COMPETITOR_CAPABILITIES
    raise ValueError("capability scope must be 'operator' or 'model'")


def capability_for(name: str, *, scope: str = "operator") -> dict[str, Any]:
    """Return a detached capability record for one known route.

    ``scope='operator'`` is the default and is the only scope used by the
    sealed matched-operator protocol.  Model-only capabilities are selected
    explicitly so they cannot accidentally enter that protocol's denominator.
    """

    canonical = canonical_name(name)
    capabilities = _capabilities_for_scope(scope)
    try:
        return deepcopy(capabilities[canonical])
    except KeyError as exc:
        raise KeyError(
            f"competitor {canonical!r} has no declared {scope} capability"
        ) from exc


def competitor_capabilities() -> dict[str, dict[str, Any]]:
    """Return detached capability records for all protocol competitors."""

    return deepcopy(COMPETITOR_CAPABILITIES)


def model_competitor_capabilities() -> dict[str, dict[str, Any]]:
    """Return detached capabilities for explicit compiled-model routes."""

    return deepcopy(MODEL_COMPETITOR_CAPABILITIES)


def _value(case: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in case:
            return case[name]
    return None


_MODE_ALIASES = {
    "block": "block_per_read",
    "per_read_block": "block_per_read",
    "operator": "standard_operator",
    "standard": "standard_operator",
    "panel": "block_panel",
}


def _present(case: Mapping[str, Any], *names: str) -> tuple[bool, Any]:
    """Return the first explicitly supplied alias, preserving ``None``."""

    for name in names:
        if name in case:
            return True, case[name]
    return False, None


def _normalise_mode(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    mode = value.strip().lower()
    return _MODE_ALIASES.get(mode, mode)


def eligibility_for(
    name: str,
    case: Mapping[str, Any] | None = None,
    *,
    scope: str = "operator",
    **fields: Any,
) -> dict[str, Any]:
    """Evaluate one cell against a competitor's declared capability.

    ``scope`` is ``"operator"`` by default; ``"model"`` selects a separate
    explicit compiled-model capability map.  ``case`` may contain ``mode``, ``rank``/``R``, ``width``/``D``,
    ``source_count``/``S`` and ``read_source_count``.  Fields supplied as
    keyword arguments are merged over that mapping.  The result includes both
    the boolean decision and a short reason suitable for a JSON report.
    A planned cell must provide its rank and width explicitly.  The adapter
    still performs the final tensor/device and oracle checks before timing.
    """

    original = name
    try:
        capabilities = _capabilities_for_scope(scope)
    except ValueError as exc:
        return {
            "competitor": str(original),
            "status": "not_applicable",
            "eligible": False,
            "eligible_denominator": False,
            "reason": str(exc),
        }
    try:
        canonical = canonical_name(name)
    except (KeyError, ValueError) as exc:
        return {
            "competitor": str(original),
            "status": "not_applicable",
            "eligible": False,
            "eligible_denominator": False,
            "reason": str(exc),
        }
    capability = capabilities.get(canonical)
    if capability is None:
        return {
            "competitor": canonical,
            "status": "not_applicable",
            "eligible": False,
            "eligible_denominator": False,
            "reason": f"{canonical} has no declared {scope} capability",
        }
    if case is not None and not isinstance(case, Mapping):
        return {
            "competitor": canonical,
            "status": "not_applicable",
            "eligible": False,
            "eligible_denominator": False,
            "reason": "eligibility case must be a mapping",
            "capability": deepcopy(capability),
        }
    merged: dict[str, Any] = dict(case or {})
    merged.update(fields)
    mode: str | None = None

    def reject(reason: str) -> dict[str, Any]:
        return {
            "competitor": canonical,
            "status": "not_applicable",
            "eligible": False,
            "eligible_denominator": False,
            "mode": mode,
            "capability_scope": scope,
            "reason": reason,
            "capability": deepcopy(capability),
        }

    mode_values = [merged[name] for name in ("mode", "schedule") if name in merged]
    if not mode_values:
        return reject(f"{canonical} requires an explicit mode")
    if any(not isinstance(value, str) or not value.strip() for value in mode_values):
        return reject("mode must be a nonempty string")
    normalized_modes = {_normalise_mode(value) for value in mode_values}
    if len(normalized_modes) != 1:
        return reject("mode aliases disagree")
    mode = normalized_modes.pop()
    if mode not in capability["modes"]:
        # The registry keeps cached and external panel routes distinct from
        # the public per-read implementation names.
        return reject(f"mode {mode!r} is outside {canonical}'s declared scope")

    dtype_present, dtype_raw = _present(merged, "dtype")
    if not dtype_present:
        return reject(f"{canonical} requires an explicit dtype")
    if not isinstance(dtype_raw, str) or not dtype_raw.strip():
        return reject("dtype must be a nonempty string")
    dtype = dtype_raw.strip().lower()
    if dtype not in capability["dtypes"]:
        return reject(
            f"dtype {dtype_raw!r} is outside {canonical}'s declared scope "
            f"({', '.join(capability['dtypes'])})"
        )

    def parse_int_aliases(
        names: tuple[str, ...], label: str, *, required: bool
    ) -> tuple[int | None, str | None]:
        values = [merged[name] for name in names if name in merged]
        if not values:
            if required:
                return None, f"{canonical} requires an explicit {label}"
            return None, None
        if not required:
            # Protocol cells include ``read_source_count=None`` for standard
            # routes to keep their geometry schema uniform.  Treat that
            # optional, non-relevant marker as omitted, but reject a mixture
            # of None and an actual alias value as ambiguous input.
            non_null = [value for value in values if value is not None]
            if not non_null:
                return None, None
            if len(non_null) != len(values):
                return None, f"{label} must be an integer"
            values = non_null
        if any(type(value) is not int for value in values):
            return None, f"{label} must be an integer"
        if len(set(values)) != 1:
            return None, f"{label} aliases disagree"
        return values[0], None

    # Parse required geometry without accepting bool-as-int or silently
    # treating malformed values as omitted.  The caller may provide the
    # canonical field or its concise S/N/D/R alias, but mode determines which
    # source count is mandatory.
    rank, rank_error = parse_int_aliases(("rank", "R"), "rank R", required=True)
    width, width_error = parse_int_aliases(("width", "D"), "width D", required=True)
    if rank_error is not None:
        return reject(rank_error)
    if width_error is not None:
        return reject(width_error)
    assert rank is not None and width is not None
    if width < 1:
        return reject("width D must be positive")
    if width > int(capability["max_width"]):
        return reject(
            f"width D={width} exceeds {canonical}'s declared limit "
            f"D<={capability['max_width']}"
        )
    if rank < 1:
        return reject("rank R must be positive")

    source_count, source_error = parse_int_aliases(
        ("source_count", "sources", "S"), "source count S", required=False
    )
    read_source_count, read_source_error = parse_int_aliases(
        ("read_source_count", "read_sources", "S_read"),
        "read source count",
        required=False,
    )
    if source_error is not None:
        return reject(source_error)
    if read_source_error is not None:
        return reject(read_source_error)
    if mode in {"standard_operator", "full"}:
        if source_count is None:
            return reject(
                f"{canonical} mode {mode!r} requires an explicit source count S"
            )
    elif mode in {"block_per_read", "block_panel"}:
        if read_source_count is None:
            # A concise S is the per-read source count when no separate total
            # or read count was supplied for a Block panel.
            s_present = any(name in merged for name in ("S",))
            if s_present and source_count is not None:
                read_source_count = source_count
            else:
                return reject(
                    f"{canonical} mode {mode!r} requires an explicit read source count"
                )

    # Gluon's checkpoint-1 source loop and feature tile are constexpr.  The
    # product below is deliberately based on the source count used by this
    # call, so Full and per-read Block cases are checked independently.  A
    # rejected case remains an explicit not_applicable row in the protocol;
    # callers must not silently fall back to another implementation.
    relevant_sources = (
        source_count
        if mode in {"full", "standard_operator"}
        else read_source_count
    )
    assert relevant_sources is not None
    compile_metrics: dict[str, int] = {}
    envelope = capability.get("compile_envelope")
    if envelope is not None:
        padded_width = 1 << (width - 1).bit_length()
        source_width_product = relevant_sources * padded_width
        static_work_score = (
            int(envelope["checkpoint1_static_work_multiplier"])
            * source_width_product
        )
        compile_metrics = {
            "padded_width": padded_width,
            "source_width_product": source_width_product,
            "static_work_score": static_work_score,
        }
        if padded_width > int(envelope["max_padded_width"]):
            return reject(
                f"{canonical} compile envelope rejects BD={padded_width} for D={width}; "
                f"maximum padded width is {envelope['max_padded_width']}"
            )
        if source_width_product > int(envelope["max_source_width_product"]):
            return reject(
                f"{canonical} compile envelope rejects S*BD={source_width_product} "
                f"(S={relevant_sources}, BD={padded_width}); maximum is "
                f"{envelope['max_source_width_product']}"
            )
        if static_work_score > int(envelope["max_checkpoint1_static_work"]):
            return reject(
                f"{canonical} checkpoint-1 static work rejects "
                f"33*S*BD={static_work_score}; maximum is "
                f"{envelope['max_checkpoint1_static_work']}"
            )

    for count, label in (
        (source_count, "source count S"),
        (read_source_count, "read source count"),
    ):
        if count is None:
            continue
        if count < 1:
            return reject(f"{label} must be positive")
        if count > int(capability["max_sources"]):
            return reject(
                f"{label}={count} exceeds {canonical}'s declared limit "
                f"S<={capability['max_sources']}"
            )

    if rank != width:
        return reject(f"{canonical} is restricted to standard R=D (got R={rank}, D={width})")

    rank_flag_present, rank_flag = _present(merged, "rank_equals_width")
    if rank_flag_present and type(rank_flag) is not bool:
        return reject("rank_equals_width must be a boolean")
    if rank_flag_present and rank_flag is not True:
        return reject(f"{canonical} is restricted to standard R=D")

    program_limit = capability.get("max_program_elements")
    if program_limit is not None:
        if capability.get("requires_power_of_two_width") and width & (width - 1):
            return reject(
                f"{canonical} requires power-of-two D for its native tl.arange tile "
                f"(got D={width})"
            )
        padded_sources = 1 << (relevant_sources - 1).bit_length()
        program_elements = padded_sources * width
        if program_elements > int(program_limit):
            return reject(
                f"{canonical} padded source-width tile has {program_elements} elements "
                f"(nextpow2(S)={padded_sources}, D={width}), exceeding the native "
                f"Triton block limit {program_limit}"
            )

    external_values = [
        merged[name]
        for name in ("external_route", "external", "explicit_external_route")
        if name in merged
    ]
    if any(type(value) is not bool for value in external_values):
        return reject("external_route must be a boolean")
    if len(set(external_values)) > 1:
        return reject("external route aliases disagree")
    external_present = bool(external_values)
    external_route = external_values[0] if external_present else False
    if external_present and external_route != capability["external_route"]:
        return reject(
            f"{canonical} does not support the requested external route"
        )

    timing_values = [
        merged[name]
        for name in ("timing", "timing_scope", "for_timing")
        if name in merged
    ]
    if any(type(value) is not bool for value in timing_values):
        return reject("timing must be a boolean")
    if len(set(timing_values)) > 1:
        return reject("timing aliases disagree")
    timing_requested = timing_values[0] if timing_values else False
    timing_limit = capability.get("timing_max_width")
    if timing_requested and timing_limit is not None and width > int(timing_limit):
        return reject(
            f"width D={width} is outside {canonical}'s native timing envelope "
            f"D<={timing_limit}"
        )

    if canonical == "native_fla_triton_checkpoint0":
        return reject("FLA Triton checkpoint 0 is diagnostic-only and excluded from denominators")
    if (
        scope == "operator"
        and canonical == "catswe_phase1"
        and mode != "standard_operator"
    ):
        return reject("Catswe exposes only the native phase-1 standard operator")

    # Full cells use the total source count; per-read Block cells use the
    # completed-source count.  A caller may supply either explicitly.  This
    # is how Liger can be eligible for a nine-source Block read while a
    # forty-nine-source Full cell remains ineligible.
    if canonical == "liger":
        relevant_sources = (
            source_count
            if mode in {"full", "standard_operator"}
            else read_source_count
        )
        if relevant_sources is None:
            return reject(f"Liger {mode} requires an explicit source count")
        if relevant_sources > 32:
            return reject(f"Liger {mode} requires S<=32 (got S={relevant_sources})")

    if not bool(capability["eligible_denominator"]):
        return reject("competitor is diagnostic-only and excluded from denominators")
    return {
        "competitor": canonical,
        "status": "eligible",
        "eligible": True,
        "eligible_denominator": True,
        "mode": mode,
        "dtype": dtype,
        "rank": rank,
        "width": width,
        "source_count": source_count,
        "read_source_count": read_source_count,
        "capability_scope": scope,
        "reason": "inside declared capability; native discovery and oracle gates remain required",
        "capability": deepcopy(capability),
        **compile_metrics,
    }


# Names used by report/runner callers that prefer a predicate style.  The
# detailed result remains available through ``eligibility_for``.
competitor_eligibility = eligibility_for
capability = capability_for


__all__ = [
    "COMPETITOR_ALIASES",
    "COMPETITOR_CAPABILITIES",
    "GLUON_COMPILE_ENVELOPE",
    "MODEL_COMPETITOR_CAPABILITIES",
    "canonical_name",
    "capability",
    "capability_for",
    "competitor_capabilities",
    "model_competitor_capabilities",
    "competitor_eligibility",
    "eligibility_for",
]
