"""Tier 3 cross-feature pairwise combination E2E tests covering Features F1-F18.

Pre-M2/M3/M4 collection safety: zero top-level imports of `sheaf_verify.*`,
`hadukenn_core.*`, `scripts.*`, `numpy`, `scipy`, `torch`, or `yaml`.
All target symbols and scientific libraries are resolved inside test bodies or local helpers.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import re
import shlex
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from tests_e2e.conftest import (
    REPO_ROOT,
    assert_readonly_unchanged,
    mcp_stdio_session,
    no_gpu_env,
    resolve_any_public_symbol,
    resolve_public_symbol,
    run_hook,
    run_sheaf_cli,
    scaffold_mini_selftest_repo,
    snapshot_readonly_hashes,
)

CANONICAL_ANCHOR = "307711a4d841c2c8fa0bbd954d15b2a585012608528c9f53b9ea08ad717bd7c4"

_KERNEL_MODULES = (
    "hadukenn_core.universal_kernel",
    "hadukenn_core.c0_universal_kernel",
    "hadukenn_core.c0_kernel",
    "hadukenn_core.kernel_ops",
    "hadukenn_core.value_graph",
    "hadukenn_core.rtx5070_frontier",
    "hadukenn_core.cuda_parity",
)

_ML_MODULES = (
    "hadukenn_core.zigzag_topology",
    "hadukenn_core.topology_ml",
    "hadukenn_core.ml_synthesis",
    "hadukenn_core.kernel_policies",
    "hadukenn_core.architecture_policy",
    "hadukenn_core.hypergraph",
    "hadukenn_core.partitions",
    "hadukenn_core.heads",
)

_RLM_MODULES = (
    "hadukenn_core.rlm_engine",
    "hadukenn_core.rlm_repl",
    "hadukenn_core.rlm_runtime",
    "hadukenn_core.rlm_context_engine",
)

_ANTIGRAVITY_PREFERRED_MODULES = (
    "hadukenn_core.antigravity_sdk",
    "hadukenn_core.antigravity_adapter",
    "hadukenn_core.antigravity",
    "hadukenn_core.antigravity_bridge",
    "hadukenn_core.antigravity_multimodal",
)

_WRITE_RUNTIME_TOOLS = frozenset(
    {
        "write_to_file",
        "replace_file_content",
        "multi_replace_file_content",
    }
)


def _bind_with_aliases(
    fn: Any,
    alias_values: list[tuple[tuple[str, ...], Any]],
    positional_fallback: tuple[Any, ...],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    sig = inspect.signature(fn)
    params = sig.parameters
    param_list = list(params.values())
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in param_list)
    pos_args: list[Any] = []
    kwargs: dict[str, Any] = {}
    used_indices: set[int] = set()

    for p in param_list:
        if p.kind != inspect.Parameter.POSITIONAL_ONLY:
            continue
        matched_val = None
        matched_idx = None
        for idx, (aliases, val) in enumerate(alias_values):
            if idx not in used_indices and p.name in aliases:
                matched_val = val
                matched_idx = idx
                break
        if matched_idx is not None:
            pos_args.append(matched_val)
            used_indices.add(matched_idx)
        elif p.default is not inspect.Parameter.empty:
            break
        else:
            sig.bind(*positional_fallback)
            return positional_fallback, {}

    for idx, (aliases, val) in enumerate(alias_values):
        if idx in used_indices:
            continue
        matched = False
        for name in aliases:
            if name in params and params[name].kind != inspect.Parameter.POSITIONAL_ONLY:
                kwargs[name] = val
                matched = True
                used_indices.add(idx)
                break
        if not matched and has_varkw:
            kwargs[aliases[0]] = val
            used_indices.add(idx)

    if len(used_indices) == len(alias_values) or all(
        p.default is not inspect.Parameter.empty
        or p.name in kwargs
        or (p.kind == inspect.Parameter.POSITIONAL_ONLY and param_list.index(p) < len(pos_args))
        or p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for p in param_list
    ):
        sig.bind(*pos_args, **kwargs)
        return tuple(pos_args), kwargs

    sig.bind(*positional_fallback)
    return positional_fallback, {}


def _call_with_aliases(
    fn: Any,
    alias_values: list[tuple[tuple[str, ...], Any]],
    positional_fallback: tuple[Any, ...],
) -> Any:
    args, kwargs = _bind_with_aliases(fn, alias_values, positional_fallback)
    return fn(*args, **kwargs)


def _bind_call(fn: Any, kwargs: dict[str, Any], positional: tuple[Any, ...] = ()) -> Any:
    sig = inspect.signature(fn)
    params = sig.parameters
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    filtered = {k: v for k, v in kwargs.items() if has_varkw or k in params}
    if all(
        p.default is not inspect.Parameter.empty
        or p.name in filtered
        or p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for p in params.values()
    ):
        return fn(**filtered)
    return fn(*positional)


def _has_symbol(package_name: str, symbol_name: str, preferred_modules: tuple[str, ...]) -> bool:
    try:
        resolve_public_symbol(package_name, symbol_name, preferred_modules)
        return True
    except AssertionError:
        return False


def _resolve_op(long_name: str, short_name: str) -> Any:
    return resolve_any_public_symbol("hadukenn_core", (long_name, short_name), _KERNEL_MODULES)


def _resolve_unit(name: str) -> Any:
    if _has_symbol("hadukenn_core", name, _KERNEL_MODULES):
        return resolve_public_symbol("hadukenn_core", name, _KERNEL_MODULES)
    return resolve_public_symbol("sheaf_verify", name)


def _verdict_passed(result: Any) -> bool:
    if isinstance(result, bool):
        return result
    for attr in ("passed", "ok", "accepted", "approved", "allowed", "valid"):
        if hasattr(result, attr) and isinstance(getattr(result, attr), bool):
            return bool(getattr(result, attr))
    if isinstance(result, tuple) and result and isinstance(result[0], bool):
        return result[0]
    if isinstance(result, dict):
        for key in ("ok", "passed", "accepted", "approved", "allowed", "valid"):
            if key in result and isinstance(result[key], bool):
                return bool(result[key])
        if "status" in result and isinstance(result["status"], str):
            st = result["status"].lower()
            if st in {"ok", "pass", "passed", "success", "allow", "approved"}:
                return True
            if st in {"error", "failed", "fail", "deny", "rejected"}:
                return False
    raise AssertionError(f"Unrecognized verdict return shape: {result!r}")


def _assert_rejected(
    fn: Any,
    *args: Any,
    extra_excs: tuple[type[BaseException], ...] = (),
    **kwargs: Any,
) -> None:
    sheaf_err = resolve_public_symbol("sheaf_verify", "SheafError", ("sheaf_verify.hashing",))
    caught = (sheaf_err, ValueError, AssertionError, *extra_excs)
    try:
        res = fn(*args, **kwargs)
    except caught:
        return
    assert _verdict_passed(res) is False, f"Expected rejection, got passing result {res!r}"


def _extract_attr_or_key(obj: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AssertionError(f"Missing expected attribute/key from {names!r} in {obj!r}")


def _load_script_module(repo_root: Path, script_rel: str, module_name: str) -> Any:
    script_path = repo_root / script_rel
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None, f"Cannot load {script_path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_hal_parts(work: Path) -> dict[str, Any]:
    cce_fn = resolve_public_symbol("sheaf_verify", "cce_mup_loss_f64", ("sheaf_verify.hal",))
    code = work / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "ref.py").write_text("def numpy_gemm_cce():\n    return None\n", encoding="utf-8")
    host = work / "host_ref.py"
    host.write_text("def numpy_gemm_cce():\n    return None\n", encoding="utf-8")
    kernel = work / "kernel.c"
    kernel.write_text("void hadukenn_tiled_cce_mup_fwd_bwd_f32(void) {}\n", encoding="utf-8")

    hidden = [[0.2, -0.1, 0.05, 0.3], [0.0, 0.4, -0.2, 0.1]]
    embedding = [
        [0.1, 0.0, -0.1, 0.2],
        [0.3, -0.2, 0.05, 0.0],
        [-0.1, 0.2, 0.1, -0.05],
    ]
    targets = [1, 0]
    mup_base_d = 4.0
    loss = cce_fn(hidden, embedding, targets, mup_base_d)
    env = {
        "python": "3.11.9",
        "cuda": "release 12.4",
        "driver": "595.99.02",
        "gpu": "NVIDIA GeForce RTX 4080 Laptop GPU (12282 MiB, sm_89)",
        "platform": "Linux-test-harness",
    }
    claim = {
        "format": "sheaf-verify.hal-claim.v1",
        "study_profile": "hal",
        "hypothesis_id": "e2e-hal-t3",
        "description": (
            "The compiled C11 tiled CCE kernel is faster than a serial Python loop on the recorded shapes."
        ),
        "performance": {
            "metric": "speedup",
            "direction": "greater",
            "baseline": "serial_python_loop",
            "baselines_reported": ["serial_python_loop", "packed_numpy"],
            "min_speedup": 1.5,
            "subject": "c11_tiled_cce",
        },
        "parity": {"max_abs": 1e-4, "ops": ["cce_mup_fwd"]},
        "guards": {"thermal_stop_celsius": 85},
    }
    parity = {
        "format": "sheaf-verify.hal-parity.v1",
        "cases": [
            {
                "name": "cce_tiny",
                "op": "cce_mup_fwd",
                "inputs": {"H": hidden, "E": embedding, "y": targets, "mup_base_d": mup_base_d},
                "kernel_outputs": {"loss": loss},
            }
        ],
    }
    timings = {
        "format": "sheaf-verify.hal-timings.v1",
        "cases": [
            {
                "name": "shape_small",
                "subject_ms": [2.0, 2.1, 1.9],
                "baselines_ms": {
                    "serial_python_loop": [10.0, 10.2, 9.8],
                    "packed_numpy": [1.0, 1.1, 0.9],
                },
            }
        ],
    }
    guards = {
        "format": "sheaf-verify.hal-guards.v1",
        "thermal": {
            "stop_celsius": 85,
            "unit": "C",
            "samples": [
                {"t": "2026-01-01T00:00:10Z", "celsius": 45},
                {"t": "2026-01-01T00:00:20Z", "celsius": 62},
            ],
        },
        "power": {
            "observed": False,
            "probe": "nvidia-smi --query-gpu=power.draw --format=csv,noheader",
            "reason": "unavailable in CPU container",
        },
    }
    return {
        "code": code,
        "host": host,
        "kernel": kernel,
        "env": env,
        "claim": claim,
        "config": {"format": "sheaf-verify.config.v1", "study": "hal-e2e-t3"},
        "metrics": {"format": "sheaf-verify.metrics.note.v1", "note": "HAL fixture; not a G0 record"},
        "parity": parity,
        "timings": timings,
        "guards": guards,
    }


def _register_hal_store(store_dir: Path, parts: dict[str, Any]) -> Any:
    store_cls = resolve_public_symbol("sheaf_verify", "Store", ("sheaf_verify.store",))
    canonical_bytes = resolve_public_symbol("sheaf_verify", "canonical_bytes", ("sheaf_verify.hashing",))
    store = store_cls(store_dir)
    store.init()
    code_sha, _ = store.register_tree(role="code", path=parts["code"], registered_at="2026-01-01T00:00:02Z")
    store.register_environment(role="environment", registered_at="2026-01-01T00:00:03Z", **parts["env"])
    host_sha, host_man = store.register(
        kind="kernel",
        role="kernel.host",
        registered_at="2026-01-01T00:00:04Z",
        payload=parts["host"].read_bytes(),
        metadata={"backend": "cpu", "entry": "numpy_gemm_cce"},
    )
    store.register(
        kind="kernel",
        role="kernel",
        registered_at="2026-01-01T00:00:05Z",
        payload=parts["kernel"].read_bytes(),
        inputs=[host_sha],
        metadata={
            "backend": "c",
            "entry": "hadukenn_tiled_cce_mup_fwd_bwd_f32",
            "host_reference_sha256": host_man["payload_sha256"],
        },
    )
    store.register(
        kind="config",
        role="config",
        registered_at="2026-01-01T00:00:06Z",
        payload=canonical_bytes(parts["config"]),
    )
    claim_sha, _ = store.register(
        kind="claim",
        role="claim",
        registered_at="2026-01-01T00:00:07Z",
        payload=canonical_bytes(parts["claim"]),
    )
    store.register(
        kind="metrics",
        role="metrics",
        registered_at="2026-01-01T00:00:08Z",
        payload=canonical_bytes(parts["metrics"]),
        inputs=[code_sha],
    )
    store.register(
        kind="log",
        role="log.parity",
        registered_at="2026-01-01T00:00:09Z",
        payload=canonical_bytes(parts["parity"]),
        inputs=[claim_sha],
    )
    store.register(
        kind="log",
        role="log.timings",
        registered_at="2026-01-01T00:00:10Z",
        payload=canonical_bytes(parts["timings"]),
        inputs=[claim_sha],
    )
    store.register(
        kind="log",
        role="log.guards",
        registered_at="2026-01-01T00:00:11Z",
        payload=canonical_bytes(parts["guards"]),
    )
    return store


def _make_inline_ctx(docs: dict[str, Any]) -> Any:
    parse_json_bytes = resolve_public_symbol("sheaf_verify", "parse_json_bytes", ("sheaf_verify.hashing",))

    class _InlineContext:
        def __init__(self, items: dict[str, Any]) -> None:
            self._raw = {k: json.dumps(v).encode("utf-8") for k, v in items.items() if v is not None}

        def has(self, role: str) -> bool:
            return role in self._raw

        def payload(self, role: str) -> bytes:
            return self._raw[role]

        def parse(self, role: str) -> Any:
            return parse_json_bytes(self._raw[role], role)

    return _InlineContext(docs)


def _build_kernel_state(
    n_steps: int = 128,
    m_streams: int = 4,
    *,
    check_properties: bool = True,
    pareto_scales: list[int] | None = None,
    seed: int = 20261108,
) -> Any:
    import numpy as np

    state_cls = resolve_public_symbol("hadukenn_core", "KernelState", _KERNEL_MODULES)
    second = _resolve_unit("SECOND")
    watt = _resolve_unit("WATT")
    hash_rate = _resolve_unit("HASH_RATE")
    dimless = _resolve_unit("DIMENSIONLESS")

    rng = np.random.default_rng(seed)
    t_base = np.linspace(1.0, float(n_steps), n_steps)
    s0 = np.sin(0.12 * t_base) + 0.45 * np.cos(0.31 * t_base) + rng.normal(0.0, 0.12, n_steps)
    s1 = np.cos(0.10 * t_base - 0.2) + 0.55 * np.sin(0.24 * t_base) + rng.normal(0.0, 0.14, n_steps)
    s2 = 0.65 * s1 * np.sin(0.18 * t_base) + rng.normal(0.0, 0.11, n_steps)
    s3 = 0.80 * s2 + 0.50 * (s0 * s1) + rng.normal(0.0, 0.11, n_steps)
    s4 = 0.45 * np.sin(0.22 * t_base) + 0.35 * s0 + rng.normal(0.0, 0.10, n_steps)

    streams_all = [
        ("m0_ai_power", {"t": t_base, "values": s0, "time_unit": second, "val_unit": watt}),
        (
            "m1_grid_freq",
            {"t": (t_base - 0.05) / 1.0001, "values": s1, "time_unit": second, "val_unit": hash_rate},
        ),
        (
            "m2_mempool_fee",
            {"t": (t_base + 0.12) / 0.9999, "values": s2, "time_unit": second, "val_unit": dimless},
        ),
        ("m3_vol_skew", {"t": t_base, "values": s3, "time_unit": second, "val_unit": dimless}),
        ("m4_thermal_headroom", {"t": t_base, "values": s4, "time_unit": second, "val_unit": dimless}),
    ]
    raw_streams = dict(streams_all[:m_streams])
    clock_relations: dict[str, dict[str, float]] = {}
    if m_streams >= 2:
        clock_relations["m1_grid_freq"] = {"offset_s": 0.05, "rate": 1.0001, "uncertainty_s": 1e-4}
    if m_streams >= 3:
        clock_relations["m2_mempool_fee"] = {"offset_s": -0.12, "rate": 0.9999, "uncertainty_s": 1e-4}

    obj_series = 0.55 * np.hypot(s1, s3) + 0.45 * np.abs(s0 * s2)
    need_spec = {
        "intent": "local_to_global_regime_and_slice_mining",
        "primary_stream": "m1_grid_freq" if m_streams >= 2 else "m0_ai_power",
        "objective_series": obj_series,
    }
    mode = {"check_properties": check_properties, "pareto_scales": pareto_scales or [256, 512]}
    return state_cls(raw_streams, clock_relations, need_spec, mode=mode)


def _run_op21(n_steps: int = 128, m_streams: int = 4, check_properties: bool = True, seed: int = 20261108) -> Any:
    compose_fn = resolve_public_symbol("hadukenn_core", "compose", _KERNEL_MODULES)
    ops = [
        _resolve_op("op_01_ingest_and_clock_sync", "op_01"),
        _resolve_op("op_02_endogenous_scale_solver", "op_02"),
        _resolve_op("op_03_selected_index_pullback", "op_03"),
        _resolve_op("op_04_events_and_interval_matrix", "op_04"),
        _resolve_op("op_05_recurrence_and_visibility_graphs", "op_05"),
        _resolve_op("op_06_coboundary_and_sheaf_resolvent", "op_06"),
        _resolve_op("op_21_value_network_graph_alignment_policy_engine", "op_21"),
    ]
    s = _build_kernel_state(n_steps=n_steps, m_streams=m_streams, check_properties=check_properties, seed=seed)
    return compose_fn(*ops)(s)


def _assert_pi5_node_coverage(policies: dict[str, Any], expected_n: int) -> None:
    coverage_keys = (
        "pi_5_covered_node_count",
        "pi_5_node_policy_weights",
        "pi_5_chunk_sizes",
        "pi_5_chunk_slices",
    )
    present = [k for k in coverage_keys if k in policies]
    assert present, f"Missing explicit pi_5 node coverage telemetry from {coverage_keys!r} in {sorted(policies)!r}"
    if "pi_5_covered_node_count" in policies:
        assert int(policies["pi_5_covered_node_count"]) == expected_n
    if "pi_5_node_policy_weights" in policies:
        assert len(policies["pi_5_node_policy_weights"]) == expected_n
    if "pi_5_chunk_sizes" in policies:
        sizes = [int(x) for x in policies["pi_5_chunk_sizes"]]
        assert sum(sizes) == expected_n and all(sz > 0 for sz in sizes)
    if "pi_5_chunk_slices" in policies:
        slices = policies["pi_5_chunk_slices"]
        covered: list[int] = []
        if all(isinstance(s, slice) for s in slices):
            for sl in slices:
                covered.extend(range(*sl.indices(expected_n)))
        elif (
            slices
            and all(isinstance(s, (tuple, list)) and len(s) == 2 and all(isinstance(v, int) for v in s) for s in slices)
            and slices[0][0] == 0
            and slices[-1][1] == expected_n
            and all(slices[i][1] == slices[i + 1][0] for i in range(len(slices) - 1))
        ):
            for start, stop in slices:
                covered.extend(range(start, stop))
        else:
            for chunk in slices:
                covered.extend(int(idx) for idx in chunk)
        assert len(covered) == expected_n and set(covered) == set(range(expected_n))


def _numpy_sm120_warp_julia_ref(c_h_in: Any, gamma_sheaf: float) -> tuple[Any, Any]:
    import numpy as np

    c = np.asarray(c_h_in, dtype=np.float32)
    n = int(c.shape[0])
    gamma = np.float32(gamma_sheaf)

    def qmul_f32(p: Any, q: Any) -> Any:
        pw, px, py, pz = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        ow = pw * qw - px * qx - py * qy - pz * qz
        ox = pw * qx + px * qw + py * qz - pz * qy
        oy = pw * qy - px * qz + py * qw + pz * qx
        oz = pw * qz + px * qy - py * qx + pz * qw
        return np.stack([ow, ox, oy, oz], axis=-1).astype(np.float32)

    q = c.copy()
    lanes = np.arange(n, dtype=np.int32) & 31
    for offset in (1, 2, 4, 8, 16):
        prev = np.zeros_like(q)
        if offset < n:
            prev[offset:] = q[:-offset]
        mask = lanes >= offset
        updated = qmul_f32(prev, q)
        q = np.where(mask[:, None], updated, q)

    for _ in range(4):
        sq = qmul_f32(q, q)
        qx, qy, qz = q[..., 1], q[..., 2], q[..., 3]
        cw, cx, cy, cz = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
        lx = np.float32(2.0) * (qy * cz - qz * cy)
        ly = np.float32(2.0) * (qz * cx - qx * cz)
        lz = np.float32(2.0) * (qx * cy - qy * cx)

        nw = sq[..., 0] + cw
        nx = sq[..., 1] + cx + gamma * lx
        ny = sq[..., 2] + cy + gamma * ly
        nz = sq[..., 3] + cz + gamma * lz

        msq = np.float32(0.25) * (nw * nw + nx * nx + ny * ny + nz * nz)
        gain = (np.float32(3.0) + msq) / (np.float32(1.0) + np.float32(3.0) * msq)
        q = np.stack([nw * gain, nx * gain, ny * gain, nz * gain], axis=-1).astype(np.float32)

    p5 = np.sqrt(q[..., 2] ** 2 + q[..., 3] ** 2).astype(np.float32)
    return q, p5


def _parse_frontmatter(text: str, source: Path) -> tuple[dict[str, Any], str]:
    normalized = text.replace("\r\n", "\n")
    assert normalized.startswith("---\n"), f"{source} missing opening YAML frontmatter fence"
    lines = normalized.split("\n")
    end_idx: int | None = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
    assert end_idx is not None, f"{source} missing closing YAML frontmatter fence"

    meta: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in lines[1:end_idx]:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if raw_line.lstrip().startswith("- ") and current_list_key is not None:
            item_raw = raw_line.lstrip()[2:].strip()
            existing = meta.get(current_list_key)
            assert isinstance(existing, list), f"{source} invalid YAML list under {current_list_key!r}"
            meta_item = item_raw.strip().strip('"').strip("'")
            existing.append(meta_item)
            continue
        assert ":" in raw_line, f"{source} malformed YAML line: {raw_line!r}"
        key, raw_val = raw_line.split(":", 1)
        key = key.strip()
        val_stripped = raw_val.strip()
        if val_stripped == "":
            meta[key] = []
            current_list_key = key
        else:
            current_list_key = None
            unquoted = val_stripped.strip('"').strip("'")
            if val_stripped.startswith("[") and val_stripped.endswith("]"):
                inner = val_stripped[1:-1].strip()
                meta[key] = [p.strip().strip('"').strip("'") for p in inner.split(",") if p.strip()] if inner else []
            elif val_stripped.lower() == "true":
                meta[key] = True
            elif val_stripped.lower() == "false":
                meta[key] = False
            else:
                meta[key] = unquoted
    return meta, "\n".join(lines[end_idx + 1 :])


def _resolve_hook_command(raw_cmd: str, base_dir: Path) -> list[str]:
    parts = shlex.split(raw_cmd)
    assert parts, f"Empty hook command: {raw_cmd!r}"
    resolved: list[str] = []
    for idx, part in enumerate(parts):
        if idx == 0 and Path(part).name in {"python", "python3"}:
            resolved.append(sys.executable)
        elif part.startswith("./") or part.startswith(".agents/"):
            candidate = (base_dir / part).resolve()
            resolved.append(str(candidate) if candidate.exists() else part)
        else:
            resolved.append(part)
    return resolved


def _matching_pretooluse_commands(hooks_path: Path, tool_name: str) -> list[list[str]]:
    data = json.loads(hooks_path.read_text(encoding="utf-8"))
    groups = data.get("PreToolUse")
    assert isinstance(groups, list) and groups, f"Missing PreToolUse groups in {hooks_path}"
    cmds: list[list[str]] = []
    for group in groups:
        matcher = group.get("matcher", "")
        assert isinstance(matcher, str) and matcher
        if re.search(matcher, tool_name):
            for handler in group.get("hooks", []):
                raw_cmd = handler.get("command")
                assert isinstance(raw_cmd, str) and raw_cmd.strip()
                cmds.append(_resolve_hook_command(raw_cmd, REPO_ROOT))
    assert cmds, f"No PreToolUse hook matched tool {tool_name!r}"
    return cmds


def _make_stdin_payload(tmp_path: Path, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "conversationId": "conv-antigravity-e2e-t3-0123456789abcdef",
        "workspacePaths": [str(REPO_ROOT)],
        "transcriptPath": str(tmp_path / "transcript.jsonl"),
        "artifactDirectoryPath": str(tmp_path / "artifacts"),
        "modelName": "gemini-2.5-pro",
        "lastUserInput": "Run Tier 3 pairwise Antigravity governance verification.",
        "toolName": tool_name,
        "toolInput": args,
        "toolCall": {"name": tool_name, "args": args},
    }


# ==============================================================================
# Tier 3 Pairwise Integration Tests (21 tests across all 18 features F1-F18)
# ==============================================================================


@pytest.mark.tier3
def test_t3_p01_f1_runner_hal_with_f6_f7_guard_parity(tmp_path: Path) -> None:
    set_probe = resolve_public_symbol("sheaf_verify", "set_probe", ("sheaf_verify.hal",))
    run_fn = resolve_public_symbol("sheaf_verify", "run", ("sheaf_verify.runner",))
    seq_clock = resolve_public_symbol("sheaf_verify", "SequenceClock", ("sheaf_verify.runner",))
    verify_store = resolve_public_symbol("sheaf_verify", "verify_store", ("sheaf_verify.verify",))
    test_key = resolve_public_symbol("sheaf_verify", "TEST_KEY", ("sheaf_verify.synthetic",))

    parts_ok = _make_hal_parts(tmp_path / "p01_ok")
    store_ok = _register_hal_store(tmp_path / "p01_ok_store", parts_ok)
    set_probe(lambda: dict(parts_ok["env"]))
    try:
        records_ok = run_fn(store_ok, key=test_key, clock=seq_clock(), profile="hal")
        rep = verify_store(store_ok, key=test_key)
    finally:
        set_probe(None)

    assert [f"HAL{i}" for i in range(7)] == [r["gate_id"] for r in records_ok]
    assert all(r["status"] == "pass" for r in records_ok)
    assert rep.ok is True and rep.problems == []

    parts_bad = _make_hal_parts(tmp_path / "p01_bad")
    parts_bad["claim"]["guards"]["thermal_stop_celsius"] = 60
    store_bad = _register_hal_store(tmp_path / "p01_bad_store", parts_bad)
    set_probe(lambda: dict(parts_bad["env"]))
    try:
        records_bad = run_fn(store_bad, key=test_key, clock=seq_clock(), profile="hal")
    finally:
        set_probe(None)

    bad_by_gate = {r["gate_id"]: r for r in records_bad}
    assert bad_by_gate["HAL4"]["status"] == "pass"
    assert bad_by_gate["HAL6"]["status"] == "fail"


@pytest.mark.tier3
def test_t3_p02_f2_selftest_audit_with_f1_f10_stdlib_and_anchor(repo_root: Path, tmp_path: Path) -> None:
    canonical_anchor = resolve_public_symbol("sheaf_verify", "CANONICAL_ANCHOR", ("sheaf_verify.selftest",))
    run_full = resolve_public_symbol("sheaf_verify", "run_full_verification", ("sheaf_verify.selftest",))
    audit_stdlib = resolve_public_symbol("sheaf_verify", "audit_python_style_stdlib", ("sheaf_verify.selftest",))

    assert canonical_anchor == CANONICAL_ANCHOR
    mini = scaffold_mini_selftest_repo(repo_root, tmp_path / "mini_p02", create_default_test=True)
    _ok_flag, report = run_full(mini)
    assert report["canonical_fixture"]["ok"] is True
    assert report["canonical_fixture"]["anchor_sha256"] == canonical_anchor
    assert report["lint"]["ok"] is True
    assert report["pytest_suite"]["failed"] == 0

    (mini / "src" / "sheaf_verify" / "bad_non_stdlib.py").write_text("import numpy as np\n", encoding="utf-8")
    issues = audit_stdlib(mini)
    assert any("numpy" in item for item in issues), f"Expected numpy stdlib violation, got {issues!r}"
    bad_ok, bad_report = run_full(mini)
    assert bad_ok is False
    assert bad_report["lint"]["ok"] is False


@pytest.mark.tier3
def test_t3_p03_f3_fraction_consensus_into_f4_post_state_validation() -> None:
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))

    ont = ont_cls(
        "t3-p03-policy",
        [rule_cls(action_name="pump", dimension="volume")],
    )
    ok_verdict = ont.evaluate_transfer_action(
        action_name="pump",
        actor_role="operator",
        src_entity="tank_a",
        dst_entity="tank_b",
        src_witnesses={
            "w1": (Fraction(1, 3), "L"),
            "w2": (Fraction(1, 3), "L"),
            "w3": ("99", "L"),
        },
        dst_initial=(Fraction(2, 3), "L"),
        transfer_amount=(Fraction(1, 6), "L"),
        claimed_post_src=(Fraction(1, 6), "L"),
        claimed_post_dst=("5/6", "L"),
    )
    assert ok_verdict.verdict == "RECOVERABLE_FAULT"
    assert ok_verdict.approved is True
    assert ok_verdict.isolated_fault_sources == ("w3",)
    assert ok_verdict.resolved_initial_src == "1/3"

    bad_verdict = ont.evaluate_transfer_action(
        action_name="pump",
        actor_role="operator",
        src_entity="tank_a",
        dst_entity="tank_b",
        src_witnesses={
            "w1": (Fraction(1, 3), "L"),
            "w2": (Fraction(1, 3), "L"),
            "w3": ("99", "L"),
        },
        dst_initial=(Fraction(2, 3), "L"),
        transfer_amount=(Fraction(1, 6), "L"),
        claimed_post_src=("1", "USD"),
        claimed_post_dst=("5/6", "L"),
    )
    assert bad_verdict.verdict == "INVALID"
    assert bad_verdict.approved is False


@pytest.mark.tier3
def test_t3_p04_f3_f4_verdicts_bound_into_f5_regulatory_bundle() -> None:
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))
    emit_fn = resolve_public_symbol("sheaf_verify", "emit_regulatory_bundle", ("sheaf_verify.compliance",))
    verify_fn = resolve_public_symbol("sheaf_verify", "verify_regulatory_bundle", ("sheaf_verify.compliance",))

    ont = ont_cls(
        "t3-p04-policy",
        [rule_cls(action_name="transfer", dimension="mass")],
    )
    v_valid = ont.evaluate_transfer_action(
        action_name="transfer",
        actor_role="operator",
        src_entity="bin_1",
        dst_entity="bin_2",
        src_witnesses={"w1": (Fraction(10), "kg"), "w2": (Fraction(10), "kg")},
        dst_initial=(Fraction(2), "kg"),
        transfer_amount=(Fraction(3), "kg"),
        claimed_post_src=(Fraction(7), "kg"),
        claimed_post_dst=(Fraction(5), "kg"),
    )
    v_tie = ont.evaluate_transfer_action(
        action_name="transfer",
        actor_role="operator",
        src_entity="bin_1",
        dst_entity="bin_2",
        src_witnesses={"w1": (10**17, "kg"), "w2": (10**17 + 1, "kg")},
        dst_initial=(Fraction(2), "kg"),
        transfer_amount=(Fraction(1), "kg"),
    )
    v_bad_dim = ont.evaluate_transfer_action(
        action_name="transfer",
        actor_role="operator",
        src_entity="bin_1",
        dst_entity="bin_2",
        src_witnesses={"w1": (Fraction(10), "kg"), "w2": (Fraction(10), "kg")},
        dst_initial=(Fraction(2), "kg"),
        transfer_amount=(Fraction(3), "kg"),
        claimed_post_src=(Fraction(7), "L"),
        claimed_post_dst=(Fraction(5), "kg"),
    )
    assert v_valid.verdict == "VALID"
    assert v_tie.verdict == "ABSTAIN_TIE"
    assert v_bad_dim.verdict == "INVALID"

    events = [v_valid.as_dict(), v_tie.as_dict(), v_bad_dim.as_dict()]
    key = b"t3-p04-regulatory-signing-key-001"
    bundle = emit_fn(
        system_id="sys-t3-p04",
        trace_head_sha256="a" * 64,
        state_digest_sha256="b" * 64,
        policy_sha256=ont.policy_sha256,
        events=events,
        created_at="2026-11-28T12:00:00Z",
        signing_key=key,
        retention_months=12,
    )
    ok_flag, errs = verify_fn(bundle, signing_key=key)
    assert ok_flag is True and errs == []

    stmt = json.loads(base64.b64decode(bundle["dsse_statement_envelope"]["payload"]).decode("utf-8"))
    pred = stmt["predicate"]
    assert pred["eu_ai_act_article_12"]["escalated_event_indices"] == [1]
    assert pred["eu_ai_act_article_12"]["refused_event_indices"] == [1, 2]
    assert pred["finra_rule_3110_worm"]["policy_snapshot_sha256"] == ont.policy_sha256

    tampered = json.loads(json.dumps(bundle))
    tampered["events_sha256"] = ("0" if bundle["events_sha256"][0] != "0" else "1") + bundle["events_sha256"][1:]
    bad_ok, bad_errs = verify_fn(tampered, signing_key=key)
    assert bad_ok is False and len(bad_errs) >= 1


@pytest.mark.tier3
def test_t3_p05_f6_hal4_finite_guard_with_f8_cluster_is_outputs(repo_root: Path, tmp_path: Path) -> None:
    import numpy as np

    eval_hal4 = resolve_public_symbol("sheaf_verify", "evaluate_hal4", ("sheaf_verify.hal",))
    cce_fn = resolve_public_symbol("sheaf_verify", "cce_mup_loss_f64", ("sheaf_verify.hal",))
    build_part = resolve_public_symbol("hadukenn_core", "build_balanced_partition", ("hadukenn_core.partitions",))
    cluster_is = resolve_public_symbol("hadukenn_core", "cluster_is_fwd_bwd", ("hadukenn_core.heads",))
    jovin = _load_script_module(repo_root, "scripts/jovin_cpu_payload.py", "e2e_jovin_t3_p05")

    rng = np.random.default_rng(305)
    nc, s_sz, d, n = 4, 4, 8, 4
    v = nc * s_sz
    part = build_part(v, nc, s_sz, [rng.integers(0, v, size=64, dtype=np.int32)], mode="mass_random", seed=305)
    H = rng.normal(0.0, 0.25, size=(n, d)).astype(np.float64)
    E = rng.normal(0.0, 0.25, size=(v, d)).astype(np.float64)
    delta_root = np.zeros((nc, d), dtype=np.float64)
    targets = rng.integers(0, v, size=n, dtype=np.int64)

    loss_is, root_nll, *_ = cluster_is(H, E, delta_root, part, targets, k_neg=nc - 1, rng=rng, compute_grad=True)
    assert np.all(np.isfinite(loss_is)) and math.isfinite(root_nll)

    mup_base_d = float(d)
    loss_ref = cce_fn(H.tolist(), E.tolist(), targets.tolist(), mup_base_d)
    assert np.allclose(loss_is, np.asarray(loss_ref, dtype=np.float64), atol=1e-5)

    parts = _make_hal_parts(tmp_path / "p05")
    parts["parity"]["cases"][0]["inputs"] = {
        "H": H.tolist(),
        "E": E.tolist(),
        "y": targets.tolist(),
        "mup_base_d": mup_base_d,
    }
    parts["parity"]["cases"][0]["kernel_outputs"] = {"loss": [float(x) for x in loss_is]}
    ctx_ok = _make_inline_ctx({"claim": parts["claim"], "log.parity": parts["parity"]})
    assert all(c.passed for c in eval_hal4(ctx_ok))
    assert jovin.all_passed(jovin.evaluate_parity(parts["parity"], parts["claim"]))

    bad_parity = json.loads(json.dumps(parts["parity"]))
    bad_parity["cases"][0]["kernel_outputs"]["loss"][0] = float("nan")
    assert jovin.all_passed(jovin.evaluate_parity(bad_parity, parts["claim"])) is False
    # Inline context using raw float NaN payload rejected by HAL4 finite guard
    ctx_nan = _make_inline_ctx({"claim": parts["claim"], "log.parity": parts["parity"]})
    raw_nan = json.dumps(parts["parity"]).replace(
        json.dumps(parts["parity"]["cases"][0]["kernel_outputs"]["loss"][0]), "NaN", 1
    )
    ctx_nan._raw["log.parity"] = raw_nan.encode("utf-8")
    assert not all(c.passed for c in eval_hal4(ctx_nan))


@pytest.mark.tier3
def test_t3_p06_f8_cluster_is_with_f15_head_policy_v128k_and_v32k_exclusion() -> None:
    import numpy as np

    build_part = resolve_public_symbol("hadukenn_core", "build_balanced_partition", ("hadukenn_core.partitions",))
    cluster_is = resolve_public_symbol("hadukenn_core", "cluster_is_fwd_bwd", ("hadukenn_core.heads",))
    flat_tied = resolve_public_symbol("hadukenn_core", "flat_tied_fwd_bwd", ("hadukenn_core.heads",))
    policy_fn = resolve_any_public_symbol(
        "hadukenn_core",
        ("validate_output_head_policy", "select_output_head_regime", "check_output_head_policy"),
        _ML_MODULES,
    )

    ok_v128k = _call_with_aliases(
        policy_fn,
        [
            (("vocab_size",), 131_072),
            (("n_clusters",), 1024),
            (("cluster_size",), 128),
            (("head_kind",), "cluster_is"),
            (("hardware_target", "hardware"), "gpu_pretrain"),
            (("batch_size",), 64),
        ],
        (131_072,),
    )
    assert _verdict_passed(ok_v128k) is True

    _assert_rejected(
        policy_fn,
        *_bind_with_aliases(
            policy_fn,
            [
                (("vocab_size",), 32_768),
                (("n_clusters",), 182),
                (("cluster_size",), 181),
                (("head_kind",), "goodman_factored"),
                (("hardware_target", "hardware"), "gpu_pretrain"),
                (("batch_size",), 64),
            ],
            (32_768,),
        )[0],
        **_bind_with_aliases(
            policy_fn,
            [
                (("vocab_size",), 32_768),
                (("n_clusters",), 182),
                (("cluster_size",), 181),
                (("head_kind",), "goodman_factored"),
                (("hardware_target", "hardware"), "gpu_pretrain"),
                (("batch_size",), 64),
            ],
            (32_768,),
        )[1],
    )

    rng = np.random.default_rng(306)
    nc, s_sz, d, n = 4, 8, 6, 8
    v = nc * s_sz
    part = build_part(v, nc, s_sz, [rng.integers(0, v, size=64, dtype=np.int32)], mode="mass_random", seed=306)
    H = rng.normal(0.0, 0.3, size=(n, d)).astype(np.float64)
    E = rng.normal(0.0, 0.3, size=(v, d)).astype(np.float64)
    delta_root = rng.normal(0.0, 0.1, size=(nc, d)).astype(np.float64)
    targets = rng.integers(0, v, size=n, dtype=np.int64)

    loss_is, _, dH_is, dE_is, _ = cluster_is(
        H, E, delta_root, part, targets, k_neg=nc - 1, rng=np.random.default_rng(9)
    )
    loss_ex, dH_ex, dE_ex = flat_tied(H, E, targets)
    assert np.allclose(loss_is, loss_ex, atol=1e-5)
    assert np.allclose(dH_is, dH_ex, atol=1e-5)
    assert np.allclose(dE_is, dE_ex, atol=1e-5)


@pytest.mark.tier3
def test_t3_p07_f9_hst_merkle_manifest_bound_in_f5_regulatory_bundle() -> None:
    tok_cls = resolve_public_symbol("sheaf_verify", "HSTTokenizer", ("sheaf_verify.hst_standard",))
    verify_hst = resolve_public_symbol("sheaf_verify", "verify_hst_manifest_integrity", ("sheaf_verify.hst_standard",))
    emit_fn = resolve_public_symbol("sheaf_verify", "emit_regulatory_bundle", ("sheaf_verify.compliance",))
    verify_bundle = resolve_public_symbol("sheaf_verify", "verify_regulatory_bundle", ("sheaf_verify.compliance",))

    corpus = ["def sheaf_step(x): return x + 1\n" * 6, "hypergraph merkle leaf count witness\n" * 6]
    tok = tok_cls.train(corpus, vocab_size=290, min_pair_count=2, min_superspan_pmi=0.2)
    manifest = tok.export_manifest()
    ok_hst, probs = verify_hst(manifest)
    assert ok_hst is True and probs == []

    merkle_root = manifest["merkle_root_sha256"]
    assert merkle_root == tok.merkle_root_sha256()
    events = [
        {"verdict": "VALID", "approved": True, "merges": len(manifest["merges"]), "merkle_root": merkle_root},
    ]
    key = b"t3-p07-hst-bundle-key-0123456789"
    bundle = emit_fn(
        system_id="hst-v1-t3-p07",
        trace_head_sha256=merkle_root,
        state_digest_sha256=merkle_root,
        policy_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest(),
        events=events,
        created_at="2026-11-28T12:05:00Z",
        signing_key=key,
    )
    ok_b, errs_b = verify_bundle(bundle, signing_key=key)
    assert ok_b is True and errs_b == []

    tampered_manifest = json.loads(json.dumps(manifest))
    tampered_manifest["merges"][0]["count"] += 1
    bad_ok, bad_probs = verify_hst(tampered_manifest)
    assert bad_ok is False and any("Merkle" in p or "merkle" in p for p in bad_probs)


@pytest.mark.tier3
def test_t3_p08_f10_siunit_and_tick_verifier_feed_f13_op01_to_op04() -> None:
    import numpy as np

    second = resolve_public_symbol("sheaf_verify", "SECOND")
    watt = resolve_public_symbol("sheaf_verify", "WATT")
    tick_fn = resolve_any_public_symbol(
        "sheaf_verify",
        ("verify_tick_monotonicity", "verify_ticks", "check_tick_monotonicity", "op_04_verify_ticks"),
    )
    state_cls = resolve_public_symbol("hadukenn_core", "KernelState", _KERNEL_MODULES)
    compose_fn = resolve_public_symbol("hadukenn_core", "compose", _KERNEL_MODULES)
    sheaf_err = resolve_public_symbol("sheaf_verify", "SheafError", ("sheaf_verify.hashing",))

    energy_unit = watt * second
    assert energy_unit.dims == (2, 1, -2, 0, 0, 0, 0)

    ops = [
        _resolve_op("op_01_ingest_and_clock_sync", "op_01"),
        _resolve_op("op_02_endogenous_scale_solver", "op_02"),
        _resolve_op("op_03_selected_index_pullback", "op_03"),
        _resolve_op("op_04_events_and_interval_matrix", "op_04"),
    ]
    s = compose_fn(*ops)(_build_kernel_state(n_steps=64, m_streams=4, seed=308))
    t_sync = np.asarray(s.synchronized_streams["t"], dtype=float).tolist()
    sig = inspect.signature(tick_fn)
    if len(sig.parameters) == 1:
        assert _verdict_passed(tick_fn(t_sync)) is True
    else:
        ticks = [round(val * 1000.0) for val in t_sync]
        res = _call_with_aliases(
            tick_fn, [(("timestamps", "t", "times"), t_sync), (("ticks", "q_ticks", "q"), ticks)], (t_sync, ticks)
        )
        assert _verdict_passed(res) is True

    bad_streams = {
        "m0": {"t": np.linspace(1.0, 16.0, 16), "values": np.sin(np.linspace(1.0, 16.0, 16)), "time_unit": watt, "val_unit": watt}
    }
    bad_state = state_cls(bad_streams, {}, {"objective_series": np.ones(16)}, mode={"check_properties": True})
    with pytest.raises((ValueError, sheaf_err, AssertionError)):
        _resolve_op("op_01_ingest_and_clock_sync", "op_01")(bad_state)


@pytest.mark.tier3
def test_t3_p09_f10_goldilocks_r1cs_matches_f13_op16_witness_and_f3_f5_ledger() -> None:
    goldilocks_p = resolve_public_symbol("sheaf_verify", "GOLDILOCKS_P")
    r1cs_fn = resolve_any_public_symbol(
        "sheaf_verify",
        ("verify_goldilocks_r1cs_dipole", "verify_goldilocks_r1cs", "goldilocks_r1cs_dipole_verify"),
    )
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))
    emit_fn = resolve_public_symbol("sheaf_verify", "emit_regulatory_bundle", ("sheaf_verify.compliance",))
    verify_bundle = resolve_public_symbol("sheaf_verify", "verify_regulatory_bundle", ("sheaf_verify.compliance",))
    compose_fn = resolve_public_symbol("hadukenn_core", "compose", _KERNEL_MODULES)

    ops = [
        _resolve_op("op_01_ingest_and_clock_sync", "op_01"),
        _resolve_op("op_02_endogenous_scale_solver", "op_02"),
        _resolve_op("op_03_selected_index_pullback", "op_03"),
        _resolve_op("op_04_events_and_interval_matrix", "op_04"),
        _resolve_op("op_05_recurrence_and_visibility_graphs", "op_05"),
        _resolve_op("op_06_coboundary_and_sheaf_resolvent", "op_06"),
        _resolve_op("op_07_linegraph_and_prefix_dynamics", "op_07"),
        _resolve_op("op_08_manifold_scan_and_so4_octonion_rotor", "op_08"),
        _resolve_op("op_09_spectral_sheaf_circuit_and_dipole_lock", "op_09"),
        _resolve_op("op_16_goldilocks_zk_r1cs_circuit_compiler", "op_16"),
    ]
    s = compose_fn(*ops)(_build_kernel_state(n_steps=64, m_streams=4, seed=309))
    witness = [float(x) for x in s.spectral_circuit["latest_12neuron_witness"]]
    assert len(witness) == 12
    zk = s.zk_goldilocks_circuit
    assert zk["prime_field_modulus"] == str(goldilocks_p)
    assert zk["r1cs_gate_residual_mod_p"] == 0
    assert zk["dipole_sum_mod_p"] == 0

    y_float = float(zk.get("y_float", 0.25))
    r1cs_res = _call_with_aliases(
        r1cs_fn,
        [(("y_float", "y"), y_float), (("dipole_acts", "witness", "acts"), witness)],
        (y_float, witness),
    )
    assert _verdict_passed(r1cs_res) is True
    res_mod = _extract_attr_or_key(r1cs_res, ("r1cs_gate_residual_mod_p", "r1cs_gate_residual", "residual"))
    dip_mod = _extract_attr_or_key(r1cs_res, ("dipole_sum_mod_p", "mod_sum"))
    assert int(res_mod) == 0 and int(dip_mod) == 0

    # Bind verified op_16 R1CS commitment into F3 compliance ontology + F5 regulatory bundle
    ont = ont_cls("t3-p09-zk-ledger", [rule_cls(action_name="zk_attest", dimension="ratio")])
    verdict = ont.evaluate_transfer_action(
        action_name="zk_attest",
        actor_role="operator",
        src_entity="r1cs_prover",
        dst_entity="audit_ledger",
        src_witnesses={"w_r1cs": (Fraction(1), "units"), "w_dipole": (Fraction(1), "units")},
        dst_initial=(Fraction(0), "units"),
        transfer_amount=(Fraction(1), "units"),
        claimed_post_src=(Fraction(0), "units"),
        claimed_post_dst=(Fraction(1), "units"),
    )
    assert verdict.verdict == "VALID" and verdict.approved is True

    commitment_sha = hashlib.sha256(str(zk["zk_stark_witness_commitment"]).encode("utf-8")).hexdigest()
    key = b"t3-p09-zk-ledger-key-01234567890"
    bundle = emit_fn(
        system_id="sys-t3-p09-zk",
        trace_head_sha256=commitment_sha,
        state_digest_sha256=commitment_sha,
        policy_sha256=ont.policy_sha256,
        events=[verdict.as_dict()],
        created_at="2026-11-28T12:10:00Z",
        signing_key=key,
    )
    ok_b, errs_b = verify_bundle(bundle, signing_key=key)
    assert ok_b is True and errs_b == []

    # Corrupted witness (truncated or NaN) fails Goldilocks verifier loudly
    bad_args, bad_kwargs = _bind_with_aliases(
        r1cs_fn, [(("y_float", "y"), y_float), (("dipole_acts", "witness", "acts"), witness[:4])], (y_float, witness[:4])
    )
    _assert_rejected(r1cs_fn, *bad_args, **bad_kwargs)


@pytest.mark.tier3
def test_t3_p10_f10_op10_crypto_receipt_matches_f13_op10_telemetry() -> None:
    cone_fn = resolve_any_public_symbol(
        "sheaf_verify",
        ("verify_sha256_causal_cone", "op_10_crypto_audit", "verify_crypto_causal_cone", "sha256_causal_cone_metrics"),
    )
    receipt_fn = resolve_any_public_symbol(
        "sheaf_verify",
        ("verify_canonical_receipt", "compute_canonical_receipt", "canonical_receipt_sha256"),
    )
    compose_fn = resolve_public_symbol("hadukenn_core", "compose", _KERNEL_MODULES)

    ops = [
        _resolve_op("op_01_ingest_and_clock_sync", "op_01"),
        _resolve_op("op_02_endogenous_scale_solver", "op_02"),
        _resolve_op("op_03_selected_index_pullback", "op_03"),
        _resolve_op("op_04_events_and_interval_matrix", "op_04"),
        _resolve_op("op_05_recurrence_and_visibility_graphs", "op_05"),
        _resolve_op("op_06_coboundary_and_sheaf_resolvent", "op_06"),
        _resolve_op("op_07_linegraph_and_prefix_dynamics", "op_07"),
        _resolve_op("op_08_manifold_scan_and_so4_octonion_rotor", "op_08"),
        _resolve_op("op_09_spectral_sheaf_circuit_and_dipole_lock", "op_09"),
        _resolve_op("op_10_slice_carry_sheaf_crypto_miner", "op_10"),
    ]
    s = compose_fn(*ops)(_build_kernel_state(n_steps=64, m_streams=4, seed=310))
    cm = s.crypto_mining_telemetry
    receipt_hex = cm["master_kernel_sha256_receipt"]
    body = cm.get("receipt_body") or cm.get("receipt_payload")
    if body is None:
        body = {
            "endogenous_params": s.endogenous_params,
            "carrier_sheets": s.carriers["num_carrier_sheets"],
            "sheaf_dirichlet_drop_pct": s.sheaf_projection["dirichlet_energy_drop_pct"],
            "mean_h1_frustration": s.sheaf_projection["mean_h1_frustration"],
            "so4_eigenrotor_ic": s.manifold_and_rotors["adaptive_so4_eigenrotor_rank_ic"],
            "f12_causal_pct": s.spectral_circuit["f12_causal_concentration_pct"],
            "witness_12": s.spectral_circuit["latest_12neuron_witness"],
        }

    if len(inspect.signature(receipt_fn).parameters) >= 2:
        assert _verdict_passed(receipt_fn(body, receipt_hex)) is True
        bad_hex = ("0" if receipt_hex[0] != "0" else "1") + receipt_hex[1:]
        _assert_rejected(receipt_fn, body, bad_hex)
    else:
        computed = receipt_fn(body)
        assert computed == receipt_hex
        assert len(computed) == 64

    cone_res = cone_fn() if len(inspect.signature(cone_fn).parameters) == 0 else cone_fn(mut_word_idx=3)
    inv_words = _extract_attr_or_key(cone_res, ("invariant_w_words",))
    assert int(inv_words) == cm["invariant_schedule_words_in_ker_deltaW"] == 17
    op10_prov = cm.get("telemetry_provenance") or cm.get("analytical_fields")
    assert op10_prov is not None, "op_10 must expose telemetry provenance separating live vs modeled metrics"


@pytest.mark.tier3
def test_t3_p11_f10_simt_bank_verifier_and_f13_sm120_warp_parity(no_gpu_env: dict[str, str]) -> None:
    import numpy as np

    assert no_gpu_env["CUDA_VISIBLE_DEVICES"] == ""
    simt_fn = resolve_any_public_symbol(
        "sheaf_verify",
        ("verify_simt_warp_and_banks", "verify_warp_butterfly", "simt_bank_conflicts", "count_bank_conflicts"),
    )
    cpu_parity_fn = resolve_any_public_symbol(
        "hadukenn_core",
        (
            "c0_rtx5070_warp_julia_and_tuple_cpu",
            "emulate_warp_julia_and_tuple_cpu",
            "cpu_warp_julia_tuple_parity",
            "verify_sm120_warp_julia_cpu_parity",
            "run_sm120_warp_julia_cpu_ref",
        ),
        _KERNEL_MODULES,
    )

    lanes = [float(i + 1) for i in range(32)]
    sig = inspect.signature(simt_fn)
    if "stride" in sig.parameters and len(sig.parameters) == 1:
        assert simt_fn(stride=32) == 31
        assert simt_fn(stride=33) == 0
    else:
        res_32 = _call_with_aliases(simt_fn, [(("lanes", "values"), lanes), (("stride",), 32)], (lanes, 32))
        res_33 = _call_with_aliases(simt_fn, [(("lanes", "values"), lanes), (("stride",), 33)], (lanes, 33))
        assert int(_extract_attr_or_key(res_32, ("conflicts", "bank_conflicts"))) == 31
        assert int(_extract_attr_or_key(res_33, ("conflicts", "bank_conflicts"))) == 0
        assert float(_extract_attr_or_key(res_33, ("reduced_sum", "butterfly_sum", "warp_sum"))) == 528.0

    rng = np.random.default_rng(311)
    c_in = (0.12 * rng.standard_normal((64, 4))).astype(np.float32)
    c_in[:, 0] = np.float32(0.85)
    gamma = 0.25
    ref_q, ref_p5 = _numpy_sm120_warp_julia_ref(c_in, gamma)
    res = _call_with_aliases(
        cpu_parity_fn,
        [(("c_H_in", "c_in", "quaternions"), c_in), (("gamma_sheaf", "gamma"), gamma)],
        (c_in, gamma),
    )
    if isinstance(res, tuple):
        q_out, p5_out = np.asarray(res[0], dtype=np.float32), np.asarray(res[1], dtype=np.float32)
    else:
        q_out = np.asarray(_extract_attr_or_key(res, ("q_out", "q_julia")), dtype=np.float32)
        p5_out = np.asarray(_extract_attr_or_key(res, ("p5_jk_rotor_out", "p5_out", "rotor_out")), dtype=np.float32)

    assert q_out.shape == (64, 4) and p5_out.shape == (64,)
    assert float(np.max(np.abs(q_out - ref_q))) <= 1e-5
    assert float(np.max(np.abs(p5_out - ref_p5))) <= 1e-5


@pytest.mark.tier3
def test_t3_p12_f11_bounds_match_f14_op21_rademacher_tikhonov_bic() -> None:
    tikhonov = resolve_public_symbol("sheaf_verify", "tikhonov_stability_rate")
    bic = resolve_public_symbol("sheaf_verify", "bic_score")
    hoeffding = resolve_public_symbol("sheaf_verify", "hoeffding_bound")

    n_steps = 128
    s = _run_op21(n_steps=n_steps, m_streams=4, seed=312)
    vga = s.value_graph_policy_alignment
    m_aligned = vga["policy_aligned_relationship_graph_metrics"]

    t_expected = float(
        _call_with_aliases(
            tikhonov,
            [(("rho", "lipschitz"), 1.0), (("lam", "reg"), 0.03), (("m", "n_samples"), n_steps)],
            (1.0, 0.03, n_steps),
        )
    )
    assert math.isclose(float(m_aligned["tikhonov_replace_one_stability_rate"]), round(t_expected, 5), abs_tol=1e-5)

    cka = float(m_aligned["centered_kernel_graph_alignment_cka"])
    bic_expected = float(
        _call_with_aliases(
            bic,
            [(("log_likelihood", "log_lik"), n_steps * math.log(max(cka, 1e-4))), (("dof", "k"), 5), (("n", "n_samples"), n_steps)],
            (n_steps * math.log(max(cka, 1e-4)), 5, n_steps),
        )
    )
    assert math.isclose(float(m_aligned["bayesian_information_criterion_bic"]), bic_expected, abs_tol=0.1)

    h_128 = float(
        _call_with_aliases(
            hoeffding,
            [(("m", "n_samples"), 128), (("delta",), 0.05), (("h_size", "num_hypotheses", "hypothesis_count"), 5)],
            (128, 0.05, 5),
        )
    )
    h_512 = float(
        _call_with_aliases(
            hoeffding,
            [(("m", "n_samples"), 512), (("delta",), 0.05), (("h_size", "num_hypotheses", "hypothesis_count"), 5)],
            (512, 0.05, 5),
        )
    )
    assert h_512 < h_128


@pytest.mark.tier3
def test_t3_p13_f11_zigzag_pi_hat_and_zp_ratio_with_f15_zigzag_pruning_and_f14_pi5() -> None:
    import numpy as np

    pi_shift = resolve_any_public_symbol(
        "sheaf_verify", ("zigzag_effective_pi", "shift_persistence_image", "verify_zigzag_pi_hat")
    )
    zp_fn = resolve_any_public_symbol(
        "sheaf_verify", ("zigzag_persistence_ratio", "zigzag_interlayer_persistence", "compute_zigzag_zp")
    )
    zigzag_sym = resolve_any_public_symbol(
        "hadukenn_core", ("ZigzagTopologyAnalyzer", "compute_zigzag_persistence", "zigzag_interlayer_analysis"), _ML_MODULES
    )
    prune_fn = resolve_any_public_symbol(
        "hadukenn_core",
        ("prune_layers_plateau", "algorithm2_prune_layers", "zigzag_prune_layers", "prune_redundant_layers"),
        _ML_MODULES,
    )

    # Run op_21 first and derive shared trajectory + pi_1 persistence from its stabilized sheaf state
    s64 = _run_op21(n_steps=64, m_streams=4, seed=313)
    vga = s64.value_graph_policy_alignment
    policies64 = vga["adaptive_policies_executed"]
    pi1_scores = policies64["pi_1_zigzag_knn_vr_inter_graph_persistence"]
    b1_op21 = [float(v) for v in policies64["pi_1_births_relative_frequency_B1"]]
    assert len(b1_op21) == 5 and all(math.isfinite(v) and v >= 0.0 for v in b1_op21)
    assert pi1_scores and all(0.0 <= float(v) <= 1.0 for v in pi1_scores.values())

    x_std = np.asarray(s64.sheaf_projection["std_mat"], dtype=float)[4:28, :4]
    h0_std = np.asarray(s64.sheaf_projection["h0_global"], dtype=float)[4:28, :4]
    x_feat = np.hstack([x_std, h0_std, x_std - h0_std])  # (24, 12) from op_06/op_21
    rng = np.random.default_rng(313)
    traj = np.zeros((8, 24, 12))
    traj[0] = x_feat
    traj[1] = x_feat + rng.normal(0.0, 0.6, (24, 12))
    plateau = np.hstack([h0_std, h0_std, x_std])
    for idx in (2, 3, 4, 5):
        traj[idx] = plateau + rng.normal(0.0, 0.004, (24, 12))
    traj[6] = x_feat + rng.normal(0.0, 1.0, (24, 12))
    traj[7] = x_feat - rng.normal(0.0, 1.0, (24, 12))

    analyzer = zigzag_sym() if isinstance(zigzag_sym, type) else zigzag_sym
    zz_res = analyzer.analyze(traj) if hasattr(analyzer, "analyze") else analyzer(traj)
    z1_bar = np.asarray(
        _extract_attr_or_key(zz_res, ("z1_bar", "z_bar", "mean_interlayer_persistence", "z1_profile")), dtype=float
    )
    assert z1_bar.shape == (8,) and np.all(np.isfinite(z1_bar))
    assert float(np.ptp(z1_bar)) > 0.02

    # Verify ZigzagTopologyAnalyzer responds to perturbed op_21 trajectory (anti-decoupling)
    traj_shuffled = traj.copy()
    for idx in range(1, 8):
        traj_shuffled[idx] = traj[idx, rng.permutation(24)]
    zz_shuf = analyzer.analyze(traj_shuffled) if hasattr(analyzer, "analyze") else analyzer(traj_shuffled)
    z1_shuf = np.asarray(
        _extract_attr_or_key(zz_shuf, ("z1_bar", "z_bar", "mean_interlayer_persistence", "z1_profile")), dtype=float
    )
    assert float(np.mean(z1_bar[2:6])) > float(np.mean(z1_shuf[2:6]))

    pruned = _call_with_aliases(
        prune_fn, [(("z1_bar", "z_bar", "scores", "persistence"), z1_bar), (("threshold",), 0.90)], (z1_bar, 0.90)
    )
    pruned_idx = (
        list(_extract_attr_or_key(pruned, ("pruned_layers", "pruned_indices", "drop_layers")))
        if (
            isinstance(pruned, dict)
            or hasattr(pruned, "pruned_layers")
            or hasattr(pruned, "pruned_indices")
            or hasattr(pruned, "drop_layers")
        )
        else list(pruned)
    )
    assert len(pruned_idx) >= 1 and 0 not in pruned_idx and 7 not in pruned_idx
    assert any(idx in (2, 3, 4, 5) for idx in pruned_idx)

    # Couple op_21 B1 births and Zigzag z1_bar into zigzag_effective_pi and zigzag_persistence_ratio
    c02 = float(b1_op21[2]) + float(z1_bar[0]) + 0.25
    c11 = float(b1_op21[0]) + 1.0
    c12 = float(b1_op21[1]) + 0.5
    c21 = float(z1_bar[2]) + 0.5
    c22 = float(z1_bar[3]) + 0.5
    grid = [
        [0.0, 0.0, c02, 0.0],
        [0.0, c11, c12, 0.0],
        [0.0, c21, c22, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    pi_hat = pi_shift(grid)
    assert math.isclose(float(pi_hat[1][1]), c11 + c12 + c21 + c22, abs_tol=1e-9)
    z_zero = _call_with_aliases(zp_fn, [(("pi_hat", "pi_eff"), pi_hat), (("beta_p",), 0)], (pi_hat, 0))
    assert float(z_zero) == 0.0
    beta_pos = max(1, round(sum(float(v) for v in pi1_scores.values()) * 10))
    z_pos = float(_call_with_aliases(zp_fn, [(("pi_hat", "pi_eff"), pi_hat), (("beta_p",), beta_pos), (("l1",), 0), (("l2",), 0)], (pi_hat, beta_pos, 0, 0)))
    assert math.isclose(z_pos, c02 / beta_pos, abs_tol=1e-9) and z_pos > 0.0

    # Also verify op_21 pi_5 handles N=17 (N % 4 != 0) without dropping remainder nodes
    s17 = _run_op21(n_steps=17, m_streams=4, check_properties=False, seed=313)
    _assert_pi5_node_coverage(s17.value_graph_policy_alignment["adaptive_policies_executed"], 17)


@pytest.mark.tier3
def test_t3_p14_f11_gini_and_split_disjointness_with_f15_smote_rf() -> None:
    import numpy as np

    gini_fn = resolve_public_symbol("sheaf_verify", "gini_impurity")
    split_fn = resolve_public_symbol("sheaf_verify", "validate_split_disjointness")
    smote_fn = resolve_any_public_symbol(
        "hadukenn_core", ("train_only_smote", "smote_resample_train", "smote_train_only"), _ML_MODULES
    )
    tree_cls = resolve_any_public_symbol("hadukenn_core", ("GiniDecisionTree", "fit_gini_tree"), _ML_MODULES)
    rf_cls = resolve_any_public_symbol(
        "hadukenn_core", ("RandomForestClassifierLite", "fit_random_forest"), _ML_MODULES
    )

    rng = np.random.default_rng(314)
    x_tr = rng.normal(0.0, 1.0, (100, 4))
    y_tr = np.zeros(100, dtype=int)
    y_tr[:10] = 1
    x_tr[:10, 0] += 3.2
    x_te = rng.normal(0.0, 1.0, (40, 4))
    y_te = np.zeros(40, dtype=int)
    y_te[:8] = 1
    x_te[:8, 0] += 3.2

    train_ids = {f"tr_{i}" for i in range(100)}
    test_ids = {f"te_{i}" for i in range(40)}
    sm_res = _call_with_aliases(
        smote_fn, [(("X_train", "X"), x_tr), (("y_train", "y"), y_tr), (("X_test",), x_te), (("seed",), 314)], (x_tr, y_tr)
    )
    if isinstance(sm_res, tuple):
        x_bal, y_bal = np.asarray(sm_res[0]), np.asarray(sm_res[1])
    else:
        x_bal = np.asarray(_extract_attr_or_key(sm_res, ("X_resampled", "X_train_resampled", "X")))
        y_bal = np.asarray(_extract_attr_or_key(sm_res, ("y_resampled", "y_train_resampled", "y")))

    n_syn = len(y_bal) - len(y_tr)
    assert n_syn > 0
    syn_ids = {f"smote_{i}" for i in range(n_syn)}
    disjoint_res = _call_with_aliases(
        split_fn,
        [(("train_ids", "train"), train_ids | syn_ids), (("test_ids", "test"), test_ids), (("synthetic_ids", "synthetic"), syn_ids)],
        (train_ids | syn_ids, test_ids, syn_ids),
    )
    assert _verdict_passed(disjoint_res) is True

    # Verify synthetic_ids & test_ids leakage is rejected even when train_ids & test_ids == empty
    leak_args, leak_kwargs = _bind_with_aliases(
        split_fn,
        [(("train_ids", "train"), train_ids), (("test_ids", "test"), test_ids | {"smote_leak"}), (("synthetic_ids", "synthetic"), syn_ids | {"smote_leak"})],
        (train_ids, test_ids | {"smote_leak"}, syn_ids | {"smote_leak"}),
    )
    _assert_rejected(split_fn, *leak_args, **leak_kwargs)

    parent_gini = float(gini_fn([int(np.sum(y_bal == 0)), int(np.sum(y_bal == 1))]))
    left_mask = x_bal[:, 0] > 1.5
    left_gini = float(gini_fn([int(np.sum(y_bal[left_mask] == 0)), int(np.sum(y_bal[left_mask] == 1))]))
    right_gini = float(gini_fn([int(np.sum(y_bal[~left_mask] == 0)), int(np.sum(y_bal[~left_mask] == 1))]))
    weighted_gini = (np.sum(left_mask) * left_gini + np.sum(~left_mask) * right_gini) / len(y_bal)
    assert weighted_gini < parent_gini

    tree = tree_cls(max_depth=4) if isinstance(tree_cls, type) else tree_cls(x_bal, y_bal, max_depth=4)
    if hasattr(tree, "fit"):
        tree.fit(x_bal, y_bal)
    assert float(tree.feature_importances_[0]) > 0.5

    rf = rf_cls(n_estimators=15, max_depth=4, seed=314) if isinstance(rf_cls, type) else rf_cls(x_bal, y_bal)
    if hasattr(rf, "fit"):
        rf.fit(x_bal, y_bal)
    preds = np.asarray(rf.predict(x_te), dtype=int)
    recall = int(np.sum((preds == 1) & (y_te == 1))) / int(np.sum(y_te == 1))
    assert recall > 0.5


@pytest.mark.tier3
def test_t3_p15_f11_elbo_bic_logsumexp_with_f15_moe_gem_and_f14_pi3() -> None:
    import numpy as np

    lse_fn = resolve_public_symbol("sheaf_verify", "logsumexp")
    bic_fn = resolve_public_symbol("sheaf_verify", "bic_score")
    elbo_fn = resolve_any_public_symbol(
        "sheaf_verify", ("verify_elbo_monotonicity", "check_elbo_monotonicity", "elbo_is_monotone")
    )
    moe_fn = resolve_any_public_symbol("hadukenn_core", ("fit_moe_gem", "MoEGEM", "moe_generalized_em"), _ML_MODULES)

    rng = np.random.default_rng(315)
    x = rng.normal(0.0, 1.0, (48, 4))
    y = 1.8 * x[:, 0] - 0.6 * x[:, 1] + rng.normal(0.0, 0.1, 48)
    moe_res = _call_with_aliases(moe_fn, [(("X",), x), (("y",), y), (("n_experts", "k"), 2), (("max_iter",), 10)], (x, y))
    moe_trace = [float(v) for v in _extract_attr_or_key(moe_res, ("elbo_trace", "free_energy_trace"))]
    assert _verdict_passed(elbo_fn(moe_trace)) is True

    s = _run_op21(n_steps=64, m_streams=4, seed=315)
    pi3_trace = [
        float(v) for v in s.value_graph_policy_alignment["adaptive_policies_executed"]["pi_3_moe_em_variational_free_energy_trajectory"]
    ]
    assert _verdict_passed(elbo_fn(pi3_trace)) is True

    moe_bic = float(_extract_attr_or_key(moe_res, ("bic", "bic_score")))
    assert math.isfinite(moe_bic)
    assert math.isfinite(float(lse_fn([1000.0, 1001.0])))
    assert math.isfinite(float(_call_with_aliases(bic_fn, [(("log_likelihood", "log_lik"), moe_trace[-1]), (("dof", "k"), 5), (("n", "n_samples"), 48)], (moe_trace[-1], 5, 48))))


@pytest.mark.tier3
def test_t3_p16_f12_siip_gate_governs_f14_op21_pi6_and_f15_bfgs() -> None:
    import numpy as np

    gate_cls = resolve_any_public_symbol("sheaf_verify", ("SiipVerifyGate", "SIIPVerifyGate", "siip_verify_gate"))
    bfgs_fn = resolve_any_public_symbol(
        "hadukenn_core", ("bfgs_optimize", "bfgs_inverse_hessian_minimize"), _ML_MODULES
    )

    gate = gate_cls(mode="formal") if "mode" in inspect.signature(gate_cls).parameters else gate_cls()
    if hasattr(gate, "set_mode"):
        gate.set_mode("formal")
    damp_fn = getattr(gate, "damp_direction", getattr(gate, "apply_hesitation_damping"))
    verify_fn = getattr(gate, "verify_commit", getattr(gate, "verify_candidate"))

    damped = damp_fn([-0.8, 0.1], [1.0, 0.0])
    damped_vec = damped[0] if isinstance(damped, tuple) and isinstance(damped[0], (list, tuple)) else damped
    assert math.isclose(float(damped_vec[0]), -0.4, abs_tol=1e-12)
    assert math.isclose(float(damped_vec[1]), 0.05, abs_tol=1e-12)

    w_star = np.array([1.5, -0.75, 0.25])
    opt_res = _call_with_aliases(
        bfgs_fn,
        [
            (("fn", "obj_fn", "loss_fn"), lambda w: float(np.sum((w - w_star) ** 2))),
            (("grad_fn",), lambda w: 2.0 * (w - w_star)),
            (("x0", "w0"), np.zeros(3)),
            (("max_iter",), 20),
        ],
        (lambda w: float(np.sum((w - w_star) ** 2)), lambda w: 2.0 * (w - w_star), np.zeros(3)),
    )
    w_opt = np.asarray(_extract_attr_or_key(opt_res, ("x", "w", "params")), dtype=float)
    f0 = float(np.sum((np.zeros(3) - w_star) ** 2))
    f_opt = float(np.sum((w_opt - w_star) ** 2))
    assert _verdict_passed(verify_fn(f_k=f0, f_cand=f_opt, x_cand=w_opt.tolist())) is True
    _assert_rejected(verify_fn, f_k=f_opt, f_cand=f_opt + 0.1, x_cand=w_opt.tolist())

    s = _run_op21(n_steps=64, m_streams=4, seed=316)
    policies = s.value_graph_policy_alignment["adaptive_policies_executed"]
    assert int(policies["pi_6_siip_formal_verified_bfgs_commits"]) >= 1


@pytest.mark.tier3
def test_t3_p17_f12_rlm_guards_govern_f16_repl_orch_and_oolong() -> None:
    policy_cls = resolve_public_symbol("sheaf_verify", "RLMBudgetPolicy")
    state_cls = resolve_public_symbol("sheaf_verify", "RLMBudgetState")
    parse_fn = resolve_public_symbol("sheaf_verify", "parse_final_directive")
    tracker_cls = resolve_public_symbol("sheaf_verify", "AntiRepetitionTracker")
    repl_cls = resolve_any_public_symbol(
        "hadukenn_core", ("SandboxedPythonREPL", "RLMContextREPL", "RLMReplEnvironment", "SandboxedRLMRepl"), _RLM_MODULES
    )
    orch_cls = resolve_any_public_symbol("hadukenn_core", ("RLMOrchestrator", "RLMEngine", "RLMRunner"), _RLM_MODULES)
    stitch_fn = resolve_any_public_symbol(
        "hadukenn_core", ("oolong_pairs_stitch", "run_oolong_pairs_stitching", "rlm_subgraph_stitch", "stitch_oolong_pairs"), _RLM_MODULES
    )

    policy = policy_cls(
        max_depth=1, max_subcalls=3, max_total_tokens=20_000, min_batch_chars=100, thinking_reserve=512
    )
    state = state_cls()
    calls = 0

    def backend(prompt: str) -> str:
        nonlocal calls
        calls += 1
        return f"OK:{len(prompt)}"

    repl = repl_cls(["doc_" + str(i) for i in range(15)], budget_policy=policy, budget_state=state, llm_backend=backend)
    _assert_rejected(repl.llm_query, "too_short", depth=1)
    assert calls == 0
    assert repl.llm_query("valid_" + "x" * 120, depth=1).startswith("OK:")

    # parse_final_directive ignores FINAL(...) inside fenced repl block
    fenced = "```repl\nprint('FINAL(spoofed)')\n```\nWorking..."
    try:
        parsed_fenced = parse_fn(fenced, {"spoofed": "no"})
    except ValueError:
        pass
    else:
        assert parsed_fenced is None or (isinstance(parsed_fenced, dict) and parsed_fenced.get("value") is None)

    tracker = tracker_cls(max_consec_no_new_var=2)
    st1 = tracker.record_turn("stitched = len(context)", ["stitched"], verified_value=15)
    assert st1.should_stop is False and st1.consec_no_new_var == 0 and st1.new_vars == ("stitched",)
    st2 = tracker.record_turn("noop turn 1", ["stitched"])
    assert st2.should_stop is False and st2.consec_no_new_var == 1
    st3 = tracker.record_turn("noop turn 2", [])
    assert st3.should_stop is True and st3.consec_no_new_var == 2 and st3.best_verified_value == 15

    orch = orch_cls(budget_policy=policy, max_turns=4, max_consec_no_new_var=2)
    turns = ["```repl\nstitched = len(context)\n```", "FINAL_VAR(stitched)"]
    turn_idx = 0

    def model_fn(_msgs: Any) -> dict[str, Any]:
        nonlocal turn_idx
        item = {"text": turns[turn_idx], "max_output_tokens": 2048, "thinking_budget": 512}
        turn_idx += 1
        return item

    res = _call_with_aliases(
        orch.run,
        [(("query", "task"), "count items"), (("context",), [f"d{i}" for i in range(15)]), (("model_fn", "llm_fn"), model_fn)],
        ("count items", [f"d{i}" for i in range(15)], model_fn),
    )
    answer = res["answer"] if isinstance(res, dict) and "answer" in res else getattr(res, "answer", res)
    assert answer == 15

    stitch_res = _call_with_aliases(
        stitch_fn,
        [(("items", "docs", "records"), [f"item_{i}" for i in range(15)]), (("n_chunks", "c_chunks", "num_chunks"), 4)],
        ([f"item_{i}" for i in range(15)], 4),
    )
    assert len(set(_extract_attr_or_key(stitch_res, ("covered_indices", "node_indices", "covered_items")))) == 15
    assert int(_extract_attr_or_key(stitch_res, ("total_pairs_evaluated", "pairs_evaluated", "num_pairs"))) == 105


@pytest.mark.tier3
def test_t3_p18_f9_hst_and_f15_chodrow_null_hypergraph_with_f16_exec_bridge() -> None:
    import numpy as np

    tok_cls = resolve_public_symbol("sheaf_verify", "HSTTokenizer", ("sheaf_verify.hst_standard",))
    corpus_doc = resolve_public_symbol("hadukenn_core", "CorpusDocument", ("hadukenn_core.hypergraph",))
    build_hg = resolve_public_symbol("hadukenn_core", "build_decontaminated_hypergraph", ("hadukenn_core.hypergraph",))
    build_null = resolve_public_symbol("hadukenn_core", "build_chodrow_null", ("hadukenn_core.hypergraph",))
    eval_knnlm = resolve_public_symbol("hadukenn_core", "evaluate_scoped_knnlm", ("hadukenn_core.hypergraph",))
    claim_fn = resolve_any_public_symbol(
        "hadukenn_core", ("validate_hypergraph_claim", "audit_chodrow_null_control"), _ML_MODULES
    )
    kern_cls = resolve_public_symbol("hadukenn_core", "IncrementalHypergraphKernel", ("hadukenn_core.exec_bridge",))
    searcher_cls = resolve_public_symbol(
        "hadukenn_core", "ActiveHypergraphTraceSearcher", ("hadukenn_core.cosimilar_engine",)
    )
    edge_spec_cls = resolve_public_symbol("hadukenn_core", "HyperedgeSpec", ("hadukenn_core.cosimilar_engine",))
    stitch_fn = resolve_any_public_symbol(
        "hadukenn_core", ("oolong_pairs_stitch", "run_oolong_pairs_stitching", "rlm_subgraph_stitch", "stitch_oolong_pairs"), _RLM_MODULES
    )

    texts = [
        "sheaf verify formal kernel merkle root proof " * 3,
        "chodrow null hypergraph scoped knnlm control " * 3,
        "zigzag persistent homology layer plateau prune " * 3,
        "oolong pairs context environment stitch trace " * 3,
    ]
    tok = tok_cls.train(texts, vocab_size=292, min_pair_count=2, min_superspan_pmi=0.2)
    manifest = tok.export_manifest()
    assert tok.merkle_root_sha256() == manifest["merkle_root_sha256"]
    assert tok.decode(tok.encode(texts[0])) == texts[0]

    train_docs = [
        corpus_doc(
            doc_id=f"tr_{i}",
            domain=f"dom_{i % 3}",
            text=f"{texts[i % len(texts)].strip()} doc_{i} shared_{i % 4} anchor_{i % 2}",
            sha256=hashlib.sha256(
                f"{texts[i % len(texts)].strip()} doc_{i} shared_{i % 4} anchor_{i % 2}".encode()
            ).hexdigest(),
        )
        for i in range(16)
    ]
    eval_docs = [
        corpus_doc(
            doc_id=f"ev_{i}",
            domain=f"dom_{i % 3}",
            text=f"eval_context_{i} shared_{i % 4} anchor_{i % 2}",
            sha256=hashlib.sha256(f"eval_context_{i} shared_{i % 4} anchor_{i % 2}".encode()).hexdigest(),
        )
        for i in range(4)
    ]
    kept_docs, hg = build_hg(train_docs, eval_docs, min_edge_size=2, max_edge_size=12, max_doc_degree=12, seed=318)
    assert kept_docs and len(hg.edges) >= 1
    null_hg = build_null(hg, seed=318)
    assert sorted(len(e) for e in hg.edges) == sorted(len(e) for e in null_hg.edges)

    assert _verdict_passed(
        _call_with_aliases(
            claim_fn,
            [(("hypergraph", "hg"), hg), (("chodrow_null", "null_hg"), null_hg), (("hyper_bpb", "metric"), 1.05), (("null_bpb",), 1.12)],
            (hg, null_hg, 1.05, 1.12),
        )
    ) is True

    rng = np.random.default_rng(318)
    ds_keys = rng.normal(0.0, 1.0, (24, 8))
    ds_vals = rng.integers(0, 6, size=24)
    ds_docs = np.arange(24) % 4
    knn_metrics = eval_knnlm(
        ds_keys[:6],
        ds_vals[:6],
        np.full(6, 1.0 / 6.0),
        [np.array([0, 1, 2], dtype=np.int64) for _ in range(6)],
        [np.array([0, 2], dtype=np.int64) for _ in range(6)],
        ds_keys,
        ds_vals,
        ds_docs,
        np.full(6, 4.0),
        np.ones(6, dtype=bool),
    )
    assert math.isfinite(float(knn_metrics["hyper_scoped_knn_rare_bpb"]))

    kern = kern_cls()
    stitch_res = _call_with_aliases(
        stitch_fn,
        [(("items", "docs", "records"), texts), (("n_chunks", "c_chunks", "num_chunks"), 2), (("exec_kernel", "kernel"), kern)],
        (texts, 2),
    )
    assert len(set(_extract_attr_or_key(stitch_res, ("covered_indices", "node_indices", "covered_items")))) == 4
    trace = kern.export_trace()
    assert len(trace) >= 1, "oolong_pairs_stitch(..., exec_kernel=kern) must emit execution trace records"
    replayed = kern_cls.replay(trace, expected_head=kern.head)
    assert replayed.state_digest() == kern.state_digest()

    searcher = searcher_cls(
        num_nodes=4,
        edges=[edge_spec_cls("e0", (0, 1, 2), ("s", "m", "o"), weight=2.5), edge_spec_cls("e1", (2, 3), ("s", "o"), weight=0.5)],
        node_salience=[2.5, 2.5, 2.0, 0.3],
    )
    win = searcher.search_optimal_window(seed_node=0, budget_k=3)
    assert len(win.selected_window) >= 1


@pytest.mark.tier3
def test_t3_p19_f17_hooks_and_mcp_enforce_f1_f10_f15_invariants(repo_root: Path, tmp_path: Path) -> None:
    store_cls = resolve_public_symbol("sheaf_verify", "Store", ("sheaf_verify.store",))
    runner_run = resolve_public_symbol("sheaf_verify", "run", ("sheaf_verify.runner",))
    seq_clock = resolve_public_symbol("sheaf_verify", "SequenceClock", ("sheaf_verify.runner",))
    synth_build = resolve_public_symbol("sheaf_verify", "build", ("sheaf_verify.synthetic",))
    test_key = resolve_public_symbol("sheaf_verify", "TEST_KEY", ("sheaf_verify.synthetic",))
    verify_store = resolve_public_symbol("sheaf_verify", "verify_store", ("sheaf_verify.verify",))

    rule_path = repo_root / ".agents" / "rules" / "10-sheaf-verify-stdlib.md"
    rule_meta, _ = _parse_frontmatter(rule_path.read_text(encoding="utf-8"), rule_path)
    assert rule_meta.get("trigger") == "glob"
    assert isinstance(rule_meta.get("globs"), str) and "sheaf_verify" in rule_meta["globs"]

    hooks_path = repo_root / ".agents" / "hooks.json"
    cmds = _matching_pretooluse_commands(hooks_path, "write_to_file")

    # (a) Reject non-stdlib import in src/sheaf_verify/ (F17 + F10)
    deny_np = _make_stdin_payload(
        tmp_path,
        "write_to_file",
        {
            "path": "src/sheaf_verify/bad_np.py",
            "file_path": "src/sheaf_verify/bad_np.py",
            "TargetFile": "src/sheaf_verify/bad_np.py",
            "content": "import numpy as np\n",
            "CodeContent": "import numpy as np\n",
        },
    )
    np_out = [run_hook(cmd, deny_np, cwd=repo_root) for cmd in cmds]
    assert all(proc.returncode == 0 for proc, _ in np_out)
    assert any(parsed.get("decision") == "deny" for _, parsed in np_out)

    # (b) Reject 182x181 on 32768 GPU (F17 + F15)
    bad_182 = 'vocab_size = 32768\nhead = "182x181"\ndevice = "cuda"\n'
    deny_182 = _make_stdin_payload(
        tmp_path,
        "write_to_file",
        {
            "path": "src/hadukenn_core/bad_head.py",
            "file_path": "src/hadukenn_core/bad_head.py",
            "TargetFile": "src/hadukenn_core/bad_head.py",
            "content": bad_182,
            "CodeContent": bad_182,
        },
    )
    out_182 = [run_hook(cmd, deny_182, cwd=repo_root) for cmd in cmds]
    assert all(proc.returncode == 0 for proc, _ in out_182)
    assert any(parsed.get("decision") == "deny" for _, parsed in out_182)

    # (c) Allow valid pure-stdlib sheaf_verify edit
    allow_stdlib = _make_stdin_payload(
        tmp_path,
        "write_to_file",
        {
            "path": "src/sheaf_verify/clean_ok.py",
            "file_path": "src/sheaf_verify/clean_ok.py",
            "TargetFile": "src/sheaf_verify/clean_ok.py",
            "content": "from __future__ import annotations\n\nimport hashlib\n",
            "CodeContent": "from __future__ import annotations\n\nimport hashlib\n",
        },
    )
    for cmd in cmds:
        proc, parsed = run_hook(cmd, allow_stdlib, cwd=repo_root)
        assert proc.returncode == 0
        assert parsed.get("decision") in {"allow", "auto_approve"}

    # (d) runner.py produces a verifiable store and stdio MCP matches canonical fixtures/passing anchor (F17 + F1)
    p19_dir = tmp_path / "p19_store"
    synth_build(p19_dir)
    p19_store = store_cls(p19_dir)
    assert verify_store(p19_store, key=test_key).ok is True
    records = runner_run(p19_store, key=test_key, clock=seq_clock("2026-01-01T01:00:00Z"))
    assert records and all("profile" not in r for r in records)
    assert verify_store(p19_store, key=test_key).ok is True

    baseline = snapshot_readonly_hashes(repo_root)
    try:
        anc_proc = run_sheaf_cli(["anchor"], store=repo_root / "fixtures" / "passing")
        assert anc_proc.returncode == 0, anc_proc.stderr
        expected_anchor = anc_proc.stdout.strip().split()[-1]
        assert expected_anchor == CANONICAL_ANCHOR
        with mcp_stdio_session() as client:
            init_resp = client.initialize()
            assert "result" in init_resp
            tools = client.list_tools()
            tool_names = [t["name"] for t in tools]
            assert "sheaf_anchor" in tool_names or any("anchor" in n.lower() for n in tool_names)
            anchor_tool = next((t for t in tools if t["name"] == "sheaf_anchor"), next(t for t in tools if "anchor" in t["name"].lower()))
            props = anchor_tool.get("inputSchema", {}).get("properties", {})
            args = {k: "fixtures/passing" for k in ("store", "store_path", "path") if k in props}
            call_resp = client.call_tool(anchor_tool["name"], args)
            assert "result" in call_resp and call_resp["result"].get("isError") is not True
            assert expected_anchor in json.dumps(call_resp["result"], sort_keys=True)
    finally:
        assert_readonly_unchanged(baseline, repo_root)


@pytest.mark.tier3
def test_t3_p20_f18_sdk_multimodal_pathway_with_f12_f15_f17_docs_budget(repo_root: Path, tmp_path: Path) -> None:
    local_cfg_cls = resolve_public_symbol("hadukenn_core", "LocalAgentConfig", _ANTIGRAVITY_PREFERRED_MODULES)
    litert_cfg_cls = resolve_public_symbol("hadukenn_core", "LiteRTAgentConfig", _ANTIGRAVITY_PREFERRED_MODULES)
    caps_cls = resolve_public_symbol("hadukenn_core", "CapabilitiesConfig", _ANTIGRAVITY_PREFERRED_MODULES)
    builtin_tools = resolve_public_symbol("hadukenn_core", "BuiltinTools", _ANTIGRAVITY_PREFERRED_MODULES)
    conv_validator = resolve_any_public_symbol(
        "hadukenn_core", ("validate_conversation_id", "is_valid_conversation_id"), _ANTIGRAVITY_PREFERRED_MODULES
    )
    schema_fn = resolve_any_public_symbol(
        "hadukenn_core", ("validate_response_schema", "verify_response_schema", "ResponseSchemaValidator"), _ANTIGRAVITY_PREFERRED_MODULES
    )
    pathway_fn = resolve_any_public_symbol(
        "hadukenn_core",
        ("run_multimodal_pathway", "evaluate_multimodal_pathway", "AntigravityMultimodalRouter", "MultiModalPathway"),
        _ANTIGRAVITY_PREFERRED_MODULES,
    )
    from_file_fn = resolve_any_public_symbol(
        "hadukenn_core", ("from_file", "load_media_attachment", "MultimodalAttachment"), _ANTIGRAVITY_PREFERRED_MODULES
    )
    gate_cls = resolve_any_public_symbol("sheaf_verify", ("SiipVerifyGate", "SIIPVerifyGate", "siip_verify_gate"))

    conv_id = "conv-antigravity-e2e-t3-p20-0123456789abcdef"
    assert _verdict_passed(conv_validator(conv_id)) is True
    _assert_rejected(conv_validator, "short_bad_id")

    ro_tools = list(builtin_tools.read_only())
    caps = _bind_call(caps_cls, {"enabled_tools": ro_tools, "enable_subagents": True}, (ro_tools,))
    local_cfg = _bind_call(local_cfg_cls, {"conversation_id": conv_id, "capabilities": caps})
    assert local_cfg is not None
    _assert_rejected(local_cfg_cls, conversation_id="short_bad_id", capabilities=caps)

    media_path = tmp_path / "t3_p20.png"
    media_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24)
    att = from_file_fn.from_file(media_path) if isinstance(from_file_fn, type) and hasattr(from_file_fn, "from_file") else from_file_fn(media_path)
    litert_cfg = _bind_call(litert_cfg_cls, {"model_path": "models/struct32k_int8", "vision_backend": "cpu", "audio_backend": "cpu"}, ("models/struct32k_int8",))
    runner = pathway_fn() if isinstance(pathway_fn, type) else pathway_fn
    exec_fn = getattr(runner, "run", getattr(runner, "evaluate", runner))

    res_ok = _bind_call(exec_fn, {"config": litert_cfg, "attachments": [att], "vocab_size": 32_768, "head_type": "dense_flat", "hardware_target": "cpu"})
    assert _verdict_passed(res_ok) is True

    sheaf_err = resolve_public_symbol("sheaf_verify", "SheafError", ("sheaf_verify.hashing",))
    try:
        res_bad = _bind_call(
            exec_fn,
            {"config": litert_cfg, "attachments": [att], "vocab_size": 32_768, "head_type": "182x181", "hardware_target": "cuda"},
        )
    except (sheaf_err, ValueError, AssertionError):
        pass
    else:
        assert _verdict_passed(res_bad) is False

    # SIIP dual-mode gate + response_schema validation
    gate = gate_cls()
    if hasattr(gate, "set_mode"):
        gate.set_mode("formal")
    verify_fn = getattr(gate, "verify_commit", getattr(gate, "verify_candidate"))
    assert _verdict_passed(verify_fn(f_k=1.0, f_cand=0.6, x_cand=[0.2, 0.3])) is True

    schema = {"type": "object", "required": ["ok", "mode"], "properties": {"ok": {"type": "boolean"}, "mode": {"type": "string"}}}
    payload = {"ok": True, "mode": str(getattr(gate, "mode", "formal"))}
    bad_payload = {"ok": "not-a-bool"}
    if isinstance(schema_fn, type):
        val_obj = _bind_call(schema_fn, {"schema": schema}, (schema,))
        check_method = getattr(val_obj, "validate", getattr(val_obj, "verify"))
        assert _verdict_passed(check_method(payload)) is True
        _assert_rejected(check_method, bad_payload)
    else:
        schema_params = inspect.signature(schema_fn).parameters
        data_key = "data" if "data" in schema_params and "payload" not in schema_params else "payload"
        assert _verdict_passed(_bind_call(schema_fn, {data_key: payload, "schema": schema}, (payload, schema))) is True
        _assert_rejected(
            lambda d: _bind_call(schema_fn, {data_key: d, "schema": schema}, (d, schema)),
            bad_payload,
        )

    # Check gate-auditor agent is read-only (F17) and docs_budget passes (F18)
    auditor_path = repo_root / ".agents" / "agents" / "gate-auditor" / "agent.md"
    auditor_meta, _ = _parse_frontmatter(auditor_path.read_text(encoding="utf-8"), auditor_path)
    assert _WRITE_RUNTIME_TOOLS.isdisjoint(set(auditor_meta["tools"]))

    budget_proc = subprocess.run(
        [sys.executable, str(repo_root / ".github" / "scripts" / "docs_budget.py")],
        cwd=repo_root,
        env=no_gpu_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=15.0,
    )
    assert budget_proc.returncode == 0, budget_proc.stderr


@pytest.mark.tier3
def test_t3_p21_f6_f7_hal_guards_and_jovin_payload_with_f13_sm120_cpu_parity(
    repo_root: Path, tmp_path: Path, no_gpu_env: dict[str, str]
) -> None:
    import numpy as np

    assert no_gpu_env["CUDA_VISIBLE_DEVICES"] == ""
    eval_hal4 = resolve_public_symbol("sheaf_verify", "evaluate_hal4", ("sheaf_verify.hal",))
    eval_hal6 = resolve_public_symbol("sheaf_verify", "evaluate_hal6", ("sheaf_verify.hal",))
    cpu_parity_fn = resolve_any_public_symbol(
        "hadukenn_core",
        (
            "c0_rtx5070_warp_julia_and_tuple_cpu",
            "emulate_warp_julia_and_tuple_cpu",
            "cpu_warp_julia_tuple_parity",
            "verify_sm120_warp_julia_cpu_parity",
            "run_sm120_warp_julia_cpu_ref",
        ),
        _KERNEL_MODULES,
    )
    jovin = _load_script_module(repo_root, "scripts/jovin_cpu_payload.py", "e2e_jovin_t3_p21")

    c_in = (0.10 * np.random.default_rng(321).standard_normal((32, 4))).astype(np.float32)
    c_in[:, 0] = np.float32(0.85)
    ref_q, ref_p5 = _numpy_sm120_warp_julia_ref(c_in, 0.25)
    res = _call_with_aliases(cpu_parity_fn, [(("c_H_in", "c_in", "quaternions"), c_in), (("gamma_sheaf", "gamma"), 0.25)], (c_in, 0.25))
    q_out = np.asarray(res[0] if isinstance(res, tuple) else _extract_attr_or_key(res, ("q_out", "q_julia")), dtype=np.float32)
    p5_out = np.asarray(res[1] if isinstance(res, tuple) else _extract_attr_or_key(res, ("p5_jk_rotor_out", "p5_out", "rotor_out")), dtype=np.float32)
    assert float(np.max(np.abs(q_out - ref_q))) <= 1e-5
    assert float(np.max(np.abs(p5_out - ref_p5))) <= 1e-5

    parts = _make_hal_parts(tmp_path / "p21")
    ctx_ok = _make_inline_ctx({"claim": parts["claim"], "log.parity": parts["parity"], "log.guards": parts["guards"]})
    assert all(c.passed for c in eval_hal4(ctx_ok))
    assert all(c.passed for c in eval_hal6(ctx_ok))
    assert jovin.all_passed(jovin.evaluate_parity(parts["parity"], parts["claim"]))
    assert jovin.all_passed(jovin.evaluate_guards(parts["guards"], parts["claim"]))

    # Tighten claim thermal stop below sample celsius=62 -> both HAL6 and jovin reject
    tight_claim = json.loads(json.dumps(parts["claim"]))
    tight_claim["guards"]["thermal_stop_celsius"] = 60
    ctx_hot = _make_inline_ctx({"claim": tight_claim, "log.guards": parts["guards"]})
    assert not all(c.passed for c in eval_hal6(ctx_hot))
    assert not jovin.all_passed(jovin.evaluate_guards(parts["guards"], tight_claim))
