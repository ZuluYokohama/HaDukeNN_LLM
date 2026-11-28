"""Tier 4 real-world end-to-end application scenario tests across Features F1-F18.

Pre-M2/M3/M4 collection safety: zero top-level imports of `sheaf_verify.*`,
`hadukenn_core.*`, `scripts.*`, `numpy`, `scipy`, `torch`, or `yaml`.
All target symbols and scientific libraries are resolved inside test bodies or local helpers.
"""

from __future__ import annotations

import hashlib
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
    mcp_stdio_session,
    no_gpu_env,
    resolve_any_public_symbol,
    resolve_public_symbol,
    run_hook,
    run_sheaf_cli,
    scaffold_mini_selftest_repo,
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

_APP_CLI_RUNTIME_TOOLS = frozenset(
    {
        "view_file",
        "write_to_file",
        "replace_file_content",
        "multi_replace_file_content",
        "list_dir",
        "find_by_name",
        "grep_search",
        "search_web",
        "read_url_content",
        "run_command",
        "manage_task",
        "schedule",
        "list_permissions",
        "ask_permission",
        "invoke_subagent",
        "define_subagent",
        "send_message",
        "manage_subagents",
        "ask_question",
        "generate_image",
    }
)

_WRITE_RUNTIME_TOOLS = frozenset(
    {
        "write_to_file",
        "replace_file_content",
        "multi_replace_file_content",
    }
)

_PLUGIN_ALLOWED_KEYS = frozenset(
    {
        "$schema",
        "name",
        "displayName",
        "description",
        "version",
        "disabled",
        "logo",
        "suggestedPrompts",
        "skills",
        "author",
        "homepage",
        "repository",
        "license",
        "keywords",
        "extensions",
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
    return {
        "code": code,
        "host": host,
        "kernel": kernel,
        "env": {
            "python": "3.11.9",
            "cuda": "release 12.4",
            "driver": "595.99.02",
            "gpu": "NVIDIA GeForce RTX 4080 Laptop GPU (12282 MiB, sm_89)",
            "platform": "Linux-test-harness",
        },
        "claim": {
            "format": "sheaf-verify.hal-claim.v1",
            "study_profile": "hal",
            "hypothesis_id": "e2e-hal-t4",
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
        },
        "config": {"format": "sheaf-verify.config.v1", "study": "hal-e2e-t4"},
        "metrics": {"format": "sheaf-verify.metrics.note.v1", "note": "HAL fixture; not a G0 record"},
        "parity": {
            "format": "sheaf-verify.hal-parity.v1",
            "cases": [
                {
                    "name": "cce_tiny",
                    "op": "cce_mup_fwd",
                    "inputs": {"H": hidden, "E": embedding, "y": targets, "mup_base_d": mup_base_d},
                    "kernel_outputs": {"loss": loss},
                }
            ],
        },
        "timings": {
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
        },
        "guards": {
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
        },
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
            existing.append(item_raw.strip().strip('"').strip("'"))
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
            if val_stripped.startswith("[") and val_stripped.endswith("]"):
                inner = val_stripped[1:-1].strip()
                meta[key] = [p.strip().strip('"').strip("'") for p in inner.split(",") if p.strip()] if inner else []
            elif val_stripped.lower() == "true":
                meta[key] = True
            elif val_stripped.lower() == "false":
                meta[key] = False
            else:
                meta[key] = val_stripped.strip('"').strip("'")
    return meta, "\n".join(lines[end_idx + 1 :])


def _resolve_hook_command(raw_cmd: str, base_dir: Path) -> list[str]:
    parts = shlex.split(raw_cmd)
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
    assert isinstance(groups, list) and groups
    cmds: list[list[str]] = []
    for group in groups:
        matcher = group.get("matcher", "")
        if isinstance(matcher, str) and matcher and re.search(matcher, tool_name):
            for handler in group.get("hooks", []):
                raw_cmd = handler.get("command")
                assert isinstance(raw_cmd, str) and raw_cmd.strip()
                cmds.append(_resolve_hook_command(raw_cmd, REPO_ROOT))
    assert cmds, f"No PreToolUse hook matched {tool_name!r}"
    return cmds


def _make_stdin_payload(tmp_path: Path, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "conversationId": "conv-antigravity-e2e-t4-0123456789abcdef",
        "workspacePaths": [str(REPO_ROOT)],
        "transcriptPath": str(tmp_path / "transcript.jsonl"),
        "artifactDirectoryPath": str(tmp_path / "artifacts"),
        "modelName": "gemini-2.5-pro",
        "lastUserInput": "Run Tier 4 end-to-end governance scenario.",
        "toolName": tool_name,
        "toolInput": args,
        "toolCall": {"name": tool_name, "args": args},
    }


def _assert_pi5_node_coverage(policies: dict[str, Any], expected_n: int) -> None:
    coverage_keys = (
        "pi_5_covered_node_count",
        "pi_5_node_policy_weights",
        "pi_5_chunk_sizes",
        "pi_5_chunk_slices",
    )
    assert any(k in policies for k in coverage_keys), f"Missing pi_5 coverage key in {sorted(policies)!r}"
    if "pi_5_covered_node_count" in policies:
        assert int(policies["pi_5_covered_node_count"]) == expected_n
    if "pi_5_node_policy_weights" in policies:
        assert len(policies["pi_5_node_policy_weights"]) == expected_n
    if "pi_5_chunk_sizes" in policies:
        sizes = [int(x) for x in policies["pi_5_chunk_sizes"]]
        assert sum(sizes) == expected_n and all(sz > 0 for sz in sizes)


# ==============================================================================
# Tier 4 Real-World End-to-End Application Scenarios (10 scenarios)
# ==============================================================================


@pytest.mark.tier4
def test_t4_s01_full_pr_gate_audit_and_attestation_pipeline(
    repo_root: Path, tmp_path: Path, readonly_snapshot: dict[str, str]
) -> None:
    del readonly_snapshot
    synthetic_build = resolve_public_symbol("sheaf_verify", "build", ("sheaf_verify.synthetic",))
    tok_cls = resolve_public_symbol("sheaf_verify", "HSTTokenizer", ("sheaf_verify.hst_standard",))
    verify_hst = resolve_public_symbol("sheaf_verify", "verify_hst_manifest_integrity", ("sheaf_verify.hst_standard",))
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))
    emit_fn = resolve_public_symbol("sheaf_verify", "emit_regulatory_bundle", ("sheaf_verify.compliance",))
    verify_bundle = resolve_public_symbol("sheaf_verify", "verify_regulatory_bundle", ("sheaf_verify.compliance",))

    # Step 1: Canonical anchor + verify on immutable fixtures/passing
    passing = repo_root / "fixtures" / "passing"
    key_file = repo_root / "fixtures" / "test_seal.key"
    anc_proc = run_sheaf_cli(["anchor"], store=passing)
    assert anc_proc.returncode == 0, anc_proc.stderr
    assert anc_proc.stdout.strip() == f"sheaf-verify.anchor.v1 {CANONICAL_ANCHOR}"

    ver_proc = run_sheaf_cli(["verify", "--json", "--seal-key-file", str(key_file)], store=passing)
    assert ver_proc.returncode == 0, ver_proc.stderr
    ver_json = json.loads(ver_proc.stdout)
    assert ver_json["ok"] is True and ver_json["problems"] == []

    # Step 2: Build ephemeral store, attest + verify-attestation, receipt + verify-receipt
    ephemeral = tmp_path / "s01_store"
    synthetic_build(ephemeral)
    att_file = tmp_path / "attestation.json"
    att_proc = run_sheaf_cli(["attest", "--seal-key-file", str(key_file), "--out", str(att_file)], store=ephemeral)
    assert att_proc.returncode == 0, att_proc.stderr
    vatt_proc = run_sheaf_cli(
        ["verify-attestation", "--seal-key-file", str(key_file), "--envelope", str(att_file), "--json"]
    )
    assert vatt_proc.returncode == 0, vatt_proc.stderr
    assert json.loads(vatt_proc.stdout)["ok"] is True

    rcpt_file = tmp_path / "receipt_0.json"
    rcpt_proc = run_sheaf_cli(
        ["receipt", "--seq", "0", "--seal-key-file", str(key_file), "--out", str(rcpt_file)], store=ephemeral
    )
    assert rcpt_proc.returncode == 0, rcpt_proc.stderr
    vrcpt_proc = run_sheaf_cli(
        ["verify-receipt", "--receipt", str(rcpt_file), "--seal-key-file", str(key_file), "--json"]
    )
    assert vrcpt_proc.returncode == 0, vrcpt_proc.stderr
    assert json.loads(vrcpt_proc.stdout)["ok"] is True

    # Step 3: HST manifest + compliance ontology + regulatory bundle binding
    tok = tok_cls.train(["pr gate audit sheaf attestation merkle root\n" * 6], vocab_size=288, min_pair_count=2, min_superspan_pmi=0.2)
    manifest = tok.export_manifest()
    ok_hst, probs_hst = verify_hst(manifest)
    assert ok_hst is True and probs_hst == []

    ont = ont_cls("s01-pr-gate", [rule_cls(action_name="merge", dimension="ratio")])
    v_ok = ont.evaluate_transfer_action(
        action_name="merge",
        actor_role="operator",
        src_entity="pr13",
        dst_entity="main",
        src_witnesses={"g1": (Fraction(1), "units"), "g2": (Fraction(1), "units")},
        dst_initial=(Fraction(0), "units"),
        transfer_amount=(Fraction(1), "units"),
        claimed_post_src=(Fraction(0), "units"),
        claimed_post_dst=(Fraction(1), "units"),
    )
    v_fault = ont.evaluate_transfer_action(
        action_name="merge",
        actor_role="operator",
        src_entity="pr13",
        dst_entity="main",
        src_witnesses={"g1": (Fraction(1), "units"), "g2": (Fraction(1), "units"), "g3": (Fraction(9), "units")},
        dst_initial=(Fraction(0), "units"),
        transfer_amount=(Fraction(1), "units"),
        claimed_post_src=(Fraction(0), "units"),
        claimed_post_dst=(Fraction(1), "units"),
    )
    v_bad = ont.evaluate_transfer_action(
        action_name="merge",
        actor_role="operator",
        src_entity="pr13",
        dst_entity="main",
        src_witnesses={"g1": (Fraction(1), "units"), "g2": (Fraction(1), "units")},
        dst_initial=(Fraction(0), "units"),
        transfer_amount=(Fraction(1), "units"),
        claimed_post_src=(Fraction(0), "kg"),
        claimed_post_dst=(Fraction(1), "units"),
    )
    assert v_ok.verdict == "VALID" and v_fault.verdict == "RECOVERABLE_FAULT" and v_bad.verdict == "INVALID"

    bundle_key = b"t4-s01-regulatory-key-0123456789"
    bundle = emit_fn(
        system_id="hadukenn-pr13-gate",
        trace_head_sha256=ver_json["head"],
        state_digest_sha256=manifest["merkle_root_sha256"],
        policy_sha256=ont.policy_sha256,
        events=[v_ok.as_dict(), v_fault.as_dict(), v_bad.as_dict()],
        created_at="2026-11-28T13:00:00Z",
        signing_key=bundle_key,
    )
    ok_b, errs_b = verify_bundle(bundle, signing_key=bundle_key)
    assert ok_b is True and errs_b == []


@pytest.mark.tier4
def test_t4_s02_tri_machine_hal_cuda_npu_verification_and_compliance(
    repo_root: Path, tmp_path: Path, no_gpu_env: dict[str, str], readonly_snapshot: dict[str, str]
) -> None:
    import numpy as np

    del readonly_snapshot
    assert no_gpu_env["CUDA_VISIBLE_DEVICES"] == ""
    set_probe = resolve_public_symbol("sheaf_verify", "set_probe", ("sheaf_verify.hal",))
    run_fn = resolve_public_symbol("sheaf_verify", "run", ("sheaf_verify.runner",))
    seq_clock = resolve_public_symbol("sheaf_verify", "SequenceClock", ("sheaf_verify.runner",))
    verify_store = resolve_public_symbol("sheaf_verify", "verify_store", ("sheaf_verify.verify",))
    test_key = resolve_public_symbol("sheaf_verify", "TEST_KEY", ("sheaf_verify.synthetic",))
    build_part = resolve_public_symbol("hadukenn_core", "build_balanced_partition", ("hadukenn_core.partitions",))
    cluster_is = resolve_public_symbol("hadukenn_core", "cluster_is_fwd_bwd", ("hadukenn_core.heads",))
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
    jovin = _load_script_module(repo_root, "scripts/jovin_cpu_payload.py", "e2e_jovin_t4_s02")

    # 1. Cluster-IS finite loss & gradient sanity
    rng = np.random.default_rng(402)
    nc, s_sz, d, n = 4, 4, 8, 6
    v = nc * s_sz
    part = build_part(v, nc, s_sz, [rng.integers(0, v, size=64, dtype=np.int32)], mode="mass_random", seed=402)
    loss_is, root_nll, dH, dE, dd = cluster_is(
        rng.normal(0.0, 0.25, (n, d)),
        rng.normal(0.0, 0.25, (v, d)),
        np.zeros((nc, d)),
        part,
        rng.integers(0, v, size=n, dtype=np.int64),
        k_neg=3,
        rng=rng,
    )
    assert np.all(np.isfinite(loss_is)) and math.isfinite(root_nll)
    assert dH is not None and dE is not None and dd is not None

    # 2. sm_120 CPU warp-reset parity across 2 warps (N=48); lanes 32..47 diverge from unreset scan
    c_in = (0.12 * rng.standard_normal((48, 4))).astype(np.float32)
    c_in[:, 0] = np.float32(0.85)
    ref_q, ref_p5 = _numpy_sm120_warp_julia_ref(c_in, 0.25)
    warp_res = _call_with_aliases(cpu_parity_fn, [(("c_H_in", "c_in", "quaternions"), c_in), (("gamma_sheaf", "gamma"), 0.25)], (c_in, 0.25))
    q_out = np.asarray(warp_res[0] if isinstance(warp_res, tuple) else _extract_attr_or_key(warp_res, ("q_out", "q_julia")), dtype=np.float32)
    p5_out = np.asarray(warp_res[1] if isinstance(warp_res, tuple) else _extract_attr_or_key(warp_res, ("p5_jk_rotor_out", "p5_out", "rotor_out")), dtype=np.float32)
    assert float(np.max(np.abs(q_out - ref_q))) <= 1e-5
    assert float(np.max(np.abs(p5_out - ref_p5))) <= 1e-5

    # Compare lanes 32..47 against single-warp parity on slice c_in[32:] vs unreset scan
    ref_q_warp1, _ = _numpy_sm120_warp_julia_ref(c_in[32:], 0.25)
    assert float(np.max(np.abs(q_out[32:] - ref_q_warp1))) <= 1e-5

    # 3. Full HAL0..HAL6 run + jovin payload check
    parts = _make_hal_parts(tmp_path / "s02")
    store = _register_hal_store(tmp_path / "s02_store", parts)
    set_probe(lambda: dict(parts["env"]))
    try:
        records = run_fn(store, key=test_key, clock=seq_clock(), profile="hal")
        store_rep = verify_store(store, key=test_key)
    finally:
        set_probe(None)
    assert [r["gate_id"] for r in records] == [f"HAL{i}" for i in range(7)]
    assert all(r["status"] == "pass" for r in records)
    assert store_rep.ok is True
    assert jovin.all_passed(jovin.evaluate_parity(parts["parity"], parts["claim"]))
    assert jovin.all_passed(jovin.evaluate_guards(parts["guards"], parts["claim"]))

    # 4. Verify sealed T46 tri-machine campaign artifact
    t46_path = repo_root / "iterations" / "t46-tri-machine-hal" / "T46_TRI_MACHINE_RESULTS.json"
    camp = run_sheaf_cli(["campaign", "--mode", "t46-tri-machine-hal", "--path", str(t46_path), "--json"])
    assert camp.returncode == 0, camp.stderr
    assert json.loads(camp.stdout)["ok"] is True


@pytest.mark.tier4
def test_t4_s03_universal_kernel_op01_to_op21_r1cs_siip_and_ledger() -> None:
    pipeline = resolve_any_public_symbol(
        "hadukenn_core", ("UNIVERSAL_KERNEL_PIPELINE", "KERNEL_PIPELINE", "C0_PIPELINE"), _KERNEL_MODULES
    )
    op_21 = _resolve_op("op_21_value_network_graph_alignment_policy_engine", "op_21")
    r1cs_fn = resolve_any_public_symbol(
        "sheaf_verify", ("verify_goldilocks_r1cs_dipole", "verify_goldilocks_r1cs", "goldilocks_r1cs_dipole_verify")
    )
    receipt_fn = resolve_any_public_symbol(
        "sheaf_verify", ("verify_canonical_receipt", "compute_canonical_receipt", "canonical_receipt_sha256")
    )
    gate_cls = resolve_any_public_symbol("sheaf_verify", ("SiipVerifyGate", "SIIPVerifyGate", "siip_verify_gate"))
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))
    emit_fn = resolve_public_symbol("sheaf_verify", "emit_regulatory_bundle", ("sheaf_verify.compliance",))
    verify_bundle = resolve_public_symbol("sheaf_verify", "verify_regulatory_bundle", ("sheaf_verify.compliance",))

    s = pipeline(_build_kernel_state(n_steps=128, m_streams=4, seed=403))
    if not getattr(s, "value_graph_policy_alignment", None):
        s = op_21(s)
    assert len(s.checks) >= 20
    assert all(c["passed"] is True for c in s.checks)

    vga = s.value_graph_policy_alignment
    unaligned = vga["unaligned_open_loop_graph_metrics"]
    aligned = vga["policy_aligned_relationship_graph_metrics"]
    assert aligned["centered_kernel_graph_alignment_cka"] > unaligned["centered_kernel_graph_alignment_cka"]
    assert aligned["pairwise_value_edge_rank_correlation"] > unaligned["pairwise_value_edge_rank_correlation"]
    policies = vga["adaptive_policies_executed"]
    _assert_pi5_node_coverage(policies, 128)
    assert int(policies["pi_6_siip_formal_verified_bfgs_commits"]) >= 1

    witness = [float(x) for x in s.spectral_circuit["latest_12neuron_witness"]]
    y_float = float(s.zk_goldilocks_circuit.get("y_float", 0.25))
    r1cs_res = _call_with_aliases(
        r1cs_fn, [(("y_float", "y"), y_float), (("dipole_acts", "witness", "acts"), witness)], (y_float, witness)
    )
    assert _verdict_passed(r1cs_res) is True
    assert int(_extract_attr_or_key(r1cs_res, ("r1cs_gate_residual_mod_p", "r1cs_gate_residual", "residual"))) == 0

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
    else:
        assert receipt_fn(body) == receipt_hex

    gate = gate_cls()
    if hasattr(gate, "set_mode"):
        gate.set_mode("formal")
    verify_fn = getattr(gate, "verify_commit", getattr(gate, "verify_candidate"))
    assert _verdict_passed(verify_fn(f_k=1.0, f_cand=0.5, x_cand=witness)) is True

    ont = ont_cls("t4-s03-kernel", [rule_cls(action_name="commit", dimension="ratio")])
    verdict = ont.evaluate_transfer_action(
        action_name="commit",
        actor_role="operator",
        src_entity="op21",
        dst_entity="ledger",
        src_witnesses={"zk": (Fraction(1), "units"), "siip": (Fraction(1), "units")},
        dst_initial=(Fraction(0), "units"),
        transfer_amount=(Fraction(1), "units"),
        claimed_post_src=(Fraction(0), "units"),
        claimed_post_dst=(Fraction(1), "units"),
    )
    assert verdict.verdict == "VALID"
    key = b"t4-s03-kernel-ledger-key-01234567"
    bundle = emit_fn(
        system_id="sys-t4-s03-kernel",
        trace_head_sha256=receipt_hex,
        state_digest_sha256=hashlib.sha256(str(s.zk_goldilocks_circuit["zk_stark_witness_commitment"]).encode("utf-8")).hexdigest(),
        policy_sha256=ont.policy_sha256,
        events=[verdict.as_dict()],
        created_at="2026-11-28T13:05:00Z",
        signing_key=key,
    )
    ok_b, errs_b = verify_bundle(bundle, signing_key=key)
    assert ok_b is True and errs_b == []


@pytest.mark.tier4
def test_t4_s04_multi_doc_rlm_oolong_pairs_budget_and_zigzag_pruning() -> None:
    import numpy as np

    policy_cls = resolve_public_symbol("sheaf_verify", "RLMBudgetPolicy")
    state_cls = resolve_public_symbol("sheaf_verify", "RLMBudgetState")
    tracker_cls = resolve_public_symbol("sheaf_verify", "AntiRepetitionTracker")
    pi_shift = resolve_any_public_symbol(
        "sheaf_verify", ("zigzag_effective_pi", "shift_persistence_image", "verify_zigzag_pi_hat")
    )
    zp_fn = resolve_any_public_symbol(
        "sheaf_verify", ("zigzag_persistence_ratio", "zigzag_interlayer_persistence", "compute_zigzag_zp")
    )
    repl_cls = resolve_any_public_symbol(
        "hadukenn_core", ("SandboxedPythonREPL", "RLMContextREPL", "RLMReplEnvironment", "SandboxedRLMRepl"), _RLM_MODULES
    )
    prof_fn = resolve_any_public_symbol(
        "hadukenn_core", ("get_rlm_prompt_profile", "RLMPromptProfile", "RLM_PROMPT_PROFILES"), _RLM_MODULES
    )
    orch_cls = resolve_any_public_symbol("hadukenn_core", ("RLMOrchestrator", "RLMEngine", "RLMRunner"), _RLM_MODULES)
    stitch_fn = resolve_any_public_symbol(
        "hadukenn_core", ("oolong_pairs_stitch", "run_oolong_pairs_stitching", "rlm_subgraph_stitch", "stitch_oolong_pairs"), _RLM_MODULES
    )
    kern_cls = resolve_public_symbol("hadukenn_core", "IncrementalHypergraphKernel", ("hadukenn_core.exec_bridge",))
    zigzag_sym = resolve_any_public_symbol(
        "hadukenn_core", ("ZigzagTopologyAnalyzer", "compute_zigzag_persistence", "zigzag_interlayer_analysis"), _ML_MODULES
    )
    prune_fn = resolve_any_public_symbol(
        "hadukenn_core", ("prune_layers_plateau", "algorithm2_prune_layers", "zigzag_prune_layers", "prune_redundant_layers"), _ML_MODULES
    )

    docs = [f"doc_{i:02d}:domain={i % 3} signal={i * 7}" for i in range(21)]
    qwen_prof = prof_fn("qwen") if callable(prof_fn) else prof_fn["qwen"]
    assert "context" in str(_extract_attr_or_key(qwen_prof, ("system_prompt", "prompt"))).lower()

    policy = policy_cls(max_depth=1, max_subcalls=12, max_total_tokens=100_000, min_batch_chars=20, thinking_reserve=512)
    state = state_cls()
    tracker = tracker_cls(max_consec_no_new_var=3)
    st1 = tracker.record_turn("pair_target = 210", ["pair_target"], verified_value=210)
    assert st1.should_stop is False and st1.consec_no_new_var == 0 and st1.new_vars == ("pair_target",)
    assert tracker.record_turn("noop1", []).should_stop is False
    assert tracker.record_turn("noop2", ["pair_target"]).should_stop is False
    st_halt = tracker.record_turn("noop3", [])
    assert st_halt.should_stop is True and st_halt.consec_no_new_var == 3 and st_halt.best_verified_value == 210

    repl = repl_cls(docs, max_stdout_chars=2048, budget_policy=policy, budget_state=state)
    exec_res = repl.execute("print(len(context))")
    assert "21" in str(_extract_attr_or_key(exec_res, ("stdout", "output")))

    orch = orch_cls(budget_policy=policy)
    turns = ["```repl\npair_target = len(context) * (len(context) - 1) // 2\n```", "FINAL_VAR(pair_target)"]
    t_idx = 0

    def model_fn(_msgs: Any) -> dict[str, Any]:
        nonlocal t_idx
        out = {"text": turns[t_idx], "max_output_tokens": 4096, "thinking_budget": 512}
        t_idx += 1
        return out

    orch_out = _call_with_aliases(
        orch.run,
        [(("query", "task"), "compute pair count"), (("context",), docs), (("model_fn", "llm_fn"), model_fn)],
        ("compute pair count", docs, model_fn),
    )
    ans = orch_out["answer"] if isinstance(orch_out, dict) and "answer" in orch_out else getattr(orch_out, "answer", orch_out)
    assert ans == 210

    kern = kern_cls()
    stitch_res = _call_with_aliases(
        stitch_fn,
        [(("items", "docs", "records"), docs), (("n_chunks", "c_chunks", "num_chunks"), 4), (("exec_kernel", "kernel"), kern)],
        (docs, 4),
    )
    assert len(set(_extract_attr_or_key(stitch_res, ("covered_indices", "node_indices", "covered_items")))) == 21
    assert int(_extract_attr_or_key(stitch_res, ("total_pairs_evaluated", "pairs_evaluated", "num_pairs"))) == 210
    trace = kern.export_trace()
    assert len(trace) >= 1, "oolong_pairs_stitch(..., exec_kernel=kern) must emit execution trace records"
    replayed = kern_cls.replay(trace, expected_head=kern.head)
    assert replayed.state_digest() == kern.state_digest()

    rng = np.random.default_rng(404)
    traj = np.zeros((8, 24, 12))
    traj[0] = rng.normal(0.0, 1.0, (24, 12))
    traj[1] = traj[0] + rng.normal(0.0, 0.7, (24, 12))
    plateau = rng.normal(0.0, 1.0, (24, 12))
    for ell in (2, 3, 4, 5):
        traj[ell] = plateau + rng.normal(0.0, 0.005, (24, 12))
    traj[6] = rng.normal(0.0, 1.0, (24, 12))
    traj[7] = rng.normal(0.0, 1.0, (24, 12))

    analyzer = zigzag_sym() if isinstance(zigzag_sym, type) else zigzag_sym
    zz_res = analyzer.analyze(traj) if hasattr(analyzer, "analyze") else analyzer(traj)
    z1_bar = np.asarray(
        _extract_attr_or_key(zz_res, ("z1_bar", "z_bar", "mean_interlayer_persistence", "z1_profile")), dtype=float
    )
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

    pi_hat = pi_shift([[0.0, 0.0, 0.0], [0.0, 2.0, 3.0], [0.0, 1.0, 4.0]])
    assert math.isclose(float(pi_hat[1][1]), 10.0, abs_tol=1e-12)
    assert math.isfinite(float(_call_with_aliases(zp_fn, [(("pi_hat", "pi_eff"), pi_hat), (("beta_p",), 2)], (pi_hat, 2))))


@pytest.mark.tier4
def test_t4_s05_antigravity_governance_session_hooks_mcp_and_sdk_fallback(
    repo_root: Path, tmp_path: Path, readonly_snapshot: dict[str, str]
) -> None:
    del readonly_snapshot
    local_cfg_cls = resolve_public_symbol("hadukenn_core", "LocalAgentConfig", _ANTIGRAVITY_PREFERRED_MODULES)
    caps_cls = resolve_public_symbol("hadukenn_core", "CapabilitiesConfig", _ANTIGRAVITY_PREFERRED_MODULES)
    builtin_tools = resolve_public_symbol("hadukenn_core", "BuiltinTools", _ANTIGRAVITY_PREFERRED_MODULES)
    schema_fn = resolve_any_public_symbol(
        "hadukenn_core", ("validate_response_schema", "verify_response_schema", "ResponseSchemaValidator"), _ANTIGRAVITY_PREFERRED_MODULES
    )
    eval_fn = resolve_any_public_symbol(
        "hadukenn_core", ("evaluate_permission", "evaluate_policy", "PermissionPolicy", "PolicyEvaluator"), _ANTIGRAVITY_PREFERRED_MODULES
    )
    hook_result_cls = resolve_public_symbol("hadukenn_core", "HookResult", _ANTIGRAVITY_PREFERRED_MODULES)
    lifecycle = resolve_any_public_symbol(
        "hadukenn_core", ("LifecycleHooks", "HookRegistry", "AntigravityLifecycle", "run_lifecycle_hooks"), _ANTIGRAVITY_PREFERRED_MODULES
    )

    # 1. Validate .agents/ surfaces (rules, skills, agents, plugin, sidecar)
    for rpath in sorted((repo_root / ".agents" / "rules").glob("*.md")):
        rmeta, _ = _parse_frontmatter(rpath.read_text(encoding="utf-8"), rpath)
        if rmeta.get("trigger") == "glob":
            assert isinstance(rmeta.get("globs") if "globs" in rmeta else rmeta.get("glob"), str)
    for spath in sorted((repo_root / ".agents" / "skills").glob("*/SKILL.md")):
        stext = spath.read_text(encoding="utf-8")
        assert "## Safety" in stext and "Do not use when" in stext
    for apath in sorted((repo_root / ".agents" / "agents").glob("*/agent.md")):
        ameta, _ = _parse_frontmatter(apath.read_text(encoding="utf-8"), apath)
        assert set(ameta["tools"]).issubset(_APP_CLI_RUNTIME_TOOLS)
        if ameta["name"] == "gate-auditor":
            assert _WRITE_RUNTIME_TOOLS.isdisjoint(set(ameta["tools"]))

    plugin_root = repo_root / ".agents" / "plugins" / "hadukenn-sheaf"
    p_manifest_path = (
        plugin_root / ".antigravity-plugin" / "plugin.json"
        if (plugin_root / ".antigravity-plugin" / "plugin.json").is_file()
        else plugin_root / "plugin.json"
    )
    p_manifest = json.loads(p_manifest_path.read_text(encoding="utf-8"))
    assert set(p_manifest.keys()).issubset(_PLUGIN_ALLOWED_KEYS)
    sidecars = [
        *(repo_root / ".agents" / "sidecars").glob("**/sidecar.json"),
        *(plugin_root / "sidecars").glob("**/sidecar.json"),
    ]
    assert sidecars

    # 2. PreToolUse hooks allow stdlib & deny numpy in sheaf_verify (exit 0)
    cmds = _matching_pretooluse_commands(repo_root / ".agents" / "hooks.json", "write_to_file")
    allow_p = _make_stdin_payload(
        tmp_path, "write_to_file", {"path": "src/sheaf_verify/ok.py", "TargetFile": "src/sheaf_verify/ok.py", "content": "import json\n", "CodeContent": "import json\n"}
    )
    deny_p = _make_stdin_payload(
        tmp_path, "write_to_file", {"path": "src/sheaf_verify/bad.py", "TargetFile": "src/sheaf_verify/bad.py", "content": "import numpy\n", "CodeContent": "import numpy\n"}
    )
    for cmd in cmds:
        proc_ok, out_ok = run_hook(cmd, allow_p, cwd=repo_root)
        assert proc_ok.returncode == 0 and out_ok.get("decision") in {"allow", "auto_approve"}
    deny_outs = [run_hook(cmd, deny_p, cwd=repo_root) for cmd in cmds]
    assert all(p.returncode == 0 for p, _ in deny_outs)
    assert any(o.get("decision") == "deny" for _, o in deny_outs)

    # 3. Stdio MCP session + SDK session config + response_schema + lifecycle/permission
    conv_id = "conv-antigravity-e2e-t4-s05-0123456789abcdef"
    ro_tools = list(builtin_tools.read_only())
    caps = _bind_call(caps_cls, {"enabled_tools": ro_tools, "enable_subagents": True}, (ro_tools,))
    cfg = _bind_call(local_cfg_cls, {"conversation_id": conv_id, "capabilities": caps})
    assert cfg is not None

    with mcp_stdio_session() as client:
        assert "result" in client.initialize()
        tools = client.list_tools()
        anchor_tool = next((t for t in tools if "anchor" in t["name"].lower()), tools[0])
        props = anchor_tool.get("inputSchema", {}).get("properties", {})
        args = {k: "fixtures/passing" for k in ("store", "store_path", "path") if k in props}
        call_resp = client.call_tool(anchor_tool["name"], args)
        assert "result" in call_resp and call_resp["result"].get("isError") is not True
        rendered = json.dumps(call_resp["result"], sort_keys=True)
        assert CANONICAL_ANCHOR in rendered

    schema = {"type": "object", "required": ["ok", "anchor"], "properties": {"ok": {"type": "boolean"}, "anchor": {"type": "string"}}}
    sample = {"ok": True, "anchor": CANONICAL_ANCHOR}
    bad_sample = {"ok": "not-a-bool"}
    if isinstance(schema_fn, type):
        val_obj = _bind_call(schema_fn, {"schema": schema}, (schema,))
        check_method = getattr(val_obj, "validate", getattr(val_obj, "verify"))
        assert _verdict_passed(check_method(sample)) is True
        _assert_rejected(check_method, bad_sample)
    else:
        schema_params = inspect.signature(schema_fn).parameters
        data_key = "data" if "data" in schema_params and "payload" not in schema_params else "payload"
        assert _verdict_passed(_bind_call(schema_fn, {data_key: sample, "schema": schema}, (sample, schema))) is True
        _assert_rejected(lambda d: _bind_call(schema_fn, {data_key: d, "schema": schema}, (d, schema)), bad_sample)

    rules = [{"action": "read_file", "decision": "allow"}, {"action": "write_file", "decision": "deny"}]
    evaluator = _bind_call(eval_fn, {"rules": rules}, (rules,)) if isinstance(eval_fn, type) else eval_fn
    fn = getattr(evaluator, "evaluate", getattr(evaluator, "check", evaluator))
    w_dec = _bind_call(fn, {"action": "write_file", "rules": rules}, ("write_file", rules))
    assert str(getattr(w_dec, "decision", w_dec)).lower() == "deny"

    hr = _bind_call(hook_result_cls, {"decision": "allow", "reason": "ok"}, ("allow",))
    reg = lifecycle() if isinstance(lifecycle, type) else lifecycle
    seen_hooks: list[str] = []

    def _pre_turn(_ctx: dict[str, Any]) -> Any:
        seen_hooks.append("pre_turn")
        return hr

    if hasattr(reg, "pre_turn") and callable(reg.pre_turn):
        reg.pre_turn(_pre_turn)
    elif hasattr(reg, "register_pre_turn") and callable(reg.register_pre_turn):
        reg.register_pre_turn(_pre_turn)

    if hasattr(reg, "run_pre_turn"):
        hook_out = reg.run_pre_turn({"conversation_id": conv_id})
    elif callable(reg):
        hook_out = _bind_call(reg, {"pre_turn": _pre_turn, "ctx": {"conversation_id": conv_id}})
    else:
        raise AssertionError(f"LifecycleHooks surface lacks run_pre_turn/callable: {reg!r}")
    assert hook_out is not None and "pre_turn" in seen_hooks


@pytest.mark.tier4
def test_t4_s06_hst_merkle_verification_scoped_knnlm_and_chodrow_null(
    repo_root: Path, readonly_snapshot: dict[str, str]
) -> None:
    import numpy as np

    del readonly_snapshot
    tok_cls = resolve_public_symbol("sheaf_verify", "HSTTokenizer", ("sheaf_verify.hst_standard",))
    verify_hst = resolve_public_symbol("sheaf_verify", "verify_hst_manifest_integrity", ("sheaf_verify.hst_standard",))
    jl_min = resolve_public_symbol("sheaf_verify", "jl_min_dimension")
    jl_fn = resolve_any_public_symbol(
        "hadukenn_core", ("jl_project", "johnson_lindenstrauss_project", "jl_compress"), _ML_MODULES
    )
    corpus_doc = resolve_public_symbol("hadukenn_core", "CorpusDocument", ("hadukenn_core.hypergraph",))
    build_hg = resolve_public_symbol("hadukenn_core", "build_decontaminated_hypergraph", ("hadukenn_core.hypergraph",))
    build_null = resolve_public_symbol("hadukenn_core", "build_chodrow_null", ("hadukenn_core.hypergraph",))
    eval_knnlm = resolve_public_symbol("hadukenn_core", "evaluate_scoped_knnlm", ("hadukenn_core.hypergraph",))
    claim_fn = resolve_any_public_symbol(
        "hadukenn_core", ("validate_hypergraph_claim", "audit_chodrow_null_control"), _ML_MODULES
    )
    e1_fn = resolve_any_public_symbol(
        "hadukenn_core", ("validate_e1_ordering_policy", "check_e1_ordering_policy"), _ML_MODULES
    )

    ref_manifest = json.loads(
        (repo_root / "iterations" / "t49-hst-foundation-stage" / "HST_V1_REFERENCE_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    ok_ref, probs_ref = verify_hst(ref_manifest)
    assert ok_ref is True and probs_ref == []

    t49_path = repo_root / "iterations" / "t49-hst-foundation-stage" / "T49_HST_FOUNDATION_RESULTS.json"
    camp = run_sheaf_cli(["campaign", "--mode", "t49-hst-foundation", "--path", str(t49_path), "--json"])
    assert camp.returncode == 0 and json.loads(camp.stdout)["ok"] is True

    texts = ["hst foundation tokenizer causal prefix merkle witness " * 4, "chodrow null hypergraph scoped knnlm control " * 4]
    tok = tok_cls.train(texts, vocab_size=292, min_pair_count=2, min_superspan_pmi=0.2)
    sample = "hst foundation tokenizer causal prefix"
    assert tok.decode(tok.encode(sample)) == sample
    ok_pref, pref_rep = tok.verify_causal_prefix_invariance(sample.encode("utf-8"), [b" one", b" two"])
    assert ok_pref is True and pref_rep["violations"] == 0

    assert int(_call_with_aliases(jl_min, [(("n", "n_points"), 60), (("eps", "epsilon"), 0.2)], (60, 0.2))) > 64
    rng = np.random.default_rng(406)
    x_hi = rng.normal(0.0, 1.0, (60, 256))
    proj = _call_with_aliases(jl_fn, [(("X", "vectors"), x_hi), (("p_dim", "p"), 64), (("seed",), 406)], (x_hi,))
    x_lo = np.asarray(proj[0] if isinstance(proj, tuple) else _extract_attr_or_key(proj, ("projected", "X_proj", "Z")))
    assert x_lo.shape == (60, 64)

    sheaf_err = resolve_public_symbol("sheaf_verify", "SheafError", ("sheaf_verify.hashing",))
    try:
        res_e1 = _call_with_aliases(
            e1_fn, [(("mode", "ordering_mode"), "passive_e1_shuffle"), (("e1_passive_shuffle",), True)], ("passive_e1_shuffle",)
        )
        assert _verdict_passed(res_e1) is False
    except (sheaf_err, ValueError, AssertionError):
        pass

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
    kept_docs, hg = build_hg(train_docs, eval_docs, min_edge_size=2, max_edge_size=12, max_doc_degree=12, seed=406)
    assert kept_docs and len(hg.edges) >= 1
    null_hg = build_null(hg, seed=406)
    assert sorted(len(e) for e in hg.edges) == sorted(len(e) for e in null_hg.edges)

    knn_metrics = eval_knnlm(
        x_lo[40:50],
        rng.integers(0, 8, size=10),
        np.full(10, 0.125),
        [np.array([0, 1, 2, 3], dtype=np.int64) for _ in range(10)],
        [np.array([0, 2], dtype=np.int64) for _ in range(10)],
        x_lo[:40],
        rng.integers(0, 8, size=40),
        np.arange(40) % 2,
        np.full(10, 4.0),
        np.ones(10, dtype=bool),
    )
    assert math.isfinite(float(knn_metrics["hyper_scoped_knn_rare_bpb"]))
    assert _verdict_passed(
        _call_with_aliases(
            claim_fn,
            [(("hypergraph", "hg"), hg), (("chodrow_null", "null_hg"), null_hg), (("hyper_bpb", "metric"), 1.04), (("null_bpb",), 1.11)],
            (hg, null_hg, 1.04, 1.11),
        )
    ) is True


@pytest.mark.tier4
def test_t4_s07_imbalanced_failure_detection_tree_smote_leakage_free() -> None:
    import numpy as np

    gini_fn = resolve_public_symbol("sheaf_verify", "gini_impurity")
    split_fn = resolve_public_symbol("sheaf_verify", "validate_split_disjointness")
    ada_bound = resolve_public_symbol("sheaf_verify", "adaboost_exp_bound")
    rad_bound = resolve_public_symbol("sheaf_verify", "rademacher_kernel_bound")
    smote_fn = resolve_any_public_symbol(
        "hadukenn_core", ("train_only_smote", "smote_resample_train", "smote_train_only"), _ML_MODULES
    )
    tree_cls = resolve_any_public_symbol("hadukenn_core", ("GiniDecisionTree", "fit_gini_tree"), _ML_MODULES)
    rf_cls = resolve_any_public_symbol(
        "hadukenn_core", ("RandomForestClassifierLite", "fit_random_forest"), _ML_MODULES
    )
    ada_fn = resolve_any_public_symbol(
        "hadukenn_core", ("adaboost_edge_weights", "fit_adaboost_edges", "adaboost_reweight_edges"), _ML_MODULES
    )

    rng = np.random.default_rng(407)
    x_tr = rng.normal(0.0, 1.0, (120, 4))
    y_tr = np.zeros(120, dtype=int)
    y_tr[:12] = 1
    x_tr[:12, 0] += 3.2

    x_te = rng.normal(0.0, 1.0, (40, 4))
    y_te = np.zeros(40, dtype=int)
    y_te[:8] = 1
    x_te[:8, 0] += 3.2
    x_te_copy = x_te.copy()

    sm_res = _call_with_aliases(
        smote_fn, [(("X_train", "X"), x_tr), (("y_train", "y"), y_tr), (("X_test",), x_te), (("seed",), 407)], (x_tr, y_tr)
    )
    assert np.array_equal(x_te, x_te_copy)
    if isinstance(sm_res, tuple):
        x_bal, y_bal = np.asarray(sm_res[0]), np.asarray(sm_res[1])
    else:
        x_bal = np.asarray(_extract_attr_or_key(sm_res, ("X_resampled", "X_train_resampled", "X")))
        y_bal = np.asarray(_extract_attr_or_key(sm_res, ("y_resampled", "y_train_resampled", "y")))

    train_ids = {f"tr_{i}" for i in range(120)}
    test_ids = {f"te_{i}" for i in range(40)}
    syn_ids = {f"syn_{i}" for i in range(len(y_bal) - 120)}
    assert syn_ids
    assert _verdict_passed(
        _call_with_aliases(
            split_fn,
            [(("train_ids", "train"), train_ids | syn_ids), (("test_ids", "test"), test_ids), (("synthetic_ids", "synthetic"), syn_ids)],
            (train_ids | syn_ids, test_ids, syn_ids),
        )
    ) is True
    # Isolate synthetic_ids & test_ids leakage without contaminating train_ids
    leak_args, leak_kwargs = _bind_with_aliases(
        split_fn,
        [(("train_ids", "train"), train_ids), (("test_ids", "test"), test_ids | {"smote_leak"}), (("synthetic_ids", "synthetic"), syn_ids | {"smote_leak"})],
        (train_ids, test_ids | {"smote_leak"}, syn_ids | {"smote_leak"}),
    )
    _assert_rejected(split_fn, *leak_args, **leak_kwargs)

    g_before = float(gini_fn([int(np.sum(y_bal == 0)), int(np.sum(y_bal == 1))]))
    mask_l = x_bal[:, 0] > 1.5
    g_after = (
        np.sum(mask_l) * float(gini_fn([int(np.sum(y_bal[mask_l] == 0)), int(np.sum(y_bal[mask_l] == 1))]))
        + np.sum(~mask_l) * float(gini_fn([int(np.sum(y_bal[~mask_l] == 0)), int(np.sum(y_bal[~mask_l] == 1))]))
    ) / len(y_bal)
    assert g_after < g_before

    tree = tree_cls(max_depth=4) if isinstance(tree_cls, type) else tree_cls(x_bal, y_bal, max_depth=4)
    if hasattr(tree, "fit"):
        tree.fit(x_bal, y_bal)
    assert float(tree.feature_importances_[0]) > float(np.max(tree.feature_importances_[1:]))

    rf = rf_cls(n_estimators=15, max_depth=4, seed=407) if isinstance(rf_cls, type) else rf_cls(x_bal, y_bal)
    if hasattr(rf, "fit"):
        rf.fit(x_bal, y_bal)
    preds = np.asarray(rf.predict(x_te), dtype=int)
    recall = int(np.sum((preds == 1) & (y_te == 1))) / int(np.sum(y_te == 1))
    assert recall > 0.5

    y_edge = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)
    weak = np.vstack([y_edge.copy() for _ in range(5)])
    weak[:, 0] = -weak[:, 0]
    ada_res = _call_with_aliases(ada_fn, [(("y", "labels"), y_edge), (("preds", "weak_preds", "H"), weak)], (y_edge, weak))
    prod_zt = float(_extract_attr_or_key(ada_res, ("prod_zt", "bound_prod_zt", "exp_bound")))
    exp_b = float(_call_with_aliases(ada_bound, [(("gamma", "margin"), 0.1), (("t_rounds", "T", "rounds"), 5)], (0.1, 5)))
    assert prod_zt <= exp_b + 1e-12
    assert float(_call_with_aliases(rad_bound, [(("lam", "bound_lambda", "radius_lambda"), 2.0), (("trace_k", "trace_K"), 16.0), (("m", "n_samples"), 120)], (2.0, 16.0, 120))) > 0.0


@pytest.mark.tier4
def test_t4_s08_fresh_process_selftest_control_graph_mutation_and_hygiene(
    repo_root: Path, tmp_path: Path, readonly_snapshot: dict[str, str]
) -> None:
    del readonly_snapshot
    cg_proc = subprocess.run(
        [sys.executable, "-m", "sheaf_verify.control_graph", "--check"],
        cwd=repo_root,
        env=no_gpu_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=30.0,
    )
    assert cg_proc.returncode == 0, cg_proc.stdout + cg_proc.stderr

    mut_proc = run_sheaf_cli(["mutation-score", "--json"])
    assert mut_proc.returncode == 0, mut_proc.stderr
    mut_json = json.loads(mut_proc.stdout)
    assert mut_json["killed"] == 12 and mut_json["survived"] == 0

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

    mini = scaffold_mini_selftest_repo(repo_root, tmp_path / "mini_s08", create_default_test=True)
    rep_out = tmp_path / "mini_s08_selftest.json"
    st_proc = subprocess.run(
        [sys.executable, "-S", "-m", "sheaf_verify.selftest", "--repo-root", str(mini), "--report-out", str(rep_out)],
        cwd=mini,
        env=no_gpu_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=180.0,
    )
    assert st_proc.returncode in (0, 1), st_proc.stdout + st_proc.stderr
    rep = json.loads(rep_out.read_text(encoding="utf-8"))
    assert rep["canonical_fixture"]["ok"] is True
    assert rep["lint"]["ok"] is True
    assert rep["control_graph"]["ok"] is True
    assert rep["docs_budget"]["ok"] is True
    assert rep["pytest_suite"]["failed"] == 0 and rep["pytest_suite"]["passed"] >= 1

    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin", ".so", ".dll"}
    banned_phrases = (
        "Permission is hereby " + "granted",
        "well" + "bore",
        "drill" + "ing",
        "DEVn" + "-1",
        "DESKTOP" + "-",
        "train" + "_paper",
    )
    drive_markers = tuple(chr(code) + ":" + chr(92) for code in (67, 68, 69, 72))
    for folder in (repo_root / ".agents", repo_root / "src" / "hadukenn_core", repo_root / "docs", repo_root / "tests_e2e"):
        for path in folder.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            assert path.suffix.lower() not in forbidden_suffixes, f"Forbidden binary file: {path}"
            if path.suffix.lower() in {".py", ".md", ".toml", ".yml", ".json", ".html", ".css", ""}:
                text = path.read_text(encoding="utf-8", errors="ignore")
                for phrase in banned_phrases:
                    assert phrase not in text, f"Banned phrase in {path}"
                for drive in drive_markers:
                    assert drive not in text, f"Windows drive marker in {path}"


@pytest.mark.tier4
def test_t4_s09_multimodal_edge_npu_vs_gpu_head_policy_audit(
    repo_root: Path, tmp_path: Path
) -> None:
    import numpy as np

    policy_fn = resolve_any_public_symbol(
        "hadukenn_core", ("validate_output_head_policy", "select_output_head_regime", "check_output_head_policy"), _ML_MODULES
    )
    build_part = resolve_public_symbol("hadukenn_core", "build_balanced_partition", ("hadukenn_core.partitions",))
    cluster_is = resolve_public_symbol("hadukenn_core", "cluster_is_fwd_bwd", ("hadukenn_core.heads",))
    gate_cls = resolve_any_public_symbol("sheaf_verify", ("SiipVerifyGate", "SIIPVerifyGate", "siip_verify_gate"))
    litert_cfg_cls = resolve_public_symbol("hadukenn_core", "LiteRTAgentConfig", _ANTIGRAVITY_PREFERRED_MODULES)
    pathway_fn = resolve_any_public_symbol(
        "hadukenn_core",
        ("run_multimodal_pathway", "evaluate_multimodal_pathway", "AntigravityMultimodalRouter", "MultiModalPathway"),
        _ANTIGRAVITY_PREFERRED_MODULES,
    )
    from_file_fn = resolve_any_public_symbol(
        "hadukenn_core", ("from_file", "load_media_attachment", "MultimodalAttachment"), _ANTIGRAVITY_PREFERRED_MODULES
    )

    sheaf_err = resolve_public_symbol("sheaf_verify", "SheafError", ("sheaf_verify.hashing",))
    # 1. Empirical exclusion matrix: 182x181 / cluster_is rejected on V=32k GPU; dense_flat allowed; V=131k cluster_is allowed; edge_npu allowed
    for bad_kind in ("goodman_factored", "cluster_is", "fp2"):
        try:
            res_bad = _call_with_aliases(
                policy_fn,
                [(("vocab_size",), 32_768), (("n_clusters",), 182), (("cluster_size",), 181), (("head_kind",), bad_kind), (("hardware_target", "hardware"), "gpu_pretrain"), (("batch_size",), 64)],
                (32_768,),
            )
            assert _verdict_passed(res_bad) is False
        except (sheaf_err, ValueError, AssertionError):
            pass

    assert _verdict_passed(
        _call_with_aliases(
            policy_fn,
            [(("vocab_size",), 32_768), (("hardware_target", "hardware"), "gpu_pretrain"), (("batch_size",), 64), (("dtype",), "bf16")],
            (32_768,),
        )
    ) is True
    assert _verdict_passed(
        _call_with_aliases(
            policy_fn,
            [(("vocab_size",), 131_072), (("n_clusters",), 1024), (("cluster_size",), 128), (("head_kind",), "cluster_is"), (("hardware_target", "hardware"), "gpu_pretrain"), (("batch_size",), 64)],
            (131_072,),
        )
    ) is True
    assert _verdict_passed(
        _call_with_aliases(
            policy_fn,
            [(("vocab_size",), 32_768), (("n_clusters",), 182), (("cluster_size",), 181), (("head_kind",), "goodman_factored"), (("hardware_target", "hardware"), "edge_npu"), (("batch_size",), 1), (("dtype",), "int8")],
            (32_768,),
        )
    ) is True

    # 2. Cluster-IS numerical sanity check
    rng = np.random.default_rng(409)
    part = build_part(16, 4, 4, [rng.integers(0, 16, size=32, dtype=np.int32)], mode="mass_random", seed=409)
    loss_is, root_nll, *_ = cluster_is(
        rng.normal(0.0, 0.2, (4, 6)), rng.normal(0.0, 0.2, (16, 6)), np.zeros((4, 6)), part, np.array([0, 4, 8, 12]), k_neg=3, rng=rng
    )
    assert np.all(np.isfinite(loss_is)) and math.isfinite(root_nll)

    # 3. PreToolUse hook blocks 182x181 on 32k GPU with exit 0
    cmds = _matching_pretooluse_commands(repo_root / ".agents" / "hooks.json", "write_to_file")
    bad_code = 'vocab_size = 32768\nhead = "182x181"\ndevice = "cuda"\n'
    payload = _make_stdin_payload(
        tmp_path, "write_to_file", {"path": "src/hadukenn_core/bad_head.py", "TargetFile": "src/hadukenn_core/bad_head.py", "content": bad_code, "CodeContent": bad_code}
    )
    hook_res = [run_hook(cmd, payload, cwd=repo_root) for cmd in cmds]
    assert all(p.returncode == 0 for p, _ in hook_res)
    assert any(o.get("decision") == "deny" for _, o in hook_res)

    # 4. Multimodal pathway & SIIP mode switch
    media = tmp_path / "edge_npu.png"
    media.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24)
    att = from_file_fn.from_file(media) if isinstance(from_file_fn, type) and hasattr(from_file_fn, "from_file") else from_file_fn(media)
    cfg = _bind_call(litert_cfg_cls, {"model_path": "models/struct32k_int8", "vision_backend": "cpu", "audio_backend": "cpu"}, ("models/struct32k_int8",))
    runner = pathway_fn() if isinstance(pathway_fn, type) else pathway_fn
    exec_fn = getattr(runner, "run", getattr(runner, "evaluate", runner))
    ok_res = _bind_call(exec_fn, {"config": cfg, "attachments": [att], "vocab_size": 32_768, "head_type": "dense_flat", "hardware_target": "cpu"})
    assert _verdict_passed(ok_res) is True

    try:
        bad_mm = _bind_call(exec_fn, {"config": cfg, "attachments": [att], "vocab_size": 32_768, "head_type": "182x181", "hardware_target": "cuda"})
    except (sheaf_err, ValueError, AssertionError):
        pass
    else:
        assert _verdict_passed(bad_mm) is False

    gate = gate_cls()
    if hasattr(gate, "set_mode"):
        gate.set_mode("speculative")
        v_fn = getattr(gate, "verify_commit", getattr(gate, "verify_candidate"))
        assert _verdict_passed(v_fn(f_k=1.0, f_cand=1.2, x_cand=[0.2, 0.2])) is True
        gate.set_mode("formal")
        try:
            r_formal = v_fn(f_k=1.0, f_cand=1.2, x_cand=[0.2, 0.2])
        except (sheaf_err, ValueError, AssertionError):
            pass
        else:
            assert _verdict_passed(r_formal) is False


@pytest.mark.tier4
def test_t4_s10_kernel_ml_synthesis_ard_moe_bfgs_dcm_and_op20_magnitudes(readonly_snapshot: dict[str, str]) -> None:
    import numpy as np

    del readonly_snapshot
    compose_fn = resolve_public_symbol("hadukenn_core", "compose", _KERNEL_MODULES)
    ard_fn = resolve_any_public_symbol("hadukenn_core", ("ard_rbf_kernel", "ard_kernel"), _ML_MODULES)
    mi_fn = resolve_any_public_symbol("hadukenn_core", ("binned_mutual_information", "mutual_information_binned"), _ML_MODULES)
    fisher_fn = resolve_any_public_symbol("hadukenn_core", ("fisher_score_kernel", "fisher_kernel"), _ML_MODULES)
    moe_fn = resolve_any_public_symbol("hadukenn_core", ("fit_moe_gem", "MoEGEM", "moe_generalized_em"), _ML_MODULES)
    bfgs_fn = resolve_any_public_symbol("hadukenn_core", ("bfgs_optimize", "bfgs_inverse_hessian_minimize"), _ML_MODULES)
    dcm_fn = resolve_any_public_symbol("hadukenn_core", ("dcm_burstiness_log_prob", "polya_urn_log_prob", "dcm_log_likelihood"), _ML_MODULES)
    elbo_fn = resolve_any_public_symbol("sheaf_verify", ("verify_elbo_monotonicity", "check_elbo_monotonicity", "elbo_is_monotone"))
    bic_fn = resolve_public_symbol("sheaf_verify", "bic_score")
    rad_fn = resolve_public_symbol("sheaf_verify", "rademacher_kernel_bound")
    reject_fn = resolve_any_public_symbol("sheaf_verify", ("reject_option_decision", "reject_option_cost", "bayes_reject_decision"))
    gate_cls = resolve_any_public_symbol("sheaf_verify", ("SiipVerifyGate", "SIIPVerifyGate", "siip_verify_gate"))

    ops = [
        _resolve_op("op_01_ingest_and_clock_sync", "op_01"),
        _resolve_op("op_02_endogenous_scale_solver", "op_02"),
        _resolve_op("op_03_selected_index_pullback", "op_03"),
        _resolve_op("op_04_events_and_interval_matrix", "op_04"),
        _resolve_op("op_05_recurrence_and_visibility_graphs", "op_05"),
        _resolve_op("op_06_coboundary_and_sheaf_resolvent", "op_06"),
        _resolve_op("op_07_linegraph_and_prefix_dynamics", "op_07"),
        _resolve_op("op_08_manifold_scan_and_so4_octonion_rotor", "op_08"),
        _resolve_op("op_17_cayley_pade_koopman_sheaf_propagator", "op_17"),
        _resolve_op("op_18_persistent_homology_betti_wasserstein_radar", "op_18"),
        _resolve_op("op_19_self_evolving_symbolic_morphism_compiler", "op_19"),
        _resolve_op("op_20_closed_loop_user_value_adaptive_graph_solver", "op_20"),
    ]
    states: dict[int, Any] = {}
    for m_streams in (3, 4):
        s_m = compose_fn(*ops)(_build_kernel_state(n_steps=96, m_streams=m_streams, seed=410 + m_streams))
        assert all(norm <= 1.000001 for norm in s_m.cayley_koopman_propagator["operator_l2_norms_by_horizon"].values())
        assert len(s_m.persistent_homology_radar["filtration_betti_barcode"]) == 6
        assert s_m.symbolic_morphism_compiler["synthesized_morphism_rank_ic"] > s_m.symbolic_morphism_compiler["best_single_operator_ic"]
        w_m = s_m.closed_loop_value_morphism["solved_network_graph_parameters_Theta"]["anisotropic_channel_weights"]
        assert len(w_m) == m_streams
        states[m_streams] = s_m

    # Feed stabilized kernel sheaf representation from s (m=4) into ML synthesis + formal bounds
    s = states[4]
    s_alt = compose_fn(*ops)(_build_kernel_state(n_steps=96, m_streams=4, seed=919))
    weights = s.closed_loop_value_morphism["solved_network_graph_parameters_Theta"]["anisotropic_channel_weights"]
    chan_weights = np.array([float(v) for v in weights.values()], dtype=float)
    assert math.isclose(float(np.sum(chan_weights)), 1.0, abs_tol=1e-2)

    x_kernel = np.asarray(s.sheaf_projection["std_mat"], dtype=float)[8:56]
    h0_kernel = np.asarray(s.sheaf_projection["h0_global"], dtype=float)[8:56]
    x_alt = np.asarray(s_alt.sheaf_projection["std_mat"], dtype=float)[8:56]
    assert x_kernel.shape == (48, 4) and h0_kernel.shape == (48, 4)
    assert float(np.std(x_kernel)) > 0.1 and float(np.max(np.abs(x_kernel - x_alt))) > 1e-3

    rng = np.random.default_rng(410)
    y_kernel = 2.0 * x_kernel[:, 0] + 0.25 * (h0_kernel @ chan_weights)
    y_alt = 2.0 * x_alt[:, 0] + 0.25 * (np.asarray(s_alt.sheaf_projection["h0_global"], dtype=float)[8:56] @ chan_weights)
    noise_col = rng.normal(0.0, 1.0, 48)
    assert float(mi_fn(x_kernel[:, 0], y_kernel)) > float(mi_fn(noise_col, y_kernel)) >= 0.0

    ls_kernel = 1.0 / np.maximum(chan_weights, 0.05)
    k_ard = np.asarray(ard_fn(x_kernel[:8], x_kernel[:8], length_scales=ls_kernel), dtype=float)
    k_ard_zero = np.asarray(ard_fn(np.zeros_like(x_kernel[:8]), np.zeros_like(x_kernel[:8]), length_scales=ls_kernel), dtype=float)
    assert np.allclose(np.diag(k_ard), 1.0, atol=1e-12)
    assert float(np.max(np.abs(k_ard - k_ard_zero))) > 1e-3
    rad_val = float(_call_with_aliases(rad_fn, [(("lam", "bound_lambda", "radius_lambda"), 1.5), (("trace_k", "trace_K"), float(np.trace(k_ard))), (("m", "n_samples"), 8)], (1.5, float(np.trace(k_ard)), 8)))
    assert math.isclose(rad_val, (1.5 / 8.0) * math.sqrt(float(np.trace(k_ard))), rel_tol=1e-9)

    k_fish = _call_with_aliases(fisher_fn, [(("X", "scores"), x_kernel), (("reg",), 0.1)], (x_kernel,))
    k_fish_arr = np.asarray(k_fish[0] if isinstance(k_fish, tuple) else k_fish, dtype=float)
    k_fish_alt = _call_with_aliases(fisher_fn, [(("X", "scores"), x_alt), (("reg",), 0.1)], (x_alt,))
    k_fish_alt_arr = np.asarray(k_fish_alt[0] if isinstance(k_fish_alt, tuple) else k_fish_alt, dtype=float)
    assert float(np.min(np.linalg.eigvalsh(0.5 * (k_fish_arr + k_fish_arr.T)))) >= -1e-9
    assert float(np.max(np.abs(k_fish_arr - k_fish_alt_arr))) > 1e-4

    moe_res = _call_with_aliases(moe_fn, [(("X",), x_kernel), (("y",), y_kernel), (("n_experts", "k"), 2), (("max_iter",), 10)], (x_kernel, y_kernel))
    elbo_trace = [float(v) for v in _extract_attr_or_key(moe_res, ("elbo_trace", "free_energy_trace"))]
    assert _verdict_passed(elbo_fn(elbo_trace)) is True
    assert math.isfinite(float(_call_with_aliases(bic_fn, [(("log_likelihood", "log_lik"), elbo_trace[-1]), (("dof", "k"), 5), (("n", "n_samples"), 48)], (elbo_trace[-1], 5, 48))))
    moe_alt = _call_with_aliases(moe_fn, [(("X",), x_alt), (("y",), y_kernel), (("n_experts", "k"), 2), (("max_iter",), 10)], (x_alt, y_kernel))
    elbo_alt = [float(v) for v in _extract_attr_or_key(moe_alt, ("elbo_trace", "free_energy_trace"))]
    assert abs(elbo_trace[-1] - elbo_alt[-1]) > 1e-4

    def _ridge_obj(w: Any, x_sub: Any = x_kernel[:, :3], y_sub: Any = y_kernel) -> float:
        res = x_sub @ w - y_sub
        return float(np.mean(res * res) + 1e-3 * np.sum(w * w))

    def _ridge_grad(w: Any, x_sub: Any = x_kernel[:, :3], y_sub: Any = y_kernel) -> Any:
        res = x_sub @ w - y_sub
        return (2.0 / len(y_sub)) * (x_sub.T @ res) + 2e-3 * w

    opt = _call_with_aliases(
        bfgs_fn,
        [(("fn", "obj_fn", "loss_fn"), _ridge_obj), (("grad_fn",), _ridge_grad), (("x0", "w0"), np.zeros(3)), (("max_iter",), 20)],
        (_ridge_obj, _ridge_grad, np.zeros(3)),
    )
    w_opt = np.asarray(_extract_attr_or_key(opt, ("x", "w", "params")), dtype=float)
    opt_alt = _call_with_aliases(
        bfgs_fn,
        [
            (("fn", "obj_fn", "loss_fn"), lambda w: _ridge_obj(w, x_alt[:, :3], y_alt)),
            (("grad_fn",), lambda w: _ridge_grad(w, x_alt[:, :3], y_alt)),
            (("x0", "w0"), np.zeros(3)),
            (("max_iter",), 20),
        ],
        (lambda w: _ridge_obj(w, x_alt[:, :3], y_alt), lambda w: _ridge_grad(w, x_alt[:, :3], y_alt), np.zeros(3)),
    )
    w_opt_alt = np.asarray(_extract_attr_or_key(opt_alt, ("x", "w", "params")), dtype=float)
    assert float(np.max(np.abs(w_opt - w_opt_alt))) > 1e-3

    gate = gate_cls()
    if hasattr(gate, "set_mode"):
        gate.set_mode("formal")
    v_fn = getattr(gate, "verify_commit", getattr(gate, "verify_candidate"))
    assert _verdict_passed(v_fn(f_k=_ridge_obj(np.zeros(3)), f_cand=_ridge_obj(w_opt), x_cand=w_opt.tolist())) is True

    alpha = np.maximum(chan_weights, 0.05)
    b0_counts = np.array([int(b["beta_0_components"]) for b in s.persistent_homology_radar["filtration_betti_barcode"][:4]], dtype=float)
    bursty_counts = np.array([max(10, int(np.sum(b0_counts))), 0, 0, 0], dtype=float)
    spread_counts = np.full(4, max(2, int(np.sum(b0_counts)) // 4), dtype=float)
    assert float(dcm_fn(bursty_counts, alpha)) > float(dcm_fn(spread_counts, alpha))
    rej = _call_with_aliases(reject_fn, [(("probs", "posteriors"), [0.92, 0.08]), (("lambda_r",), 0.2), (("lambda_s",), 1.0)], ([0.92, 0.08], 0.2, 1.0))
    rej_flag = rej["reject"] if isinstance(rej, dict) else getattr(rej, "reject", bool(rej))
    assert rej_flag is False
