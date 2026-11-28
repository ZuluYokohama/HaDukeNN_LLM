"""Unified-memory Snapdragon X Plus planner and T-46 local deployment suite."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hadukenn_core import torch_bridge
from hadukenn_core.fast_ops import (
    OpCounter,
    benchmark_kernel_parity,
    detect_oryon_capabilities,
    run_axpy_f32,
    run_dot_f32,
    run_int8_gemm,
    run_nf4_matvec,
)
from hadukenn_core.hst_foundation_lm import STAGED_FOUNDATION_LADDER
from hadukenn_core.npu_qnn_backend import close_all_htp_sessions, run_int8_matmul_htp
from hadukenn_core.rtx5070_frontier import NF4_CODEBOOK
from hadukenn_core.sheaf_bridge import build_sheaf_bridge_csr
from sheaf_verify.anticheat import audit_kernel_source

T46_SCHEMA = "sheaf-verify.t46-snapdragon-local.v1"
T46_VERSION = "0.9.0"
STRUCT32K_VOCAB_SIZE = 32768
HEAD_KIND_FLAT_DENSE_32K = "flat_dense_struct32k"

_BANNED_DRIVE_PREFIXES = tuple(f"{d}:" + "\\" for d in ("C", "D", "E", "H", "c", "d", "e", "h"))
_WIN_DRIVE_PREFIX_RE = re.compile(r"[A-Za-z]:" + re.escape(chr(92)))

_TIER_BLUEPRINTS: tuple[dict[str, Any], ...] = (
    {
        "name": "51M",
        "n_layers": 8,
        "d_model": 576,
        "d_ff": 1536,
        "tied_head": True,
        "backbone_quant": "nf4_int4",
        "runtime_overhead_gib": 0.12,
        "power_fraction": 0.56,
    },
    {
        "name": "125M",
        "n_layers": 12,
        "d_model": 768,
        "d_ff": 2560,
        "tied_head": True,
        "backbone_quant": "nf4_int4",
        "runtime_overhead_gib": 0.24,
        "power_fraction": 0.68,
    },
    {
        "name": "1.5B",
        "n_layers": 28,
        "d_model": 1792,
        "d_ff": 6656,
        "tied_head": False,
        "backbone_quant": "nf4_int4",
        "runtime_overhead_gib": 0.34,
        "power_fraction": 0.84,
    },
    {
        "name": "3.2B",
        "n_layers": 28,
        "d_model": 3072,
        "d_ff": 7680,
        "tied_head": False,
        "backbone_quant": "int8_npu",
        "runtime_overhead_gib": 0.60,
        "power_fraction": 0.96,
    },
    {
        "name": "8B",
        "n_layers": 32,
        "d_model": 4096,
        "d_ff": 14336,
        "tied_head": False,
        "backbone_quant": "nf4_int4",
        "runtime_overhead_gib": 1.24,
        "power_fraction": 1.00,
    },
)

_HST_RUNTIME_OVERHEAD_GIB: dict[str, float] = {
    "HST-Proxy-15M": 0.12,
    "HST-Core-125M": 0.24,
    "HST-Edge-360M": 0.30,
    "HST-Base-1.3B": 0.45,
    "HST-Wide-3.2B": 0.60,
}


def _require_finite(val: Any, name: str) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{name} must be a finite real number, got {type(val).__name__}")
    try:
        out = float(val)
    except OverflowError as exc:
        raise ValueError(f"{name} overflows float64") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {val!r}")
    return 0.0 if out == 0.0 else out


def _require_nonbool_int(val: Any, name: str, *, min_value: int | None = None) -> int:
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValueError(f"{name} must be an integer, got {type(val).__name__}")
    if min_value is not None and val < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {val}")
    return val


def _sanitize_text(text: str) -> str:
    return _WIN_DRIVE_PREFIX_RE.sub("[win-path]/", text)


def _sanitize_obj(value: Any) -> Any:
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return _require_finite(value, "float_field")
    if isinstance(value, list | tuple):
        return [_sanitize_obj(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _sanitize_obj(v) for k, v in value.items()}
    return _sanitize_text(str(value))


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize payload into deterministic UTF-8 canonical JSON without NaN/Inf."""
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return text.encode("utf-8")


def compute_manifest_sha256(doc: dict[str, Any]) -> str:
    """Compute canonical SHA-256 over a T-46 report excluding ``manifest_sha256``."""
    body = {k: v for k, v in doc.items() if k != "manifest_sha256"}
    return hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def _read_proc_meminfo_available_gib() -> float:
    """Read Linux ``/proc/meminfo`` ``MemAvailable`` dynamically in GiB."""
    meminfo = Path("/proc/meminfo")
    try:
        if not meminfo.is_file():
            return 0.0
        text = meminfo.read_text(encoding="utf-8", errors="ignore")
    except (FileNotFoundError, OSError):
        return 0.0

    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return float(int(parts[1])) / (1024.0 * 1024.0)
    return 0.0


@dataclass(frozen=True)
class SnapdragonXPlusSpec:
    """Hardware envelope for Qualcomm Snapdragon X Plus (X1P-42-100) unified LPDDR5x."""

    soc: str = "X1P-42-100"
    soc_name: str = "Qualcomm Snapdragon X Plus (X1P-42-100)"
    cores: int = 8
    max_ghz: float = 3.24
    bandwidth_gb_s: float = 135.0
    tdp_watts: float = 14.5
    npu_tops_int8: float = 45.0
    physical_ram_gib: float = 15.61
    wsl2_mem_available_gib: float | None = None

    def __post_init__(self) -> None:
        if self.wsl2_mem_available_gib is None:
            raw_avail = _read_proc_meminfo_available_gib()
            if (
                not isinstance(self.physical_ram_gib, bool)
                and isinstance(self.physical_ram_gib, (int, float))
                and math.isfinite(float(self.physical_ram_gib))
                and float(self.physical_ram_gib) > 0.0
            ):
                raw_avail = min(raw_avail, float(self.physical_ram_gib))
            object.__setattr__(self, "wsl2_mem_available_gib", round(raw_avail, 4))

    def validate(self) -> None:
        _require_nonbool_int(self.cores, "cores", min_value=1)
        if _require_finite(self.max_ghz, "max_ghz") <= 0.0:
            raise ValueError("max_ghz must be > 0")
        if _require_finite(self.bandwidth_gb_s, "bandwidth_gb_s") <= 0.0:
            raise ValueError("bandwidth_gb_s must be > 0")
        if _require_finite(self.tdp_watts, "tdp_watts") <= 0.0:
            raise ValueError("tdp_watts must be > 0")
        if _require_finite(self.npu_tops_int8, "npu_tops_int8") <= 0.0:
            raise ValueError("npu_tops_int8 must be > 0")
        phys_gib = _require_finite(self.physical_ram_gib, "physical_ram_gib")
        if phys_gib <= 0.0:
            raise ValueError("physical_ram_gib must be > 0")
        avail_gib = _require_finite(self.wsl2_mem_available_gib, "wsl2_mem_available_gib")
        if avail_gib < 0.0 or avail_gib > phys_gib:
            raise ValueError(f"wsl2_mem_available_gib must be in [0, {phys_gib}], got {avail_gib}")


def plan_snapdragon_local_tiers(
    spec: SnapdragonXPlusSpec | None = None,
    *,
    vocab_size: int = STRUCT32K_VOCAB_SIZE,
) -> list[dict[str, Any]]:
    """Model the 5 unified LPDDR5x tiers (51M..8B) preserving struct32k dense flat heads."""
    eff_spec = SnapdragonXPlusSpec() if spec is None else spec
    if not isinstance(eff_spec, SnapdragonXPlusSpec):
        raise ValueError("spec must be a SnapdragonXPlusSpec instance")
    eff_spec.validate()
    v_size = _require_nonbool_int(vocab_size, "vocab_size", min_value=1)
    avail_gib = float(eff_spec.wsl2_mem_available_gib or 0.0)
    phys_gib = float(eff_spec.physical_ram_gib)
    gib = float(1024**3)

    plans: list[dict[str, Any]] = []
    for bp in _TIER_BLUEPRINTS:
        n_layers = int(bp["n_layers"])
        d_model = int(bp["d_model"])
        d_ff = int(bp["d_ff"])
        tied = bool(bp["tied_head"])
        quant = str(bp["backbone_quant"])

        backbone_params = n_layers * (4 * d_model * d_model + 3 * d_model * d_ff + 2 * d_model)
        head_embed_params = v_size * d_model * (1 if tied else 2)
        total_params = backbone_params + head_embed_params
        total_params_m = round(float(total_params) / 1e6, 2)

        bytes_per_backbone_param = 1.0 if quant == "int8_npu" else 0.5625
        backbone_bytes = float(backbone_params) * bytes_per_backbone_param
        flat_head_bytes = float(v_size * d_model) * 2.0  # FP16 flat dense struct32k head
        required_ram_gib = round(
            (backbone_bytes + flat_head_bytes) / gib + float(bp["runtime_overhead_gib"]),
            4,
        )

        active_gb_per_tok = max((backbone_bytes + flat_head_bytes) / 1e9, 1e-6)
        effective_bw_gb_s = float(eff_spec.bandwidth_gb_s) * 0.72
        bandwidth_tok_s = round(effective_bw_gb_s / active_gb_per_tok, 2)
        power_w = round(min(float(eff_spec.tdp_watts), float(eff_spec.tdp_watts) * float(bp["power_fraction"])), 2)

        plans.append(
            {
                "name": str(bp["name"]),
                "tier": str(bp["name"]),
                "n_layers": n_layers,
                "d_model": d_model,
                "d_ff": d_ff,
                "vocab_size": v_size,
                "head_kind": HEAD_KIND_FLAT_DENSE_32K,
                "backbone_quant": quant,
                "total_params_m": total_params_m,
                "required_ram_gib": required_ram_gib,
                "fits_physical_ram": bool(required_ram_gib <= phys_gib),
                "fits_wsl2_mem_available": bool(required_ram_gib <= avail_gib),
                "bandwidth_tok_s": bandwidth_tok_s,
                "power_w": power_w,
                "live_executed": False,
            }
        )
    return plans


def plan_hst_foundation_snapdragon(
    wsl2_cap_gib: float = 7.5,
    host_slab_gib: float = 13.0,
    spec: SnapdragonXPlusSpec | None = None,
) -> dict[str, Any]:
    """Map STAGED_FOUNDATION_LADDER (15M..3.2B) to unified LPDDR5x WSL2 and host slab budgets."""
    wsl2_cap = _require_finite(wsl2_cap_gib, "wsl2_cap_gib")
    if wsl2_cap <= 0.0:
        raise ValueError("wsl2_cap_gib must be > 0")
    host_slab = _require_finite(host_slab_gib, "host_slab_gib")
    if host_slab <= 0.0:
        raise ValueError("host_slab_gib must be > 0")
    eff_spec = SnapdragonXPlusSpec() if spec is None else spec
    if not isinstance(eff_spec, SnapdragonXPlusSpec):
        raise ValueError("spec must be a SnapdragonXPlusSpec instance")
    eff_spec.validate()

    avail_gib = float(eff_spec.wsl2_mem_available_gib or 0.0)
    gib = float(1024**3)
    plans: list[dict[str, Any]] = []

    for tier in STAGED_FOUNDATION_LADDER:
        d = int(tier.d_model)
        n_layers = int(tier.n_layers)
        d_ff = int(tier.d_ff)
        v_size = int(tier.vocab_size)
        backbone_params = n_layers * (4 * d * d + 3 * d * d_ff + 2 * d) + d
        embed_params = v_size * d
        total_params = backbone_params + embed_params
        total_params_m = round(float(total_params) / 1e6, 2)

        bf16_train_gib = round(float(tier.est_bf16_train_vram_gib), 4)
        overhead = _HST_RUNTIME_OVERHEAD_GIB.get(tier.tier_id, 0.30)
        int8_npu_gib = round((float(backbone_params) * 1.0 + float(embed_params) * 2.0) / gib + overhead, 4)
        nf4_cpu_gib = round((float(backbone_params) * 0.5625 + float(embed_params) * 2.0) / gib + overhead, 4)

        fits_wsl2_cap = bool(bf16_train_gib <= wsl2_cap)
        fits_wsl2_avail = bool(bf16_train_gib <= avail_gib)
        fits_host_slab = bool(bf16_train_gib <= host_slab)
        int8_fits_wsl2 = bool(int8_npu_gib <= wsl2_cap)
        int8_fits_avail = bool(int8_npu_gib <= avail_gib)
        int8_fits_host = bool(int8_npu_gib <= host_slab)
        nf4_fits_host = bool(nf4_cpu_gib <= host_slab)

        if fits_wsl2_cap:
            rec_place = "wsl2_oryon_bf16_train_or_int8_npu"
            rec_quant = "bf16"
        elif fits_host_slab:
            rec_place = "host_slab_bf16_or_wsl2_int8_npu"
            rec_quant = "int8_npu_wsl2_or_bf16_host"
        elif int8_fits_host:
            rec_place = "host_slab_int8_npu_looped_edge"
            rec_quant = "int8_npu"
        elif nf4_fits_host:
            rec_place = "host_slab_nf4_cpu_offload"
            rec_quant = "nf4_cpu"
        else:
            rec_place = "multi_gpu_cloud_offload"
            rec_quant = "bf16_sharded"

        plans.append(
            {
                "tier_id": str(tier.tier_id),
                "d_model": d,
                "n_layers": n_layers,
                "n_heads": int(tier.n_heads),
                "d_ff": d_ff,
                "vocab_size": v_size,
                "head_regime": str(tier.head_regime),
                "target_hardware": str(tier.target_hardware),
                "backbone_params": backbone_params,
                "embed_params": embed_params,
                "total_params": total_params,
                "total_params_m": total_params_m,
                "bf16_train_gib": bf16_train_gib,
                "int8_npu_gib": int8_npu_gib,
                "nf4_cpu_gib": nf4_cpu_gib,
                "fits_wsl2_cap": fits_wsl2_cap,
                "fits_wsl2_mem_available": fits_wsl2_avail,
                "fits_host_slab": fits_host_slab,
                "int8_npu_fits_wsl2_cap": int8_fits_wsl2,
                "int8_npu_fits_wsl2_mem_available": int8_fits_avail,
                "int8_npu_fits_host_slab": int8_fits_host,
                "nf4_cpu_fits_host_slab": nf4_fits_host,
                "recommended_placement": rec_place,
                "recommended_quantization": rec_quant,
            }
        )

    summary = {
        "total_tiers": len(plans),
        "bf16_wsl2_cap_fitting_tiers": [p["tier_id"] for p in plans if p["fits_wsl2_cap"]],
        "bf16_wsl2_mem_available_fitting_tiers": [p["tier_id"] for p in plans if p["fits_wsl2_mem_available"]],
        "bf16_host_slab_fitting_tiers": [p["tier_id"] for p in plans if p["fits_host_slab"]],
        "int8_npu_wsl2_cap_fitting_tiers": [p["tier_id"] for p in plans if p["int8_npu_fits_wsl2_cap"]],
        "int8_npu_host_slab_fitting_tiers": [p["tier_id"] for p in plans if p["int8_npu_fits_host_slab"]],
        "nf4_cpu_host_slab_fitting_tiers": [p["tier_id"] for p in plans if p["nf4_cpu_fits_host_slab"]],
        "tiled_cce_tiers": [p["tier_id"] for p in plans if p["head_regime"] == "tiled_cce"],
        "cluster_is_tiers": [p["tier_id"] for p in plans if p["head_regime"] != "tiled_cce"],
    }
    return _sanitize_obj(
        {
            "schema": "sheaf-verify.hst-snapdragon-plan.v1",
            "hardware": asdict(eff_spec),
            "wsl2_cap_gib": wsl2_cap,
            "host_slab_gib": host_slab,
            "wsl2_mem_available_gib": avail_gib,
            "tiers": plans,
            "summary": summary,
        }
    )


def _pack_nf4_weights(weight_fp: np.ndarray, group_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    m_rows, k_cols = weight_fp.shape
    thresholds = 0.5 * (NF4_CODEBOOK[:-1] + NF4_CODEBOOK[1:])
    n_groups = k_cols // group_size
    blocks = weight_fp.astype(np.float32).reshape(m_rows, n_groups, group_size)
    scales = np.maximum(np.max(np.abs(blocks), axis=-1), 1e-12).astype(np.float32)
    normed = blocks / scales[:, :, None]
    codes = np.searchsorted(thresholds, normed).astype(np.uint8).reshape(m_rows, k_cols)
    packed = ((codes[:, 0::2] << 4) | codes[:, 1::2]).astype(np.uint8)
    return packed, scales


def _dequant_nf4_matvec_ref(
    x_f32: np.ndarray,
    qweights_u8: np.ndarray,
    scales_f32: np.ndarray,
    group_size: int = 64,
) -> np.ndarray:
    m_rows = qweights_u8.shape[0]
    k_cols = qweights_u8.shape[1] * 2
    hi = (qweights_u8 >> 4) & 0x0F
    lo = qweights_u8 & 0x0F
    codes = np.empty((m_rows, k_cols), dtype=np.uint8)
    codes[:, 0::2] = hi
    codes[:, 1::2] = lo
    vals = NF4_CODEBOOK[codes].reshape(m_rows, k_cols // group_size, group_size)
    w_fp = (vals * scales_f32[:, :, None]).reshape(m_rows, k_cols)
    return (w_fp @ x_f32).astype(np.float32)


def _run_torch_bridge_check() -> tuple[bool, list[str], bool]:
    tb_path = Path(__file__).resolve().with_name("torch_bridge.py")
    rep = audit_kernel_source(tb_path.read_text(encoding="utf-8"))
    violations = [_sanitize_text(str(v)) for v in rep.violations]
    audit_ok = bool(rep.ok) and len(violations) == 0

    live_checked = False
    if torch_bridge.TORCH_AVAILABLE:
        import torch

        torch.manual_seed(46)
        mod = torch_bridge.NF4FrozenLinearTorch(torch.randn(16, 64), block_size=64, lora_rank=4)
        out = mod(torch.randn(2, 64))
        if not bool(torch.all(torch.isfinite(out)).item()):
            audit_ok = False
            violations.append("torch_bridge NF4FrozenLinearTorch produced non-finite output")
        live_checked = True
    return audit_ok, violations, live_checked


def _run_dot_axpy_check(
    rng: np.random.Generator,
    dim: int,
    counter: OpCounter,
) -> tuple[float, float]:
    dot_a = rng.normal(0.0, 1.0, size=(dim,)).astype(np.float32)
    dot_b = rng.normal(0.0, 1.0, size=(dim,)).astype(np.float32)
    dot_val = float(run_dot_f32(dot_a, dot_b, counter=counter))
    dot_ref = float(np.dot(dot_a.astype(np.float64), dot_b.astype(np.float64)))
    dot_err = abs(dot_val - dot_ref)
    if not (
        math.isfinite(dot_val)
        and math.isfinite(dot_err)
        and math.isclose(dot_val, dot_ref, rel_tol=1e-4, abs_tol=1e-3)
    ):
        raise AssertionError(
            f"dot_f32 parity check failed (dim={dim}): val={dot_val!r}, ref={dot_ref!r}, err={dot_err!r}"
        )

    axpy_out = run_axpy_f32(1.5, dot_a, dot_b, counter=counter)
    axpy_ref = (np.float32(1.5) * dot_a + dot_b).astype(np.float32)
    axpy_err = float(np.max(np.abs(axpy_out - axpy_ref)))
    if (
        axpy_out.shape != axpy_ref.shape
        or not bool(np.all(np.isfinite(axpy_out)))
        or not math.isfinite(axpy_err)
        or not bool(np.allclose(axpy_out, axpy_ref, rtol=1e-5, atol=1e-5))
    ):
        raise AssertionError(f"axpy_f32 parity check failed (dim={dim}): max_abs_err={axpy_err!r}")
    return dot_err, axpy_err


def run_t46_snapdragon_local_suite(
    *,
    dcvs_mode: str = "burst",
    close_npu_sessions: bool = True,
    include_tiled_cce_51m: bool = False,
) -> dict[str, Any]:
    """Execute the T-46 Snapdragon X Plus local E2E verification suite on fitting tiers."""
    if dcvs_mode not in {"burst", "power_saver"}:
        raise ValueError(f"Unsupported dcvs_mode {dcvs_mode!r}; expected 'burst' or 'power_saver'")

    spec = SnapdragonXPlusSpec()
    tiers = plan_snapdragon_local_tiers(spec, vocab_size=STRUCT32K_VOCAB_SIZE)
    caps = _sanitize_obj(detect_oryon_capabilities())
    counter = OpCounter()

    rng = np.random.default_rng(46)
    tier_51m_fits = any(t["name"] == "51M" and t["fits_wsl2_mem_available"] for t in tiers)
    tier_125m_fits = any(t["name"] == "125M" and t["fits_wsl2_mem_available"] for t in tiers)

    # 1. Oryon C11 NEON FP32 + NF4 MatVec + INT8 GEMM parity-before-timing (51M d_model=576)
    dot_err, axpy_err = _run_dot_axpy_check(rng, 576, counter)

    w_nf4_fp = rng.normal(0.0, 0.35, size=(256, 576)).astype(np.float32)
    x_nf4 = rng.normal(0.0, 0.5, size=(576,)).astype(np.float32)
    qw_u8, scales_f32 = _pack_nf4_weights(w_nf4_fp, group_size=64)
    nf4_bench = benchmark_kernel_parity(
        "nf4_matvec_51m",
        lambda: run_nf4_matvec(x_nf4, qw_u8, scales_f32, group_size=64, counter=counter),
        lambda: _dequant_nf4_matvec_ref(x_nf4, qw_u8, scales_f32, group_size=64),
        warmup=1,
        iters=3,
        rtol=1e-3,
        atol=1e-4,
        assert_parity=True,
        counter=counter,
    )

    a_gemm = rng.integers(-64, 64, size=(128, 576), dtype=np.int8)
    b_gemm = rng.integers(-64, 64, size=(576, 128), dtype=np.int8)
    gemm_bench = benchmark_kernel_parity(
        "int8_gemm_oryon",
        lambda: run_int8_gemm(a_gemm, b_gemm, mode="auto", counter=counter),
        lambda: run_int8_gemm(a_gemm, b_gemm, mode="numpy"),
        warmup=1,
        iters=3,
        rtol=0.0,
        atol=0.0,
        assert_parity=True,
        counter=counter,
    )

    # 2. Tiered INT8 NPU (HTP QNN -> Oryon i8mm -> NumPy reference)
    a_npu = rng.integers(-32, 32, size=(256, 512), dtype=np.int8)
    b_npu = rng.integers(-32, 32, size=(512, 512), dtype=np.int8)
    live_125m_executed = False
    live_125m_info: dict[str, Any] = {"executed": False, "d_model": 768}
    err_125m_int8 = 0.0
    parity_125m_ok = True
    try:
        res_auto = run_int8_matmul_htp(a_npu, b_npu, force_tier="auto", counter=counter, dcvs_mode=dcvs_mode)
        res_i8mm = run_int8_matmul_htp(a_npu, b_npu, force_tier="oryon_i8mm", counter=counter)
        res_np = run_int8_matmul_htp(a_npu, b_npu, force_tier="numpy_ref", counter=counter)
        if tier_125m_fits:
            dot_err_125, axpy_err_125 = _run_dot_axpy_check(rng, 768, counter)
            w_nf4_125 = rng.normal(0.0, 0.35, size=(256, 768)).astype(np.float32)
            x_nf4_125 = rng.normal(0.0, 0.5, size=(768,)).astype(np.float32)
            qw_125, sc_125 = _pack_nf4_weights(w_nf4_125, group_size=64)
            nf4_125 = benchmark_kernel_parity(
                "nf4_matvec_125m",
                lambda: run_nf4_matvec(x_nf4_125, qw_125, sc_125, group_size=64, counter=counter),
                lambda: _dequant_nf4_matvec_ref(x_nf4_125, qw_125, sc_125, group_size=64),
                warmup=1,
                iters=2,
                rtol=1e-3,
                atol=1e-4,
                assert_parity=True,
                counter=counter,
            )
            a_gemm_125 = rng.integers(-64, 64, size=(128, 768), dtype=np.int8)
            b_gemm_125 = rng.integers(-64, 64, size=(768, 128), dtype=np.int8)
            gemm_125 = benchmark_kernel_parity(
                "int8_gemm_125m",
                lambda: run_int8_gemm(a_gemm_125, b_gemm_125, mode="auto", counter=counter),
                lambda: run_int8_gemm(a_gemm_125, b_gemm_125, mode="numpy"),
                warmup=1,
                iters=2,
                rtol=0.0,
                atol=0.0,
                assert_parity=True,
                counter=counter,
            )
            a_npu_125 = rng.integers(-32, 32, size=(128, 768), dtype=np.int8)
            b_npu_125 = rng.integers(-32, 32, size=(768, 128), dtype=np.int8)
            res_125_npu = run_int8_matmul_htp(
                a_npu_125, b_npu_125, force_tier="auto", counter=counter, dcvs_mode=dcvs_mode
            )
            ref_125_npu = a_npu_125.astype(np.int32) @ b_npu_125.astype(np.int32)
            npu_125_err = float(np.max(np.abs(res_125_npu["output"] - ref_125_npu)))
            npu_125_parity = bool(np.array_equal(res_125_npu["output"], ref_125_npu)) and bool(
                res_125_npu["canary_ok"]
            )
            err_125m_int8 = max(float(gemm_125["max_abs_err"]), npu_125_err)
            parity_125m_ok = bool(gemm_125["parity_ok"] and npu_125_parity)
            live_125m_executed = True
            live_125m_info = {
                "executed": True,
                "d_model": 768,
                "npu_tier_used": str(res_125_npu["tier_used"]),
                "canary_ok": bool(res_125_npu["canary_ok"]),
                "parity_ok": parity_125m_ok,
                "nf4_speedup_vs_numpy": float(nf4_125["speedup_x"]),
                "gemm_speedup_vs_numpy": float(gemm_125["speedup_x"]),
                "dot_abs_err": round(float(dot_err_125), 8),
                "axpy_max_abs_err": round(float(axpy_err_125), 8),
            }
    finally:
        if close_npu_sessions:
            close_all_htp_sessions()

    np_ref = res_np["output"]
    auto_parity_ok = bool(np.array_equal(res_auto["output"], np_ref))
    i8mm_parity_ok = bool(np.array_equal(res_i8mm["output"], np_ref))
    int8_max_abs_err = float(
        max(
            float(np.max(np.abs(res_auto["output"] - np_ref))),
            float(np.max(np.abs(res_i8mm["output"] - np_ref))),
            float(gemm_bench["max_abs_err"]),
            err_125m_int8,
        )
    )
    int8_parity_ok = bool(
        gemm_bench["parity_ok"]
        and auto_parity_ok
        and i8mm_parity_ok
        and parity_125m_ok
        and int8_max_abs_err == 0.0
    )

    # Optional 51M flat dense V=32,768 tiled CCE check for local_e2e_runner
    tiled_cce_51m_ok = True
    if include_tiled_cce_51m and tier_51m_fits:
        from hadukenn_core.fast_ops import run_tiled_cce_avx512

        h_cce = rng.normal(0.0, 0.2, size=(4, 576)).astype(np.float32)
        e_cce = rng.normal(0.0, 0.05, size=(STRUCT32K_VOCAB_SIZE, 576)).astype(np.float32)
        targets_cce = rng.integers(0, STRUCT32K_VOCAB_SIZE, size=(4,), dtype=np.int32)
        loss_cce, _, _ = run_tiled_cce_avx512(h_cce, e_cce, targets_cce, tile_V=512, compute_grad=False)
        tiled_cce_51m_ok = bool(loss_cce.shape == (4,) and np.all(np.isfinite(loss_cce)))
        if not tiled_cce_51m_ok:
            raise AssertionError("tiled_cce_51m produced non-finite loss or wrong shape")

    for tier in tiers:
        if tier["name"] == "51M" and tier_51m_fits:
            tier["live_executed"] = True
        elif tier["name"] == "125M" and tier_125m_fits and live_125m_executed:
            tier["live_executed"] = True
        else:
            tier["live_executed"] = False

    # 3. Chunked sparse CSR Sheaf-Bridge at V=32,768 (run once per suite invocation)
    oracle = rng.normal(0.0, 1.0, size=(STRUCT32K_VOCAB_SIZE, 64)).astype(np.float32)
    csr_res = build_sheaf_bridge_csr(
        oracle,
        k=8,
        reg=1e-3,
        mst_weight=0.15,
        chunk_size=1024,
        dtype="float16",
    )
    sheaf_bridge_csr = {
        "vocab_size": int(csr_res["vocab_size"]),
        "lle_k": int(csr_res["lle_k"]),
        "dtype": str(csr_res["dtype"]),
        "lle_nnz": int(csr_res["lle_nnz"]),
        "mst_edges": int(csr_res["mst_edges"]),
        "nnz": int(csr_res["nnz"]),
        "csr_bytes": int(csr_res["csr_bytes"]),
        "peak_ram_bytes": int(csr_res["peak_ram_bytes"]),
        "within_budget": bool(csr_res["within_budget"]),
        "dense_flat_head_preserved": bool(csr_res["dense_flat_head_preserved"]),
    }

    # 4. Torch-bridge static AST + optional live CPU audit
    torch_audit_ok, torch_violations, torch_live_checked = _run_torch_bridge_check()

    oryon_speedup = float(gemm_bench["speedup_x"])

    execution_tiers: dict[str, Any] = {
        "auto_selected": {
            "tier_used": str(res_auto["tier_used"]),
            "canary_ok": bool(res_auto["canary_ok"]),
            "parity_ok": auto_parity_ok,
            "speedup_vs_numpy": float(res_auto["speedup_vs_numpy"]),
            "providers": _sanitize_obj(res_auto.get("providers", [])),
        },
        "oryon_i8mm": {
            "tier_used": str(res_i8mm["tier_used"]),
            "canary_ok": bool(res_i8mm["canary_ok"]),
            "parity_ok": i8mm_parity_ok,
            "speedup_vs_numpy": float(res_i8mm["speedup_vs_numpy"]),
            "gemm_speedup_vs_numpy": oryon_speedup,
            "nf4_speedup_vs_numpy": float(nf4_bench["speedup_x"]),
            "dot_abs_err": round(float(dot_err), 8),
            "axpy_max_abs_err": round(float(axpy_err), 8),
        },
        "numpy_ref": {
            "tier_used": str(res_np["tier_used"]),
            "canary_ok": bool(res_np["canary_ok"]),
            "parity_ok": True,
            "numpy_ref_ms": float(res_np["numpy_ref_ms"]),
        },
        "tiled_cce_51m_ok": tiled_cce_51m_ok,
        "tier_125m_check": live_125m_info,
    }

    hw_dict = asdict(spec)
    hw_dict["oryon_capabilities"] = caps

    report: dict[str, Any] = {
        "schema": T46_SCHEMA,
        "version": T46_VERSION,
        "dcvs_mode": dcvs_mode,
        "hardware": hw_dict,
        "tiers": tiers,
        "sheaf_bridge_csr": sheaf_bridge_csr,
        "torch_bridge_audit_ok": torch_audit_ok,
        "torch_bridge_violations": torch_violations,
        "torch_bridge_live_checked": torch_live_checked,
        "int8_parity_ok": int8_parity_ok,
        "int8_max_abs_err": int8_max_abs_err,
        "oryon_simd_speedup_vs_numpy": oryon_speedup,
        "nf4_matvec_speedup_vs_numpy": float(nf4_bench["speedup_x"]),
        "op_counter": _sanitize_obj(counter.as_dict()),
        "execution_tiers": execution_tiers,
        "npu_tier_used": str(res_auto["tier_used"]),
        "npu_canary_ok": bool(res_auto["canary_ok"]),
        "no_wslconfig_modified": True,
        "no_network_downloads": True,
    }
    report = _sanitize_obj(report)
    report["manifest_sha256"] = compute_manifest_sha256(report)
    return report
