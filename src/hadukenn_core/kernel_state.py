"""Kernel state container, SI rational units, and operator composition helpers."""

from __future__ import annotations

import os
import platform
from collections.abc import Callable
from typing import Any

import numpy as np

from sheaf_verify.formal_verifiers import (
    DIMENSIONLESS,
    GOLDILOCKS_P,
    HASH_RATE,
    MAX_TICK_ABS,
    PLANCK_TIME_S,
    SECOND,
    WATT,
    QFraction,
    SIUnit,
)

__all__ = [
    "DIMENSIONLESS",
    "GOLDILOCKS_P",
    "HASH_RATE",
    "KernelState",
    "MAX_TICK_ABS",
    "PLANCK_TIME_S",
    "QFraction",
    "SECOND",
    "SIUnit",
    "WATT",
    "build_default_streams",
    "compose",
    "detect_execution_environment",
]


def detect_execution_environment() -> dict[str, Any]:
    """Probe host OS, CPU architecture, Python/NumPy versions, and accelerator devices."""
    return {
        "os": platform.system(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cuda_device_detected": os.path.exists("/dev/nvidiactl"),
        "tenstorrent_kmd_detected": os.path.exists("/dev/tenstorrent"),
    }


class KernelState:
    """Mutable container passed through the declarative ``compose(*operators)`` pipeline."""

    def __init__(
        self,
        raw_streams: dict[str, Any],
        clock_relations: dict[str, Any],
        articulated_need: dict[str, Any],
        mode: dict[str, Any] | None = None,
    ) -> None:
        self.raw_streams: dict[str, Any] = raw_streams
        self.clock_relations: dict[str, Any] = clock_relations
        self.need: dict[str, Any] = articulated_need
        self.mode: dict[str, Any] = (
            dict(mode)
            if mode is not None
            else {"check_properties": True, "pareto_scales": [1024, 2048, 4096]}
        )
        self.env_info: dict[str, Any] = detect_execution_environment()
        self.synchronized_streams: dict[str, Any] = {}
        self.endogenous_params: dict[str, Any] = {}
        self.carriers: dict[str, Any] = {}
        self.interval_stats: dict[str, Any] = {}
        self.network_stats: dict[str, Any] = {}
        self.sheaf_projection: dict[str, Any] = {}
        self.dynamics_and_wavelets: dict[str, Any] = {}
        self.manifold_and_rotors: dict[str, Any] = {}
        self.spectral_circuit: dict[str, Any] = {}
        self.crypto_mining_telemetry: dict[str, Any] = {}
        self.tig_pareto_frontier: dict[str, Any] = {}
        self.relational_decompositions: dict[str, Any] = {}
        self.simt_gpu_auditor: dict[str, Any] = {}
        self.tig_8challenge_portfolio: dict[str, Any] = {}
        self.zk_goldilocks_circuit: dict[str, Any] = {}
        self.cayley_koopman_propagator: dict[str, Any] = {}
        self.persistent_homology_radar: dict[str, Any] = {}
        self.symbolic_morphism_compiler: dict[str, Any] = {}
        self.closed_loop_value_morphism: dict[str, Any] = {}
        self.value_graph_policy_alignment: dict[str, Any] = {}
        self.checks: list[dict[str, Any]] = []

    def record_check(self, name: str, kind: str, passed: bool, detail: str) -> None:
        # "empirical" checks (wall-clock timing etc.) are recorded but never fatal.
        if not passed and kind != "empirical" and self.mode.get("check_properties", True):
            raise AssertionError(f"Kernel invariant check failed [{name} ({kind})]: {detail}")
        self.checks.append(
            {
                "invariant": name,
                "tag": kind,
                "passed": bool(passed),
                "detail": detail,
            }
        )


def compose(*operators: Callable[[KernelState], KernelState]) -> Callable[[KernelState], KernelState]:
    """Compose a sequence of kernel operators into a single left-to-right pipeline."""

    def _pipeline(state: KernelState) -> KernelState:
        for op in operators:
            state = op(state)
        return state

    return _pipeline


def build_default_streams(
    n_steps: int = 160,
    seed: int = 20261108,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build deterministic 4-stream multi-clock test inputs using an isolated RNG."""
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2, got {n_steps}")

    rng = np.random.default_rng(seed)
    t_base = np.linspace(1.0, float(n_steps), n_steps)
    s_ai_power = np.sin(0.12 * t_base) + 0.45 * np.cos(0.31 * t_base) + rng.normal(0.0, 0.12, n_steps)
    s_grid_freq = np.cos(0.10 * t_base - 0.2) + 0.55 * np.sin(0.24 * t_base) + rng.normal(0.0, 0.14, n_steps)
    s_mempool_fee = 0.65 * s_grid_freq * np.sin(0.18 * t_base) + rng.normal(0.0, 0.11, n_steps)
    s_vol_skew = 0.80 * s_mempool_fee + 0.50 * (s_ai_power * s_grid_freq) + rng.normal(0.0, 0.11, n_steps)

    raw_streams: dict[str, Any] = {
        "m0_ai_power": {"t": t_base, "values": s_ai_power, "time_unit": SECOND, "val_unit": WATT},
        "m1_grid_freq": {
            "t": (t_base - 0.05) / 1.0001,
            "values": s_grid_freq,
            "time_unit": SECOND,
            "val_unit": HASH_RATE,
        },
        "m2_mempool_fee": {
            "t": (t_base + 0.12) / 0.9999,
            "values": s_mempool_fee,
            "time_unit": SECOND,
            "val_unit": DIMENSIONLESS,
        },
        "m3_vol_skew": {"t": t_base, "values": s_vol_skew, "time_unit": SECOND, "val_unit": DIMENSIONLESS},
    }
    clock_relations: dict[str, Any] = {
        "m1_grid_freq": {"offset_s": 0.05, "rate": 1.0001, "uncertainty_s": 1e-4},
        "m2_mempool_fee": {"offset_s": -0.12, "rate": 0.9999, "uncertainty_s": 1e-4},
    }
    target_need_series = 0.55 * np.hypot(s_grid_freq, s_vol_skew) + 0.45 * np.abs(s_ai_power * s_mempool_fee)
    need_spec: dict[str, Any] = {
        "intent": "local_to_global_regime_and_slice_mining",
        "primary_stream": "m1_grid_freq",
        "objective_series": target_need_series,
    }
    return raw_streams, clock_relations, need_spec
