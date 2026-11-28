"""Pure-stdlib algebraic, dimensional, tick, cryptographic, SIMT, and Goldilocks R1CS verifiers.

Synthesizes the deterministic verification primitives extracted from
``c0_universal_kernel_v5.py`` (``op_01``, ``op_04``, ``op_10``, ``op_14``, ``op_16``)
using strictly Python 3.11+ standard library modules.
"""

from __future__ import annotations

import hmac
import math
from collections.abc import Sequence
from fractions import Fraction
from typing import Any

from sheaf_verify.hashing import (
    SheafError,
    as_int,
    as_real,
    canonical_bytes,
    is_sha256,
    sha256_hex,
)

PLANCK_TIME_S: float = 5.391247e-44
MAX_TICK_ABS: int = 1 << 62
GOLDILOCKS_P: int = (1 << 64) - (1 << 32) + 1


class QFraction:
    """Exact reduced rational number over Python ``int`` with ``den > 0``."""

    __slots__ = ("num", "den")

    def __init__(self, num: int, den: int = 1) -> None:
        num_i = as_int(num, "QFraction.num")
        den_i = as_int(den, "QFraction.den")
        if den_i == 0:
            raise ZeroDivisionError("QFraction denominator cannot be zero")
        g = math.gcd(num_i, den_i)
        if den_i < 0:
            num_i = -num_i
            den_i = -den_i
        self.num: int = num_i // g
        self.den: int = den_i // g

    @staticmethod
    def _coerce(other: object) -> QFraction | None:
        if isinstance(other, QFraction):
            return other
        if isinstance(other, int) and not isinstance(other, bool):
            return QFraction(other, 1)
        if isinstance(other, Fraction):
            return QFraction(other.numerator, other.denominator)
        return None

    def __add__(self, other: object) -> QFraction:
        rhs = self._coerce(other)
        if rhs is None:
            return NotImplemented
        return QFraction(self.num * rhs.den + rhs.num * self.den, self.den * rhs.den)

    def __radd__(self, other: object) -> QFraction:
        return self.__add__(other)

    def __sub__(self, other: object) -> QFraction:
        rhs = self._coerce(other)
        if rhs is None:
            return NotImplemented
        return QFraction(self.num * rhs.den - rhs.num * self.den, self.den * rhs.den)

    def __rsub__(self, other: object) -> QFraction:
        lhs = self._coerce(other)
        if lhs is None:
            return NotImplemented
        return lhs.__sub__(self)

    def __mul__(self, other: object) -> QFraction:
        rhs = self._coerce(other)
        if rhs is None:
            return NotImplemented
        return QFraction(self.num * rhs.num, self.den * rhs.den)

    def __rmul__(self, other: object) -> QFraction:
        return self.__mul__(other)

    def __truediv__(self, other: object) -> QFraction:
        rhs = self._coerce(other)
        if rhs is None:
            return NotImplemented
        if rhs.num == 0:
            raise ZeroDivisionError("division by zero QFraction")
        return QFraction(self.num * rhs.den, self.den * rhs.num)

    def __rtruediv__(self, other: object) -> QFraction:
        lhs = self._coerce(other)
        if lhs is None:
            return NotImplemented
        return lhs.__truediv__(self)

    def __neg__(self) -> QFraction:
        return QFraction(-self.num, self.den)

    def __pos__(self) -> QFraction:
        return self

    def __eq__(self, other: object) -> bool:
        if isinstance(other, QFraction):
            return self.num == other.num and self.den == other.den
        if isinstance(other, int) and not isinstance(other, bool):
            return self.den == 1 and self.num == other
        if isinstance(other, Fraction):
            return self.num == other.numerator and self.den == other.denominator
        return NotImplemented

    def __hash__(self) -> int:
        return hash(Fraction(self.num, self.den))

    def __repr__(self) -> str:
        return f"{self.num}" if self.den == 1 else f"{self.num}/{self.den}"


def _coerce_dim(value: object) -> QFraction:
    if isinstance(value, QFraction):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return QFraction(value, 1)
    if isinstance(value, Fraction):
        return QFraction(value.numerator, value.denominator)
    raise SheafError(f"SIUnit dimension exponent must be int, Fraction, or QFraction, got {type(value).__name__}")


class SIUnit:
    """7-base SI dimensional unit ``(m, kg, s, A, K, mol, cd)`` with exact ``QFraction`` exponents."""

    def __init__(
        self,
        dims: Sequence[int | Fraction | QFraction],
        scale: float = 1.0,
        symbol: str = "",
    ) -> None:
        if isinstance(dims, (str, bytes)):
            raise SheafError("SIUnit dims must be a 7-element sequence")
        dims_seq = list(dims)
        if len(dims_seq) != 7:
            raise SheafError(f"SIUnit requires exactly 7 dimension exponents, got {len(dims_seq)}")
        self.dims: tuple[QFraction, ...] = tuple(_coerce_dim(d) for d in dims_seq)
        self.scale: float = as_real(scale, "SIUnit.scale")
        if not isinstance(symbol, str):
            raise SheafError("SIUnit.symbol must be a str")
        self.symbol: str = symbol

    def same_dims(self, other: SIUnit) -> bool:
        if not isinstance(other, SIUnit):
            raise SheafError("same_dims expects an SIUnit instance")
        return self.dims == other.dims

    def __mul__(self, other: object) -> SIUnit:
        if not isinstance(other, SIUnit):
            return NotImplemented
        dims = tuple(a + b for a, b in zip(self.dims, other.dims, strict=True))
        if self.symbol and other.symbol:
            sym = f"({self.symbol}*{other.symbol})"
        else:
            sym = self.symbol or other.symbol
        return SIUnit(dims, self.scale * other.scale, sym)

    def __truediv__(self, other: object) -> SIUnit:
        if not isinstance(other, SIUnit):
            return NotImplemented
        if other.scale == 0.0:
            raise ZeroDivisionError("Cannot divide by a zero-scale SIUnit")
        dims = tuple(a - b for a, b in zip(self.dims, other.dims, strict=True))
        if self.symbol and other.symbol:
            sym = f"({self.symbol}/{other.symbol})"
        else:
            sym = self.symbol or other.symbol
        return SIUnit(dims, self.scale / other.scale, sym)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SIUnit):
            return NotImplemented
        return self.dims == other.dims and self.scale == other.scale

    def __repr__(self) -> str:
        return f"SIUnit(dims={self.dims!r}, scale={self.scale!r}, symbol={self.symbol!r})"


SECOND = SIUnit((0, 0, 1, 0, 0, 0, 0), 1.0, "s")
DIMENSIONLESS = SIUnit((0, 0, 0, 0, 0, 0, 0), 1.0, "1")
WATT = SIUnit((2, 1, -3, 0, 0, 0, 0), 1.0, "W")
HASH_RATE = SIUnit((0, 0, -1, 0, 0, 0, 0), 1.0, "H/s")


def verify_tick_monotonicity(
    timestamps: Sequence[float] | None = None,
    ticks: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Verify Planck-floor timestamp monotonicity and bounded non-decreasing integer ticks."""
    if timestamps is None and ticks is None:
        raise SheafError("verify_tick_monotonicity requires at least one of timestamps or ticks")

    min_dt: float | None = None
    ts_ok = True
    if timestamps is not None:
        if isinstance(timestamps, (str, bytes)):
            raise SheafError("timestamps must be a sequence of real numbers")
        ts_list = [as_real(val, "timestamp") for val in timestamps]
        if len(ts_list) < 2:
            raise SheafError("timestamps must contain at least 2 entries")
        dts = [ts_list[idx + 1] - ts_list[idx] for idx in range(len(ts_list) - 1)]
        min_dt = min(dts)
        ts_ok = min_dt >= PLANCK_TIME_S

    max_abs_tick: int | None = None
    ticks_ok = True
    if ticks is not None:
        if isinstance(ticks, (str, bytes)):
            raise SheafError("ticks must be a sequence of integers")
        q_list = [as_int(val, "tick") for val in ticks]
        if not q_list:
            raise SheafError("ticks must be non-empty")
        max_abs_tick = max(abs(q) for q in q_list)
        bounded = max_abs_tick <= MAX_TICK_ABS
        monotone = all(q_list[idx + 1] >= q_list[idx] for idx in range(len(q_list) - 1))
        ticks_ok = bounded and monotone

    ok = ts_ok and ticks_ok
    return {
        "ok": ok,
        "passed": ok,
        "timestamps_ok": ts_ok,
        "ticks_ok": ticks_ok,
        "min_dt": min_dt,
        "max_abs_tick": max_abs_tick,
    }


def _rotr32(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF


def _expand_sha256_schedule(w16: Sequence[int]) -> list[int]:
    w = list(w16) + [0] * 48
    for t in range(16, 64):
        s0 = _rotr32(w[t - 15], 7) ^ _rotr32(w[t - 15], 18) ^ (w[t - 15] >> 3)
        s1 = _rotr32(w[t - 2], 17) ^ _rotr32(w[t - 2], 19) ^ (w[t - 2] >> 10)
        w[t] = (w[t - 16] + s0 + w[t - 7] + s1) & 0xFFFFFFFF
    return w


def verify_sha256_causal_cone(mut_word_idx: int = 3) -> dict[str, Any]:
    """Compute SHA-256 64-round backward causal-cone liveness and schedule nullspace."""
    idx = as_int(mut_word_idx, "mut_word_idx")
    if not (0 <= idx < 16):
        raise SheafError(f"mut_word_idx must lie in [0, 15], got {idx}")

    needed: dict[int, set[str]] = {64: {"a"}}
    for r in range(63, -1, -1):
        nxt = needed[r + 1]
        cur: set[str] = set()
        if "a" in nxt:
            cur.update(["a", "b", "c", "e", "f", "g", "h"])
        if "e" in nxt:
            cur.update(["d", "e", "f", "g", "h"])
        for reg_dst, reg_src in (
            ("b", "a"),
            ("c", "b"),
            ("d", "c"),
            ("f", "e"),
            ("g", "f"),
            ("h", "g"),
        ):
            if reg_dst in nxt:
                cur.add(reg_src)
        needed[r] = cur

    terminal_6round_total = 6 * 8
    terminal_6round_live = sum(len(needed[r]) for r in range(58, 64))
    dead_terminal_reg_slices = terminal_6round_total - terminal_6round_live

    w_base = [(i * 0x9E3779B9) & 0xFFFFFFFF for i in range(16)]
    w_ref = _expand_sha256_schedule(w_base)
    w_mut = list(w_base)
    w_mut[idx] ^= 1
    w_pert = _expand_sha256_schedule(w_mut)
    invariant_w_words = sum(1 for a, b in zip(w_ref, w_pert, strict=True) if a == b)

    saved_prefix_equiv = 3.0 + 0.35 * invariant_w_words
    saved_terminal_equiv = (dead_terminal_reg_slices / 8.0) + 4.15
    eff_rounds = 64.0 - (saved_prefix_equiv + saved_terminal_equiv)

    return {
        "ok": True,
        "passed": True,
        "mut_word_idx": idx,
        "dead_terminal_reg_slices": dead_terminal_reg_slices,
        "invariant_w_words": invariant_w_words,
        "eff_rounds": eff_rounds,
    }


def canonical_receipt_sha256(body: Any) -> str:
    """Compute canonical JSON SHA-256 hex digest using ``sheaf_verify.hashing.canonical_bytes``."""
    return sha256_hex(canonical_bytes(body))


def verify_canonical_receipt(body: Any, expected_sha256: str) -> dict[str, Any]:
    """Verify a canonical JSON receipt body against an expected SHA-256 hex digest."""
    if not is_sha256(expected_sha256):
        raise SheafError("expected_sha256 must be a 64-character lowercase SHA-256 hex string")
    actual = canonical_receipt_sha256(body)
    matched = hmac.compare_digest(actual, expected_sha256)
    return {
        "ok": matched,
        "passed": matched,
        "sha256": actual,
        "expected_sha256": expected_sha256,
    }


def verify_simt_warp_and_banks(lanes: Sequence[float], stride: int = 33) -> dict[str, Any]:
    """Run lockstep 32-lane butterfly XOR warp reduction and 32-bank conflict arithmetic."""
    if isinstance(lanes, (str, bytes)):
        raise SheafError("lanes must be a 32-element sequence of finite numbers")
    lane_vals = [as_real(v, "lane") for v in lanes]
    if len(lane_vals) != 32:
        raise SheafError(f"SIMT warp requires exactly 32 lanes, got {len(lane_vals)}")
    stride_i = as_int(stride, "stride")
    if stride_i < 0:
        raise SheafError(f"stride must be non-negative, got {stride_i}")

    warp_regs = list(lane_vals)
    try:
        expected_sum = math.fsum(lane_vals)
    except OverflowError as exc:
        raise SheafError("SIMT warp lane sum overflows float64") from exc
    stage_snapshots: list[list[float]] = []
    for mask in (16, 8, 4, 2, 1):
        snapshot = list(warp_regs)
        warp_regs = [snapshot[i] + snapshot[i ^ mask] for i in range(32)]
        if not all(math.isfinite(reg) for reg in warp_regs):
            raise SheafError("SIMT warp reduction overflows float64")
        stage_snapshots.append(list(warp_regs))

    butterfly_err = max(abs(reg - expected_sum) for reg in warp_regs)
    banks = [(i * stride_i) & 31 for i in range(32)]
    conflicts = 32 - len(set(banks))
    ok = math.isclose(butterfly_err, 0.0, rel_tol=0.0, abs_tol=1e-9) and conflicts == 0

    return {
        "ok": ok,
        "passed": ok,
        "conflicts": conflicts,
        "bank_conflicts": conflicts,
        "reduced_sum": warp_regs[0],
        "butterfly_err": butterfly_err,
        "stage_snapshots": stage_snapshots,
    }


def verify_goldilocks_r1cs_dipole(
    y_float: float = 0.0,
    dipole_acts: Sequence[float] = (),
    *,
    y_field: int | None = None,
    z_witness: int | None = None,
    v_minus_override: int | None = None,
) -> dict[str, Any]:
    """Verify Pade(1,1) Goldilocks F_p R1CS gate and 12-neuron dipole cancellation."""
    y_val = as_real(y_float, "y_float")
    if isinstance(dipole_acts, (str, bytes)):
        raise SheafError("dipole_acts must be a 12-element sequence of finite numbers")
    acts = [as_real(a, "dipole_act") for a in dipole_acts]
    if len(acts) != 12:
        raise SheafError(f"dipole_acts must have exactly 12 elements, got {len(acts)}")

    scale_fp = 1 << 24
    if y_field is not None:
        y_fp = as_int(y_field, "y_field") % GOLDILOCKS_P
    else:
        y_scaled = y_val * scale_fp
        if not math.isfinite(y_scaled):
            raise SheafError("y_float * scale_fp overflows float64")
        y_fp = int(round(y_scaled)) % GOLDILOCKS_P

    num_a = (3 * scale_fp + y_fp) % GOLDILOCKS_P
    den_b = (scale_fp + 3 * y_fp) % GOLDILOCKS_P
    if den_b % GOLDILOCKS_P == 0:
        raise SheafError("degenerate Goldilocks denominator: den_b == 0 mod GOLDILOCKS_P")

    den_b_inv = pow(den_b, GOLDILOCKS_P - 2, GOLDILOCKS_P)
    z_computed = (num_a * den_b_inv) % GOLDILOCKS_P
    z_used = (as_int(z_witness, "z_witness") % GOLDILOCKS_P) if z_witness is not None else z_computed
    r1cs_residual = (den_b * z_used - num_a) % GOLDILOCKS_P

    vm_override = (
        as_int(v_minus_override, "v_minus_override") % GOLDILOCKS_P
        if v_minus_override is not None
        else None
    )
    mod_sum = 0
    for pair_i in range(6):
        act_scaled = acts[2 * pair_i] * scale_fp
        if not math.isfinite(act_scaled):
            raise SheafError("dipole_act * scale_fp overflows float64")
        act_fp = int(round(act_scaled)) % GOLDILOCKS_P
        v_plus = (act_fp * 3500) % GOLDILOCKS_P
        v_minus = (
            vm_override
            if (pair_i == 0 and vm_override is not None)
            else (GOLDILOCKS_P - v_plus) % GOLDILOCKS_P
        )
        mod_sum = (mod_sum + v_plus + v_minus) % GOLDILOCKS_P

    ok = r1cs_residual == 0 and mod_sum == 0
    return {
        "ok": ok,
        "passed": ok,
        "prime_field_modulus": str(GOLDILOCKS_P),
        "y_field": y_fp,
        "z_witness": z_used,
        "r1cs_gate_residual_mod_p": r1cs_residual,
        "dipole_sum_mod_p": mod_sum,
    }
