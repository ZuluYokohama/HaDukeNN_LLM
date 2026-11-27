"""Certified rational/interval arithmetic and multi-precision drift envelopes.

Standard IEEE-754 binary64 arithmetic is non-associative and susceptible to
catastrophic cancellation near gate boundaries. ``CertifiedBall`` tracks an
exact ``fractions.Fraction`` value alongside outward-directed IEEE-754
``[low, high]`` enclosures via ``math.nextafter`` so boundary predicates hold
only when the entire error enclosure satisfies the inequality.

This module also defines unit-roundoff specifications across modern AI
accelerators (FP64 down to MXFP8 and NVFP4), tree-reduction vs. sequential
accumulation drift bounds, learner-sampler RL precision parity checks, and
hardware benchmark roofline eligibility checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from sheaf_verify.checks import Criterion, bad, ok
from sheaf_verify.hashing import SheafError, as_int, as_real, fmt_float


@dataclass(frozen=True)
class PrecisionSpec:
    name: str
    mantissa_bits: int
    unit_roundoff: float
    has_tensor_core_mma: bool


# Unit roundoff u = 2^{-(p)} where p = 1 + mantissa_bits (significand precision).
PRECISION_SPECS: dict[str, PrecisionSpec] = {
    "fp64": PrecisionSpec("fp64", 52, 2.0**-53, True),
    "fp32": PrecisionSpec("fp32", 23, 2.0**-24, False),
    "tf32": PrecisionSpec("tf32", 10, 2.0**-11, True),
    "fp16": PrecisionSpec("fp16", 10, 2.0**-11, True),
    "bf16": PrecisionSpec("bf16", 7, 2.0**-8, True),
    "fp8_e4m3": PrecisionSpec("fp8_e4m3", 3, 2.0**-4, True),
    "mxfp8": PrecisionSpec("mxfp8", 3, 2.0**-4, True),
    "fp8_e5m2": PrecisionSpec("fp8_e5m2", 2, 2.0**-3, True),
    "nvfp4": PrecisionSpec("nvfp4", 1, 2.0**-2, True),
}

# Pre-Ampere / non-Tensor-Core shader architectures unsuitable for compiled BF16 wall-clock claims.
NON_TENSOR_CORE_ARCHS = frozenset({"sm_60", "sm_61", "sm_75"})
NON_BF16_GPU_MARKERS = ("1660", "1650", "1080", "1070", "1060", "tu116", "tu117")


def _outward_bounds(exact: Fraction, extra_radius: Fraction = Fraction(0)) -> tuple[float, float]:
    if extra_radius < 0:
        raise SheafError("interval radius must be non-negative")
    lo_exact = exact - extra_radius
    hi_exact = exact + extra_radius
    lo_f = float(lo_exact)
    hi_f = float(hi_exact)
    if Fraction(lo_f) > lo_exact:
        lo_f = math.nextafter(lo_f, -math.inf)
    if Fraction(hi_f) < hi_exact:
        hi_f = math.nextafter(hi_f, math.inf)
    return lo_f, hi_f


@dataclass(frozen=True)
class CertifiedBall:
    """Midpoint-radius rational interval with outward-rounded float bounds."""

    center: Fraction
    radius: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        if self.radius < 0:
            raise SheafError("CertifiedBall radius must be non-negative")

    @classmethod
    def from_number(cls, value: int | float | str | Fraction, ulp_slack: int = 0) -> CertifiedBall:
        if isinstance(value, bool):
            raise SheafError("boolean is not a real number")
        if isinstance(value, Fraction):
            exact = value
        elif isinstance(value, int):
            exact = Fraction(value)
        elif isinstance(value, str):
            try:
                exact = Fraction(value)
            except ValueError as error:
                raise SheafError(f"invalid rational literal: {value!r}") from error
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise SheafError("CertifiedBall requires a finite number")
            exact = Fraction(value)
        else:
            raise SheafError("unsupported numeric type for CertifiedBall")
        if ulp_slack < 0:
            raise SheafError("ulp_slack must be non-negative")
        if ulp_slack == 0:
            return cls(exact, Fraction(0))
        ref = float(exact)
        step = abs(math.nextafter(ref, math.inf) - ref)
        if not math.isfinite(step) or step == 0.0:
            step = 2.0**-1074
        return cls(exact, Fraction(step) * ulp_slack)

    @property
    def low_exact(self) -> Fraction:
        return self.center - self.radius

    @property
    def high_exact(self) -> Fraction:
        return self.center + self.radius

    def float_enclosure(self) -> tuple[float, float]:
        return _outward_bounds(self.center, self.radius)

    @property
    def midpoint_float(self) -> float:
        return float(self.center)

    @property
    def radius_float(self) -> float:
        lo, hi = self.float_enclosure()
        mid = float(self.center)
        return max(mid - lo, hi - mid)

    def contains_zero(self) -> bool:
        return self.low_exact <= 0 <= self.high_exact

    def __add__(self, other: CertifiedBall) -> CertifiedBall:
        return CertifiedBall(self.center + other.center, self.radius + other.radius)

    def __sub__(self, other: CertifiedBall) -> CertifiedBall:
        return CertifiedBall(self.center - other.center, self.radius + other.radius)

    def __neg__(self) -> CertifiedBall:
        return CertifiedBall(-self.center, self.radius)

    def __mul__(self, other: CertifiedBall) -> CertifiedBall:
        corners = (
            self.low_exact * other.low_exact,
            self.low_exact * other.high_exact,
            self.high_exact * other.low_exact,
            self.high_exact * other.high_exact,
        )
        lo, hi = min(corners), max(corners)
        return CertifiedBall((lo + hi) / 2, (hi - lo) / 2)

    def __truediv__(self, other: CertifiedBall) -> CertifiedBall:
        if other.contains_zero():
            raise SheafError("CertifiedBall division enclosure straddles zero")
        inv_lo = Fraction(1, 1) / other.high_exact
        inv_hi = Fraction(1, 1) / other.low_exact
        inv = CertifiedBall((inv_lo + inv_hi) / 2, (inv_hi - inv_lo) / 2)
        return self * inv

    def sqrt(self) -> CertifiedBall:
        if self.low_exact < 0:
            raise SheafError("CertifiedBall sqrt requires a non-negative enclosure")
        if self.high_exact == 0:
            return CertifiedBall(Fraction(0), Fraction(0))
        lo_f = math.sqrt(float(self.low_exact))
        lo_f = max(0.0, math.nextafter(lo_f, -math.inf))
        hi_f = math.nextafter(math.sqrt(float(self.high_exact)), math.inf)
        while Fraction(lo_f) * Fraction(lo_f) > self.low_exact and lo_f > 0.0:
            lo_f = math.nextafter(lo_f, -math.inf)
        while Fraction(hi_f) * Fraction(hi_f) < self.high_exact:
            hi_f = math.nextafter(hi_f, math.inf)
        lo_q, hi_q = Fraction(lo_f), Fraction(hi_f)
        return CertifiedBall((lo_q + hi_q) / 2, (hi_q - lo_q) / 2)

    def strictly_less_than(self, bound: CertifiedBall | float | int | Fraction) -> bool:
        rhs = bound if isinstance(bound, CertifiedBall) else CertifiedBall.from_number(bound)
        return self.high_exact < rhs.low_exact

    def certified_at_most(self, bound: CertifiedBall | float | int | Fraction) -> bool:
        rhs = bound if isinstance(bound, CertifiedBall) else CertifiedBall.from_number(bound)
        return self.high_exact <= rhs.low_exact

    def strictly_greater_than(self, bound: CertifiedBall | float | int | Fraction) -> bool:
        rhs = bound if isinstance(bound, CertifiedBall) else CertifiedBall.from_number(bound)
        return self.low_exact > rhs.high_exact


def certified_bpb_identity(
    total_bits: float,
    raw_bytes: int,
    per_seed: list[float],
    *,
    rel_tol: Fraction = Fraction(1, 1_000_000_000),
) -> Criterion:
    """Verify mean(per_seed) == total_bits / raw_bytes using exact rational balls."""
    if raw_bytes <= 0 or not per_seed:
        return bad("numerics.bpb_ball", "raw_bytes and per_seed must be positive/non-empty")
    seed_balls = [CertifiedBall.from_number(val, ulp_slack=2) for val in per_seed]
    sum_ball = CertifiedBall(Fraction(0))
    for item in seed_balls:
        sum_ball = sum_ball + item
    mean_ball = sum_ball / CertifiedBall.from_number(len(per_seed))
    ratio_ball = CertifiedBall.from_number(total_bits, ulp_slack=2) / CertifiedBall.from_number(raw_bytes)
    diff = mean_ball - ratio_ball
    gap = max(abs(diff.low_exact), abs(diff.high_exact))
    scale = max(Fraction(1), abs(mean_ball.center))
    if gap <= rel_tol * scale:
        return ok(
            "numerics.bpb_ball",
            f"certified ball mean {fmt_float(mean_ball.midpoint_float)} encloses total_bits/raw_bytes within {float(gap):.3e}",
        )
    return bad(
        "numerics.bpb_ball",
        f"certified ball discrepancy {float(gap):.3e} exceeds tolerance {float(rel_tol * scale):.3e}",
    )


def reduction_drift_envelope(dtype: str, count: int, *, mode: str = "tree") -> float:
    """Higham backward-error factor gamma_k = (k*u) / (1 - k*u) for sum reductions."""
    key = dtype.strip().lower()
    if key not in PRECISION_SPECS:
        raise SheafError(f"unknown precision dtype {dtype!r}")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise SheafError("reduction count must be a positive integer")
    if count == 1:
        return 0.0
    u = PRECISION_SPECS[key].unit_roundoff
    if mode == "tree":
        steps = math.ceil(math.log2(count))
    elif mode == "sequential":
        steps = count - 1
    else:
        raise SheafError("reduction mode must be 'tree' or 'sequential'")
    ku = steps * u
    if ku >= 1.0:
        return math.inf
    return ku / (1.0 - ku)


def learner_sampler_parity_criteria(record: dict) -> list[Criterion]:
    """Audit distributed RLVR learner vs. sampler precision-grid alignment."""
    if not isinstance(record, dict):
        return [bad("precision.grid", "precision parity record is missing")]
    learner = str(record.get("learner_dtype", "")).strip().lower()
    sampler = str(record.get("sampler_dtype", "")).strip().lower()
    unmatched = record.get("unmatched_layers", [])
    batch_invariant = record.get("batch_invariant_reductions")
    criteria = []
    if learner not in PRECISION_SPECS or sampler not in PRECISION_SPECS:
        return [bad("precision.dtypes", "learner_dtype and sampler_dtype must be known precision formats")]
    if learner == sampler:
        criteria.append(ok("precision.parity", f"learner and sampler match on {learner}"))
    else:
        criteria.append(bad("precision.parity", f"learner ({learner}) and sampler ({sampler}) precision mismatch risks KL drift"))
    if isinstance(unmatched, list) and len(unmatched) == 0:
        criteria.append(ok("precision.layers", "all linear projections share uniform quantization"))
    else:
        criteria.append(bad("precision.layers", "silent mixed-precision layers left unmatched"))
    if batch_invariant is True:
        criteria.append(ok("precision.batch_invariance", "pairwise tree-reduction batch invariance enforced"))
    else:
        criteria.append(bad("precision.batch_invariance", "non-associative dynamic batch reductions are not locked"))
    return criteria


def hardware_wallclock_criteria(
    *,
    gpu_name: str,
    sm_arch: str,
    dtype: str,
    benchmark_role: str,
    solo_device: bool = True,
) -> list[Criterion]:
    """Ensure wall-clock throughput claims never run on non-Tensor-Core or contended GPUs."""
    role = benchmark_role.strip().lower()
    arch = sm_arch.strip().lower()
    dtype_norm = dtype.strip().lower()
    name_norm = gpu_name.strip().lower()
    criteria = []
    has_tc = arch not in NON_TENSOR_CORE_ARCHS and not any(tag in name_norm for tag in NON_BF16_GPU_MARKERS)
    if role == "wallclock":
        if not has_tc:
            criteria.append(bad(
                "hardware.tensor_cores",
                f"GPU {gpu_name!r} ({sm_arch}) lacks modern Tensor Cores / native BF16 and cannot carry wall-clock claims",
            ))
        else:
            criteria.append(ok("hardware.tensor_cores", f"GPU {gpu_name!r} ({sm_arch}) has native Tensor Core support"))
        if not solo_device:
            criteria.append(bad("hardware.solo", "wall-clock benchmark must run solo without side-job contention"))
        else:
            criteria.append(ok("hardware.solo", "wall-clock benchmark ran solo with side jobs paused"))
        if dtype_norm == "bf16" and not has_tc:
            criteria.append(bad("hardware.bf16", f"{sm_arch} emulates BF16 on shaders"))
        else:
            criteria.append(ok("hardware.dtype", f"dtype {dtype_norm} is hardware-native"))
    elif role in ("eval_fp32", "knn_lm"):
        if dtype_norm == "bf16" and not has_tc:
            criteria.append(bad("hardware.eval_dtype", f"side-job {role} on {sm_arch} must use fp32, not emulated bf16"))
        else:
            criteria.append(ok("hardware.eval_isolation", f"side-job {role} safely isolated in {dtype_norm}"))
    else:
        criteria.append(bad("hardware.role", f"unknown benchmark_role {benchmark_role!r}"))
    return criteria


def drift_gate_criterion(observed_rel_err: float, dtype: str, reduction_n: int, *, safety_factor: float = 2.0) -> Criterion:
    """Check whether observed cross-hardware drift stays inside the tree-reduction envelope."""
    err = as_real(observed_rel_err, "observed_rel_err")
    n = as_int(reduction_n, "reduction_n")
    if err < 0 or safety_factor <= 0:
        return bad("numerics.drift_envelope", "observed error must be non-negative and safety_factor positive")
    bound = safety_factor * reduction_drift_envelope(dtype, n, mode="tree")
    if CertifiedBall.from_number(err, ulp_slack=1).certified_at_most(bound):
        return ok("numerics.drift_envelope", f"rel drift {fmt_float(err)} <= certified {dtype} tree bound {fmt_float(bound)}")
    return bad("numerics.drift_envelope", f"rel drift {fmt_float(err)} exceeds certified {dtype} tree bound {fmt_float(bound)}")
