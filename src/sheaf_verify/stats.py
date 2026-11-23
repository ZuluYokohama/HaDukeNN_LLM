"""Welch tests, bootstrap intervals, and Hedges' g.

The p-value is the two-tailed survival function of a Student t random
variable. The incomplete-beta continued fraction is evaluated in this
module so a reviewer can recompute it without SciPy. Bootstrap resampling
uses ``random.Random``, which is deterministic for a recorded seed on
CPython.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from sheaf_verify.hashing import SheafError


@dataclass(frozen=True)
class TTest:
    t: float
    df: float
    p: float


def mean(values: list[float]) -> float:
    if not values:
        raise SheafError("mean of an empty sample")
    return math.fsum(values) / len(values)


def sample_variance(values: list[float]) -> float:
    if len(values) < 2:
        raise SheafError("variance needs at least two observations")
    center = mean(values)
    return math.fsum((item - center) ** 2 for item in values) / (len(values) - 1)


def welch(a: list[float], b: list[float]) -> TTest:
    """Two-sample t test without assuming equal variances. Difference is mean(a) - mean(b)."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        raise SheafError("Welch's test needs at least two observations in each sample")
    m1, m2 = mean(a), mean(b)
    v1, v2 = sample_variance(a), sample_variance(b)
    se2 = v1 / n1 + v2 / n2
    if se2 == 0.0:
        df = float(n1 + n2 - 2)
        if m1 == m2:
            return TTest(0.0, df, 1.0)
        return TTest(math.copysign(math.inf, m1 - m2), df, 0.0)
    t_stat = (m1 - m2) / math.sqrt(se2)
    left = (v1 / n1) ** 2 / (n1 - 1)
    right = (v2 / n2) ** 2 / (n2 - 1)
    df = se2 ** 2 / (left + right)
    return TTest(t_stat, df, student_t_two_tailed(t_stat, df))


def paired_t(a: list[float], b: list[float]) -> TTest:
    if len(a) != len(b) or len(a) < 2:
        raise SheafError("paired t test needs two equal samples of length at least 2")
    diffs = [left - right for left, right in zip(a, b)]
    n = len(diffs)
    center = mean(diffs)
    var = sample_variance(diffs)
    df = float(n - 1)
    if var == 0.0:
        if center == 0.0:
            return TTest(0.0, df, 1.0)
        return TTest(math.copysign(math.inf, center), df, 0.0)
    t_stat = center / math.sqrt(var / n)
    return TTest(t_stat, df, student_t_two_tailed(t_stat, df))


def hedges_g(a: list[float], b: list[float]) -> float:
    """Bias-corrected standardized mean difference, mean(a) - mean(b)."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        raise SheafError("Hedges' g needs at least two observations in each sample")
    pooled = ((n1 - 1) * sample_variance(a) + (n2 - 1) * sample_variance(b)) / (n1 + n2 - 2)
    correction = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    if pooled == 0.0:
        if mean(a) == mean(b):
            return 0.0
        return math.copysign(math.inf, mean(a) - mean(b))
    return ((mean(a) - mean(b)) / math.sqrt(pooled)) * correction


def bootstrap_mean_difference(
    a: list[float],
    b: list[float],
    *,
    seed: int,
    draws: int,
    paired: bool,
) -> tuple[float, float]:
    """Percentile interval of mean(a) - mean(b).

    The percentile uses linear interpolation at index ``(n - 1) * p`` of the
    sorted replicates. Paired resampling draws shared indexes. Independent
    resampling draws each arm separately.
    """
    if draws < 1:
        raise SheafError("bootstrap needs at least one draw")
    if paired and len(a) != len(b):
        raise SheafError("paired bootstrap needs equal sample lengths")
    rng = random.Random(seed)
    n_a, n_b = len(a), len(b)
    replicates = []
    for _ in range(draws):
        if paired:
            indexes = [rng.randrange(n_a) for _ in range(n_a)]
            replicates.append(mean([a[index] - b[index] for index in indexes]))
        else:
            sample_a = [a[rng.randrange(n_a)] for _ in range(n_a)]
            sample_b = [b[rng.randrange(n_b)] for _ in range(n_b)]
            replicates.append(mean(sample_a) - mean(sample_b))
    replicates.sort()
    return _percentile(replicates, 0.025), _percentile(replicates, 0.975)


def student_t_two_tailed(t_stat: float, df: float) -> float:
    if df <= 0 or not math.isfinite(df):
        raise SheafError("Student t degrees of freedom must be positive")
    if math.isinf(t_stat):
        return 0.0
    if t_stat == 0.0:
        return 1.0
    x = df / (df + t_stat * t_stat)
    return regularized_incomplete_beta(x, df / 2.0, 0.5)


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        raise SheafError("incomplete beta parameters must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    log_front = a * math.log(x) + b * math.log(1.0 - x) - log_beta
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_front) * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta, modified Lentz form.

    This is the Numerical Recipes evaluation, with terms rescaled so a large
    degrees-of-freedom value does not overflow the unscaled recurrence.
    """
    tiny = 1e-300
    epsilon = 3e-14
    max_iter = 500
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            return h
    raise SheafError("incomplete beta continued fraction did not converge")


def one_sided_less(test: TTest) -> float:
    """Lower-tail probability. The stored Welch p-value is two-tailed."""
    if math.isinf(test.t):
        return 0.0 if test.t < 0 else 1.0
    if test.t <= 0:
        return min(1.0, test.p / 2.0)
    return min(1.0, 1.0 - test.p / 2.0)


def student_t_critical(two_tailed_p: float, df: float) -> float:
    """Positive t such that the two-tailed p-value equals ``two_tailed_p``."""
    if not 0.0 < two_tailed_p < 1.0:
        raise SheafError("critical value needs a two-tailed probability in (0, 1)")
    lo = 0.0
    hi = 1.0
    while student_t_two_tailed(hi, df) > two_tailed_p:
        hi *= 2.0
        if hi > 1e6:
            raise SheafError("Student-t critical value did not bracket")
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if student_t_two_tailed(mid, df) > two_tailed_p:
            lo = mid
        else:
            hi = mid
    return hi


def slope_interval(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Ordinary-least-squares slope and its 95% Student-t interval.

    Returns ``(slope, low, high)``. A perfect line has a zero-width interval
    at the slope, which excludes 0 whenever the slope itself is not 0.
    """
    n = len(xs)
    if n != len(ys) or n < 3:
        raise SheafError("slope needs at least three paired observations")
    mx = mean(xs)
    my = mean(ys)
    sxx = math.fsum((item - mx) ** 2 for item in xs)
    if sxx == 0.0:
        raise SheafError("slope is undefined when x does not vary")
    slope = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    sse = math.fsum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    scale = max(1.0, math.fsum(item * item for item in ys))
    if sse <= 1e-18 * scale:
        return slope, slope, slope
    df = n - 2
    se = math.sqrt((sse / df) / sxx)
    critical = student_t_critical(0.05, float(df))
    return slope, slope - critical * se, slope + critical * se


def _percentile(sorted_values: list[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight
