"""Sequential Anytime-Valid Inference (SAVI), Empirical-Bayes paired tests, and LTT calibration.

Extends fixed-horizon Student-t statistics with three zero-dependency statistical
engines tailored to small-seed LLM scaling ladders and sequential verification:
1. Game-theoretic Kelly-betting e-processes ``E_t = prod_{i=1}^t (1 + lambda_i z_i)``
   preserving Type-I error ``P_{H_0}(exists t >= 1: E_t >= 1/alpha) <= alpha``
   under optional stopping via Ville's inequality, paired with a dual futility
   e-process ``E_t^futile >= futility_threshold`` (default 10.0, <= 10% false
   futility risk against ``min_effect``) and e-BH multiple-testing control under
   arbitrary dependence (e.g., correlated sheaf edges).
2. Empirical-Bayes (Smyth) tier-pooled paired-seed t-tests solving the ``n=3``
   Welch degrees-of-freedom trap (``df=2..4`` requiring 4.6--9.9 SE) while
   preserving pre-registered Welch side-by-side.
3. Learn-Then-Test (LTT) Hoeffding-Bentkus conformal risk calibration over
   monotone threshold grids without Bonferroni loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from sheaf_verify.hashing import SheafError
from sheaf_verify.stats import TTest, mean, one_sided_less, paired_t, sample_variance, student_t_two_tailed, welch


@dataclass(frozen=True)
class EProcessResult:
    trajectory: tuple[float, ...]
    futility_trajectory: tuple[float, ...]
    final_e: float
    max_e: float
    max_futility_e: float
    stopped_at: int | None
    rejected_null: bool
    futile: bool
    alpha: float


def kelly_e_process(
    gains: list[float],
    *,
    scale_bound: float,
    alpha: float = 0.01,
    max_bet: float = 0.75,
    min_effect: float | None = None,
    futility_threshold: float = 10.0,
) -> EProcessResult:
    """Predictable plug-in Kelly betting e-process for ``H_0: E[gain_i] <= 0`` vs ``H_1: E[gain_i] > 0``.

    Simultaneously tracks a dual futility supermartingale against ``H_target: E[gain_i] >= min_effect``
    (default ``0.2 * scale_bound``), declaring futility when ``max_futility_e >= futility_threshold``
    (at ``futility_threshold = 10.0``, Ville's inequality caps false-futility Type-II error at <= 10%).
    """
    if not gains:
        raise SheafError("kelly_e_process requires at least one observation")
    if scale_bound <= 0 or not math.isfinite(scale_bound):
        raise SheafError("scale_bound must be a finite positive number")
    if not 0.0 < alpha < 1.0:
        raise SheafError("alpha must lie in (0, 1)")
    if not 0.0 < max_bet < 1.0:
        raise SheafError("max_bet must lie in (0, 1)")
    if futility_threshold <= 1.0:
        raise SheafError("futility_threshold must exceed 1.0")

    eff = 0.2 * scale_bound if min_effect is None else float(min_effect)
    if eff < 0 or eff >= scale_bound:
        raise SheafError("min_effect must satisfy 0 <= min_effect < scale_bound")

    threshold = 1.0 / alpha
    wealth = 1.0
    peak = 1.0
    fut_wealth = 1.0
    fut_peak = 1.0
    stopped: int | None = None
    traj: list[float] = []
    fut_traj: list[float] = []
    sum_z = 0.0
    sum_z2 = 0.25
    sum_f = 0.0
    sum_f2 = 0.25
    fut_span = scale_bound + eff

    for idx, raw in enumerate(gains, 1):
        val = float(raw)
        if not math.isfinite(val):
            raise SheafError("gain observation must be finite")
        z = max(-1.0, min(1.0, val / scale_bound))
        if idx == 1:
            bet = 0.5 * max_bet
            f_bet = 0.5 * max_bet
        else:
            mu_hat = sum_z / (idx - 1)
            v_hat = sum_z2 / (idx - 1)
            bet = min(max_bet, max(0.0, mu_hat / max(v_hat, 1e-6)))
            f_mu = sum_f / (idx - 1)
            f_v = sum_f2 / (idx - 1)
            f_bet = min(max_bet, max(0.0, f_mu / max(f_v, 1e-6)))

        wealth *= 1.0 + bet * z
        peak = max(peak, wealth)
        traj.append(wealth)
        if stopped is None and wealth >= threshold:
            stopped = idx

        # Dual futility increment: under E[val] >= eff, E[eff - val] <= 0; bounded above by (eff + scale_bound) <= fut_span
        f_inc = max(-1.0, min(1.0, (eff - val) / fut_span))
        fut_wealth *= 1.0 + f_bet * f_inc
        fut_peak = max(fut_peak, fut_wealth)
        fut_traj.append(fut_wealth)

        sum_z += z
        sum_z2 += z * z
        sum_f += f_inc
        sum_f2 += f_inc * f_inc

    futile = (stopped is None) and (fut_peak >= futility_threshold)
    return EProcessResult(
        trajectory=tuple(traj),
        futility_trajectory=tuple(fut_traj),
        final_e=wealth,
        max_e=peak,
        max_futility_e=fut_peak,
        stopped_at=stopped,
        rejected_null=stopped is not None,
        futile=futile,
        alpha=alpha,
    )


def merge_dependent_e_values(e_values: list[float]) -> float:
    """Arithmetic mean of e-values remains an exact valid e-value under arbitrary dependence."""
    if not e_values:
        raise SheafError("e_values must be non-empty")
    for val in e_values:
        if isinstance(val, bool) or not isinstance(val, (int, float)) or val < 0 or not math.isfinite(float(val)):
            raise SheafError("every e-value must be a finite non-negative real")
    return math.fsum(float(v) for v in e_values) / len(e_values)


def ebh_discoveries(e_values: list[float], *, alpha: float = 0.05) -> list[int]:
    """Wang & Ramdas e-BH procedure controlling FDR at ``alpha`` under arbitrary dependence."""
    if not 0.0 < alpha < 1.0:
        raise SheafError("alpha must lie in (0, 1)")
    m = len(e_values)
    if m == 0:
        return []
    for val in e_values:
        if isinstance(val, bool) or not isinstance(val, (int, float)) or val < 0 or not math.isfinite(float(val)):
            raise SheafError("every e-value must be finite and non-negative")
    indexed = sorted(enumerate(float(v) for v in e_values), key=lambda pair: pair[1], reverse=True)
    k_star = 0
    for rank, (_orig_idx, ev) in enumerate(indexed, 1):
        if (rank * ev) / m >= 1.0 / alpha:
            k_star = rank
    return sorted(indexed[i][0] for i in range(k_star))


@dataclass(frozen=True)
class EmpiricalBayesContrast:
    mean_diff: float
    welch_t: float
    welch_df: float
    welch_p_one_sided: float
    paired_se: float
    paired_t: float
    paired_df: float
    paired_p_one_sided: float
    pearson_r: float
    eb_posterior_var: float
    eb_se: float
    eb_t: float
    eb_df: float
    eb_p_one_sided: float


def _pearson_r(a: list[float], b: list[float]) -> float:
    ma, mb = mean(a), mean(b)
    sa = math.fsum((x - ma) ** 2 for x in a)
    sb = math.fsum((y - mb) ** 2 for y in b)
    if sa <= 0.0 or sb <= 0.0:
        return 0.0
    cov = math.fsum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    return max(-1.0, min(1.0, cov / math.sqrt(sa * sb)))


def empirical_bayes_paired_t(
    treatment: list[float],
    control: list[float],
    *,
    prior_df: float = 12.0,
    prior_pair_var: float | None = None,
    historical_pair_diffs: list[list[float]] | None = None,
) -> EmpiricalBayesContrast:
    """Side-by-side Welch, Paired-Seed, and Smyth Empirical-Bayes moderated paired test."""
    if len(treatment) != len(control) or len(treatment) < 2:
        raise SheafError("treatment and control must have equal length >= 2")
    n = len(treatment)
    w_res = welch(treatment, control)
    p_res = paired_t(treatment, control)
    diffs = [t - c for t, c in zip(treatment, control, strict=True)]
    d_mean = mean(diffs)
    s2_d = sample_variance(diffs)
    r = _pearson_r(treatment, control)

    if historical_pair_diffs:
        valid_vars = []
        total_df = 0.0
        for past in historical_pair_diffs:
            if len(past) >= 2:
                df_i = len(past) - 1
                valid_vars.append((df_i, sample_variance(past)))
                total_df += df_i
        if not valid_vars:
            raise SheafError("historical_pair_diffs contained no valid samples of length >= 2")
        prior_df = total_df
        prior_pair_var = math.fsum(df_i * v_i for df_i, v_i in valid_vars) / total_df
    elif prior_pair_var is None:
        raise SheafError("provide either prior_pair_var or historical_pair_diffs")

    if prior_df < 0 or prior_pair_var <= 0:
        raise SheafError("prior_df must be >= 0 and prior_pair_var > 0")

    df_sample = float(n - 1)
    eb_df = prior_df + df_sample
    eb_post_var = (prior_df * prior_pair_var + df_sample * s2_d) / eb_df
    eb_se = math.sqrt(eb_post_var / n)
    eb_t = d_mean / eb_se if eb_se > 0 else math.copysign(math.inf, d_mean)
    eb_two = student_t_two_tailed(eb_t, eb_df)
    eb_test = TTest(eb_t, eb_df, eb_two)

    return EmpiricalBayesContrast(
        mean_diff=d_mean,
        welch_t=w_res.t,
        welch_df=w_res.df,
        welch_p_one_sided=one_sided_less(w_res),
        paired_se=math.sqrt(s2_d / n),
        paired_t=p_res.t,
        paired_df=p_res.df,
        paired_p_one_sided=one_sided_less(p_res),
        pearson_r=r,
        eb_posterior_var=eb_post_var,
        eb_se=eb_se,
        eb_t=eb_t,
        eb_df=eb_df,
        eb_p_one_sided=one_sided_less(eb_test),
    )


def hoeffding_bentkus_p(losses: list[float], risk_bound: float) -> float:
    """Finite-sample valid p-value for ``H_0: E[L] >= risk_bound`` on ``[0, 1]`` bounded losses."""
    if not losses:
        raise SheafError("losses must be non-empty")
    if not 0.0 < risk_bound < 1.0:
        raise SheafError("risk_bound must lie strictly in (0, 1)")
    n = len(losses)
    vals = []
    for item in losses:
        v = float(item)
        if not 0.0 <= v <= 1.0:
            raise SheafError("LTT losses must be bounded in [0, 1]")
        vals.append(v)
    r_hat = math.fsum(vals) / n
    if r_hat >= risk_bound:
        return 1.0
    eps = max(r_hat, 1e-15)
    kl = eps * math.log(eps / risk_bound) + (1.0 - eps) * math.log((1.0 - eps) / (1.0 - risk_bound))
    p_hoeffding = math.exp(-n * kl)
    k = math.floor(n * r_hat)
    p_binom = 0.0
    for j in range(k + 1):
        log_prob = (
            math.lgamma(n + 1)
            - math.lgamma(j + 1)
            - math.lgamma(n - j + 1)
            + j * math.log(risk_bound)
            + (n - j) * math.log(1.0 - risk_bound)
        )
        p_binom += math.exp(log_prob)
    p_bentkus = min(1.0, math.e * p_binom)
    return min(1.0, p_hoeffding, p_bentkus)


@dataclass(frozen=True)
class LTTCalibrationResult:
    certified_thresholds: tuple[float, ...]
    p_values: dict[float, float]
    selected_threshold: float | None


def ltt_calibrate_thresholds(
    threshold_losses: list[tuple[float, list[float]]],
    *,
    risk_bound: float,
    delta: float = 0.05,
) -> LTTCalibrationResult:
    """Learn-Then-Test (Angelopoulos et al.) monotone fixed-sequence calibration."""
    if not threshold_losses:
        raise SheafError("threshold_losses must be non-empty")
    if not 0.0 < delta < 1.0:
        raise SheafError("delta must lie in (0, 1)")
    certified: list[float] = []
    p_map: dict[float, float] = {}
    still_valid = True
    for thresh, losses in threshold_losses:
        pval = hoeffding_bentkus_p(losses, risk_bound)
        p_map[float(thresh)] = pval
        if still_valid and pval <= delta:
            certified.append(float(thresh))
        else:
            still_valid = False
    selected = certified[-1] if certified else None
    return LTTCalibrationResult(
        certified_thresholds=tuple(certified),
        p_values=p_map,
        selected_threshold=selected,
    )
