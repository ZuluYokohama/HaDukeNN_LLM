"""Pure-stdlib generalization, information-theoretic, topological, and split-integrity guards.

Synthesizes scalar/algebraic bounds and data-integrity verifiers from:
- Petersen (2022), *Mathematics of Machine Learning* (Hoeffding, Rademacher, Tikhonov, JL, AdaBoost)
- soulmachine (2017), *Machine Learning Cheat Sheet* (log-sum-exp, BIC, ELBO, reject-option)
- Gardinazzi et al. (2025), *Zigzag Persistent Topological Features in LLMs* (effective PI, Z_p)
- Alam, Decision Trees & Random Forest Predictive Maintenance (Gini impurity, split disjointness)
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from fractions import Fraction
from typing import Any

from sheaf_verify.hashing import SheafError, as_int, as_real


def hoeffding_bound(m: int, delta: float = 0.05, h_size: int = 1) -> float:
    """Finite-hypothesis Hoeffding generalization bound: ``sqrt((log|H| + log(2/delta)) / (2m))``."""
    m_i = as_int(m, "m")
    h_i = as_int(h_size, "h_size")
    delta_f = as_real(delta, "delta")
    if m_i < 1:
        raise SheafError(f"m must be >= 1, got {m_i}")
    if h_i < 1:
        raise SheafError(f"h_size must be >= 1, got {h_i}")
    if not (0.0 < delta_f < 1.0):
        raise SheafError(f"delta must lie strictly in (0, 1), got {delta_f}")
    try:
        return math.sqrt((math.log(h_i) + (math.log(2.0) - math.log(delta_f))) / (2.0 * m_i))
    except (OverflowError, ValueError) as exc:
        raise SheafError("hoeffding_bound numerical overflow") from exc


def rademacher_kernel_bound(lam: float, trace_k: float, m: int, *, s: float = 0.0) -> float:
    """RKHS Rademacher complexity bound (Petersen Prop 14.1): ``(lam / m) * sqrt(Tr(K)) + s / sqrt(m)``."""
    lam_f = as_real(lam, "lam")
    tr_f = as_real(trace_k, "trace_k")
    m_i = as_int(m, "m")
    s_f = as_real(s, "s")
    if lam_f < 0.0:
        raise SheafError(f"lam must be >= 0, got {lam_f}")
    if tr_f < 0.0:
        raise SheafError(f"trace_k must be >= 0, got {tr_f}")
    if m_i < 1:
        raise SheafError(f"m must be >= 1, got {m_i}")
    if s_f < 0.0:
        raise SheafError(f"s must be >= 0, got {s_f}")
    try:
        val = (lam_f / m_i) * math.sqrt(tr_f) + (s_f / math.sqrt(m_i))
    except (OverflowError, ValueError) as exc:
        raise SheafError("rademacher_kernel_bound numerical overflow") from exc
    if not math.isfinite(val):
        raise SheafError("rademacher_kernel_bound result is non-finite")
    return val


def tikhonov_stability_rate(rho: float, lam: float, m: int) -> float:
    """Tikhonov on-average replace-one stability bound (Petersen Cor 9.1): ``2 * rho^2 / (lam * m)``."""
    rho_f = as_real(rho, "rho")
    lam_f = as_real(lam, "lam")
    m_i = as_int(m, "m")
    if rho_f < 0.0:
        raise SheafError(f"rho must be >= 0, got {rho_f}")
    if lam_f <= 0.0:
        raise SheafError(f"lam must be > 0, got {lam_f}")
    if m_i < 1:
        raise SheafError(f"m must be >= 1, got {m_i}")
    try:
        rate = 2.0 * (rho_f / lam_f) * (rho_f / m_i)
    except (ZeroDivisionError, OverflowError) as exc:
        raise SheafError("tikhonov_stability_rate numerical overflow") from exc
    if not math.isfinite(rate):
        raise SheafError("tikhonov_stability_rate result is non-finite")
    return rate


def tikhonov_optimal_lambda(rho: float, bound_b: float, m: int) -> float:
    """Optimal Tikhonov regularization parameter ``sqrt(2 * rho^2 / (B^2 * m))``."""
    rho_f = as_real(rho, "rho")
    b_f = as_real(bound_b, "bound_b")
    m_i = as_int(m, "m")
    if rho_f < 0.0:
        raise SheafError(f"rho must be >= 0, got {rho_f}")
    if b_f <= 0.0:
        raise SheafError(f"bound_b must be > 0, got {b_f}")
    if m_i < 1:
        raise SheafError(f"m must be >= 1, got {m_i}")
    try:
        lam = math.sqrt(2.0 / m_i) * (rho_f / b_f)
    except (ZeroDivisionError, OverflowError) as exc:
        raise SheafError("tikhonov_optimal_lambda numerical overflow") from exc
    if not math.isfinite(lam):
        raise SheafError("tikhonov_optimal_lambda result is non-finite")
    return lam


def jl_min_dimension(n: int, eps: float) -> int:
    """Johnson-Lindenstrauss target dimension lower bound (Petersen Thm 20.1): ``ceil(20 * log(n) / eps^2)``."""
    n_i = as_int(n, "n")
    eps_f = as_real(eps, "eps")
    if n_i < 2:
        raise SheafError(f"n must be >= 2, got {n_i}")
    if not (0.0 < eps_f < 0.5):
        raise SheafError(f"eps must lie strictly in (0, 0.5), got {eps_f}")
    try:
        dim_val = (20.0 * math.log(n_i) / eps_f) / eps_f
    except (ZeroDivisionError, OverflowError) as exc:
        raise SheafError("jl_min_dimension numerical overflow/underflow") from exc
    if not math.isfinite(dim_val):
        raise SheafError("jl_min_dimension result is non-finite")
    return math.ceil(dim_val)


def adaboost_exp_bound(gamma: float, t_rounds: int) -> float:
    """AdaBoost exponential training-error bound (Petersen Thm 18.1): ``exp(-2 * gamma^2 * T)``."""
    gamma_f = as_real(gamma, "gamma")
    t_i = as_int(t_rounds, "t_rounds")
    if not (0.0 < gamma_f < 0.5):
        raise SheafError(f"gamma must lie strictly in (0, 0.5), got {gamma_f}")
    if t_i < 1:
        raise SheafError(f"t_rounds must be >= 1, got {t_i}")
    try:
        return math.exp(-2.0 * (gamma_f ** 2) * t_i)
    except (OverflowError, ValueError) as exc:
        raise SheafError("adaboost_exp_bound numerical overflow") from exc


def logsumexp(values: Sequence[float]) -> float:
    """Numerically stable log-sum-exp via max-shift trick."""
    if isinstance(values, (str, bytes)):
        raise SheafError("values must be a sequence of finite real numbers")
    vals = [as_real(v, "value") for v in values]
    if not vals:
        raise SheafError("logsumexp requires at least one value")
    b = max(vals)
    return b + math.log(math.fsum(math.exp(x - b) for x in vals))


def bic_score(log_likelihood: float, dof: int, n: int) -> float:
    """Bayesian Information Criterion: ``log_likelihood - 0.5 * dof * log(n)``."""
    ll_f = as_real(log_likelihood, "log_likelihood")
    dof_i = as_int(dof, "dof")
    n_i = as_int(n, "n")
    if dof_i < 0:
        raise SheafError(f"dof must be >= 0, got {dof_i}")
    if n_i < 1:
        raise SheafError(f"n must be >= 1, got {n_i}")
    try:
        val = ll_f - 0.5 * dof_i * math.log(n_i)
    except (OverflowError, ValueError) as exc:
        raise SheafError("bic_score numerical overflow") from exc
    if not math.isfinite(val):
        raise SheafError("bic_score result is non-finite")
    return val


def verify_elbo_monotonicity(elbo_trace: Sequence[float], *, tol: float = 1e-12) -> dict[str, Any]:
    """Verify variational ELBO / free-energy trajectory is finite and monotone non-decreasing."""
    tol_f = as_real(tol, "tol")
    if tol_f < 0.0:
        raise SheafError(f"tol must be >= 0, got {tol_f}")
    if isinstance(elbo_trace, (str, bytes)):
        raise SheafError("elbo_trace must be a sequence of finite real numbers")
    vals = [as_real(v, "elbo") for v in elbo_trace]
    if not vals:
        raise SheafError("elbo_trace must be non-empty")
    deltas = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
    min_delta = min(deltas) if deltas else 0.0
    ok = all(d >= -tol_f for d in deltas)
    return {
        "ok": ok,
        "passed": ok,
        "min_delta": min_delta,
        "steps": len(vals),
    }


def reject_option_decision(
    probs: Sequence[float],
    lambda_r: float,
    lambda_s: float = 1.0,
) -> dict[str, Any]:
    """Optimal Bayes reject-option decision under substitution cost ``lambda_s`` and reject cost ``lambda_r``."""
    if isinstance(probs, (str, bytes)):
        raise SheafError("probs must be a sequence of probabilities")
    p_list = [as_real(p, "prob") for p in probs]
    if not p_list:
        raise SheafError("probs must be non-empty")
    if any(p < 0.0 for p in p_list):
        raise SheafError("probabilities must be non-negative")
    total = math.fsum(p_list)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise SheafError(f"probabilities must sum to 1.0, got {total}")

    lr = as_real(lambda_r, "lambda_r")
    ls = as_real(lambda_s, "lambda_s")
    if ls <= 0.0:
        raise SheafError(f"lambda_s must be > 0, got {ls}")
    if lr < 0.0 or lr >= ls:
        raise SheafError(f"costs must satisfy 0 <= lambda_r < lambda_s, got lambda_r={lr}, lambda_s={ls}")

    max_p = max(p_list)
    best_idx = p_list.index(max_p)
    threshold = 1.0 - (lr / ls)
    risk = ls * (1.0 - max_p)
    reject = (not math.isclose(risk, lr, rel_tol=1e-12, abs_tol=0.0)) and (risk > lr)
    return {
        "ok": True,
        "reject": reject,
        "predicted_class": None if reject else best_idx,
        "max_prob": max_p,
        "confidence_threshold": threshold,
    }


def _validate_square_nonneg_grid(grid: Sequence[Sequence[float]], label: str) -> list[list[float]]:
    if isinstance(grid, (str, bytes)):
        raise SheafError(f"{label} must be a 2D square matrix")
    rows = list(grid)
    n = len(rows)
    if n == 0:
        raise SheafError(f"{label} must be non-empty")
    mat: list[list[float]] = []
    for row in rows:
        if isinstance(row, (str, bytes)):
            raise SheafError(f"{label} rows must be sequences of finite real numbers")
        vals = [as_real(v, label) for v in row]
        if len(vals) != n:
            raise SheafError(f"{label} must be square ({n}x{n}), got row of length {len(vals)}")
        if any(v < 0.0 for v in vals):
            raise SheafError(f"{label} entries must be non-negative")
        mat.append(vals)
    return mat


def _check_finite(val: float, msg: str) -> float:
    if not math.isfinite(val):
        raise SheafError(msg)
    return val


def zigzag_effective_pi(pi_grid: Sequence[Sequence[float]]) -> list[list[float]]:
    """Collapse zigzag persistence image ``PI_p`` onto model-layer grid ``PI_hat_p`` (Paper 2 Eq. 4)."""
    mat = _validate_square_nonneg_grid(pi_grid, "pi_grid")
    n = len(mat)
    out_n = (n // 2) + 1
    pi_hat = [[0.0 for _ in range(out_n)] for _ in range(out_n)]
    for b in range(n):
        b_hat = (b + 1) // 2
        for d in range(n):
            d_hat = (d + 1) // 2
            pi_hat[b_hat][d_hat] = _check_finite(
                pi_hat[b_hat][d_hat] + mat[b][d],
                "zigzag_effective_pi sum overflowed to non-finite",
            )
    return pi_hat


def zigzag_persistence_ratio(
    pi_hat: Sequence[Sequence[float]],
    beta_p: int,
    l1: int = 0,
    l2: int = 1,
) -> float:
    """Compute inter-layer persistence ratio ``Z_p(l1, l2)`` with exact ``beta_p == 0 -> 0.0`` guard."""
    mat = _validate_square_nonneg_grid(pi_hat, "pi_hat")
    beta_i = as_int(beta_p, "beta_p")
    if beta_i < 0:
        raise SheafError(f"beta_p must be >= 0, got {beta_i}")
    l1_i = as_int(l1, "l1")
    l2_i = as_int(l2, "l2")
    n = len(mat)
    if not (0 <= l1_i < n and 0 <= l2_i < n):
        raise SheafError(f"layer indices ({l1_i}, {l2_i}) out of bounds for {n}x{n} pi_hat")
    if beta_i == 0:
        return 0.0
    m1 = min(l1_i, l2_i)
    m2 = max(l1_i, l2_i)
    try:
        total = _check_finite(
            math.fsum(mat[i][j] for i in range(0, m1 + 1) for j in range(m2 + 1, n)),
            "zigzag_persistence_ratio sum is non-finite",
        )
    except OverflowError as exc:
        raise SheafError("zigzag_persistence_ratio sum numerical overflow") from exc
    return _check_finite(
        float(Fraction(total) / beta_i),
        "zigzag_persistence_ratio result is non-finite",
    )


def gini_impurity(counts: Sequence[float]) -> float:
    """Compute Gini impurity ``1 - sum_c p_c^2`` from non-negative counts or probabilities."""
    if isinstance(counts, (str, bytes)):
        raise SheafError("counts must be a sequence of non-negative numbers")
    vals = [as_real(c, "count") for c in counts]
    if not vals:
        raise SheafError("gini_impurity requires at least one entry")
    if any(c < 0.0 for c in vals):
        raise SheafError("gini_impurity counts must be non-negative")
    scale = max(vals)
    if scale <= 0.0:
        raise SheafError("gini_impurity total must be strictly positive")
    scaled = [c / scale for c in vals]
    total = math.fsum(scaled)
    return 1.0 - math.fsum((c / total) ** 2 for c in scaled)


def validate_split_disjointness(
    train_ids: Iterable[Any],
    test_ids: Iterable[Any],
    synthetic_ids: Iterable[Any] = (),
) -> dict[str, Any]:
    """Verify train, test, and synthetic oversampled IDs have zero leakage into test_ids."""
    for label, seq in (("train_ids", train_ids), ("test_ids", test_ids), ("synthetic_ids", synthetic_ids)):
        if isinstance(seq, (str, bytes)):
            raise SheafError(f"{label} must be an iterable of IDs, not str/bytes")
    tr = set(train_ids)
    te = set(test_ids)
    syn = set(synthetic_ids)
    if not tr or not te:
        raise SheafError("train_ids and test_ids must both be non-empty")
    train_test_overlap = tr & te
    syn_test_overlap = syn & te
    if train_test_overlap or syn_test_overlap:
        raise SheafError(
            f"split leakage detected into test_ids: "
            f"train&test={len(train_test_overlap)}, synthetic&test={len(syn_test_overlap)}"
        )
    return {
        "ok": True,
        "passed": True,
        "n_train": len(tr),
        "n_test": len(te),
        "n_synthetic": len(syn),
    }
