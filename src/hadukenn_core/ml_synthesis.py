"""Petersen, soulmachine, and Alam ML synthesis guarded by M2 pure-stdlib invariants.

Reuses M2 primitives from ``sheaf_verify``:
- ``SheafError`` (``sheaf_verify.hashing``)
- ``adaboost_exp_bound``, ``bic_score``, ``gini_impurity``, ``jl_min_dimension``,
  ``logsumexp``, ``validate_split_disjointness``, ``verify_elbo_monotonicity``
  (``sheaf_verify.learning_guards``)
- ``SiipVerifyGate`` (``sheaf_verify.rlm_guards``)
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
from scipy.special import gammaln

from sheaf_verify.hashing import SheafError
from sheaf_verify.learning_guards import (
    adaboost_exp_bound,
    bic_score,
    gini_impurity,
    jl_min_dimension,
    logsumexp,
    validate_split_disjointness,
    verify_elbo_monotonicity,
)
from sheaf_verify.rlm_guards import SiipVerifyGate


def _as_finite_array(val: Any, label: str, *, allow_pos_inf: bool = False) -> np.ndarray:
    try:
        arr = np.asarray(val, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SheafError(f"{label} must be a numeric array") from exc
    if arr.size == 0:
        raise SheafError(f"{label} must be non-empty")
    if allow_pos_inf:
        if bool(np.any(np.isnan(arr))) or bool(np.any(np.isneginf(arr))):
            raise SheafError(f"{label} must not contain NaN or -inf")
    else:
        if not bool(np.all(np.isfinite(arr))):
            raise SheafError(f"{label} must contain only finite values")
    return arr


# ==============================================================================
# Petersen (2022) — Diffusion Maps, Normalized Cut, JL Projection, AdaBoost
# ==============================================================================


def compute_diffusion_maps(
    W: Any = None,
    t: int = 2,
    p_dim: int = 4,
    *,
    affinity: Any = None,
    t_diff: int | None = None,
    p: int | None = None,
) -> dict[str, Any]:
    """Diffusion Maps embedding (Petersen Def 20.2-20.3, Prop 20.1-20.2)."""
    w_raw = W if W is not None else affinity
    if w_raw is None:
        raise SheafError("compute_diffusion_maps requires W / affinity")
    w = _as_finite_array(w_raw, "W")
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise SheafError(f"W must be square 2D, got shape {w.shape}")
    if np.any(w < -1e-12):
        raise SheafError("Affinity matrix W must be non-negative")
    w = np.maximum(0.5 * (w + w.T), 0.0)

    n = w.shape[0]
    t_val = int(t_diff if t_diff is not None else t)
    p_val = int(p if p is not None else p_dim)
    if t_val < 1:
        raise SheafError(f"Diffusion time t must be >= 1, got {t_val}")
    if p_val < 1:
        raise SheafError(f"p_dim must be >= 1, got {p_val}")
    if n <= p_val:
        raise SheafError(f"Diffusion Maps requires N > p_dim, got N={n}, p_dim={p_val}")

    deg = np.sum(w, axis=1)
    inv_sqrt = np.zeros(n, dtype=np.float64)
    active = deg > 0.0
    inv_sqrt[active] = 1.0 / np.sqrt(deg[active])

    s_mat = (w * inv_sqrt[:, None]) * inv_sqrt[None, :]
    s_mat = 0.5 * (s_mat + s_mat.T)
    evals, evecs = np.linalg.eigh(s_mat)
    evals = evals[::-1]
    evecs = evecs[:, ::-1]

    lam_k = evals[1 : p_val + 1]
    phi_k = evecs[:, 1 : p_val + 1] * inv_sqrt[:, None]
    embedding = phi_k * (lam_k[None, :] ** t_val)

    return {
        "embedding": embedding,
        "phi": embedding,
        "Phi": embedding,
        "eigenvalues": lam_k,
        "isolated_nodes": int(np.sum(~active)),
    }


diffusion_map_embedding = compute_diffusion_maps
diffusion_maps = compute_diffusion_maps


def normalized_cut_fiedler(
    W: Any = None,
    *,
    affinity: Any = None,
) -> dict[str, Any]:
    """Normalized Cut spectral bipartition via Fiedler vector (Petersen Lecture 19.3)."""
    w_raw = W if W is not None else affinity
    if w_raw is None:
        raise SheafError("normalized_cut_fiedler requires W / affinity")
    w = _as_finite_array(w_raw, "W")
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise SheafError(f"W must be square 2D, got shape {w.shape}")
    if w.shape[0] < 2:
        raise SheafError("normalized_cut_fiedler requires N >= 2")
    if np.any(w < -1e-12):
        raise SheafError("Affinity matrix W must be non-negative")
    w = np.maximum(0.5 * (w + w.T), 0.0)

    n = w.shape[0]
    deg = np.sum(w, axis=1)
    nz_idx = np.nonzero(deg > 0.0)[0]

    fiedler = np.zeros(n, dtype=np.float64)
    if nz_idx.size < 2:
        labels = np.arange(n, dtype=int) % 2
        return {
            "labels": labels,
            "partition": labels,
            "clusters": labels,
            "ncut": 0.0,
            "ncut_val": 0.0,
            "ncut_value": 0.0,
            "fiedler": fiedler,
        }

    w_sub = w[np.ix_(nz_idx, nz_idx)]
    deg_sub = deg[nz_idx]
    inv_sqrt_sub = 1.0 / np.sqrt(deg_sub)
    s_sub = (w_sub * inv_sqrt_sub[:, None]) * inv_sqrt_sub[None, :]
    l_sub = np.eye(nz_idx.size, dtype=np.float64) - 0.5 * (s_sub + s_sub.T)
    _, evecs = np.linalg.eigh(l_sub)
    f_sub = evecs[:, 1] * inv_sqrt_sub
    fiedler[nz_idx] = f_sub

    # Sweep Cheeger thresholds over active nodes to minimize exact Ncut
    order = np.argsort(f_sub)
    sorted_vals = f_sub[order]
    candidates: list[float] = [0.0, float(np.median(f_sub))]
    for i in range(sorted_vals.size - 1):
        if sorted_vals[i + 1] > sorted_vals[i] + 1e-12:
            candidates.append(0.5 * float(sorted_vals[i] + sorted_vals[i + 1]))

    best_labels_sub = (f_sub >= 0.0).astype(int)
    best_ncut = math.inf
    for thresh in candidates:
        cand_sub = (f_sub >= thresh).astype(int)
        if np.all(cand_sub == 0) or np.all(cand_sub == 1):
            continue
        m0 = cand_sub == 0
        m1 = ~m0
        vol0 = float(np.sum(deg_sub[m0]))
        vol1 = float(np.sum(deg_sub[m1]))
        if vol0 <= 0.0 or vol1 <= 0.0:
            continue
        cut_val = float(np.sum(w_sub[np.ix_(m0, m1)]))
        ncut_cand = cut_val * ((1.0 / vol0) + (1.0 / vol1))
        if ncut_cand < best_ncut:
            best_ncut = ncut_cand
            best_labels_sub = cand_sub

    if not math.isfinite(best_ncut):
        best_labels_sub = np.zeros(nz_idx.size, dtype=int)
        best_labels_sub[order[nz_idx.size // 2 :]] = 1
        m0 = best_labels_sub == 0
        m1 = ~m0
        vol0 = float(np.sum(deg_sub[m0]))
        vol1 = float(np.sum(deg_sub[m1]))
        cut_val = float(np.sum(w_sub[np.ix_(m0, m1)]))
        best_ncut = cut_val * ((1.0 / max(vol0, 1e-12)) + (1.0 / max(vol1, 1e-12)))

    labels = np.zeros(n, dtype=int)
    labels[nz_idx] = best_labels_sub
    return {
        "labels": labels,
        "partition": labels,
        "clusters": labels,
        "ncut": float(best_ncut),
        "ncut_val": float(best_ncut),
        "ncut_value": float(best_ncut),
        "fiedler": fiedler,
    }


compute_ncut_fiedler = normalized_cut_fiedler
ncut_fiedler_partition = normalized_cut_fiedler


def jl_project(
    X: Any = None,
    p_dim: int | None = None,
    seed: int = 13,
    *,
    vectors: Any = None,
    p: int | None = None,
    eps: float = 0.25,
) -> dict[str, Any]:
    """Johnson-Lindenstrauss Gaussian random projection (Petersen Thm 20.1)."""
    x_raw = X if X is not None else vectors
    if x_raw is None:
        raise SheafError("jl_project requires X / vectors")
    arr = _as_finite_array(x_raw, "X")
    if arr.ndim != 2:
        raise SheafError(f"X must be 2D, got shape {arr.shape}")
    p_val = int(p if p is not None else (p_dim if p_dim is not None else 0))
    if p_val < 1:
        raise SheafError(f"p_dim must be >= 1, got {p_val}")

    n, d = arr.shape
    rng = np.random.default_rng(int(seed))
    proj_mat = rng.standard_normal((d, p_val)) / math.sqrt(float(p_val))
    z = arr @ proj_mat
    jl_dim_bound = jl_min_dimension(max(n, 2), float(eps))

    return {
        "projected": z,
        "X_proj": z,
        "Z": z,
        "p_dim": p_val,
        "jl_min_dimension": jl_dim_bound,
    }


johnson_lindenstrauss_project = jl_project
jl_compress = jl_project


def adaboost_edge_weights(
    y: Any = None,
    preds: Any = None,
    *,
    labels: Any = None,
    weak_preds: Any = None,
    H: Any = None,
) -> dict[str, Any]:
    """AdaBoost exponential edge/sample reweighting and Z_t bound (Petersen Thm 18.1)."""
    y_raw = y if y is not None else labels
    h_raw = preds if preds is not None else (weak_preds if weak_preds is not None else H)
    if y_raw is None or h_raw is None:
        raise SheafError("adaboost_edge_weights requires y and preds")

    y_arr = _as_finite_array(y_raw, "y").ravel()
    h_arr = _as_finite_array(h_raw, "preds")
    if h_arr.ndim != 2:
        raise SheafError(f"preds must be 2D (T, N), got shape {h_arr.shape}")
    n = y_arr.size
    if h_arr.shape[1] != n:
        if h_arr.shape[0] == n:
            h_arr = h_arr.T
        else:
            raise SheafError(f"preds shape {h_arr.shape} incompatible with y length {n}")
    if not bool(np.all(np.isin(y_arr, (-1.0, 1.0)))):
        raise SheafError("y labels must be in {-1, +1}")
    if not bool(np.all(np.isin(h_arr, (-1.0, 1.0)))):
        raise SheafError("preds must be in {-1, +1}")

    t_rounds = h_arr.shape[0]
    if t_rounds < 1:
        raise SheafError("AdaBoost requires at least 1 weak learner")

    weights = np.full(n, 1.0 / float(n), dtype=np.float64)
    alphas: list[float] = []
    z_factors: list[float] = []
    min_margin = 0.5

    for t_idx in range(t_rounds):
        ht = h_arr[t_idx]
        eps_t = float(np.sum(weights[ht != y_arr]))
        if eps_t > 0.5 + 1e-6 or (t_idx == 0 and eps_t >= 0.5 - 1e-12):
            raise SheafError(
                f"AdaBoost weak learner error eps_t={eps_t:.6f} >= 0.5 violates Petersen Thm 18.1"
            )
        if eps_t >= 0.5 - 1e-9:
            # Reweighted sample saturation at 0.5 after an earlier informative round
            eps_eff = 0.5 - 1e-9
        else:
            eps_eff = max(eps_t, 1e-12)
        margin_t = 0.5 - eps_eff
        min_margin = min(min_margin, margin_t)
        alpha_t = 0.5 * math.log((1.0 - eps_eff) / eps_eff)
        unnorm = weights * np.exp(-alpha_t * y_arr * ht)
        z_t = float(np.sum(unnorm))
        if not math.isfinite(z_t) or z_t <= 0.0:
            raise SheafError("AdaBoost normalization factor Z_t became non-positive")
        weights = unnorm / z_t
        alphas.append(alpha_t)
        z_factors.append(z_t)

    prod_zt = float(math.prod(z_factors))
    gamma_bound = min(max(min_margin, 1e-15), 0.5 - 1e-12)
    exp_theory = adaboost_exp_bound(gamma_bound, t_rounds)
    if prod_zt > exp_theory + 1e-12:
        raise SheafError(
            f"AdaBoost prod_zt={prod_zt:.6f} exceeded Petersen Thm 18.1 bound={exp_theory:.6f}"
        )

    return {
        "alphas": np.asarray(alphas, dtype=np.float64),
        "edge_alphas": np.asarray(alphas, dtype=np.float64),
        "weights": weights,
        "z_factors": np.asarray(z_factors, dtype=np.float64),
        "prod_zt": prod_zt,
        "bound_prod_zt": prod_zt,
        "exp_bound": prod_zt,
        "theoretical_exp_bound": exp_theory,
    }


fit_adaboost_edges = adaboost_edge_weights
adaboost_reweight_edges = adaboost_edge_weights


# ==============================================================================
# soulmachine (2017) — ARD, MI, Fisher, MoE-GEM, BFGS+SIIP, DCM
# ==============================================================================


def ard_rbf_kernel(
    X1: Any = None,
    X2: Any = None,
    length_scales: Any = None,
    *,
    X: Any = None,
    x1: Any = None,
    x: Any = None,
    Y: Any = None,
    x2: Any = None,
    y: Any = None,
    ell: Any = None,
    scales: Any = None,
) -> np.ndarray:
    """ARD squared-exponential kernel (soulmachine Eq. 14.2) with inf-scale feature pruning."""
    a_raw = X1 if X1 is not None else (X if X is not None else (x1 if x1 is not None else x))
    b_raw = X2 if X2 is not None else (Y if Y is not None else (x2 if x2 is not None else y))
    ls_raw = length_scales if length_scales is not None else (ell if ell is not None else scales)
    if a_raw is None or b_raw is None or ls_raw is None:
        raise SheafError("ard_rbf_kernel requires X1, X2, and length_scales")

    a = _as_finite_array(a_raw, "X1")
    b = _as_finite_array(b_raw, "X2")
    ls = _as_finite_array(ls_raw, "length_scales", allow_pos_inf=True).ravel()
    if a.ndim != 2 or b.ndim != 2:
        raise SheafError("X1 and X2 must be 2D arrays")
    if a.shape[1] != b.shape[1] or ls.size != a.shape[1]:
        raise SheafError(f"Dimension mismatch: X1={a.shape}, X2={b.shape}, length_scales={ls.shape}")
    if np.any(ls <= 0.0):
        raise SheafError("length_scales must be strictly positive (or +inf)")

    # Compute scaled differences safely so inf dims drop cleanly and tiny ls (0 < ls < 1e-162) avoid 0 * inf = NaN
    diff = a[:, None, :] - b[None, :, :]
    inf_mask = np.isinf(ls)[None, None, :]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scaled = np.where(inf_mask | (diff == 0.0), 0.0, diff / ls[None, None, :])
        sq = np.sum(scaled * scaled, axis=-1)
    return np.exp(-0.5 * sq)


ard_kernel = ard_rbf_kernel


def _make_mi_bin_edges(arr: np.ndarray, n_bins_val: int) -> np.ndarray | None:
    uniq = np.unique(arr)
    if uniq.size < 2:
        return None
    if uniq.size <= n_bins_val:
        mids = 0.5 * (uniq[:-1] + uniq[1:])
        edges = np.unique(np.concatenate(([uniq[0]], mids, [uniq[-1]])))
        if edges.size >= 2:
            return edges
    q_grid = np.linspace(0.0, 1.0, n_bins_val + 1)
    edges = np.unique(np.quantile(arr, q_grid))
    if edges.size < 2:
        edges = np.unique(np.linspace(float(uniq[0]), float(uniq[-1]), n_bins_val + 1))
    return edges if edges.size >= 2 else None


def binned_mutual_information(x: Any, y: Any, n_bins: int = 8) -> float:
    """Binned mutual information I(X; Y) in bits (soulmachine Eq. 2.54)."""
    x_arr = _as_finite_array(x, "x").ravel()
    y_arr = _as_finite_array(y, "y").ravel()
    if x_arr.size != y_arr.size or x_arr.size < 2:
        raise SheafError("x and y must be 1D arrays of equal length >= 2")
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or int(n_bins) < 2:
        raise SheafError(f"n_bins must be an integer >= 2, got {n_bins!r}")

    n_bins_val = int(n_bins)
    x_edges = _make_mi_bin_edges(x_arr, n_bins_val)
    y_edges = _make_mi_bin_edges(y_arr, n_bins_val)
    if x_edges is None or y_edges is None:
        return 0.0

    hist, _, _ = np.histogram2d(x_arr, y_arr, bins=(x_edges, y_edges))
    total = float(np.sum(hist))
    if total <= 0.0:
        return 0.0
    pxy = hist / total
    px = np.sum(pxy, axis=1, keepdims=True)
    py = np.sum(pxy, axis=0, keepdims=True)
    outer = px @ py
    nz = pxy > 0.0
    mi = float(np.sum(pxy[nz] * np.log2(pxy[nz] / outer[nz])))
    return max(0.0, mi)


mutual_information_binned = binned_mutual_information


def fisher_score_kernel(
    X: Any = None,
    reg: float = 0.1,
    *,
    scores: Any = None,
) -> np.ndarray:
    """Fisher score kernel G F^{-1} G^T (soulmachine Eq. 14.16-14.18) with N < D singularity guard."""
    g_raw = X if X is not None else scores
    if g_raw is None:
        raise SheafError("fisher_score_kernel requires X / scores")
    g = _as_finite_array(g_raw, "X")
    if g.ndim != 2:
        raise SheafError(f"X must be 2D, got shape {g.shape}")
    n, d = g.shape
    reg_f = float(reg)
    if not math.isfinite(reg_f) or reg_f < 0.0:
        raise SheafError(f"reg must be finite and >= 0, got {reg_f}")
    if n < d and reg_f <= 0.0:
        raise SheafError(f"Singular empirical Fisher matrix for N={n} < D={d} with reg={reg_f} <= 0")

    fisher = (g.T @ g) / float(n) + reg_f * np.eye(d, dtype=np.float64)
    if reg_f == 0.0:
        cond = float(np.linalg.cond(fisher))
        if not math.isfinite(cond) or cond > 1e12:
            raise SheafError("Empirical Fisher matrix is singular or ill-conditioned with reg == 0.0")

    try:
        sol = np.linalg.solve(fisher, g.T)
    except np.linalg.LinAlgError as exc:
        raise SheafError("Empirical Fisher matrix solve failed") from exc

    k_mat = g @ sol
    return 0.5 * (k_mat + k_mat.T)


fisher_kernel = fisher_score_kernel


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_v = np.exp(shifted)
    return exp_v / np.sum(exp_v, axis=1, keepdims=True)


def fit_moe_gem(
    X: Any,
    y: Any,
    n_experts: int = 2,
    max_iter: int = 10,
    reg: float = 1e-2,
    *,
    k: int | None = None,
) -> dict[str, Any]:
    """Mixture of Experts trained via Generalized EM with monotone ELBO & BIC (soulmachine Ch. 11)."""
    x_arr = _as_finite_array(X, "X")
    y_arr = _as_finite_array(y, "y").ravel()
    if x_arr.ndim != 2:
        raise SheafError(f"X must be 2D, got shape {x_arr.shape}")
    n, d = x_arr.shape
    if y_arr.size != n:
        raise SheafError(f"y length {y_arr.size} != N={n}")

    k_exp = int(k if k is not None else n_experts)
    if k_exp < 1:
        raise SheafError(f"n_experts must be >= 1, got {k_exp}")
    iters = int(max_iter)
    if iters < 1:
        raise SheafError(f"max_iter must be >= 1, got {iters}")

    phi = np.hstack([x_arr, np.ones((n, 1), dtype=np.float64)])
    d_aug = d + 1
    rng = np.random.default_rng(17)

    # Ridge base fit + small deterministic expert diversity
    ridge_mat = phi.T @ phi + float(reg) * np.eye(d_aug, dtype=np.float64)
    w_base = np.linalg.solve(ridge_mat, phi.T @ y_arr)
    w_experts = np.vstack([w_base + rng.normal(0.0, 0.05, size=d_aug) for _ in range(k_exp)])
    v_gate = rng.normal(0.0, 0.02, size=(d_aug, k_exp))
    sigma2 = np.full(k_exp, max(float(np.var(y_arr)), 1e-2), dtype=np.float64)

    def _e_step(w_cur: np.ndarray, v_cur: np.ndarray, s2_cur: np.ndarray) -> tuple[np.ndarray, float]:
        gate_probs = np.maximum(_softmax_rows(phi @ v_cur), 1e-12)
        log_gate = np.log(gate_probs)
        log_joint = np.empty((n, k_exp), dtype=np.float64)
        for j in range(k_exp):
            res = y_arr - (phi @ w_cur[j])
            log_joint[:, j] = log_gate[:, j] - 0.5 * (
                math.log(2.0 * math.pi * s2_cur[j]) + (res * res) / s2_cur[j]
            )
        lse = np.array([logsumexp(log_joint[i].tolist()) for i in range(n)], dtype=np.float64)
        resp = np.exp(log_joint - lse[:, None])
        resp /= np.sum(resp, axis=1, keepdims=True)
        return resp, float(np.sum(lse))

    resp, cur_elbo = _e_step(w_experts, v_gate, sigma2)
    elbo_trace: list[float] = [cur_elbo]

    for _ in range(iters):
        # M-step 1: Update expert weights & variances; accept only if marginal ELBO does not decrease
        w_cand = w_experts.copy()
        s2_cand = sigma2.copy()
        for j in range(k_exp):
            rj = resp[:, j]
            r_sum = float(np.sum(rj))
            if r_sum <= 1e-8:
                continue
            phi_w = phi * rj[:, None]
            lhs = phi.T @ phi_w + 1e-6 * np.eye(d_aug, dtype=np.float64)
            rhs = phi.T @ (rj * y_arr)
            w_cand[j] = np.linalg.solve(lhs, rhs)
            res_j = y_arr - (phi @ w_cand[j])
            s2_cand[j] = max(float(np.sum(rj * (res_j * res_j)) / r_sum), 1e-4)

        resp_tmp, elbo_tmp = _e_step(w_cand, v_gate, s2_cand)
        if elbo_tmp >= cur_elbo:
            w_experts, sigma2 = w_cand, s2_cand
            resp, cur_elbo = resp_tmp, elbo_tmp

        # M-step 2: Generalized EM gating ascent with backtracking preserving monotonicity
        gate_probs = _softmax_rows(phi @ v_gate)
        grad_v = phi.T @ (resp - gate_probs) - float(reg) * v_gate
        step = 0.05
        for _bt in range(6):
            v_cand = v_gate + step * grad_v
            resp_v, elbo_v = _e_step(w_experts, v_cand, sigma2)
            if elbo_v >= cur_elbo:
                v_gate = v_cand
                resp, cur_elbo = resp_v, elbo_v
                break
            step *= 0.5

        elbo_trace.append(cur_elbo)

    mono = verify_elbo_monotonicity(elbo_trace, tol=1e-12)
    if not mono["passed"]:
        raise SheafError(f"MoE GEM violated ELBO monotonicity: min_delta={mono['min_delta']}")

    dof = k_exp * (2 * d_aug + 1)
    bic_val = bic_score(elbo_trace[-1], dof, n)

    return {
        "elbo_trace": np.asarray(elbo_trace, dtype=np.float64),
        "free_energy_trace": np.asarray(elbo_trace, dtype=np.float64),
        "bic": float(bic_val),
        "bic_score": float(bic_val),
        "responsibilities": resp,
        "weights": w_experts,
        "gating_weights": v_gate,
        "sigma2": sigma2,
    }


MoEGEM = fit_moe_gem
moe_generalized_em = fit_moe_gem


def bfgs_step(
    B_inv: Any = None,
    s: Any = None,
    y: Any = None,
    *,
    H_inv: Any = None,
    delta_k: Any = None,
    y_k: Any = None,
    tol: float = 1e-12,
) -> np.ndarray:
    """Inverse-Hessian BFGS Sherman-Morrison update with strict s^T y > 0 guard."""
    b_raw = B_inv if B_inv is not None else H_inv
    s_raw = s if s is not None else delta_k
    y_raw = y if y is not None else y_k
    if b_raw is None or s_raw is None or y_raw is None:
        raise SheafError("bfgs_step requires B_inv, s, and y")

    b_mat = _as_finite_array(b_raw, "B_inv")
    s_vec = _as_finite_array(s_raw, "s").ravel()
    y_vec = _as_finite_array(y_raw, "y").ravel()
    if b_mat.ndim != 2 or b_mat.shape[0] != b_mat.shape[1]:
        raise SheafError(f"B_inv must be square 2D, got shape {b_mat.shape}")
    if not bool(np.allclose(b_mat, b_mat.T, atol=1e-9)):
        raise SheafError("B_inv must be symmetric in bfgs_step")
    b_sym = 0.5 * (b_mat + b_mat.T)
    if float(np.min(np.linalg.eigvalsh(b_sym))) <= 0.0:
        raise SheafError("B_inv must be symmetric positive definite in bfgs_step")
    d = b_sym.shape[0]
    if s_vec.size != d or y_vec.size != d:
        raise SheafError(f"Dimension mismatch in bfgs_step: B_inv={b_mat.shape}, s={s_vec.shape}, y={y_vec.shape}")

    sy = float(np.dot(s_vec, y_vec))
    if not math.isfinite(sy) or sy <= float(tol):
        raise SheafError(f"BFGS secant curvature condition violated: s^T y = {sy:.6e} <= {tol:.6e}")

    rho = 1.0 / sy
    eye = np.eye(d, dtype=np.float64)
    v_mat = eye - rho * np.outer(s_vec, y_vec)
    b_next = v_mat @ b_sym @ v_mat.T + rho * np.outer(s_vec, s_vec)
    b_next = 0.5 * (b_next + b_next.T)
    if not bool(np.all(np.isfinite(b_next))):
        raise SheafError("bfgs_step produced non-finite B_inv")
    return b_next


bfgs_inverse_hessian_update = bfgs_step
bfgs_update_inverse_hessian = bfgs_step


def bfgs_optimize(
    fn: Callable[[np.ndarray], float] | None = None,
    grad_fn: Callable[[np.ndarray], Any] | None = None,
    x0: Any = None,
    max_iter: int = 25,
    *,
    obj_fn: Callable[[np.ndarray], float] | None = None,
    loss_fn: Callable[[np.ndarray], float] | None = None,
    w0: Any = None,
    tol: float = 1e-8,
    siip_gate: SiipVerifyGate | None = None,
) -> dict[str, Any]:
    """Inverse-Hessian BFGS optimizer with SIIP verification gate and s^T y <= 0 SPD guard."""
    f_fn = fn if fn is not None else (obj_fn if obj_fn is not None else loss_fn)
    x_raw = x0 if x0 is not None else w0
    if f_fn is None or grad_fn is None or x_raw is None:
        raise SheafError("bfgs_optimize requires fn, grad_fn, and x0")
    if int(max_iter) < 1:
        raise SheafError(f"max_iter must be >= 1, got {max_iter}")

    gate = siip_gate if siip_gate is not None else SiipVerifyGate(mode="formal")
    x = _as_finite_array(x_raw, "x0").ravel().copy()
    d = x.size
    b_inv = np.eye(d, dtype=np.float64)

    f_curr = float(f_fn(x))
    g_curr = _as_finite_array(grad_fn(x), "grad").ravel()
    if not math.isfinite(f_curr):
        raise SheafError("Initial objective value is non-finite")

    prev_dir: np.ndarray | None = None
    n_iter = 0

    for it in range(int(max_iter)):
        if float(np.linalg.norm(g_curr)) <= float(tol):
            break
        n_iter = it + 1

        p_dir = -(b_inv @ g_curr)
        if prev_dir is not None:
            p_dir = np.asarray(gate.damp_direction(p_dir.tolist(), prev_dir.tolist()), dtype=np.float64)
        if float(np.dot(g_curr, p_dir)) >= 0.0:
            p_dir = -g_curr

        step_accepted = False
        x_cand = x
        f_cand = f_curr
        alpha = 1.0
        for _ls in range(24):
            trial_x = x + alpha * p_dir
            if not bool(np.all(np.isfinite(trial_x))):
                alpha *= 0.5
                continue
            trial_f = float(f_fn(trial_x))
            if not math.isfinite(trial_f):
                alpha *= 0.5
                continue
            verdict = gate.verify_commit(f_curr, trial_f, trial_x.tolist())
            if verdict["accepted"]:
                x_cand = trial_x
                f_cand = trial_f
                step_accepted = True
                break
            alpha *= 0.5

        if not step_accepted:
            break

        s_vec = x_cand - x
        g_next = _as_finite_array(grad_fn(x_cand), "grad").ravel()
        y_vec = g_next - g_curr

        # Guard negative/zero secant curvature s^T y <= 0 so B_inv remains strictly SPD
        if float(np.dot(s_vec, y_vec)) > 1e-12:
            try:
                b_cand = bfgs_step(b_inv, s_vec, y_vec)
                if float(np.min(np.linalg.eigvalsh(b_cand))) > 1e-12:
                    b_inv = b_cand
            except SheafError:
                pass

        prev_dir = s_vec
        x = x_cand
        f_curr = f_cand
        g_curr = g_next

    return {
        "x": x,
        "w": x,
        "params": x,
        "B_inv": b_inv,
        "H_inv": b_inv,
        "hess_inv": b_inv,
        "fun": float(f_curr),
        "n_iter": n_iter,
        "hesitation_damping_events": gate.hesitation_damping_events,
    }


bfgs_inverse_hessian_minimize = bfgs_optimize


def dcm_burstiness_log_prob(counts: Any, alpha: Any) -> float:
    """Dirichlet-Compound Multinomial / Polya urn log-probability (soulmachine Eq. 3.47)."""
    c = _as_finite_array(counts, "counts").ravel()
    a = _as_finite_array(alpha, "alpha").ravel()
    if c.size != a.size or c.size == 0:
        raise SheafError("counts and alpha must have equal non-zero length")
    if np.any(c < 0.0):
        raise SheafError("counts must be non-negative")
    if np.any(a <= 0.0):
        raise SheafError("alpha concentration parameters must be strictly positive")

    n_tot = float(np.sum(c))
    a_tot = float(np.sum(a))
    log_multinom = float(gammaln(n_tot + 1.0) - np.sum(gammaln(c + 1.0)))
    log_polya = float(gammaln(a_tot) - gammaln(n_tot + a_tot) + np.sum(gammaln(c + a) - gammaln(a)))
    val = log_multinom + log_polya
    if not math.isfinite(val):
        raise SheafError("dcm_burstiness_log_prob produced non-finite output")
    return val


polya_urn_log_prob = dcm_burstiness_log_prob
dcm_log_likelihood = dcm_burstiness_log_prob


# ==============================================================================
# Alam — Train-Only SMOTE, Gini Decision Tree, Random Forest
# ==============================================================================


def train_only_smote(
    X_train: Any = None,
    y_train: Any = None,
    X_test: Any = None,
    train_ids: Iterable[Any] | None = None,
    test_ids: Iterable[Any] | None = None,
    k_neighbors: int = 5,
    seed: int = 13,
    *,
    X: Any = None,
    y: Any = None,
) -> dict[str, Any]:
    """Leakage-free train-only SMOTE oversampling validated with M2 ``validate_split_disjointness``."""
    del X_test  # Never read or mutate held-out test inputs
    x_raw = X_train if X_train is not None else X
    y_raw = y_train if y_train is not None else y
    if x_raw is None or y_raw is None:
        raise SheafError("train_only_smote requires X_train and y_train")

    if train_ids is not None or test_ids is not None:
        tr_ids = train_ids if train_ids is not None else ()
        te_ids = test_ids if test_ids is not None else ()
        validate_split_disjointness(tr_ids, te_ids)

    x_tr = _as_finite_array(x_raw, "X_train")
    y_tr = np.asarray(y_raw)
    if np.issubdtype(y_tr.dtype, np.number):
        if not bool(np.all(np.isfinite(y_tr))):
            raise SheafError("y_train must contain only finite values")
    elif y_tr.dtype == object:
        for val in y_tr.ravel().tolist():
            if isinstance(val, (float, np.floating)) and not math.isfinite(float(val)):
                raise SheafError("y_train must contain only finite values")
    if x_tr.ndim != 2:
        raise SheafError(f"X_train must be 2D, got shape {x_tr.shape}")
    if y_tr.ndim != 1 or y_tr.shape[0] != x_tr.shape[0]:
        raise SheafError("y_train must be 1D with length matching X_train")
    if int(k_neighbors) < 1:
        raise SheafError(f"k_neighbors must be >= 1, got {k_neighbors}")

    classes, counts = np.unique(y_tr, return_counts=True)
    if classes.size < 2:
        raise SheafError("train_only_smote requires at least 2 distinct classes in y_train")

    max_cnt = int(np.max(counts))
    rng = np.random.default_rng(int(seed))
    syn_x_list: list[np.ndarray] = []
    syn_y_list: list[np.ndarray] = []

    for cls, cnt_val in zip(classes.tolist(), counts.tolist(), strict=True):
        cnt = int(cnt_val)
        if cnt >= max_cnt:
            continue
        n_needed = max_cnt - cnt
        x_min = x_tr[y_tr == cls]
        if cnt == 1:
            syn_pts = np.repeat(x_min, n_needed, axis=0)
        else:
            diff = x_min[:, None, :] - x_min[None, :, :]
            dists = np.sum(diff * diff, axis=-1)
            np.fill_diagonal(dists, np.inf)
            k_eff = min(int(k_neighbors), cnt - 1)
            nn_idx = np.argsort(dists, axis=1, kind="mergesort")[:, :k_eff]

            base_picks = rng.integers(0, cnt, size=n_needed)
            nbr_offsets = rng.integers(0, k_eff, size=n_needed)
            nbr_picks = nn_idx[base_picks, nbr_offsets]
            lam = rng.uniform(0.0, 1.0, size=(n_needed, 1))
            syn_pts = x_min[base_picks] + lam * (x_min[nbr_picks] - x_min[base_picks])

        syn_x_list.append(syn_pts)
        syn_y_list.append(np.full(n_needed, cls, dtype=y_tr.dtype))

    if syn_x_list:
        x_bal = np.vstack([x_tr, *syn_x_list])
        y_bal = np.concatenate([y_tr, *syn_y_list])
        n_syn = int(y_bal.size - y_tr.size)
    else:
        x_bal = x_tr.copy()
        y_bal = y_tr.copy()
        n_syn = 0

    return {
        "X_resampled": x_bal,
        "X_train_resampled": x_bal,
        "X": x_bal,
        "y_resampled": y_bal,
        "y_train_resampled": y_bal,
        "y": y_bal,
        "n_synthetic": n_syn,
    }


smote_resample_train = train_only_smote
smote_train_only = train_only_smote


class _TreeNode:
    __slots__ = ("feature", "threshold", "left", "right", "prediction")

    def __init__(
        self,
        *,
        feature: int | None = None,
        threshold: float | None = None,
        left: _TreeNode | None = None,
        right: _TreeNode | None = None,
        prediction: int = 0,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction


class GiniDecisionTree:
    """Binary axis-aligned CART classifier using M2 ``gini_impurity`` (Alam)."""

    def __init__(
        self,
        max_depth: int = 4,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | int | None = None,
        seed: int = 13,
    ) -> None:
        if int(max_depth) < 1:
            raise SheafError(f"max_depth must be >= 1, got {max_depth}")
        if int(min_samples_split) < 2:
            raise SheafError(f"min_samples_split must be >= 2, got {min_samples_split}")
        if int(min_samples_leaf) < 1:
            raise SheafError(f"min_samples_leaf must be >= 1, got {min_samples_leaf}")
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.seed = int(seed)
        self.feature_importances_: np.ndarray = np.array([], dtype=np.float64)
        self._root: _TreeNode | None = None
        self._classes: np.ndarray = np.array([], dtype=int)
        self._n_features: int = 0

    def _select_features(self, n_features: int, rng: np.random.Generator) -> np.ndarray:
        if self.max_features is None:
            return np.arange(n_features, dtype=int)
        if self.max_features == "sqrt":
            k = max(1, int(math.sqrt(n_features)))
        elif self.max_features == "log2":
            k = max(1, int(math.log2(n_features)))
        else:
            k = min(n_features, max(1, int(self.max_features)))
        if k >= n_features:
            return np.arange(n_features, dtype=int)
        return np.sort(rng.choice(n_features, size=k, replace=False))

    def fit(self, X: Any, y: Any) -> GiniDecisionTree:
        x_arr = _as_finite_array(X, "X")
        y_raw = np.asarray(y)
        if x_arr.ndim != 2 or y_raw.ndim != 1 or x_arr.shape[0] != y_raw.shape[0]:
            raise SheafError("X (N, D) and y (N,) shape mismatch in GiniDecisionTree.fit")

        classes, y_idx = np.unique(y_raw, return_inverse=True)
        self._classes = classes
        n_classes = int(classes.size)
        n_features = x_arr.shape[1]
        self._n_features = int(n_features)
        raw_imp = np.zeros(n_features, dtype=np.float64)
        rng = np.random.default_rng(self.seed)

        def _node_gini(counts: np.ndarray) -> float:
            pos = [float(c) for c in counts.tolist() if c > 0]
            return gini_impurity(pos) if pos else 0.0

        def _build(x_sub: np.ndarray, y_sub: np.ndarray, depth: int) -> _TreeNode:
            counts = np.bincount(y_sub, minlength=n_classes)
            majority = int(np.argmax(counts))
            n_node = int(y_sub.size)
            if depth >= self.max_depth or n_node < self.min_samples_split or np.sum(counts > 0) <= 1:
                return _TreeNode(prediction=majority)

            parent_gini = _node_gini(counts)
            best_gain = 0.0
            best_feat: int | None = None
            best_thresh = 0.0

            feat_candidates = self._select_features(n_features, rng)
            for feat in feat_candidates.tolist():
                col = x_sub[:, feat]
                order = np.argsort(col, kind="mergesort")
                col_s = col[order]
                y_s = y_sub[order]

                left_counts = np.zeros(n_classes, dtype=np.int64)
                right_counts = counts.copy()
                for i in range(n_node - 1):
                    c_lbl = int(y_s[i])
                    left_counts[c_lbl] += 1
                    right_counts[c_lbl] -= 1
                    n_l = i + 1
                    n_r = n_node - n_l
                    if col_s[i] == col_s[i + 1]:
                        continue
                    if n_l < self.min_samples_leaf or n_r < self.min_samples_leaf:
                        continue
                    g_l = _node_gini(left_counts)
                    g_r = _node_gini(right_counts)
                    gain = parent_gini - ((n_l * g_l + n_r * g_r) / float(n_node))
                    if gain > best_gain:
                        best_gain = gain
                        best_feat = feat
                        best_thresh = 0.5 * float(col_s[i] + col_s[i + 1])

            if best_feat is None or best_gain <= 1e-12:
                return _TreeNode(prediction=majority)

            raw_imp[best_feat] += float(n_node) * best_gain
            left_mask = x_sub[:, best_feat] <= best_thresh
            return _TreeNode(
                feature=best_feat,
                threshold=best_thresh,
                left=_build(x_sub[left_mask], y_sub[left_mask], depth + 1),
                right=_build(x_sub[~left_mask], y_sub[~left_mask], depth + 1),
                prediction=majority,
            )

        self._root = _build(x_arr, y_idx, 0)
        tot_imp = float(np.sum(raw_imp))
        if tot_imp > 0.0:
            self.feature_importances_ = raw_imp / tot_imp
        else:
            self.feature_importances_ = np.full(n_features, 1.0 / float(n_features), dtype=np.float64)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self._root is None:
            raise SheafError("GiniDecisionTree must be fit before calling predict")
        x_arr = _as_finite_array(X, "X")
        if x_arr.ndim != 2:
            raise SheafError("X must be 2D in GiniDecisionTree.predict")
        if x_arr.shape[1] != self._n_features:
            raise SheafError(
                f"Feature count mismatch in GiniDecisionTree.predict: "
                f"expected {self._n_features}, got {x_arr.shape[1]}"
            )

        preds = np.empty(x_arr.shape[0], dtype=self._classes.dtype)
        for i in range(x_arr.shape[0]):
            node = self._root
            row = x_arr[i]
            while node.feature is not None and node.threshold is not None:
                node = node.left if row[node.feature] <= node.threshold else node.right  # type: ignore[assignment]
            preds[i] = self._classes[node.prediction]
        return preds


def fit_gini_tree(
    X: Any,
    y: Any,
    *,
    max_depth: int = 4,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    seed: int = 13,
) -> GiniDecisionTree:
    """Fit and return a :class:`GiniDecisionTree`."""
    return GiniDecisionTree(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        seed=seed,
    ).fit(X, y)


class RandomForestClassifierLite:
    """Lightweight bootstrap Random Forest over :class:`GiniDecisionTree` (Alam)."""

    def __init__(
        self,
        n_estimators: int = 15,
        max_depth: int = 4,
        max_features: str | int | None = "sqrt",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        seed: int = 13,
    ) -> None:
        if int(n_estimators) < 1:
            raise SheafError(f"n_estimators must be >= 1, got {n_estimators}")
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.max_features = max_features
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.seed = int(seed)
        self.trees_: list[GiniDecisionTree] = []
        self.feature_importances_: np.ndarray = np.array([], dtype=np.float64)
        self._classes: np.ndarray = np.array([], dtype=int)
        self._n_features: int = 0

    def fit(self, X: Any, y: Any) -> RandomForestClassifierLite:
        x_arr = _as_finite_array(X, "X")
        y_raw = np.asarray(y)
        if x_arr.ndim != 2 or y_raw.ndim != 1 or x_arr.shape[0] != y_raw.shape[0]:
            raise SheafError("X (N, D) and y (N,) shape mismatch in RandomForestClassifierLite.fit")

        self._classes = np.unique(y_raw)
        n, d = x_arr.shape
        self._n_features = int(d)
        rng = np.random.default_rng(self.seed)
        self.trees_ = []
        imp_sum = np.zeros(d, dtype=np.float64)

        for idx in range(self.n_estimators):
            boot = rng.integers(0, n, size=n)
            tree = GiniDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                seed=self.seed + idx + 1,
            ).fit(x_arr[boot], y_raw[boot])
            self.trees_.append(tree)
            imp_sum += tree.feature_importances_

        tot = float(np.sum(imp_sum))
        self.feature_importances_ = (
            imp_sum / tot if tot > 0.0 else np.full(d, 1.0 / float(d), dtype=np.float64)
        )
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not self.trees_:
            raise SheafError("RandomForestClassifierLite must be fit before calling predict")
        x_arr = _as_finite_array(X, "X")
        if x_arr.ndim != 2:
            raise SheafError("X must be 2D in RandomForestClassifierLite.predict")
        if x_arr.shape[1] != self._n_features:
            raise SheafError(
                f"Feature count mismatch in RandomForestClassifierLite.predict: "
                f"expected {self._n_features}, got {x_arr.shape[1]}"
            )
        votes = np.vstack([t.predict(x_arr) for t in self.trees_])  # (n_estimators, n_samples)
        preds = np.empty(x_arr.shape[0], dtype=self._classes.dtype)
        for j in range(x_arr.shape[0]):
            vals, cnts = np.unique(votes[:, j], return_counts=True)
            preds[j] = vals[int(np.argmax(cnts))]
        return preds


def fit_random_forest(
    X: Any,
    y: Any,
    *,
    n_estimators: int = 15,
    max_depth: int = 4,
    max_features: str | int | None = "sqrt",
    seed: int = 13,
) -> RandomForestClassifierLite:
    """Fit and return a :class:`RandomForestClassifierLite`."""
    return RandomForestClassifierLite(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        seed=seed,
    ).fit(X, y)


def compute_rademacher_and_tikhonov_certificates(
    lam: float,
    trace_k: float,
    m: int,
    rho: float = 1.0,
) -> dict[str, float]:
    """Convenience bridge exposing Petersen Rademacher & Tikhonov certificates."""
    from sheaf_verify.learning_guards import rademacher_kernel_bound, tikhonov_stability_rate

    return {
        "rademacher_kernel_bound": rademacher_kernel_bound(lam, trace_k, m),
        "tikhonov_stability_rate": tikhonov_stability_rate(rho, max(lam, 1e-12), m),
    }


__all__ = [
    "GiniDecisionTree",
    "MoEGEM",
    "RandomForestClassifierLite",
    "adaboost_edge_weights",
    "adaboost_reweight_edges",
    "ard_kernel",
    "ard_rbf_kernel",
    "bfgs_inverse_hessian_minimize",
    "bfgs_inverse_hessian_update",
    "bfgs_optimize",
    "bfgs_step",
    "bfgs_update_inverse_hessian",
    "binned_mutual_information",
    "compute_diffusion_maps",
    "compute_ncut_fiedler",
    "compute_rademacher_and_tikhonov_certificates",
    "dcm_burstiness_log_prob",
    "dcm_log_likelihood",
    "diffusion_map_embedding",
    "diffusion_maps",
    "fisher_kernel",
    "fisher_score_kernel",
    "fit_adaboost_edges",
    "fit_gini_tree",
    "fit_moe_gem",
    "fit_random_forest",
    "jl_compress",
    "jl_project",
    "johnson_lindenstrauss_project",
    "moe_generalized_em",
    "mutual_information_binned",
    "ncut_fiedler_partition",
    "normalized_cut_fiedler",
    "polya_urn_log_prob",
    "smote_resample_train",
    "smote_train_only",
    "train_only_smote",
]
