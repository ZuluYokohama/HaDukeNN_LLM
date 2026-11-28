"""Structured and memory-frugal LLM output heads with analytical BLAS/AVX-512 gradients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hadukenn_core.fast_ops import get_fast_head_lib, run_tiled_cce_avx512
from hadukenn_core.partitions import ClusterPartition


def fast_scatter_add_2d(out: np.ndarray, indices: np.ndarray, values: np.ndarray) -> None:
    """BLAS-accelerated `np.add.at(out, indices, values)` for 2D `[M, D]` tables."""
    m_rows = out.shape[0]
    n_items = indices.shape[0]
    if n_items == 0:
        return
    if m_rows <= 8192:
        sel = np.zeros((m_rows, n_items), dtype=values.dtype)
        sel[indices, np.arange(n_items)] = 1.0
        out += sel @ values
    else:
        np.add.at(out, indices, values)


def _log_softmax(x: np.ndarray, axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    mx = np.max(x, axis=axis, keepdims=True)
    shifted = x - mx
    ex = np.exp(shifted)
    sum_ex = np.sum(ex, axis=axis, keepdims=True)
    sm = ex / sum_ex
    lsm = shifted - np.log(sum_ex)
    return lsm, sm


@dataclass
class CountShrunkSpanComposer:
    """
    H-Fix 1 (`#47`): Variance-preserving `1 / sqrt(k_s)` base-token sum plus
    Empirical-Bayes count-shrunk span residual `gamma_s = n_s / (n_s + kappa)`.
    """

    span_parts: list[tuple[int, ...]]
    span_counts: np.ndarray
    kappa: float = 400.0

    def __post_init__(self) -> None:
        counts = np.maximum(self.span_counts.astype(np.float32), 0.0)
        self.gamma = counts / (counts + float(self.kappa))
        self.inv_sqrt_k = np.array(
            [1.0 / (len(p) ** 0.5) for p in self.span_parts],
            dtype=np.float32,
        )

    def compose(self, E_base: np.ndarray, delta_span: np.ndarray) -> tuple[np.ndarray, float]:
        k_span, d = delta_span.shape
        if k_span == 0:
            return E_base, 0.0
        comp = np.zeros((k_span, d), dtype=E_base.dtype)
        for s_idx, parts in enumerate(self.span_parts):
            comp[s_idx] = self.inv_sqrt_k[s_idx] * np.sum(E_base[list(parts)], axis=0)
        scaled_res = self.gamma[:, None] * delta_span
        u_span = comp + scaled_res
        comp_energy = np.sum(comp * comp, axis=1)
        res_energy = np.sum(scaled_res * scaled_res, axis=1)
        share = float(np.mean(res_energy / np.maximum(comp_energy + res_energy, 1e-12)))
        return np.vstack([E_base, u_span]), share

    def backward(
        self,
        dU_full: np.ndarray,
        v_base: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        dE_base = dU_full[:v_base].copy()
        dU_span = dU_full[v_base:]
        d_delta = self.gamma[:, None] * dU_span
        for s_idx, parts in enumerate(self.span_parts):
            g = self.inv_sqrt_k[s_idx] * dU_span[s_idx]
            for b in parts:
                dE_base[b] += g
        return dE_base, d_delta


def flat_tied_fwd_bwd(
    H: np.ndarray,
    E: np.ndarray,
    targets: np.ndarray,
    compute_grad: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    logits = H @ E.T
    lsm, sm = _log_softmax(logits, axis=1)
    n = H.shape[0]
    idx = np.arange(n)
    loss = -lsm[idx, targets]
    if not compute_grad:
        return loss, None, None
    grad_logits = sm
    grad_logits[idx, targets] -= 1.0
    grad_logits *= 1.0 / float(n)
    dH = grad_logits @ E
    dE = grad_logits.T @ H
    return loss, dH, dE


def tiled_cce_fwd_bwd(
    H: np.ndarray,
    E: np.ndarray,
    targets: np.ndarray,
    tile_V: int = 256,
    compute_grad: bool = True,
    use_native: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if use_native and get_fast_head_lib() is not None and H.dtype == np.float32 and E.dtype == np.float32:
        return run_tiled_cce_avx512(H, E, targets, tile_V=tile_V, compute_grad=compute_grad)

    n, _ = H.shape
    v_total = E.shape[0]
    m_run = np.full(n, -1e30, dtype=np.float64)
    s_run = np.zeros(n, dtype=np.float64)
    z_y = np.sum(H * E[targets], axis=1, dtype=np.float64)

    for v0 in range(0, v_total, tile_V):
        v1 = min(v0 + tile_V, v_total)
        z_tile = H.astype(np.float64) @ E[v0:v1].astype(np.float64).T
        m_tile = np.max(z_tile, axis=1)
        m_new = np.maximum(m_run, m_tile)
        s_run = s_run * np.exp(m_run - m_new) + np.sum(np.exp(z_tile - m_new[:, None]), axis=1)
        m_run = m_new

    lse = m_run + np.log(s_run)
    loss = (lse - z_y).astype(H.dtype)
    if not compute_grad:
        return loss, None, None

    dH = np.zeros_like(H, dtype=np.float64)
    dE = np.zeros_like(E, dtype=np.float64)
    inv_n = 1.0 / float(n)
    for v0 in range(0, v_total, tile_V):
        v1 = min(v0 + tile_V, v_total)
        e_tile = E[v0:v1].astype(np.float64)
        z_tile = H.astype(np.float64) @ e_tile.T
        p_tile = np.exp(z_tile - lse[:, None]) * inv_n
        in_tile = (targets >= v0) & (targets < v1)
        if np.any(in_tile):
            rows = np.nonzero(in_tile)[0]
            cols = targets[rows] - v0
            p_tile[rows, cols] -= inv_n
        dH += p_tile @ e_tile
        dE[v0:v1] += p_tile.T @ H.astype(np.float64)

    return loss, dH.astype(H.dtype), dE.astype(E.dtype)


def factored_fp2_fwd_bwd(
    H: np.ndarray,
    E: np.ndarray,
    delta_root: np.ndarray,
    partition: ClusterPartition,
    targets: np.ndarray,
    compute_grad: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    n, d = H.shape
    nc, s_sz = partition.n_clusters, partition.cluster_size
    cu = partition.cluster_units
    tc = partition.token_to_cluster[targets]
    tm = partition.token_to_member[targets]
    idx = np.arange(n)

    E_cl = E[cu]
    C_root = np.mean(E_cl, axis=1) + delta_root
    root_logits = H @ C_root.T
    root_lsm, root_sm = _log_softmax(root_logits, axis=1)

    E_tc = E_cl[tc]
    intra_logits = (E_tc @ H[:, :, None]).squeeze(-1)
    intra_lsm, intra_sm = _log_softmax(intra_logits, axis=1)

    loss = -(root_lsm[idx, tc] + intra_lsm[idx, tm])
    if not compute_grad:
        return loss, None, None, None

    inv_n = 1.0 / float(n)
    g_root = root_sm
    g_root[idx, tc] -= 1.0
    g_root *= inv_n

    g_intra = intra_sm
    g_intra[idx, tm] -= 1.0
    g_intra *= inv_n

    dH = g_root @ C_root + (g_intra[:, None, :] @ E_tc).squeeze(1)
    dC_root = g_root.T @ H
    d_delta_root = dC_root.copy()

    dE_cl_flat = np.broadcast_to((dC_root / float(s_sz))[:, None, :], (nc, s_sz, d)).reshape(nc, s_sz * d).copy()
    contrib_flat = (g_intra[:, :, None] * H[:, None, :]).reshape(n, s_sz * d)
    fast_scatter_add_2d(dE_cl_flat, tc, contrib_flat)

    dE = np.empty_like(E)
    dE[cu] = dE_cl_flat.reshape(nc, s_sz, d)
    return loss, dH, dE, d_delta_root


def cluster_is_fwd_bwd(
    H: np.ndarray,
    E: np.ndarray,
    delta_root: np.ndarray,
    partition: ClusterPartition,
    targets: np.ndarray,
    k_neg: int = 3,
    rng: np.random.Generator | None = None,
    compute_grad: bool = True,
) -> tuple[np.ndarray, float, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if rng is None:
        rng = np.random.default_rng(0)
    n, d = H.shape
    nc, s_sz = partition.n_clusters, partition.cluster_size
    k_eff = min(k_neg, nc - 1)
    cu = partition.cluster_units
    tc = partition.token_to_cluster[targets]
    tm = partition.token_to_member[targets]
    idx = np.arange(n)

    E_cl = E[cu]
    C_root = np.mean(E_cl, axis=1) + delta_root

    root_logits = H @ C_root.T
    root_lsm, root_sm = _log_softmax(root_logits, axis=1)
    root_nll = float(-np.mean(root_lsm[idx, tc]))

    eps_smooth = 0.15
    q_cond = root_sm.copy()
    q_cond[idx, tc] = 0.0
    row_sum = np.maximum(np.sum(q_cond, axis=1, keepdims=True), 1e-8)
    q_cond /= row_sum
    unif_other = 1.0 / float(nc - 1)
    q_tilde = (1.0 - eps_smooth) * q_cond + eps_smooth * unif_other
    q_tilde[idx, tc] = 0.0
    q_tilde /= np.sum(q_tilde, axis=1, keepdims=True)

    u_noise = np.clip(rng.random((n, nc), dtype=np.float32), 1e-7, 1.0 - 1e-7)
    gumbel = np.log(np.maximum(q_tilde, 1e-12)) - np.log(-np.log(u_noise))
    gumbel[idx, tc] = -1e30
    if k_eff >= nc - 1:
        neg_c = np.argpartition(-gumbel, kth=k_eff - 1, axis=1)[:, :k_eff]
        tau = None
    else:
        top_kp1 = np.argpartition(-gumbel, kth=k_eff, axis=1)[:, : k_eff + 1]
        top_vals = np.take_along_axis(gumbel, top_kp1, axis=1)
        order = np.argsort(-top_vals, axis=1)
        top_sorted = np.take_along_axis(top_kp1, order, axis=1)
        vals_sorted = np.take_along_axis(top_vals, order, axis=1)
        neg_c = top_sorted[:, :k_eff]
        tau = vals_sorted[:, k_eff : k_eff + 1]

    active_c = np.concatenate([tc[:, None], neg_c], axis=1)
    active_cols = (1 + k_eff) * s_sz
    E_act_flat = E_cl[active_c].reshape(n, active_cols, d)
    flat_logits = (E_act_flat @ H[:, :, None]).squeeze(-1)

    if tau is None:
        log_w_neg = np.zeros((n, k_eff), dtype=flat_logits.dtype)
    else:
        q_neg = np.take_along_axis(q_tilde, neg_c, axis=1)
        log_x = np.log(np.maximum(q_neg, 1e-12)).astype(np.float64) - tau.astype(np.float64)
        p_incl = -np.expm1(-np.exp(np.clip(log_x, -80.0, 40.0)))
        log_w_neg = (-np.log(np.maximum(p_incl, 1e-12))).astype(flat_logits.dtype)

    blk_view = flat_logits.reshape(n, 1 + k_eff, s_sz)
    blk_view[:, 1:, :] += log_w_neg[:, :, None]

    lsm, sm = _log_softmax(flat_logits, axis=1)
    loss = -lsm[idx, tm]

    if not compute_grad:
        return loss, root_nll, None, None, None

    inv_n = 1.0 / float(n)
    g_flat = sm
    g_flat[idx, tm] -= 1.0
    g_flat *= inv_n

    dH = (g_flat[:, None, :] @ E_act_flat).squeeze(1)
    g_blk = g_flat.reshape(n, 1 + k_eff, s_sz)
    dE_cl_flat = np.zeros((nc, s_sz * d), dtype=E.dtype)
    contrib = (g_blk[:, :, :, None] * H[:, None, None, :]).reshape(n, 1 + k_eff, s_sz * d)
    for blk in range(1 + k_eff):
        fast_scatter_add_2d(dE_cl_flat, active_c[:, blk], contrib[:, blk])

    dE = np.empty_like(E)
    dE[cu] = dE_cl_flat.reshape(nc, s_sz, d)

    g_root = root_sm.copy()
    g_root[idx, tc] -= 1.0
    g_root *= inv_n
    d_delta_root = g_root.T @ H
    return loss, root_nll, dH, dE, d_delta_root


def cluster_routed_eval_nll(
    H: np.ndarray,
    E: np.ndarray,
    delta_root: np.ndarray,
    partition: ClusterPartition,
    targets: np.ndarray,
    top_m_clusters: int = 8,
    dispersion_diag: np.ndarray | None = None,
) -> np.ndarray:
    n, d = H.shape
    nc, s_sz = partition.n_clusters, partition.cluster_size
    cu = partition.cluster_units
    E_cl = E[cu]
    mean_c = np.mean(E_cl, axis=1)
    if dispersion_diag is None:
        dispersion_diag = np.var(E_cl, axis=1)

    cumulant_scores = H @ (mean_c + 0.25 * delta_root).T + 0.5 * ((H * H) @ dispersion_diag.T)
    m_keep = min(top_m_clusters, nc)
    top_c = np.argpartition(-cumulant_scores, kth=m_keep - 1, axis=1)[:, :m_keep]

    E_top = E_cl[top_c].reshape(n, m_keep * s_sz, d)
    exact_dots = (E_top @ H[:, :, None]).squeeze(-1)
    mask_rem = np.ones((n, nc), dtype=bool)
    rows = np.arange(n)[:, None]
    mask_rem[rows, top_c] = False
    rem_scores = np.where(mask_rem, cumulant_scores + np.log(float(s_sz)), -1e30)

    all_terms = np.concatenate([exact_dots, rem_scores], axis=1)
    mx = np.max(all_terms, axis=1)
    lse = mx + np.log(np.sum(np.exp(all_terms - mx[:, None]), axis=1))
    z_y = np.sum(H * E[targets], axis=1)
    return np.maximum(lse - z_y, 0.0)


@dataclass(frozen=True)
class TiedRootShortcutLayout:
    vocab_size: int
    n_root_tokens: int
    n_tail_clusters: int
    tail_cluster_size: int
    root_tokens: np.ndarray
    tail_units: np.ndarray
    is_root_token: np.ndarray
    token_root_slot: np.ndarray
    token_tail_cluster: np.ndarray
    token_tail_member: np.ndarray


def build_tied_root_shortcut_layout(
    vocab_size: int,
    n_root_tokens: int,
    n_tail_clusters: int,
    tail_cluster_size: int,
    train_streams: list[np.ndarray],
) -> TiedRootShortcutLayout:
    if n_root_tokens + n_tail_clusters * tail_cluster_size != vocab_size:
        raise ValueError("Require R_root + NC_tail * S_tail == V")
    counts = np.zeros(vocab_size, dtype=np.int64)
    for seq in train_streams:
        if seq.size > 0:
            np.add.at(counts, seq, 1)
    order = np.argsort(-counts, kind="mergesort")
    root_tokens = np.sort(order[:n_root_tokens]).astype(np.int32)
    tail_pool = np.sort(order[n_root_tokens:]).astype(np.int32)

    tail_units = tail_pool.reshape(n_tail_clusters, tail_cluster_size).copy()
    is_root = np.zeros(vocab_size, dtype=bool)
    is_root[root_tokens] = True
    root_slot = np.zeros(vocab_size, dtype=np.int32)
    root_slot[root_tokens] = np.arange(n_root_tokens, dtype=np.int32)
    tail_c = np.full(vocab_size, -1, dtype=np.int32)
    tail_m = np.zeros(vocab_size, dtype=np.int32)
    for c in range(n_tail_clusters):
        members = tail_units[c]
        root_slot[members] = n_root_tokens + c
        tail_c[members] = c
        tail_m[members] = np.arange(tail_cluster_size, dtype=np.int32)

    return TiedRootShortcutLayout(
        vocab_size=vocab_size,
        n_root_tokens=n_root_tokens,
        n_tail_clusters=n_tail_clusters,
        tail_cluster_size=tail_cluster_size,
        root_tokens=root_tokens,
        tail_units=tail_units,
        is_root_token=is_root,
        token_root_slot=root_slot,
        token_tail_cluster=tail_c,
        token_tail_member=tail_m,
    )


def tied_root_shortcut_fwd_bwd(
    H: np.ndarray,
    E: np.ndarray,
    delta_tail: np.ndarray,
    layout: TiedRootShortcutLayout,
    targets: np.ndarray,
    compute_grad: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    n, d = H.shape
    r_num = layout.n_root_tokens
    nc_t = layout.n_tail_clusters
    s_t = layout.tail_cluster_size
    idx = np.arange(n)

    E_root_toks = E[layout.root_tokens]
    E_tail_cl = E[layout.tail_units]
    C_tail = np.mean(E_tail_cl, axis=1) + delta_tail
    W_root = np.vstack([E_root_toks, C_tail])

    root_logits = H @ W_root.T
    root_lsm, root_sm = _log_softmax(root_logits, axis=1)
    r_slot = layout.token_root_slot[targets]
    loss = -root_lsm[idx, r_slot]

    in_tail = ~layout.is_root_token[targets]
    tc_safe = np.maximum(layout.token_tail_cluster[targets], 0)
    tm_safe = layout.token_tail_member[targets]
    E_tc = E_tail_cl[tc_safe]
    tail_logits = (E_tc @ H[:, :, None]).squeeze(-1)
    tail_lsm, tail_sm = _log_softmax(tail_logits, axis=1)
    loss = loss + np.where(in_tail, -tail_lsm[idx, tm_safe], 0.0)

    if not compute_grad:
        return loss, None, None, None

    inv_n = 1.0 / float(n)
    g_root = root_sm
    g_root[idx, r_slot] -= 1.0
    g_root *= inv_n

    g_tail = tail_sm
    g_tail[idx, tm_safe] -= 1.0
    g_tail *= (in_tail.astype(H.dtype) * inv_n)[:, None]

    dH = g_root @ W_root + (g_tail[:, None, :] @ E_tc).squeeze(1)
    dW_root = g_root.T @ H
    dE_root_toks = dW_root[:r_num]
    dC_tail = dW_root[r_num:]
    d_delta_tail = dC_tail.copy()

    dE_tail_flat = np.broadcast_to((dC_tail / float(s_t))[:, None, :], (nc_t, s_t, d)).reshape(nc_t, s_t * d).copy()
    contrib_flat = (g_tail[:, :, None] * H[:, None, :]).reshape(n, s_t * d)
    fast_scatter_add_2d(dE_tail_flat, tc_safe, contrib_flat)

    dE = np.empty_like(E)
    dE[layout.root_tokens] = dE_root_toks
    dE[layout.tail_units] = dE_tail_flat.reshape(nc_t, s_t, d)
    return loss, dH, dE, d_delta_tail
