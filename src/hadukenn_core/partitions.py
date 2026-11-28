"""Exact-bijection spectral partitioners (`NC * S == V`): mass-random, syntagmatic, and paradigmatic."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import svds


@dataclass(frozen=True)
class ClusterPartition:
    """Verified zero-pad bijection `cluster_units[NC, S]` over `0..V-1`."""

    mode: str
    vocab_size: int
    n_clusters: int
    cluster_size: int
    cluster_units: np.ndarray  # [NC, S] int32 exact permutation
    token_to_cluster: np.ndarray  # [V] int32
    token_to_member: np.ndarray  # [V] int32
    cluster_entropy_bits: float
    cluster_prob_max: float

    def verify_bijection(self, max_p1_cluster_size: int = 181) -> dict[str, object]:
        flat = self.cluster_units.reshape(-1)
        is_perm = bool(
            self.cluster_units.shape == (self.n_clusters, self.cluster_size)
            and self.n_clusters * self.cluster_size == self.vocab_size
            and np.array_equal(np.sort(flat), np.arange(self.vocab_size, dtype=np.int32))
        )
        # Check inverse lookup tables
        reconstructed = self.cluster_units[self.token_to_cluster, self.token_to_member]
        inv_ok = bool(np.array_equal(reconstructed, np.arange(self.vocab_size, dtype=np.int32)))
        return {
            "mode": self.mode,
            "bijection_ok": is_perm and inv_ok,
            "zero_pad_slots": int(self.n_clusters * self.cluster_size - self.vocab_size) == 0,
            "p1_size_gate_ok": self.cluster_size <= max_p1_cluster_size,
            "cluster_size_min": self.cluster_size,
            "cluster_size_max": self.cluster_size,
            "cluster_entropy_bits": round(self.cluster_entropy_bits, 6),
            "cluster_prob_max": round(self.cluster_prob_max, 6),
        }


def _balanced_assign_from_features(
    feats: np.ndarray,
    counts: np.ndarray,
    n_clusters: int,
    cluster_size: int,
    rng: np.random.Generator,
    n_iters: int = 12,
) -> np.ndarray:
    """
    Capacity-constrained balanced spherical k-means guaranteeing every cluster
    receives `cluster_size` tokens (`NC * S == V`, zero `-1` sentinels).
    """
    v_size, dim = feats.shape
    if n_clusters * cluster_size != v_size:
        raise ValueError(f"NC*S ({n_clusters}*{cluster_size}) != V ({v_size})")

    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    x = feats / np.maximum(norms, 1e-8)

    # Frequency-weighted k-means++ initialization
    weights = np.maximum(counts.astype(np.float64), 1.0)
    weights /= weights.sum()
    centroids = np.empty((n_clusters, dim), dtype=np.float32)
    first = int(rng.choice(v_size, p=weights))
    centroids[0] = x[first]
    min_dist_sq = np.sum((x - centroids[0]) ** 2, axis=1)
    for c in range(1, n_clusters):
        probs = weights * np.maximum(min_dist_sq, 1e-12)
        probs_sum = float(probs.sum())
        idx = int(rng.choice(v_size, p=probs / probs_sum)) if probs_sum > 0 else int(rng.integers(0, v_size))
        centroids[c] = x[idx]
        dist_sq = np.sum((x - centroids[c]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)

    cluster_units = np.empty((n_clusters, cluster_size), dtype=np.int32)
    for _ in range(n_iters):
        sim = x @ centroids.T  # [V, NC]
        # Order tokens by margin between best and second-best cluster so contested tokens pick first
        part = np.partition(sim, -2, axis=1)
        margin = part[:, -1] - part[:, -2]
        order = np.argsort(-margin * np.log1p(counts + 1.0))

        caps = np.zeros(n_clusters, dtype=np.int32)
        for u in order.tolist():
            row = sim[u]
            ranked_c = np.argsort(-row)
            for c in ranked_c.tolist():
                if caps[c] < cluster_size:
                    cluster_units[c, caps[c]] = u
                    caps[c] += 1
                    break

        # Update normalized centroids
        for c in range(n_clusters):
            members = cluster_units[c]
            w = np.sqrt(np.maximum(counts[members], 1.0))[:, None].astype(np.float32)
            mean_vec = np.sum(x[members] * w, axis=0)
            norm = float(np.linalg.norm(mean_vec))
            if norm > 1e-8:
                centroids[c] = mean_vec / norm

    # Deterministic sort inside each cluster
    for c in range(n_clusters):
        cluster_units[c] = np.sort(cluster_units[c])
    return cluster_units


def _finalize_partition(
    mode: str,
    vocab_size: int,
    n_clusters: int,
    cluster_size: int,
    cluster_units: np.ndarray,
    counts: np.ndarray,
) -> ClusterPartition:
    tok_to_c = np.empty(vocab_size, dtype=np.int32)
    tok_to_m = np.empty(vocab_size, dtype=np.int32)
    for c in range(n_clusters):
        members = cluster_units[c]
        tok_to_c[members] = c
        tok_to_m[members] = np.arange(cluster_size, dtype=np.int32)

    total = max(float(counts.sum()), 1.0)
    c_mass = np.array([float(counts[cluster_units[c]].sum()) / total for c in range(n_clusters)], dtype=np.float64)
    entropy = -sum(p * math.log2(p) for p in c_mass if p > 0.0)
    return ClusterPartition(
        mode=mode,
        vocab_size=vocab_size,
        n_clusters=n_clusters,
        cluster_size=cluster_size,
        cluster_units=cluster_units,
        token_to_cluster=tok_to_c,
        token_to_member=tok_to_m,
        cluster_entropy_bits=float(entropy),
        cluster_prob_max=float(c_mass.max()),
    )


def build_balanced_partition(
    vocab_size: int,
    n_clusters: int,
    cluster_size: int,
    train_streams: list[np.ndarray],
    mode: str = "paradigmatic_spectral",
    embed_dim: int = 24,
    seed: int = 13,
) -> ClusterPartition:
    """
    Build an exact `NC x S == V` vocabulary bijection under:
      - `mass_random`: frequency-stratified mass-matched random partition (null baseline)
      - `syntagmatic_spectral`: Zhou hypergraph Laplacian spectral embedding over window co-occurrence
      - `paradigmatic_spectral`: SVD of smoothed PPMI left/right context transition distributions
    """
    if n_clusters * cluster_size != vocab_size:
        raise ValueError(f"Exact bijection requires n_clusters * cluster_size == vocab_size ({n_clusters}*{cluster_size} != {vocab_size})")
    rng = np.random.default_rng(seed)

    counts = np.zeros(vocab_size, dtype=np.int64)
    for seq in train_streams:
        if seq.size > 0:
            np.add.at(counts, seq, 1)

    if mode == "mass_random":
        # Stratify tokens by frequency rank into S bands of size NC, then randomly permute each band across clusters
        rank_order = np.argsort(-counts, kind="mergesort")
        cluster_units = np.empty((n_clusters, cluster_size), dtype=np.int32)
        for band in range(cluster_size):
            band_tokens = rank_order[band * n_clusters : (band + 1) * n_clusters].copy()
            rng.shuffle(band_tokens)
            cluster_units[:, band] = band_tokens
        for c in range(n_clusters):
            cluster_units[c] = np.sort(cluster_units[c])
        return _finalize_partition(mode, vocab_size, n_clusters, cluster_size, cluster_units, counts)

    k_svd = min(embed_dim, vocab_size - 2)
    if mode == "syntagmatic_spectral":
        # First-order syntagmatic hyperedges: fixed 24-token windows -> incidence matrix H [V, E]
        rows: list[int] = []
        cols: list[int] = []
        edge_id = 0
        win_len = 24
        for seq in train_streams:
            n = int(seq.shape[0])
            for start in range(0, n, win_len):
                uniq = np.unique(seq[start : start + win_len])
                if uniq.size >= 3:
                    rows.extend(uniq.tolist())
                    cols.extend([edge_id] * int(uniq.size))
                    edge_id += 1
        if edge_id < k_svd + 2:
            feats = rng.standard_normal((vocab_size, k_svd)).astype(np.float32)
        else:
            vals = np.ones(len(rows), dtype=np.float32)
            H_inc: csr_matrix = coo_matrix((vals, (rows, cols)), shape=(vocab_size, edge_id)).tocsr()
            dv = np.asarray(H_inc.sum(axis=1)).reshape(-1)
            de = np.asarray(H_inc.sum(axis=0)).reshape(-1)
            dv_inv_sqrt = 1.0 / np.sqrt(np.maximum(dv, 1.0))
            de_inv_sqrt = 1.0 / np.sqrt(np.maximum(de, 1.0))
            B_norm = diags(dv_inv_sqrt) @ H_inc @ diags(de_inv_sqrt)
            u_vecs, _, _ = svds(B_norm.astype(np.float64), k=k_svd + 1, random_state=seed)
            # Drop trivial principal mode (index -1 in ascending svds order)
            feats = u_vecs[:, :-1].astype(np.float32)
        cluster_units = _balanced_assign_from_features(feats, counts, n_clusters, cluster_size, rng)
        return _finalize_partition(mode, vocab_size, n_clusters, cluster_size, cluster_units, counts)

    if mode == "paradigmatic_spectral":
        # Second-order paradigmatic PPMI: tokens sharing identical preceding 1-gram & hashed 2-gram contexts
        n_ctx = min(max(vocab_size * 2, 1024), 4096)
        r_list: list[np.ndarray] = []
        c_list: list[np.ndarray] = []
        for seq in train_streams:
            if seq.size < 3:
                continue
            y = seq[2:]
            prev1 = seq[1:-1]
            prev2 = ((seq[:-2].astype(np.int64) * 1315423911 + prev1.astype(np.int64)) % (n_ctx // 2)) + (n_ctx // 2)
            ctx1 = prev1 % (n_ctx // 2)
            r_list.append(y)
            c_list.append(ctx1.astype(np.int32))
            r_list.append(y)
            c_list.append(prev2.astype(np.int32))

        if not r_list:
            feats = rng.standard_normal((vocab_size, k_svd)).astype(np.float32)
        else:
            all_r = np.concatenate(r_list)
            all_c = np.concatenate(c_list)
            ones = np.ones(all_r.shape[0], dtype=np.float64)
            M: csr_matrix = coo_matrix((ones, (all_r, all_c)), shape=(vocab_size, n_ctx)).tocsr()
            M.sum_duplicates()
            total_pairs = max(float(M.sum()), 1.0)
            p_u = np.maximum(np.asarray(M.sum(axis=1)).reshape(-1), 1.0) ** 0.75
            p_c = np.maximum(np.asarray(M.sum(axis=0)).reshape(-1), 1.0)
            coo = M.tocoo()
            ppmi_vals = np.maximum(
                0.0,
                np.log((coo.data * total_pairs) / (p_u[coo.row] * p_c[coo.col] + 1e-12)),
            )
            P_mat = coo_matrix((ppmi_vals, (coo.row, coo.col)), shape=(vocab_size, n_ctx)).tocsr()
            u_vecs, s_vals, _ = svds(P_mat, k=k_svd, random_state=seed)
            feats = (u_vecs * np.sqrt(np.maximum(s_vals, 1e-8))[None, :]).astype(np.float32)
        cluster_units = _balanced_assign_from_features(feats, counts, n_clusters, cluster_size, rng)
        return _finalize_partition(mode, vocab_size, n_clusters, cluster_size, cluster_units, counts)

    raise ValueError(f"Unknown partition mode: {mode!r}")
