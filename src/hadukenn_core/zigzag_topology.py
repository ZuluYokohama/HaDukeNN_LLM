"""Zigzag persistent topological features in LLMs (Paper 2: Gardinazzi et al., arXiv:2410.11042v3).

Implements:
- ``build_knn_vr_complex``: k-NN Vietoris-Rips complex supporting both ``op21_vr_cap``
  (upper-radius cap ``D <= r_cap``) and ``app_d_short_filter`` (Paper 2 App. D short-edge removal
  ``D > r_thresh``) on ``(N, d)`` point clouds or ``(N, N)`` distance matrices.
- ``ZigzagTopologyAnalyzer``: exact F_2 clique-complex homology H_1 (including 2-simplex triangle
  boundaries so collinear trajectories satisfy beta_1 == 0) and exact Carlsson-de Silva zigzag
  persistence collapsed via M2 ``zigzag_effective_pi`` and scored via ``zigzag_persistence_ratio``.
- ``prune_layers_plateau``: Algorithm 2 10%-plateau layer pruning with App. G.1 readout protection.
- ``diagnose_token_shuffling``: App. F analytical shuffling diagnostic (``is_diagnostic_only=True``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from sheaf_verify.hashing import SheafError
from sheaf_verify.learning_guards import zigzag_effective_pi, zigzag_persistence_ratio


def _as_finite_float_array(arr: Any, label: str) -> np.ndarray:
    try:
        out = np.asarray(arr, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SheafError(f"{label} must be a numeric array") from exc
    if out.size == 0 or not bool(np.all(np.isfinite(out))):
        raise SheafError(f"{label} must be non-empty and finite")
    return out


def _pairwise_distances_from_input(arr: np.ndarray, *, is_distance_matrix: bool | None) -> np.ndarray:
    if arr.ndim != 2:
        raise SheafError(f"Expected 2D array, got shape {arr.shape}")
    n = arr.shape[0]
    if n < 2:
        raise SheafError(f"Expected at least 2 points, got {n}")

    use_dist = is_distance_matrix
    if use_dist is None:
        use_dist = bool(
            arr.shape[0] == arr.shape[1]
            and np.all(arr >= -1e-12)
            and np.allclose(arr, arr.T, atol=1e-9)
            and np.allclose(np.diag(arr), 0.0, atol=1e-9)
        )

    if use_dist:
        if arr.shape[0] != arr.shape[1]:
            raise SheafError(f"Distance matrix must be square, got {arr.shape}")
        if np.any(arr < -1e-12):
            raise SheafError("Distance matrix entries must be non-negative")
        if not bool(np.allclose(arr, arr.T, atol=1e-9)):
            raise SheafError("Distance matrix must be symmetric")
        dist = np.maximum(0.5 * (arr + arr.T), 0.0)
        np.fill_diagonal(dist, 0.0)
        return dist

    diff = arr[:, None, :] - arr[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, 0.0)
    return dist


def _sym_knn_mask(dist: np.ndarray, k_nn: int) -> np.ndarray:
    n = dist.shape[0]
    k_eff = min(k_nn, n - 1)
    d_work = dist.copy()
    np.fill_diagonal(d_work, np.inf)
    order = np.argsort(d_work, axis=1, kind="mergesort")[:, :k_eff]
    directed = np.zeros((n, n), dtype=bool)
    rows = np.arange(n)[:, None]
    directed[rows, order] = True
    mask = directed | directed.T
    np.fill_diagonal(mask, False)
    return mask


def _connected_components_count(adj: np.ndarray) -> int:
    n = adj.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    iu, ju = np.nonzero(np.triu(adj, k=1))
    for u, v in zip(iu.tolist(), ju.tolist(), strict=True):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[rv] = ru
    return len({find(i) for i in range(n)})


def _f2_rank(rows: list[int]) -> int:
    """Compute F_2 rank of a list of integer bitmask rows."""
    basis: dict[int, int] = {}
    rank = 0
    for row in rows:
        r = row
        while r:
            msb = r.bit_length() - 1
            piv = basis.get(msb)
            if piv is None:
                basis[msb] = r
                rank += 1
                break
            r ^= piv
    return rank


def _extract_clique_complex(adj: np.ndarray) -> dict[str, Any]:
    """Build 2-skeleton clique complex K = (V, E, T) and exact F_2 cycle/boundary spaces."""
    n = adj.shape[0]
    iu, ju = np.nonzero(np.triu(adj, k=1))
    edges: list[tuple[int, int]] = list(zip(iu.tolist(), ju.tolist(), strict=True))
    edge_to_idx = {e: idx for idx, e in enumerate(edges)}

    # Spanning forest (tree edges vs chord edges) via Union-Find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree_edges: list[tuple[int, int]] = []
    chord_edges: list[tuple[int, int]] = []
    chord_idx_map: dict[tuple[int, int], int] = {}

    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[rv] = ru
            tree_edges.append((u, v))
        else:
            c_idx = len(chord_edges)
            chord_edges.append((u, v))
            chord_idx_map[(u, v)] = c_idx

    n_comp = len({find(i) for i in range(n)})
    n_chords = len(chord_edges)

    # Adjacency list for fast triangle enumeration (u < v < w)
    nbrs: list[set[int]] = [set() for _ in range(n)]
    for u, v in edges:
        nbrs[u].add(v)
        nbrs[v].add(u)

    triangles: list[tuple[int, int, int]] = []
    boundary_chord_masks: list[int] = []
    for u, v in edges:
        common = nbrs[u] & nbrs[v]
        for w in common:
            if w > v:
                triangles.append((u, v, w))
                mask = 0
                for e_tri in ((u, v), (u, w), (v, w)):
                    c_i = chord_idx_map.get(e_tri)
                    if c_i is not None:
                        mask |= 1 << c_i
                if mask:
                    boundary_chord_masks.append(mask)

    # Echelon basis of B_1(K) inside Z_1(K) ~= F_2^{n_chords}
    b1_basis: dict[int, int] = {}
    for row in boundary_chord_masks:
        r = row
        while r:
            msb = r.bit_length() - 1
            piv = b1_basis.get(msb)
            if piv is None:
                b1_basis[msb] = r
                break
            r ^= piv

    # Build fundamental cycle masks over full E for each chord edge
    tree_adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in tree_edges:
        tree_adj[u].append(v)
        tree_adj[v].append(u)

    def _tree_path_edge_mask(src: int, dst: int) -> int:
        stack: list[tuple[int, int]] = [(src, -1)]
        prev: dict[int, int] = {src: -1}
        while stack:
            cur, par = stack.pop()
            if cur == dst:
                break
            for nb in tree_adj[cur]:
                if nb != par and nb not in prev:
                    prev[nb] = cur
                    stack.append((nb, cur))
        e_mask = 0
        cur = dst
        while cur != src:
            par = prev[cur]
            e = (par, cur) if par < cur else (cur, par)
            e_mask |= 1 << edge_to_idx[e]
            cur = par
        return e_mask

    # Basis of H_1(K): chord indices whose bit is NOT a pivot of b1_basis
    pivot_chords = set(b1_basis.keys())
    h1_basis_chords = [c for c in range(n_chords) if c not in pivot_chords]
    h1_cycle_edge_masks: list[int] = []
    for c in h1_basis_chords:
        u, v = chord_edges[c]
        c_mask = (1 << edge_to_idx[(u, v)]) | _tree_path_edge_mask(u, v)
        h1_cycle_edge_masks.append(c_mask)

    beta_1 = len(h1_basis_chords)
    return {
        "n": n,
        "beta_0": n_comp,
        "edges": edges,
        "edge_to_idx": edge_to_idx,
        "chord_edges": chord_edges,
        "chord_idx_map": chord_idx_map,
        "triangles": triangles,
        "b1_basis": b1_basis,
        "h1_basis_chords": h1_basis_chords,
        "h1_cycle_edge_masks": h1_cycle_edge_masks,
        "beta_1": beta_1,
    }


def _induced_h1_map(src_complex: dict[str, Any], dst_complex: dict[str, Any]) -> list[int]:
    """Compute F_2 linear map H_1(src) -> H_1(dst) as a list of integer bitmasks (one per src basis col).

    Wait: returns ``rows`` indexed by src basis vector ``i`` (0..dim_src-1), where bit ``j`` of
    ``rows[i]`` is 1 iff basis vector ``i`` of ``H_1(src)`` maps to component ``j`` of ``H_1(dst)``.
    """
    dst_edge_to_idx: dict[tuple[int, int], int] = dst_complex["edge_to_idx"]
    dst_chord_map: dict[tuple[int, int], int] = dst_complex["chord_idx_map"]
    dst_b1_basis: dict[int, int] = dst_complex["b1_basis"]
    dst_h1_chords: list[int] = dst_complex["h1_basis_chords"]
    dst_chord_to_h1 = {c: j for j, c in enumerate(dst_h1_chords)}

    src_edges: list[tuple[int, int]] = src_complex["edges"]
    # Precompute bit mapping from src edge index -> dst chord index (or None if tree edge in dst)
    src_e_to_dst_chord: list[int | None] = []
    for e in src_edges:
        if e not in dst_edge_to_idx:
            raise SheafError("Zigzag intersection complex must be a subcomplex of endpoint complex")
        src_e_to_dst_chord.append(dst_chord_map.get(e))

    mapped_rows: list[int] = []
    for cyc_mask in src_complex["h1_cycle_edge_masks"]:
        # Project cycle onto dst chord coordinates
        z_chords = 0
        m = cyc_mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            dc = src_e_to_dst_chord[e_idx]
            if dc is not None:
                z_chords ^= 1 << dc
            m ^= lsb

        # Reduce z_chords modulo dst_b1_basis (pivots sorted descending)
        for piv_bit in sorted(dst_b1_basis.keys(), reverse=True):
            if (z_chords >> piv_bit) & 1:
                z_chords ^= dst_b1_basis[piv_bit]

        # Remaining bits live strictly in dst_h1_chords
        h1_vec = 0
        rem = z_chords
        while rem:
            lsb = rem & -rem
            c_bit = lsb.bit_length() - 1
            h_j = dst_chord_to_h1[c_bit]
            h1_vec |= 1 << h_j
            rem ^= lsb
        mapped_rows.append(h1_vec)
    return mapped_rows


def _zigzag_h1_intervals(
    layer_complexes: list[dict[str, Any]],
    inter_complexes: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    """Compute exact Carlsson-de Silva F_2 zigzag persistence intervals over H_1.

    Indices ``0 .. 2L-2`` correspond to:
      even ``2*l`` -> ``V_l = H_1(K_l)``
      odd ``2*l+1`` -> ``W_l = H_1(K_l cap K_{l+1})``
    with alternating inclusions ``V_l <- W_l -> V_{l+1}``.
    """
    n_layers = len(layer_complexes)
    m_steps = 2 * n_layers - 1
    if n_layers == 0:
        return []

    def _priority(birth: int) -> tuple[int, int]:
        # Right-filtration order: odd births descending (youngest first), then even births ascending
        if birth % 2 == 1:
            return (0, -birth)
        return (1, birth)

    # Active basis at current step: list of birth times aligned to current space basis vectors 0..dim-1
    dim_0 = layer_complexes[0]["beta_1"]
    active_births: list[int] = [0] * dim_0
    # Vector representation in current space: standard basis vectors (1 << j)
    active_vecs: list[int] = [1 << j for j in range(dim_0)]
    intervals: list[tuple[int, int]] = []

    for l_idx in range(n_layers - 1):
        v_left = layer_complexes[l_idx]
        w_mid = inter_complexes[l_idx]
        v_right = layer_complexes[l_idx + 1]

        odd_idx = 2 * l_idx + 1
        even_right_idx = 2 * l_idx + 2

        # 1. Backward arrow: V_l <- W_l (k = odd_idx)
        dim_v_left = v_left["beta_1"]
        f_rows = _induced_h1_map(w_mid, v_left)  # len = dim_w, bits in 0..dim_v_left-1

        # Express active_vecs of V_l in priority-sorted order: active_vecs[p] is p-th basis vector
        # Build change-of-basis from V_l standard coords to active-basis coords
        # Since active_vecs form a basis of F_2^{dim_v_left}, invert the matrix!
        if dim_v_left > 0:
            # Matrix M[row=std_i, col=active_p] -> augment with identity to get M^{-1}
            aug = [0] * dim_v_left
            for p_idx, vec in enumerate(active_vecs):
                aug[p_idx] = vec
            # Express each active_vec in standard coords -> we want coord(u) in active_vecs
            # Solve active_vecs^T * c = u over F_2 by building rows = active_vecs with RHS = (1<<p)
            inv_basis: dict[int, tuple[int, int]] = {}
            for p_idx, vec in enumerate(active_vecs):
                r_vec = vec
                r_rhs = 1 << p_idx
                while r_vec:
                    msb = r_vec.bit_length() - 1
                    piv = inv_basis.get(msb)
                    if piv is None:
                        inv_basis[msb] = (r_vec, r_rhs)
                        break
                    r_vec ^= piv[0]
                    r_rhs ^= piv[1]

            def _to_active_coords(u: int, ib: dict[int, tuple[int, int]] = inv_basis) -> int:
                acc = 0
                cur = u
                while cur:
                    msb = cur.bit_length() - 1
                    piv_vec, piv_rhs = ib[msb]
                    cur ^= piv_vec
                    acc ^= piv_rhs
                return acc
        else:
            def _to_active_coords(u: int) -> int:
                return 0

        # Order active basis indices by priority ascending (smallest priority first; largest right-most)
        prio_order = sorted(range(dim_v_left), key=lambda idx: _priority(active_births[idx]))
        # Map active coord p -> prio_pos q (0..dim_v_left-1)
        p_to_q = [0] * dim_v_left
        for q_pos, p_idx in enumerate(prio_order):
            p_to_q[p_idx] = q_pos

        # Eliminate W_l images right-to-left (largest prio_pos q first)
        row_pairs: list[tuple[int, int]] = []  # (image_in_q_coords, w_vec)
        for w_j, f_u in enumerate(f_rows):
            a_coords = _to_active_coords(f_u)
            q_coords = 0
            tmp = a_coords
            while tmp:
                lsb = tmp & -tmp
                p_bit = lsb.bit_length() - 1
                q_coords |= 1 << p_to_q[p_bit]
                tmp ^= lsb
            row_pairs.append((q_coords, 1 << w_j))

        pivots_q: dict[int, tuple[int, int]] = {}  # q_pos -> (q_coords, w_vec)
        ker_w_vecs: list[int] = []
        for q_c, w_v in row_pairs:
            qc = q_c
            wv = w_v
            while qc:
                msb = qc.bit_length() - 1  # largest q_pos in qc
                piv = pivots_q.get(msb)
                if piv is None:
                    pivots_q[msb] = (qc, wv)
                    break
                qc ^= piv[0]
                wv ^= piv[1]
            else:
                ker_w_vecs.append(wv)

        # Any q_pos not in pivots_q dies at odd_idx - 1 (even_left)
        new_active_births: list[int] = []
        new_active_vecs: list[int] = []
        for w_ker in ker_w_vecs:
            new_active_births.append(odd_idx)
            new_active_vecs.append(w_ker)
        for q_pos, p_idx in enumerate(prio_order):
            if q_pos in pivots_q:
                new_active_births.append(active_births[p_idx])
                new_active_vecs.append(pivots_q[q_pos][1])
            else:
                intervals.append((active_births[p_idx], odd_idx - 1))

        active_births = new_active_births
        active_vecs = new_active_vecs

        # 2. Forward arrow: W_l -> V_{l+1} (k = even_right_idx)
        dim_v_right = v_right["beta_1"]
        g_rows = _induced_h1_map(w_mid, v_right)  # len = dim_w, bits in 0..dim_v_right-1

        def _apply_g(w_vec: int, rows: list[int] = g_rows) -> int:
            out = 0
            cur = w_vec
            while cur:
                lsb = cur & -cur
                w_j = lsb.bit_length() - 1
                out ^= rows[w_j]
                cur ^= lsb
            return out

        # Process active basis of W_l in priority order (smallest priority claims V_{l+1} first)
        w_prio_order = sorted(range(len(active_vecs)), key=lambda idx: _priority(active_births[idx]))
        im_basis: dict[int, int] = {}
        fwd_births: list[int] = []
        fwd_vecs: list[int] = []

        for idx in w_prio_order:
            u_orig = _apply_g(active_vecs[idx])
            u = u_orig
            while u:
                msb = u.bit_length() - 1
                piv = im_basis.get(msb)
                if piv is None:
                    im_basis[msb] = u
                    fwd_births.append(active_births[idx])
                    fwd_vecs.append(u_orig)
                    break
                u ^= piv
            else:
                # Linearly dependent in V_{l+1}: dies at odd_idx
                intervals.append((active_births[idx], odd_idx))

        # Extend to full basis of V_{l+1} with new births at even_right_idx
        for std_j in range(dim_v_right):
            cand = 1 << std_j
            u = cand
            while u:
                msb = u.bit_length() - 1
                piv = im_basis.get(msb)
                if piv is None:
                    im_basis[msb] = u
                    fwd_births.append(even_right_idx)
                    fwd_vecs.append(cand)
                    break
                u ^= piv

        active_births = fwd_births
        active_vecs = fwd_vecs

    # Remaining active bars die at the end m_steps - 1
    end_idx = m_steps - 1
    for b in active_births:
        intervals.append((b, end_idx))

    return intervals


def build_knn_vr_complex(
    X_or_D: Any = None,
    k_nn: int = 6,
    radius_quantile: float = 0.22,
    radius: float | None = None,
    filter_mode: str = "op21_vr_cap",
    *,
    points: Any = None,
    X: Any = None,
    k: int | None = None,
    r_thresh: float | None = None,
    mode: str | None = None,
    is_distance_matrix: bool | None = None,
) -> dict[str, Any]:
    """Build k-NN Vietoris-Rips graph/complex with ``op21_vr_cap`` or ``app_d_short_filter``."""
    raw = X_or_D if X_or_D is not None else (points if points is not None else X)
    if raw is None:
        raise SheafError("build_knn_vr_complex requires X_or_D / points / X")
    arr = _as_finite_float_array(raw, "X_or_D")
    dist = _pairwise_distances_from_input(arr, is_distance_matrix=is_distance_matrix)
    n = dist.shape[0]

    k_param = k if k is not None else k_nn
    if isinstance(k_param, bool) or not isinstance(k_param, (int, np.integer)):
        raise SheafError(f"k_nn must be an integer, got {k_param!r}")
    k_eff_req = int(k_param)
    if k_eff_req < 1:
        raise SheafError(f"k_nn must be >= 1, got {k_eff_req}")

    q_val = float(radius_quantile)
    if not (math.isfinite(q_val) and 0.0 <= q_val <= 1.0):
        raise SheafError(f"radius_quantile must lie in [0, 1], got {q_val}")

    mode_raw = (mode if mode is not None else filter_mode).strip().lower()
    if mode_raw in {"op21_vr_cap", "upper_radius", "vr_cap"}:
        canonical_mode = "op21_vr_cap"
    elif mode_raw in {"app_d_short_filter", "short_filter"}:
        canonical_mode = "app_d_short_filter"
    else:
        raise SheafError(f"Unknown filter_mode: {mode_raw!r}")

    knn_mask = _sym_knn_mask(dist, k_eff_req)
    tri_dist = dist[np.triu_indices(n, k=1)]

    r_param = r_thresh if r_thresh is not None else radius
    if r_param is not None:
        r_val = float(r_param)
        if not math.isfinite(r_val) or r_val < 0.0:
            raise SheafError(f"radius must be finite and >= 0, got {r_val}")
    else:
        r_val = float(np.quantile(tri_dist, q_val))

    if canonical_mode == "op21_vr_cap":
        mask = knn_mask & (dist <= r_val + 1e-12)
    else:
        # Paper 2 App. D: filter OUT short edges (length <= r_val), retaining dist > r_val
        mask = knn_mask & (dist > r_val)

    np.fill_diagonal(mask, False)
    mask = mask | mask.T

    cpx = _extract_clique_complex(mask)
    return {
        "mask": mask,
        "adjacency": mask.astype(np.int8),
        "radius": float(r_val),
        "r_thresh": float(r_val),
        "beta_0": int(cpx["beta_0"]),
        "beta_1": int(cpx["beta_1"]),
        "n_edges": int(len(cpx["edges"])),
        "n_triangles": int(len(cpx["triangles"])),
        "filter_mode": canonical_mode,
    }


build_knn_vr_graph = build_knn_vr_complex
knn_vr_complex = build_knn_vr_complex
knn_vr_graph = build_knn_vr_complex


class ZigzagTopologyAnalyzer:
    """Exact F_2 zigzag persistent homology analyzer across LLM layer trajectories."""

    def __init__(self, k_nn: int = 4, alpha: float = -1.0, radius_quantile: float | None = None) -> None:
        if isinstance(k_nn, bool) or not isinstance(k_nn, (int, np.integer)) or int(k_nn) < 1:
            raise SheafError(f"k_nn must be an integer >= 1, got {k_nn!r}")
        if not math.isfinite(float(alpha)):
            raise SheafError(f"alpha must be finite, got {alpha}")
        if radius_quantile is not None and not (0.0 <= float(radius_quantile) <= 1.0):
            raise SheafError(f"radius_quantile must lie in [0, 1], got {radius_quantile}")
        self.k_nn = int(k_nn)
        self.alpha = float(alpha)
        self.radius_quantile = float(radius_quantile) if radius_quantile is not None else None

    def analyze(self, traj: Any) -> dict[str, Any]:
        arr = _as_finite_float_array(traj, "traj")
        if arr.ndim != 3:
            raise SheafError(f"traj must be 3D (L, N, d), got shape {arr.shape}")
        n_layers, n_pts, _ = arr.shape
        if n_layers < 2:
            raise SheafError(f"traj must have at least 2 layers, got {n_layers}")
        if n_pts < 3:
            raise SheafError(f"traj must have at least 3 points per layer, got {n_pts}")

        layer_masks: list[np.ndarray] = []
        layer_complexes: list[dict[str, Any]] = []
        for l_idx in range(n_layers):
            pts = arr[l_idx]
            dist = _pairwise_distances_from_input(pts, is_distance_matrix=False)
            mask = _sym_knn_mask(dist, self.k_nn)
            if self.radius_quantile is not None:
                tri = dist[np.triu_indices(n_pts, k=1)]
                r_cap = float(np.quantile(tri, self.radius_quantile))
                mask = mask & (dist <= r_cap + 1e-12)
                np.fill_diagonal(mask, False)
            layer_masks.append(mask)
            layer_complexes.append(_extract_clique_complex(mask))

        inter_complexes: list[dict[str, Any]] = []
        for l_idx in range(n_layers - 1):
            inter_mask = layer_masks[l_idx] & layer_masks[l_idx + 1]
            inter_complexes.append(_extract_clique_complex(inter_mask))

        intervals = _zigzag_h1_intervals(layer_complexes, inter_complexes)

        grid_sz = 2 * n_layers - 1
        pi_grid = [[0.0 for _ in range(grid_sz)] for _ in range(grid_sz)]
        for b, d in intervals:
            pi_grid[b][d] += 1.0

        pi_hat_list = zigzag_effective_pi(pi_grid)
        pi_hat = np.asarray(pi_hat_list, dtype=np.float64)
        beta_1_arr = np.array([int(c["beta_1"]) for c in layer_complexes], dtype=np.int64)

        # Paper 2 Eq. 5-8: Z_1(l1, l2) via M2 zigzag_persistence_ratio and B_1(l)
        z1_bar = np.zeros(n_layers, dtype=np.float64)
        b1_freq = np.zeros(n_layers, dtype=np.float64)

        for ell in range(n_layers):
            total_pi_ell = float(np.sum(pi_hat[ell]))
            w_sum = 0.0
            z_weighted = 0.0
            b_weighted = 0.0
            for ell_i in range(n_layers):
                if ell_i == ell:
                    continue
                omega = abs(ell - ell_i) ** self.alpha
                w_sum += omega
                z_pair = zigzag_persistence_ratio(pi_hat_list, int(beta_1_arr[ell]), ell, ell_i)
                z_weighted += omega * z_pair
                b_weighted += omega * float(pi_hat[ell, ell_i])
            if w_sum > 0.0:
                z1_bar[ell] = z_weighted / w_sum
                if total_pi_ell > 0.0:
                    b1_freq[ell] = b_weighted / (w_sum * total_pi_ell)

        return {
            "z1_bar": z1_bar,
            "z_bar": z1_bar,
            "mean_interlayer_persistence": z1_bar,
            "z1_profile": z1_bar,
            "b1_freq": b1_freq,
            "births_relative_frequency": b1_freq,
            "b1": b1_freq,
            "pi_hat": pi_hat,
            "beta_1": beta_1_arr,
        }

    compute = analyze

    def __call__(self, traj: Any) -> dict[str, Any]:
        return self.analyze(traj)


def compute_zigzag_persistence(traj: Any, k_nn: int = 4, alpha: float = -1.0) -> dict[str, Any]:
    """Functional wrapper around :class:`ZigzagTopologyAnalyzer`."""
    return ZigzagTopologyAnalyzer(k_nn=k_nn, alpha=alpha).analyze(traj)


zigzag_interlayer_analysis = compute_zigzag_persistence


def prune_layers_plateau(
    z1_bar: Any = None,
    threshold: float = 0.90,
    *,
    z_bar: Any = None,
    scores: Any = None,
    persistence: Any = None,
    protect_readout_layers: int = 1,
) -> dict[str, Any]:
    """Algorithm 2 10%-plateau layer pruning with App. G.1 readout protection."""
    raw = z1_bar if z1_bar is not None else (z_bar if z_bar is not None else (scores if scores is not None else persistence))
    if raw is None:
        raise SheafError("prune_layers_plateau requires z1_bar / z_bar / scores / persistence")
    arr = _as_finite_float_array(raw, "z1_bar").ravel()
    t_val = float(threshold)
    if not (math.isfinite(t_val) and 0.0 < t_val <= 1.0):
        raise SheafError(f"threshold must lie in (0, 1], got {t_val}")
    if isinstance(protect_readout_layers, bool) or not isinstance(protect_readout_layers, (int, np.integer)):
        raise SheafError(
            f"protect_readout_layers must be a non-negative integer, got {protect_readout_layers!r}"
        )
    protect_val = int(protect_readout_layers)
    if protect_val < 0:
        raise SheafError(f"protect_readout_layers must be >= 0, got {protect_val}")

    n_layers = int(arr.size)
    max_z = float(np.max(arr))
    if n_layers < 3 or max_z <= 0.0:
        return {
            "pruned_layers": [],
            "pruned_indices": [],
            "drop_layers": [],
            "kept_layers": list(range(n_layers)),
            "threshold": t_val,
            "max_z1": max_z,
        }

    # App. G.1 readout guard: never prune layer 0, L-1, or L-2 when L >= 4
    extra_protect = protect_val if n_layers >= 4 else 0
    upper_bound = max(1, n_layers - 1 - extra_protect)

    # Paper 2 Algorithm 2: z1_bar[l] >= max(z1_bar) * threshold over non-protected interior layers
    cutoff = max_z * t_val
    pruned = [idx for idx in range(1, upper_bound) if float(arr[idx]) >= cutoff]

    pruned_set = set(pruned)
    kept = [idx for idx in range(n_layers) if idx not in pruned_set]
    return {
        "pruned_layers": pruned,
        "pruned_indices": pruned,
        "drop_layers": pruned,
        "kept_layers": kept,
        "threshold": t_val,
        "max_z1": max_z,
    }


algorithm2_prune_layers = prune_layers_plateau
zigzag_prune_layers = prune_layers_plateau
prune_redundant_layers = prune_layers_plateau


def diagnose_token_shuffling(
    structured: Any = None,
    shuffled: Any = None,
    *,
    structured_traj: Any = None,
    shuffled_traj: Any = None,
    k_nn: int = 4,
    alpha: float = -1.0,
) -> dict[str, Any]:
    """Paper 2 App. F token-shuffling diagnostic comparing structured vs shuffled trajectories."""
    s_raw = structured if structured is not None else structured_traj
    sh_raw = shuffled if shuffled is not None else shuffled_traj
    if s_raw is None or sh_raw is None:
        raise SheafError("diagnose_token_shuffling requires structured and shuffled trajectories")

    analyzer = ZigzagTopologyAnalyzer(k_nn=k_nn, alpha=alpha)
    res_s = analyzer.analyze(s_raw)
    res_sh = analyzer.analyze(sh_raw)

    z_s = np.asarray(res_s["z1_bar"], dtype=np.float64)
    z_sh = np.asarray(res_sh["z1_bar"], dtype=np.float64)
    n_layers = int(z_s.size)
    mid_start = max(0, n_layers // 4)
    mid_end = max(mid_start + 1, n_layers - 1)

    s_mid = float(np.mean(z_s[mid_start:mid_end]))
    sh_mid = float(np.mean(z_sh[mid_start:mid_end]))
    s_mean = float(np.mean(z_s))
    sh_mean = float(np.mean(z_sh))

    return {
        "structured_mid_late_z1": s_mid,
        "shuffled_mid_late_z1": sh_mid,
        "structured_mean_z1": s_mean,
        "shuffled_mean_z1": sh_mean,
        "structured_z1": s_mid,
        "shuffled_z1": sh_mid,
        "delta_z1": s_mid - sh_mid,
        "is_diagnostic_only": True,
        "diagnostic_only": True,
    }


app_f_shuffling_diagnostic = diagnose_token_shuffling
zigzag_shuffling_diagnostic = diagnose_token_shuffling
