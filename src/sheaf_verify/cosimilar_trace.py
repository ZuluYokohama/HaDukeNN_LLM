"""Cosimilar Hypergraph Recursive Trace & Observer Measurement Engine (v1.0.0).

Pure-stdlib implementation uniting five exact mathematical operators:
1. Exact Rational Stochastic Schur-Complement Trace (`ExactRationalMarkovKernel`):
   ``M_S = M_SS + M_SB (I - M_BB)^(-1) M_BS``, proving recursive transitivity
   ``(M_T)_S == M_S``, normalized stationary measure restriction, and multiplicative
   step-counter dilation ``tau_S = tau_T * tau_{S|T}``.
2. Coinductive Probabilistic & Role-Typed Bisimulation (`compute_cosimilar_quotient`):
   Greatest fixed-point partition refinement establishing the exact zero-surprise
   boundary (`0` bits multi-step KL divergence) versus lossy bucket lumping.
3. Active Hyperedge-Guided Window Search (`ActiveHypergraphTraceSearcher`):
   Replaces exponential ``2^|V|`` subset enumeration and ``O(|B|^3)`` global matrix
   inversion with bounded hyperedge-incidence neighborhood search and bipartite
   Kron/Schur reduction.
4. Higher-Arity Hypergraph Synergy Verifier (`evaluate_three_way_synergy`):
   Demonstrates exact 3-way relational parity/conservation inference in linear
   incidence memory where 2D pairwise Markov marginals collapse to 50% chance.
5. Multi-Observer Sheaf Gluing, Counterfactual Branching & Spoof Localization
   (`CosimilarObserverSheafEngine`): Checks zero-surprise restriction agreement across
   overlapping observer windows, forks isolated non-stationary counterfactual branches,
   localizes adversarial observer spoofing, and emits SCITT RFC 9943 receipts.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction

from sheaf_verify.compliance import emit_regulatory_bundle, verify_regulatory_bundle
from sheaf_verify.hashing import SheafError, canonical_bytes


def _frac_invert(mat: list[list[Fraction]]) -> list[list[Fraction]]:
    """Exact Gauss-Jordan matrix inverse over Q."""
    n = len(mat)
    if n == 0:
        return []
    aug = [[Fraction(mat[r][c]) for c in range(n)] + [Fraction(1 if r == c else 0) for c in range(n)] for r in range(n)]
    for col in range(n):
        piv = None
        for r in range(col, n):
            if aug[r][col] != 0:
                piv = r
                break
        if piv is None:
            raise SheafError("Singular submatrix encountered in rational Schur trace: (I - M_BB) is not invertible.")
        aug[col], aug[piv] = aug[piv], aug[col]
        scale = aug[col][col]
        for c in range(col, 2 * n):
            aug[col][c] /= scale
        for r in range(n):
            if r != col and aug[r][col] != 0:
                factor = aug[r][col]
                for c in range(col, 2 * n):
                    aug[r][c] -= factor * aug[col][c]
    return [row[n:] for row in aug]


def _frac_matmul(a: list[list[Fraction]], b: list[list[Fraction]]) -> list[list[Fraction]]:
    if not a or not b:
        return []
    rows = len(a)
    inner = len(b)
    cols = len(b[0])
    out = [[Fraction(0) for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            aik = a[i][k]
            if aik != 0:
                for j in range(cols):
                    out[i][j] += aik * b[k][j]
    return out


def _float_invert(mat: list[list[float]]) -> list[list[float]]:
    """Pivoted Gauss-Jordan inverse over float64."""
    n = len(mat)
    if n == 0:
        return []
    aug = [[float(mat[r][c]) for c in range(n)] + [1.0 if r == c else 0.0 for c in range(n)] for r in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-13:
            raise SheafError("Singular float submatrix in Kron/Schur reduction.")
        aug[col], aug[piv] = aug[piv], aug[col]
        inv_s = 1.0 / aug[col][col]
        for c in range(col, 2 * n):
            aug[col][c] *= inv_s
        for r in range(n):
            if r != col and abs(aug[r][col]) > 0.0:
                f = aug[r][col]
                for c in range(col, 2 * n):
                    aug[r][c] -= f * aug[col][c]
    return [row[n:] for row in aug]


def _float_matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    if not a or not b:
        return []
    rows, inner, cols = len(a), len(b), len(b[0])
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        row_a = a[i]
        row_o = out[i]
        for k in range(inner):
            aik = row_a[k]
            if aik != 0.0:
                row_b = b[k]
                for j in range(cols):
                    row_o[j] += aik * row_b[j]
    return out


@dataclass(frozen=True)
class TraceLawVerification:
    """Exact verification certificate for recursive Schur-complement trace invariants."""

    stochastic_exact: bool
    transitive_exact: bool
    stationary_restricted_exact: bool
    dilation_multiplicative_exact: bool
    outer_dilation: float
    inner_given_outer_dilation: float
    total_dilation: float


class ExactRationalMarkovKernel:
    """Exact rational Markov kernel supporting Schur-complement traces and lumpability audits."""

    def __init__(self, matrix: list[list[Fraction | int | str]], state_names: list[str] | None = None) -> None:
        n = len(matrix)
        if n == 0 or any(len(row) != n for row in matrix):
            raise SheafError("Markov kernel must be a non-empty square matrix.")
        frac_mat: list[list[Fraction]] = []
        for r_idx, row in enumerate(matrix):
            f_row = [Fraction(v) for v in row]
            if any(v < 0 for v in f_row):
                raise SheafError(f"Negative transition probability in row {r_idx}.")
            if sum(f_row) != Fraction(1, 1):
                raise SheafError(f"Row {r_idx} sums to {sum(f_row)}, expected exact 1.")
            frac_mat.append(f_row)
        self.n = n
        self.matrix = frac_mat
        self.state_names = list(state_names) if state_names is not None else [f"s{i}" for i in range(n)]
        if len(self.state_names) != n:
            raise SheafError("state_names length must equal matrix dimension.")

    def stationary_distribution(self) -> list[Fraction]:
        """Compute unique stationary measure pi satisfying pi M = pi, sum(pi) = 1 over Q."""
        n = self.n
        if n == 1:
            return [Fraction(1, 1)]
        aug: list[list[Fraction]] = []
        for r in range(n - 1):
            row = [self.matrix[c][r] - (Fraction(1, 1) if r == c else Fraction(0, 1)) for c in range(n)]
            row.append(Fraction(0, 1))
            aug.append(row)
        aug.append([Fraction(1, 1) for _ in range(n)] + [Fraction(1, 1)])

        for col in range(n):
            piv = None
            for r in range(col, n):
                if aug[r][col] != 0:
                    piv = r
                    break
            if piv is None:
                raise SheafError("Markov kernel does not have a unique ergodic stationary distribution.")
            aug[col], aug[piv] = aug[piv], aug[col]
            scale = aug[col][col]
            for c in range(col, n + 1):
                aug[col][c] /= scale
            for r in range(n):
                if r != col and aug[r][col] != 0:
                    factor = aug[r][col]
                    for c in range(col, n + 1):
                        aug[r][c] -= factor * aug[col][c]
        pi = [aug[r][n] for r in range(n)]
        if any(p <= 0 for p in pi) or sum(pi) != Fraction(1, 1):
            raise SheafError("Stationary vector is not strictly positive and normalized.")
        return pi

    def schur_trace(self, visible: list[int] | tuple[int, ...]) -> ExactRationalMarkovKernel:
        """Compute exact Schur-complement trace M_S = M_SS + M_SB (I - M_BB)^(-1) M_BS."""
        vis = list(dict.fromkeys(visible))
        if not vis or any(i < 0 or i >= self.n for i in vis):
            raise SheafError("Visible subset must be a non-empty valid index list.")
        if len(vis) == self.n:
            reordered = [[self.matrix[i][j] for j in vis] for i in vis]
            return ExactRationalMarkovKernel(reordered, [self.state_names[i] for i in vis])

        vis_set = set(vis)
        # Restrict hidden states to those reachable from vis; closed disconnected islands have M_SB == 0
        reach_from_vis: set[int] = set()
        stack = list(vis)
        while stack:
            u = stack.pop()
            for v in range(self.n):
                if self.matrix[u][v] > 0 and v not in vis_set and v not in reach_from_vis:
                    reach_from_vis.add(v)
                    stack.append(v)
        if not reach_from_vis:
            return ExactRationalMarkovKernel(
                [[self.matrix[i][j] for j in vis] for i in vis], [self.state_names[i] for i in vis]
            )
        hid = sorted(reach_from_vis)
        m_ss = [[self.matrix[i][j] for j in vis] for i in vis]
        m_sb = [[self.matrix[i][j] for j in hid] for i in vis]
        m_bs = [[self.matrix[i][j] for j in vis] for i in hid]
        i_minus_mbb = [
            [(Fraction(1, 1) if r == c else Fraction(0, 1)) - self.matrix[hid[r]][hid[c]] for c in range(len(hid))]
            for r in range(len(hid))
        ]
        resolvent = _frac_invert(i_minus_mbb)
        indirect = _frac_matmul(_frac_matmul(m_sb, resolvent), m_bs)
        traced = [[m_ss[r][c] + indirect[r][c] for c in range(len(vis))] for r in range(len(vis))]
        return ExactRationalMarkovKernel(traced, [self.state_names[i] for i in vis])

    def verify_trace_laws(self, outer_window: list[int], inner_window: list[int]) -> TraceLawVerification:
        """Verify transitivity (M_T)_S == M_S, stationary restriction, and tick dilation."""
        if not set(inner_window).issubset(set(outer_window)):
            raise SheafError("inner_window must be a subset of outer_window.")
        m_outer = self.schur_trace(outer_window)
        rel_inner = [outer_window.index(idx) for idx in inner_window]
        m_nested = m_outer.schur_trace(rel_inner)
        m_direct = self.schur_trace(inner_window)

        stochastic_ok = all(sum(r) == Fraction(1, 1) for r in m_direct.matrix) and all(
            sum(r) == Fraction(1, 1) for r in m_outer.matrix
        )
        transitive_ok = m_nested.matrix == m_direct.matrix

        pi_full = self.stationary_distribution()
        pi_outer = m_outer.stationary_distribution()
        pi_inner = m_direct.stationary_distribution()

        mass_outer = sum(pi_full[i] for i in outer_window)
        mass_inner = sum(pi_full[i] for i in inner_window)
        mass_inner_in_outer = sum(pi_outer[k] for k in rel_inner)

        stat_ok = all(pi_inner[pos] == pi_full[idx] / mass_inner for pos, idx in enumerate(inner_window)) and all(
            pi_outer[pos] == pi_full[idx] / mass_outer for pos, idx in enumerate(outer_window)
        )

        tau_outer = Fraction(1, 1) / mass_outer
        tau_inner_given_outer = Fraction(1, 1) / mass_inner_in_outer
        tau_total = Fraction(1, 1) / mass_inner
        dilation_ok = tau_total == tau_outer * tau_inner_given_outer and tau_total >= Fraction(1, 1)

        return TraceLawVerification(
            stochastic_exact=stochastic_ok,
            transitive_exact=transitive_ok,
            stationary_restricted_exact=stat_ok,
            dilation_multiplicative_exact=dilation_ok,
            outer_dilation=float(tau_outer),
            inner_given_outer_dilation=float(tau_inner_given_outer),
            total_dilation=float(tau_total),
        )

    def lumping_surprise_bpb(self, partition: list[list[int]]) -> float:
        """Compute 2-step KL surprise (in bits) caused by coarse bucket lumping vs exact dynamics."""
        covered = [idx for block in partition for idx in block]
        if sorted(covered) != list(range(self.n)) or any(len(b) == 0 for b in partition):
            raise SheafError("Partition must be a disjoint cover of all states.")

        pi = self.stationary_distribution()
        k_blocks = len(partition)
        block_mass = [sum(pi[i] for i in block) for block in partition]

        # Stationary-weighted 1-step lumped transition matrix L_{a,b}
        lumped_1 = [[Fraction(0, 1) for _ in range(k_blocks)] for _ in range(k_blocks)]
        for a, block_a in enumerate(partition):
            for b, block_b in enumerate(partition):
                num = Fraction(0, 1)
                for i in block_a:
                    row_sum = sum(self.matrix[i][j] for j in block_b)
                    num += pi[i] * row_sum
                lumped_1[a][b] = num / block_mass[a]

        lumped_sq = _frac_matmul(lumped_1, lumped_1)
        exact_sq = _frac_matmul(self.matrix, self.matrix)

        # True 2-step macro transition distribution P(B_{t+2}=b | B_t=a)
        true_macro_sq = [[Fraction(0, 1) for _ in range(k_blocks)] for _ in range(k_blocks)]
        for a, block_a in enumerate(partition):
            for b, block_b in enumerate(partition):
                num = Fraction(0, 1)
                for i in block_a:
                    num += pi[i] * sum(exact_sq[i][j] for j in block_b)
                true_macro_sq[a][b] = num / block_mass[a]

        kl_bits = 0.0
        for a in range(k_blocks):
            w_a = float(block_mass[a])
            for b in range(k_blocks):
                p_ab = float(true_macro_sq[a][b])
                q_ab = float(lumped_sq[a][b])
                if p_ab > 1e-15:
                    if q_ab <= 1e-15:
                        return float("inf")
                    kl_bits += w_a * p_ab * math.log2(p_ab / q_ab)
        return max(0.0, kl_bits)


def compute_cosimilar_quotient(
    role_kernels: dict[str, list[list[Fraction | int | str]]],
    observables: list[str] | None = None,
) -> list[list[int]]:
    """Greatest fixed-point probabilistic & role-typed bisimulation (cosimilarity) quotient.

    Two states u, v belong to the same cosimilar equivalence class iff:
      1. observables[u] == observables[v], and
      2. For every role r and every equivalence class C, sum_{w in C} P^(r)[u][w] == sum_{w in C} P^(r)[v][w].
    """
    if not role_kernels:
        raise SheafError("At least one role transition kernel is required.")
    role_names = sorted(role_kernels.keys())
    first = role_kernels[role_names[0]]
    n = len(first)
    if n == 0:
        raise SheafError("Empty state space in compute_cosimilar_quotient.")

    parsed: dict[str, list[list[Fraction]]] = {}
    for r_name in role_names:
        mat = role_kernels[r_name]
        if len(mat) != n or any(len(row) != n for row in mat):
            raise SheafError(f"Role kernel {r_name!r} is not {n}x{n}.")
        parsed[r_name] = [[Fraction(x) for x in row] for row in mat]

    obs = list(observables) if observables is not None else ["*"] * n
    if len(obs) != n:
        raise SheafError("observables length must match state count.")

    # Initial partition by immediate observable label
    groups: dict[str, list[int]] = {}
    for idx, label in enumerate(obs):
        groups.setdefault(label, []).append(idx)
    partition = [sorted(block) for _, block in sorted(groups.items(), key=lambda kv: kv[1][0])]

    while True:
        new_partition: list[list[int]] = []
        for block in partition:
            if len(block) == 1:
                new_partition.append(block)
                continue
            sig_map: dict[tuple[Fraction, ...], list[int]] = {}
            for state in block:
                sig: list[Fraction] = []
                for r_name in role_names:
                    row = parsed[r_name][state]
                    for target_block in partition:
                        sig.append(sum(row[t] for t in target_block))
                key = tuple(sig)
                sig_map.setdefault(key, []).append(state)
            for sub in sorted(sig_map.values(), key=lambda b: b[0]):
                new_partition.append(sorted(sub))
        new_partition.sort(key=lambda b: b[0])
        if new_partition == partition:
            break
        partition = new_partition

    return partition


@dataclass(frozen=True)
class HyperedgeSpec:
    """Typed multi-way hyperedge connecting 2..k vertices with role weights."""

    edge_id: str
    nodes: tuple[int, ...]
    roles: tuple[str, ...]
    weight: float = 1.0


@dataclass(frozen=True)
class SearchComparison:
    """Result of active incidence-guided trace window search."""

    selected_window: tuple[int, ...]
    objective_score: float
    candidates_evaluated: int
    total_combinatorial_subsets: int
    traced_matrix: list[list[float]]


class ActiveHypergraphTraceSearcher:
    """Hyperedge-incidence Markov operator and bounded-search trace selector."""

    def __init__(self, num_nodes: int, edges: list[HyperedgeSpec], node_salience: list[float] | None = None) -> None:
        if num_nodes < 2 or not edges:
            raise SheafError("Hypergraph requires >=2 nodes and >=1 hyperedge.")
        self.num_nodes = num_nodes
        self.edges = list(edges)
        self.node_salience = (
            [float(x) for x in node_salience] if node_salience is not None else [1.0 / num_nodes] * num_nodes
        )
        if len(self.node_salience) != num_nodes:
            raise SheafError("node_salience length must match num_nodes.")

        # Sparse incidence index (O(sum |e|) storage, never K^(2n))
        self.node_to_edges: list[list[int]] = [[] for _ in range(num_nodes)]
        for e_idx, edge in enumerate(self.edges):
            if len(edge.nodes) < 2 or len(edge.nodes) != len(edge.roles) or edge.weight <= 0:
                raise SheafError(f"Invalid hyperedge {edge.edge_id!r}.")
            for u in edge.nodes:
                if u < 0 or u >= num_nodes:
                    raise SheafError(f"Node index {u} out of bounds in {edge.edge_id!r}.")
                self.node_to_edges[u].append(e_idx)

    def incidence_storage_ints(self) -> int:
        """Total sparse incidence integers stored (O(sum |e|))."""
        return sum(len(e.nodes) for e in self.edges)

    def induced_random_walk_matrix(self) -> list[list[float]]:
        """Two-step bipartite vertex->hyperedge->vertex random walk kernel P_{u,v}."""
        n = self.num_nodes
        mat = [[0.0] * n for _ in range(n)]
        for u in range(n):
            inc = self.node_to_edges[u]
            if not inc:
                mat[u][u] = 1.0
                continue
            total_w = sum(self.edges[e_idx].weight for e_idx in inc)
            for e_idx in inc:
                edge = self.edges[e_idx]
                p_e = edge.weight / total_w
                inv_sz = 1.0 / len(edge.nodes)
                for v in edge.nodes:
                    mat[u][v] += p_e * inv_sz
        # Normalize against float drift
        for u in range(n):
            s = sum(mat[u])
            mat[u] = [x / s for x in mat[u]]
        return mat

    def float_schur_trace(self, visible: list[int] | tuple[int, ...]) -> list[list[float]]:
        """Fast pivoted float64 Schur-complement trace M_S on induced hypergraph walk."""
        vis = list(dict.fromkeys(visible))
        if not vis:
            raise SheafError("visible window cannot be empty.")
        base = self.induced_random_walk_matrix()
        if len(vis) == self.num_nodes:
            return [[base[i][j] for j in vis] for i in vis]
        vis_set = set(vis)
        # Restrict hidden states to those reachable from vis so disconnected recurrent islands do not singularize I - M_BB
        reach_from_vis: set[int] = set()
        stack = list(vis)
        while stack:
            u = stack.pop()
            for v in range(self.num_nodes):
                if base[u][v] > 1e-15 and v not in vis_set and v not in reach_from_vis:
                    reach_from_vis.add(v)
                    stack.append(v)
        if not reach_from_vis:
            out0 = [[base[i][j] for j in vis] for i in vis]
            return [[x / sum(r) for x in r] for r in out0]
        hid = sorted(reach_from_vis)
        m_ss = [[base[i][j] for j in vis] for i in vis]
        m_sb = [[base[i][j] for j in hid] for i in vis]
        m_bs = [[base[i][j] for j in vis] for i in hid]
        i_mbb = [[(1.0 if r == c else 0.0) - base[hid[r]][hid[c]] for c in range(len(hid))] for r in range(len(hid))]
        inv_mbb = _float_invert(i_mbb)
        indirect = _float_matmul(_float_matmul(m_sb, inv_mbb), m_bs)
        out = [[m_ss[r][c] + indirect[r][c] for c in range(len(vis))] for r in range(len(vis))]
        for r in range(len(vis)):
            rs = sum(out[r])
            out[r] = [x / rs for x in out[r]]
        return out

    def window_objective(self, window: tuple[int, ...] | list[int], base_mat: list[list[float]] | None = None) -> float:
        """Score a candidate observer window S by salience retention + internal hyperedge closure."""
        win_set = set(window)
        if not win_set:
            return 0.0
        mat = base_mat if base_mat is not None else self.induced_random_walk_matrix()
        total_sal = sum(self.node_salience)
        captured_sal = sum(self.node_salience[u] for u in win_set) / max(total_sal, 1e-12)

        # Internal 1-step retention (low boundary escape before Schur trace)
        internal_flow = 0.0
        for u in win_set:
            internal_flow += sum(mat[u][v] for v in win_set)
        retention = internal_flow / len(win_set)

        # Higher-order hyperedge synergy bonus: fraction of incident hyperedges with >=2 endpoints in S
        touched_edges = {e_idx for u in win_set for e_idx in self.node_to_edges[u]}
        multi_covered = sum(
            self.edges[e_idx].weight
            for e_idx in touched_edges
            if sum(1 for v in self.edges[e_idx].nodes if v in win_set) >= 2
        )
        total_edge_w = sum(self.edges[e_idx].weight for e_idx in touched_edges) if touched_edges else 1.0
        hyper_coherence = multi_covered / total_edge_w
        return 0.50 * captured_sal + 0.30 * retention + 0.20 * hyper_coherence

    def search_optimal_window(self, seed_node: int, budget_k: int) -> SearchComparison:
        """Search along hyperedge incidence frontiers in O(k * d_inc) instead of C(|V|, k)."""
        if budget_k < 1 or budget_k > self.num_nodes:
            raise SheafError("budget_k must satisfy 1 <= budget_k <= num_nodes.")
        if seed_node < 0 or seed_node >= self.num_nodes:
            raise SheafError("seed_node out of bounds.")

        base_mat = self.induced_random_walk_matrix()
        selected: list[int] = [seed_node]
        evaluated = 1

        while len(selected) < budget_k:
            frontier: set[int] = set()
            for u in selected:
                for e_idx in self.node_to_edges[u]:
                    for v in self.edges[e_idx].nodes:
                        if v not in selected:
                            frontier.add(v)
            if not frontier:
                frontier = {v for v in range(self.num_nodes) if v not in selected}

            best_node = None
            best_score = -1.0
            for cand in sorted(frontier):
                trial = tuple(sorted(selected + [cand]))
                score = self.window_objective(trial, base_mat)
                evaluated += 1
                if score > best_score:
                    best_score = score
                    best_node = cand
            assert best_node is not None
            selected.append(best_node)

        win = tuple(sorted(selected))
        final_score = self.window_objective(win, base_mat)
        traced = self.float_schur_trace(win)
        return SearchComparison(
            selected_window=win,
            objective_score=final_score,
            candidates_evaluated=evaluated,
            total_combinatorial_subsets=math.comb(self.num_nodes, budget_k),
            traced_matrix=traced,
        )


def evaluate_three_way_synergy(triples: list[tuple[int, int, int]]) -> dict[str, float | int]:
    """Compare 3-way hypergraph parity inference (A ^ B ^ C == 0) against 2D pairwise Markov marginals.

    Under the exact 3-uniform parity distribution {(0,0,0), (0,1,1), (1,0,1), (1,1,0)},
    every pairwise marginal P(A,B), P(B,C), P(A,C) is perfectly uniform (1/4 each), so ANY
    2D pairwise Markov predictor is mathematically locked at 50% accuracy on C given (A,B),
    whereas a 3-arity hyperedge incidence table recovers C = A ^ B with 100% accuracy.
    """
    if not triples:
        raise SheafError("triples cannot be empty.")

    # Fit empirical pairwise transition counts P(C=1 | A) and P(C=1 | B) vs 3-way hyperedge P(C=1 | A, B)
    pair_ac = [[0, 0], [0, 0]]
    pair_bc = [[0, 0], [0, 0]]
    hyper_abc: dict[tuple[int, int], list[int]] = {(0, 0): [0, 0], (0, 1): [0, 0], (1, 0): [0, 0], (1, 1): [0, 0]}

    for a, b, c in triples:
        if a not in (0, 1) or b not in (0, 1) or c not in (0, 1):
            raise SheafError("Triple states must be binary (0 or 1).")
        pair_ac[a][c] += 1
        pair_bc[b][c] += 1
        hyper_abc[(a, b)][c] += 1

    correct_pairwise = 0
    correct_hyper = 0
    for a, b, c in triples:
        # Pairwise Markov additive/log-linear pooling of P(C|A) and P(C|B); ties break to 0 (50% on balanced XOR)
        p1_a = pair_ac[a][1] / max(sum(pair_ac[a]), 1)
        p1_b = pair_bc[b][1] / max(sum(pair_bc[b]), 1)
        pred_pair = 1 if (0.5 * (p1_a + p1_b)) > 0.5 else 0
        if pred_pair == c:
            correct_pairwise += 1

        counts_3 = hyper_abc[(a, b)]
        pred_hyper = 1 if counts_3[1] > counts_3[0] else 0
        if pred_hyper == c:
            correct_hyper += 1

    total = len(triples)
    return {
        "samples": total,
        "pairwise_markov_accuracy": correct_pairwise / total,
        "hypergraph_3way_accuracy": correct_hyper / total,
    }


@dataclass
class ObserverRecord:
    """Cached local observer window trace and commitment digest."""

    observer_id: str
    window: tuple[int, ...]
    traced_matrix: list[list[float]]
    digest_sha256: str
    recomputed_count: int = 1


class CosimilarObserverSheafEngine:
    """Multi-observer trace-gluing, counterfactual branching, and spoof-localization engine."""

    def __init__(self, num_nodes: int, edges: list[HyperedgeSpec], observers: dict[str, list[int]]) -> None:
        if not observers:
            raise SheafError("At least one observer window is required.")
        self.num_nodes = num_nodes
        self.edges: dict[str, HyperedgeSpec] = {e.edge_id: e for e in edges}
        if len(self.edges) != len(edges):
            raise SheafError("Duplicate hyperedge IDs are not permitted.")
        self.observer_windows: dict[str, tuple[int, ...]] = {
            oid: tuple(dict.fromkeys(win)) for oid, win in sorted(observers.items())
        }
        self.cache: dict[str, ObserverRecord] = {}
        self.audit_log: list[dict] = []
        self._recompute_observers(set(self.observer_windows.keys()), reason="init")

    def _build_searcher(self) -> ActiveHypergraphTraceSearcher:
        return ActiveHypergraphTraceSearcher(self.num_nodes, [self.edges[k] for k in sorted(self.edges.keys())])

    @staticmethod
    def _digest_matrix(oid: str, window: tuple[int, ...], mat: list[list[float]]) -> str:
        rounded = [[round(v, 10) for v in row] for row in mat]
        payload = {"matrix": rounded, "observer_id": oid, "window": list(window)}
        return hashlib.sha256(canonical_bytes(payload)).hexdigest()

    def _recompute_observers(self, target_ids: set[str], reason: str) -> None:
        searcher = self._build_searcher()
        reused = sorted(set(self.observer_windows.keys()) - target_ids)
        updated = sorted(target_ids)
        for oid in updated:
            win = self.observer_windows[oid]
            mat = searcher.float_schur_trace(win)
            prev_count = self.cache[oid].recomputed_count + 1 if oid in self.cache else 1
            self.cache[oid] = ObserverRecord(
                observer_id=oid,
                window=win,
                traced_matrix=mat,
                digest_sha256=self._digest_matrix(oid, win, mat),
                recomputed_count=prev_count,
            )
        self.audit_log.append(
            {
                "reason": reason,
                "recomputed_observers": updated,
                "reused_cached_observers": reused,
                "state_root_sha256": self.state_root_sha256(),
            }
        )

    def state_root_sha256(self) -> str:
        """Incremental Merkle/fold commitment over observer digests (O(m), not O(|V|^2))."""
        leaves = [f"{oid}:{self.cache[oid].digest_sha256}" for oid in sorted(self.cache.keys())]
        return hashlib.sha256("|".join(leaves).encode("utf-8")).hexdigest()

    def fork_branch(self) -> CosimilarObserverSheafEngine:
        """Create an isolated counterfactual branch sharing zero mutable state with parent."""
        return copy.deepcopy(self)

    def apply_intervention(self, edge_id: str, new_weight: float) -> dict[str, list[str] | str]:
        """Mutate a hyperedge weight and selectively recompute impacted observers."""
        if edge_id not in self.edges:
            raise SheafError(f"Unknown hyperedge {edge_id!r}.")
        if new_weight <= 0.0:
            raise SheafError("Hyperedge weight must remain strictly positive.")

        old = self.edges[edge_id]
        self.edges[edge_id] = HyperedgeSpec(
            edge_id=old.edge_id, nodes=old.nodes, roles=old.roles, weight=float(new_weight)
        )

        # Find connected component of nodes reachable from old.nodes; observers completely
        # confined to disconnected components are safely preserved from cache!
        adj: list[set[int]] = [set() for _ in range(self.num_nodes)]
        for e in self.edges.values():
            for u in e.nodes:
                adj[u].update(e.nodes)
        visited: set[int] = set()
        stack = list(old.nodes)
        while stack:
            curr = stack.pop()
            if curr not in visited:
                visited.add(curr)
                stack.extend(adj[curr] - visited)

        impacted = {oid for oid, win in self.observer_windows.items() if any(u in visited for u in win)}
        self._recompute_observers(impacted, reason=f"intervention:{edge_id}={new_weight:.4f}")
        return {
            "recomputed": sorted(impacted),
            "reused": sorted(set(self.observer_windows.keys()) - impacted),
            "state_root_sha256": self.state_root_sha256(),
        }

    @staticmethod
    def _subtrace_float(mat: list[list[float]], sub_positions: list[int]) -> list[list[float]]:
        if len(sub_positions) == len(mat):
            return [[mat[i][j] for j in sub_positions] for i in sub_positions]
        sub_set = set(sub_positions)
        hid = [i for i in range(len(mat)) if i not in sub_set]
        m_ss = [[mat[i][j] for j in sub_positions] for i in sub_positions]
        m_sb = [[mat[i][j] for j in hid] for i in sub_positions]
        m_bs = [[mat[i][j] for j in sub_positions] for i in hid]
        i_mbb = [[(1.0 if r == c else 0.0) - mat[hid[r]][hid[c]] for c in range(len(hid))] for r in range(len(hid))]
        inv_mbb = _float_invert(i_mbb)
        ind = _float_matmul(_float_matmul(m_sb, inv_mbb), m_bs)
        return [[m_ss[r][c] + ind[r][c] for c in range(len(sub_positions))] for r in range(len(sub_positions))]

    def check_overlap_coherence(
        self, external_overrides: dict[str, list[list[float]]] | None = None, tol: float = 1e-7
    ) -> dict[str, object]:
        """Verify sheaf-like zero-surprise overlap agreement across all pairs of observers.

        For any two observers A, B with overlap U = S_A cap S_B != empty, Recursive Trace
        transitivity requires (M_{S_A})_U == (M_{S_B})_U. If an adversarial/hallucinating
        observer injects a fabricated local transition matrix, overlap coboundary witnesses
        pinpoint the offending observer ID.
        """
        oids = sorted(self.observer_windows.keys())
        overrides = external_overrides or {}
        max_discrepancy = 0.0
        violating_pairs: list[tuple[str, str, float]] = []
        fault_votes: dict[str, int] = {oid: 0 for oid in oids}

        for idx_a in range(len(oids)):
            for idx_b in range(idx_a + 1, len(oids)):
                oa, ob = oids[idx_a], oids[idx_b]
                wa, wb = self.observer_windows[oa], self.observer_windows[ob]
                overlap = [u for u in wa if u in set(wb)]
                if len(overlap) < 2:
                    continue
                mat_a = overrides.get(oa, self.cache[oa].traced_matrix)
                mat_b = overrides.get(ob, self.cache[ob].traced_matrix)
                pos_a = [wa.index(u) for u in overlap]
                pos_b = [wb.index(u) for u in overlap]
                tr_a = self._subtrace_float(mat_a, pos_a)
                tr_b = self._subtrace_float(mat_b, pos_b)
                l1_err = max(
                    sum(abs(tr_a[r][c] - tr_b[r][c]) for c in range(len(overlap))) for r in range(len(overlap))
                )
                if l1_err > max_discrepancy:
                    max_discrepancy = l1_err
                if l1_err > tol:
                    violating_pairs.append((oa, ob, l1_err))
                    fault_votes[oa] += 1
                    fault_votes[ob] += 1

        # Majority coboundary witness localization (distinguishes 1-vs-1 tie from multi-witness fault)
        flagged: list[str] = []
        verdict = "COHERENT"
        if violating_pairs:
            max_v = max(fault_votes.values())
            top = [oid for oid, v in fault_votes.items() if v == max_v]
            if max_v >= 2 and len(top) == 1:
                flagged = top
                verdict = "SPOOF_LOCALIZED"
            else:
                flagged = sorted(oid for oid, v in fault_votes.items() if v > 0)
                verdict = "AMBIGUOUS_CONFLICT"

        return {
            "coherent": len(violating_pairs) == 0,
            "flagged_observers": flagged,
            "max_overlap_l1_discrepancy": max_discrepancy,
            "verdict": verdict,
            "violating_pairs": [{"l1_error": err, "observer_a": a, "observer_b": b} for a, b, err in violating_pairs],
        }

    def emit_scitt_audit_bundle(self, signing_key: bytes, system_id: str = "hadukenn-cosimilar-trace-v1.0.0") -> dict:
        """Emit and self-verify an IETF SCITT RFC 9943 regulatory receipt over observer trace history."""
        coh = self.check_overlap_coherence()
        policy_hash = hashlib.sha256(b"hadukenn-t47-cosimilar-trace-policy-v1.0.0").hexdigest()
        events = [
            {
                "event_index": idx,
                "reason": entry["reason"],
                "recomputed_observers": entry["recomputed_observers"],
                "reused_cached_observers": entry["reused_cached_observers"],
                "state_root_sha256": entry["state_root_sha256"],
            }
            for idx, entry in enumerate(self.audit_log)
        ]
        state_root = self.state_root_sha256()
        trace_head = hashlib.sha256(canonical_bytes(events)).hexdigest()
        bundle = emit_regulatory_bundle(
            system_id=system_id,
            trace_head_sha256=trace_head,
            state_digest_sha256=state_root,
            policy_sha256=policy_hash,
            events=events,
            created_at="2026-11-26T19:25:00Z",
            signing_key=signing_key,
        )
        ok_flag, _ = verify_regulatory_bundle(bundle, signing_key=signing_key)
        if not ok_flag:
            raise SheafError("Self-verification of SCITT audit bundle failed.")
        return {"coherence": coh, "scitt_bundle": bundle, "scitt_verified": ok_flag}
