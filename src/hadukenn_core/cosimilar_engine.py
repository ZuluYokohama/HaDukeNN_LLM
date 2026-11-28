"""Multi-seed empirical campaign runner for the v1.0.0 Cosimilar Trace Engine (T-47).

Benchmarks HaDukeNN's Cosimilar Hypergraph Recursive Trace & Observer Engine
head-to-head against:
  1. Naive 1-step bucket lumping (Goodman class-factored style) vs exact cosimilar
     quotients and Schur-complement path traces across 5 seeds.
  2. 2D Cartesian-product Markov fusion matrices (O(K^(2n))) and 2D pairwise Markov
     marginal shadows vs sparse k-ary hypergraph incidence representations.
  3. Exhaustive C(|V|, k) observer-window subset search vs active hyperedge-incidence
     guided trace search.
  4. Unaudited mutable observer state vs sheaf-glued counterfactual branching,
     disconnected-component cache reuse, spoof localization, and SCITT RFC 9943 proofs.
"""

from __future__ import annotations

import json
import math
import random
import time
from fractions import Fraction
from pathlib import Path

from sheaf_verify.cosimilar_trace import (
    ActiveHypergraphTraceSearcher,
    CosimilarObserverSheafEngine,
    ExactRationalMarkovKernel,
    HyperedgeSpec,
    compute_cosimilar_quotient,
    evaluate_three_way_synergy,
)

SEEDS = (13, 21, 34, 55, 89)


def _build_cosimilar_block_kernel(seed: int) -> tuple[ExactRationalMarkovKernel, list[list[int]], list[list[int]]]:
    """Construct an 8-state rational Markov chain with exact 4-class cosimilar symmetry.

    Pairs (0,1), (2,3), (4,5), (6,7) are probabilistically bisimilar (exact cosimilar
    quotient has 4 classes -> 0.000000 bits KL surprise), whereas merging into 2 coarse
    buckets [[0,1,2,3], [4,5,6,7]] breaks lumpability and incurs positive KL surprise!
    """
    rng = random.Random(seed)
    # 4x4 macro transition weights among cosimilar blocks C0, C1, C2, C3
    macro_rows: list[list[Fraction]] = []
    for a in range(4):
        raw = [rng.randint(1, 9) for _ in range(4)]
        # Make strongly asymmetric across C0 vs C1 so merging {C0, C1} violates lumpability
        if a == 0:
            raw = [9, 1, 1, 1]
        elif a == 1:
            raw = [1, 1, 8, 2]
        tot = sum(raw)
        macro_rows.append([Fraction(x, tot) for x in raw])

    mat8: list[list[Fraction]] = [[Fraction(0, 1) for _ in range(8)] for _ in range(8)]
    for b_src in range(4):
        for offset in (0, 1):
            u = 2 * b_src + offset
            for b_dst in range(4):
                p_block = macro_rows[b_src][b_dst]
                # Asymmetric intra-target split that still sums to exact p_block on each target block!
                split = Fraction(1, 3) if (offset == 0) else Fraction(2, 3)
                mat8[u][2 * b_dst] = p_block * split
                mat8[u][2 * b_dst + 1] = p_block * (Fraction(1, 1) - split)

    cosimilar_partition = [[0, 1], [2, 3], [4, 5], [6, 7]]
    coarse_noncosimilar = [[0, 1, 2, 3], [4, 5, 6, 7]]
    return ExactRationalMarkovKernel(mat8), cosimilar_partition, coarse_noncosimilar


def run_t47_cosimilar_campaign() -> dict:
    """Execute the 5-seed T-47 Cosimilar Hypergraph Trace benchmark campaign."""
    t_start = time.perf_counter()

    # Realm 1: Exact Schur Trace Laws & Cosimilar vs Coarse Lumping Surprise (5 seeds)
    trace_checks_passed = 0
    cosimilar_surprises: list[float] = []
    coarse_surprises: list[float] = []
    dilations: list[float] = []

    for seed in SEEDS:
        kern, cosim_part, coarse_part = _build_cosimilar_block_kernel(seed)
        laws = kern.verify_trace_laws(outer_window=[0, 1, 2, 3, 4, 5], inner_window=[0, 2, 4])
        discovered_quotient = compute_cosimilar_quotient(
            {"step": kern.matrix},
            observables=["obs_0"] * 4 + ["obs_1"] * 4,
        )
        if (
            laws.stochastic_exact
            and laws.transitive_exact
            and laws.stationary_restricted_exact
            and laws.dilation_multiplicative_exact
            and discovered_quotient == cosim_part
        ):
            trace_checks_passed += 1
        dilations.append(laws.total_dilation)
        cosimilar_surprises.append(kern.lumping_surprise_bpb(discovered_quotient))
        coarse_surprises.append(kern.lumping_surprise_bpb(coarse_part))

    # Realm 2: Higher-Order Hypergraph vs 2D Cartesian Markov Product & Pairwise Shadows
    # (a) 3-way XOR parity synergy across 5 seeds (400 samples/seed)
    pair_accs: list[float] = []
    hyper_accs: list[float] = []
    for seed in SEEDS:
        rng = random.Random(seed + 100)
        valid_xor = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
        # Balanced blocks + deterministic shuffle -> zero finite-sample bias
        samples = (valid_xor * 100)[:]
        rng.shuffle(samples)
        syn = evaluate_three_way_synergy(samples)
        pair_accs.append(float(syn["pairwise_markov_accuracy"]))
        hyper_accs.append(float(syn["hypergraph_3way_accuracy"]))

    # (b) State-space memory scaling for n=9 interacting ternary agents (K=3 -> K^n = 19,683 states)
    n_agents, k_states = 9, 3
    cartesian_states = k_states ** n_agents
    cartesian_matrix_entries = cartesian_states * cartesian_states  # 387,420,489 float64 entries!
    cartesian_bytes = cartesian_matrix_entries * 8

    hg_edges_9 = [
        HyperedgeSpec(
            edge_id=f"he_{i}",
            nodes=(i % n_agents, (i + 1) % n_agents, (i + 2) % n_agents),
            roles=("a", "b", "c"),
        )
        for i in range(n_agents * 4)
    ]
    hg_9 = ActiveHypergraphTraceSearcher(num_nodes=n_agents * k_states, edges=[
        HyperedgeSpec(
            edge_id=e.edge_id,
            nodes=tuple(u * k_states + (idx % k_states) for idx, u in enumerate(e.nodes)),
            roles=e.roles,
        )
        for e in hg_edges_9
    ])
    incidence_bytes = hg_9.incidence_storage_ints() * 8
    memory_compression_ratio = cartesian_bytes / max(incidence_bytes, 1)

    # Realm 3: Active Hyperedge-Guided Window Search vs Exhaustive C(|V|, k) Subset Search
    # Test optimality ratio on |V|=18, k=4 (C(18,4) = 3,060 subsets) across 5 seeds + speed on |V|=64, k=8
    optimality_ratios: list[float] = []
    eval_reduction_ratios: list[float] = []
    large_search_ms: list[float] = []

    for seed in SEEDS:
        rng = random.Random(seed + 200)
        v_small, k_small = 18, 4
        sal = [0.05 + rng.random() for _ in range(v_small)]
        # Plant a high-salience connected 4-node hyperedge cluster {2, 5, 9, 13}
        for hot in (2, 5, 9, 13):
            sal[hot] = 4.5
        edges_small = [
            HyperedgeSpec("hot_core", (2, 5, 9, 13), ("r0", "r1", "r2", "r3"), weight=4.0),
            HyperedgeSpec("hot_sub1", (2, 5, 7), ("r0", "r1", "r2"), weight=1.5),
            HyperedgeSpec("hot_sub2", (9, 13, 15), ("r0", "r1", "r2"), weight=1.5),
        ]
        for i in range(v_small):
            edges_small.append(
                HyperedgeSpec(f"ring_{i}", (i, (i + 1) % v_small, (i + 3) % v_small), ("u", "v", "w"), weight=0.6)
            )
        searcher_s = ActiveHypergraphTraceSearcher(v_small, edges_small, sal)
        res_s = searcher_s.search_optimal_window(seed_node=2, budget_k=k_small)

        # Brute-force C(18, 4) = 3,060 subsets containing or not containing seed
        import itertools

        base_m = searcher_s.induced_random_walk_matrix()
        brute_best = max(
            searcher_s.window_objective(comb, base_m) for comb in itertools.combinations(range(v_small), k_small)
        )
        optimality_ratios.append(res_s.objective_score / brute_best)
        eval_reduction_ratios.append(res_s.total_combinatorial_subsets / res_s.candidates_evaluated)

        # Large hypergraph |V|=64, k=8 (C(64,8) = 4,426,165,368 subsets!)
        v_large, k_large = 64, 8
        sal_l = [0.1 + rng.random() for _ in range(v_large)]
        edges_l = [
            HyperedgeSpec(
                f"he_l_{i}",
                (i, (i + 1) % v_large, (i + 4) % v_large, (i + 9) % v_large),
                ("a", "b", "c", "d"),
                weight=1.0 + (i % 3) * 0.5,
            )
            for i in range(v_large)
        ]
        searcher_l = ActiveHypergraphTraceSearcher(v_large, edges_l, sal_l)
        t0 = time.perf_counter()
        _ = searcher_l.search_optimal_window(seed_node=0, budget_k=k_large)
        large_search_ms.append((time.perf_counter() - t0) * 1000.0)

    # Realm 4: Multi-Observer Sheaf Coherence, Disconnected Cache Reuse, Fork Isolation & Spoof Detection
    spoof_detected_seeds = 0
    fork_isolation_seeds = 0
    scitt_passed_seeds = 0
    selective_reuse_seeds = 0

    for seed in SEEDS:
        # Two disconnected 6-node hypergraph islands (0..5 and 6..11) to verify selective cache reuse!
        edges_obs = [
            HyperedgeSpec("isl0_a", (0, 1, 2, 3), ("a", "b", "c", "d"), weight=2.0),
            HyperedgeSpec("isl0_b", (1, 2, 4, 5), ("a", "b", "c", "d"), weight=1.5),
            HyperedgeSpec("isl0_c", (0, 3, 4, 5), ("a", "b", "c", "d"), weight=1.2),
            HyperedgeSpec("isl1_a", (6, 7, 8, 9), ("a", "b", "c", "d"), weight=2.0),
            HyperedgeSpec("isl1_b", (7, 8, 10, 11), ("a", "b", "c", "d"), weight=1.7),
        ]
        observers = {
            "obs_alpha": [0, 1, 2, 3],
            "obs_beta": [1, 2, 3, 4],
            "obs_gamma": [0, 2, 3, 5],
            "obs_island2": [6, 7, 8, 10],
        }
        eng = CosimilarObserverSheafEngine(12, edges_obs, observers)
        base_root = eng.state_root_sha256()
        base_coh = eng.check_overlap_coherence()

        # Fork counterfactual branch and mutate isl0_a -> obs_island2 must be reused from cache!
        branch = eng.fork_branch()
        iv = branch.apply_intervention("isl0_a", new_weight=5.5 + 0.1 * seed)
        branch_coh = branch.check_overlap_coherence()

        if (
            eng.state_root_sha256() == base_root
            and branch.state_root_sha256() != base_root
            and bool(base_coh["coherent"])
            and bool(branch_coh["coherent"])
        ):
            fork_isolation_seeds += 1

        if iv["reused"] == ["obs_island2"] and len(iv["recomputed"]) == 3:
            selective_reuse_seeds += 1

        # Inject a fabricated local Markov trace on obs_beta -> overlap
        # witnesses with alpha & gamma must catch obs_beta!
        spoof_mat = [list(row) for row in branch.cache["obs_beta"].traced_matrix]
        spoof_mat[0] = [0.70, 0.10, 0.10, 0.10]
        spoof_coh = branch.check_overlap_coherence(external_overrides={"obs_beta": spoof_mat})
        if (
            spoof_coh["coherent"] is False
            and spoof_coh["verdict"] == "SPOOF_LOCALIZED"
            and spoof_coh["flagged_observers"] == ["obs_beta"]
        ):
            spoof_detected_seeds += 1

        key = f"hadukenn-t47-secret-key-seed-{seed}".encode()
        bundle_res = branch.emit_scitt_audit_bundle(signing_key=key)
        if bundle_res["scitt_verified"] is True:
            scitt_passed_seeds += 1

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    return {
        "campaign_id": "T-47",
        "elapsed_ms": round(elapsed_ms, 3),
        "realm_1_trace_and_cosimilarity": {
            "coarse_lumping_mean_surprise_bpb": round(sum(coarse_surprises) / len(coarse_surprises), 6),
            "coarse_lumping_min_surprise_bpb": round(min(coarse_surprises), 6),
            "cosimilar_quotient_max_surprise_bpb": round(max(cosimilar_surprises), 12),
            "mean_step_dilation_factor": round(sum(dilations) / len(dilations), 4),
            "seeds_passing_all_4_trace_laws": trace_checks_passed,
        },
        "realm_2_hypergraph_vs_2d_markov": {
            "cartesian_2d_bytes_n9_k3": cartesian_bytes,
            "cartesian_2d_states_n9_k3": cartesian_states,
            "hypergraph_3way_xor_accuracy": round(sum(hyper_accs) / len(hyper_accs), 6),
            "hypergraph_incidence_bytes_n9_k3": incidence_bytes,
            "memory_compression_factor": round(memory_compression_ratio, 1),
            "pairwise_2d_markov_xor_accuracy": round(sum(pair_accs) / len(pair_accs), 6),
        },
        "realm_3_active_window_search": {
            "c_64_8_total_subsets": math.comb(64, 8),
            "mean_eval_reduction_factor_v18": round(sum(eval_reduction_ratios) / len(eval_reduction_ratios), 2),
            "mean_large_v64_k8_search_ms": round(sum(large_search_ms) / len(large_search_ms), 3),
            "mean_optimality_ratio_vs_bruteforce_v18": round(sum(optimality_ratios) / len(optimality_ratios), 6),
        },
        "realm_4_observer_sheaf_and_interventions": {
            "fork_isolation_and_coherence_seeds": fork_isolation_seeds,
            "scitt_rfc9943_verified_seeds": scitt_passed_seeds,
            "selective_disconnected_cache_reuse_seeds": selective_reuse_seeds,
            "spoof_observer_localized_seeds": spoof_detected_seeds,
        },
        "schema": "sheaf-verify.cosimilar-trace.v1",
        "seeds": list(SEEDS),
        "version": "1.0.0",
    }


def write_t47_artifacts(repo_root: Path) -> Path:
    """Execute T-47 campaign and persist deterministic JSON artifact under iterations/."""
    out_dir = repo_root / "iterations" / "t47-cosimilar-trace"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = run_t47_cosimilar_campaign()
    out_path = out_dir / "T47_COSIMILAR_TRACE_RESULTS.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    path = write_t47_artifacts(root)
    print(f"Wrote {path}")
