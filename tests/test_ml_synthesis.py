"""Unit, invariant, and policy-exclusion tests for ``ml_synthesis`` and ``architecture_policy``."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import pearsonr, spearmanr

from hadukenn_core.architecture_policy import (
    validate_e1_ordering_policy,
    validate_hypergraph_claim,
    validate_output_head_policy,
    validate_partition_policy,
)
from hadukenn_core.hypergraph import (
    CorpusDocument,
    build_chodrow_null,
    build_decontaminated_hypergraph,
)
from hadukenn_core.ml_synthesis import (
    GiniDecisionTree,
    RandomForestClassifierLite,
    adaboost_edge_weights,
    ard_rbf_kernel,
    bfgs_optimize,
    bfgs_step,
    binned_mutual_information,
    compute_diffusion_maps,
    dcm_burstiness_log_prob,
    fisher_score_kernel,
    fit_moe_gem,
    jl_project,
    normalized_cut_fiedler,
    train_only_smote,
)
from hadukenn_core.partitions import build_balanced_partition
from sheaf_verify.hashing import SheafError
from sheaf_verify.learning_guards import (
    adaboost_exp_bound,
    validate_split_disjointness,
    verify_elbo_monotonicity,
)
from sheaf_verify.rlm_guards import SiipVerifyGate


def test_petersen_diffusion_maps_ncut_isolated_node_jl_and_adaboost() -> None:
    c0 = np.array([[0.12 * i + 0.005 * (i**2), 0.14 * i + 0.04 * ((i * 3) % 5)] for i in range(8)], dtype=float)
    c1 = np.array([[3.2 + 0.12 * i, 0.14 * i + 0.04 * ((i * 2 + 1) % 5)] for i in range(8)], dtype=float)
    pts = np.vstack([c0, c1])
    diff_sq = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1)
    w = np.exp(-diff_sq / (1.1**2))
    np.fill_diagonal(w, 0.0)

    d_res = compute_diffusion_maps(w, t=2, p_dim=4)
    phi = d_res["embedding"]
    assert phi.shape == (16, 4)

    deg = np.sum(w, axis=1)
    p_mat = w / deg[:, None]
    p2 = p_mat @ p_mat
    iu, ju = np.triu_indices(16, k=1)
    rw_dist = np.array([np.sum(((p2[i] - p2[j]) ** 2) / deg) for i, j in zip(iu, ju, strict=True)])
    emb_dist = np.sum((phi[iu] - phi[ju]) ** 2, axis=1)
    assert float(pearsonr(emb_dist, rw_dist).statistic) > 0.95
    assert float(spearmanr(emb_dist, rw_dist).statistic) > 0.90

    n_res = normalized_cut_fiedler(w)
    planted = np.array([0] * 8 + [1] * 8)
    agree = max(float(np.mean(n_res["labels"] == planted)), float(np.mean(n_res["labels"] != planted)))
    assert agree >= 0.95
    assert n_res["ncut"] < 0.25

    # Isolated node (D_ii == 0) finiteness and N <= p_dim guard
    w_iso = w.copy()
    w_iso[15, :] = 0.0
    w_iso[:, 15] = 0.0
    iso_diff = compute_diffusion_maps(w_iso, t=2, p_dim=4)
    assert np.all(np.isfinite(iso_diff["embedding"]))
    assert iso_diff["isolated_nodes"] == 1
    assert math.isfinite(normalized_cut_fiedler(w_iso)["ncut"])

    with pytest.raises(SheafError):
        compute_diffusion_maps(np.ones((4, 4)) - np.eye(4), t=2, p_dim=4)

    # JL projection + AdaBoost bound
    rng = np.random.default_rng(42)
    x_hi = rng.normal(0.0, 1.0, (40, 128))
    jl_res = jl_project(x_hi, p_dim=48, seed=42)
    x_lo = jl_res["projected"]
    assert x_lo.shape == (40, 48)
    assert jl_res["jl_min_dimension"] > 48
    iu_j, ju_j = np.triu_indices(40, k=1)
    ratio = np.median(
        np.linalg.norm(x_lo[iu_j] - x_lo[ju_j], axis=1)
        / np.maximum(np.linalg.norm(x_hi[iu_j] - x_hi[ju_j], axis=1), 1e-9)
    )
    assert 0.65 <= float(ratio) <= 1.35

    y_edge = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)
    weak = np.vstack([y_edge.copy() for _ in range(5)])
    weak[:, 0] = -weak[:, 0]
    ada_res = adaboost_edge_weights(y_edge, weak)
    assert np.all(ada_res["alphas"] > 0.0)
    assert ada_res["prod_zt"] <= adaboost_exp_bound(0.1, 5) + 1e-12


def test_soulmachine_ard_mi_fisher_moe_bfgs_and_dcm() -> None:
    rng = np.random.default_rng(77)
    x = rng.normal(0.0, 1.0, (36, 4))
    y = 2.0 * x[:, 0] + 0.1 * rng.normal(0.0, 1.0, 36)

    assert binned_mutual_information(x[:, 0], y) > binned_mutual_information(x[:, 3], y) >= 0.0
    assert binned_mutual_information(np.ones(10), y[:10]) == 0.0

    k_inf = ard_rbf_kernel(x[:6], x[:6], length_scales=np.array([1.0, 1.0, 1.0, float("inf")]))
    k_3d = ard_rbf_kernel(x[:6, :3], x[:6, :3], length_scales=np.array([1.0, 1.0, 1.0]))
    assert np.allclose(k_inf, k_3d, atol=1e-12)
    assert np.allclose(np.diag(k_inf), 1.0, atol=1e-12)

    # Fisher kernel rejects reg <= 0 when N < D
    x_under = rng.normal(0.0, 1.0, (4, 12))
    with pytest.raises(SheafError):
        fisher_score_kernel(x_under, reg=0.0)
    k_fish = fisher_score_kernel(x_under, reg=0.25)
    assert k_fish.shape == (4, 4)
    assert float(np.min(np.linalg.eigvalsh(0.5 * (k_fish + k_fish.T)))) >= -1e-9

    # MoE GEM monotone ELBO & BIC
    moe = fit_moe_gem(x, y, n_experts=2, max_iter=10)
    assert verify_elbo_monotonicity(moe["elbo_trace"])["passed"] is True
    assert math.isfinite(moe["bic"]) and moe["bic"] != 0.0

    # BFGS step rejects s^T y <= 0; bfgs_optimize keeps B_inv SPD on -sum(w^2) and converges on convex
    with pytest.raises(SheafError):
        bfgs_step(np.eye(2), np.array([1.0, 0.0]), np.array([-0.5, 0.0]))
    neg_res = bfgs_optimize(lambda w: float(-np.sum(w * w)), lambda w: -2.0 * w, np.array([1.0, 0.5]), max_iter=5)
    assert float(np.min(np.linalg.eigvalsh(neg_res["B_inv"]))) > 0.0

    w_star = np.array([1.5, -0.75, 0.25])
    opt = bfgs_optimize(
        lambda w: float(np.sum((w - w_star) ** 2)),
        lambda w: 2.0 * (w - w_star),
        np.zeros(3),
        max_iter=25,
        siip_gate=SiipVerifyGate(mode="formal"),
    )
    assert float(np.linalg.norm(opt["x"] - w_star)) < 1e-4

    alpha = np.array([0.2, 0.2, 0.2, 0.2])
    assert dcm_burstiness_log_prob(np.array([10, 0, 0, 0]), alpha) > dcm_burstiness_log_prob(
        np.array([3, 3, 2, 2]), alpha
    )


def test_alam_train_only_smote_disjointness_gini_tree_and_rf() -> None:
    rng = np.random.default_rng(88)
    x_tr = rng.normal(0.0, 1.0, (80, 4))
    y_tr = np.zeros(80, dtype=int)
    y_tr[:10] = 1
    x_tr[:10, 0] += 3.2

    x_te = rng.normal(0.0, 1.0, (30, 4))
    y_te = np.zeros(30, dtype=int)
    y_te[:6] = 1
    x_te[:6, 0] += 3.2
    x_te_copy = x_te.copy()

    train_ids = {f"tr_{i}" for i in range(80)}
    test_ids = {f"te_{i}" for i in range(30)}

    with pytest.raises(SheafError):
        train_only_smote(x_tr, y_tr, X_test=x_te, train_ids={"a", "b"}, test_ids={"b", "c"})
    with pytest.raises(SheafError):
        train_only_smote(np.ones((8, 3)), np.zeros(8, dtype=int))

    sm = train_only_smote(x_tr, y_tr, X_test=x_te, train_ids=train_ids, test_ids=test_ids, seed=88)
    assert np.array_equal(x_te, x_te_copy)
    assert sm["n_synthetic"] > 0
    syn_ids = {f"syn_{i}" for i in range(sm["n_synthetic"])}
    assert validate_split_disjointness(train_ids | syn_ids, test_ids, syn_ids)["passed"] is True

    x_bal, y_bal = sm["X_resampled"], sm["y_resampled"]
    tree = GiniDecisionTree(max_depth=4).fit(x_bal, y_bal)
    assert math.isclose(float(np.sum(tree.feature_importances_)), 1.0, abs_tol=1e-6)
    assert float(tree.feature_importances_[0]) > 0.5

    rf = RandomForestClassifierLite(n_estimators=15, max_depth=4, seed=88).fit(x_bal, y_bal)
    preds = rf.predict(x_te)
    recall = int(np.sum((preds == 1) & (y_te == 1))) / int(np.sum(y_te == 1))
    assert recall > 0.5


@pytest.mark.parametrize(
    ("head_kind", "vocab_size", "n_clusters", "cluster_size", "hardware", "batch_size", "dtype", "should_pass"),
    [
        ("goodman_factored", 32_768, 182, 181, "gpu_pretrain", 64, "bf16", False),
        ("cluster_is", 32_768, 182, 181, "gpu_pretrain", 64, "bf16", False),
        ("fp2", 32_768, 182, 181, "gpu_pretrain", 64, "bf16", False),
        (None, 32_768, None, None, "gpu_pretrain", 64, "bf16", True),
        ("cluster_is", 131_072, 1024, 128, "gpu_pretrain", 64, "bf16", True),
        ("goodman_factored", 32_768, 182, 181, "edge_npu", 1, "int8", True),
        ("goodman_factored", 32_768, 182, 181, "edge_npu", 4, "int8", False),
    ],
)
def test_agents_md_output_head_policy_matrix(
    head_kind: str | None,
    vocab_size: int,
    n_clusters: int | None,
    cluster_size: int | None,
    hardware: str,
    batch_size: int,
    dtype: str,
    should_pass: bool,
) -> None:
    if should_pass:
        res = validate_output_head_policy(
            vocab_size=vocab_size,
            n_clusters=n_clusters,
            cluster_size=cluster_size,
            head_kind=head_kind,
            hardware_target=hardware,
            batch_size=batch_size,
            dtype=dtype,
        )
        assert res["ok"] is True and res["passed"] is True
    else:
        with pytest.raises(SheafError):
            validate_output_head_policy(
                vocab_size=vocab_size,
                n_clusters=n_clusters,
                cluster_size=cluster_size,
                head_kind=head_kind,
                hardware_target=hardware,
                batch_size=batch_size,
                dtype=dtype,
            )


def test_chodrow_null_hypergraph_claim_e1_policy_and_partition_bridge() -> None:
    train_docs = [
        CorpusDocument(f"tr_{i}", f"dom_{i % 3}", f"token_{i} shared_{i % 4} anchor_{i % 2}", f"sha_tr_{i}")
        for i in range(16)
    ]
    eval_docs = [
        CorpusDocument(f"ev_{i}", f"dom_{i % 3}", f"eval_{i} shared_{i % 4} anchor_{i % 2}", f"sha_ev_{i}")
        for i in range(4)
    ]
    _, hg = build_decontaminated_hypergraph(train_docs, eval_docs, min_edge_size=2, max_edge_size=12, seed=13)
    null_hg = build_chodrow_null(hg, n_swaps_factor=4, seed=13)

    with pytest.raises(SheafError):
        validate_hypergraph_claim(hg, None, hyper_bpb=1.05)
    with pytest.raises(SheafError):
        validate_hypergraph_claim(hg, null_hg, hyper_bpb=1.10, null_bpb=1.09)
    ok_hg = validate_hypergraph_claim(hg, null_hg, hyper_bpb=1.04, null_bpb=1.11)
    assert ok_hg["ok"] is True and ok_hg["delta_bpb"] > 0.0

    with pytest.raises(SheafError):
        validate_e1_ordering_policy("passive_e1_shuffle", e1_passive_shuffle=True)
    assert validate_e1_ordering_policy("hyper")["ok"] is True

    rng = np.random.default_rng(13)
    part = build_balanced_partition(32, 4, 8, [rng.integers(0, 32, size=48, dtype=np.int32)], mode="mass_random")
    with pytest.raises(SheafError):
        validate_partition_policy(part, head_kind="cluster_is", hardware_target="gpu_pretrain")
    assert validate_partition_policy(part, head_kind="goodman_factored", hardware_target="edge_npu", batch_size=1, dtype="int8")["bijection_ok"] is True


def test_adaboost_perfect_and_bad_learners_and_skewed_mi_and_ard_tiny() -> None:
    # AdaBoost eps_t == 0.0 (perfect weak learner): no ZeroDivisionError, bound holds
    y_ab = np.array([1, 1, -1, -1, 1, -1], dtype=float)
    res_perf = adaboost_edge_weights(y_ab, np.vstack([y_ab, y_ab]))
    assert np.all(np.isfinite(res_perf["alphas"])) and np.all(res_perf["alphas"] > 0.0)
    assert res_perf["prod_zt"] <= res_perf["theoretical_exp_bound"] + 1e-12

    # AdaBoost eps_t >= 0.5 on round 0 or > 0.5 rejected with SheafError
    h_tie = np.array([[1, 1, -1, 1, -1, 1]], dtype=float)  # 3/6 = 0.5 error
    h_bad = np.array([[1, -1, 1, 1, -1, 1]], dtype=float)  # 5/6 > 0.5 error
    with pytest.raises(SheafError):
        adaboost_edge_weights(y_ab, h_tie)
    with pytest.raises(SheafError):
        adaboost_edge_weights(y_ab, h_bad)
    with pytest.raises(SheafError):
        adaboost_edge_weights(y_ab, -y_ab.reshape(1, -1))

    # Skewed discrete/binary MI does not collapse to 0.0
    x_bin = np.array([0, 0, 0, 0, 0, 1], dtype=float)
    mi_bin = binned_mutual_information(x_bin, x_bin, n_bins=4)
    expected_h = -(5.0 / 6.0) * math.log2(5.0 / 6.0) - (1.0 / 6.0) * math.log2(1.0 / 6.0)
    assert math.isclose(mi_bin, expected_h, abs_tol=1e-9)

    # ARD tiny positive length_scale (< 1e-162) avoids 0 * inf = NaN
    pts = np.array([[0.0, 1.0], [1.0, 2.0]])
    k_tiny = ard_rbf_kernel(pts, pts, length_scales=np.array([1e-200, float("inf")]))
    assert np.all(np.isfinite(k_tiny))
    assert np.allclose(np.diag(k_tiny), 1.0, atol=1e-12)


def test_tree_rf_feature_mismatch_smote_nan_bfgs_spd_and_policy_guards() -> None:
    rng = np.random.default_rng(5)
    x4 = rng.normal(size=(36, 4))
    y4 = (x4[:, 3] > 0.0).astype(int)
    dt = GiniDecisionTree(max_depth=3).fit(x4, y4)
    rf = RandomForestClassifierLite(n_estimators=5, max_depth=3, seed=5).fit(x4, y4)

    for model in (dt, rf):
        with pytest.raises(SheafError):
            model.predict(x4[:, :2])
        with pytest.raises(SheafError):
            model.predict(np.hstack([x4, x4[:, :2]]))

    # train_only_smote rejects NaN / Inf in numeric y_train
    x_sm = np.vstack([rng.normal(0.0, 1.0, (18, 2)), rng.normal(2.0, 1.0, (2, 2))])
    y_nan = np.array([0.0] * 18 + [1.0, float("nan")])
    y_inf = np.array([0.0] * 18 + [1.0, float("inf")])
    with pytest.raises(SheafError):
        train_only_smote(x_sm, y_nan)
    with pytest.raises(SheafError):
        train_only_smote(x_sm, y_inf)

    # bfgs_step rejects non-symmetric and non-SPD B_inv
    s = np.array([1.0, 0.0])
    y = np.array([1.0, 0.0])
    with pytest.raises(SheafError):
        bfgs_step(np.diag([1.0, -2.0]), s, y)
    with pytest.raises(SheafError):
        bfgs_step(np.array([[1.0, 1.0], [0.0, 1.0]]), s, y)

    # validate_output_head_policy rejects negative / bool n_clusters & cluster_size
    with pytest.raises(SheafError):
        validate_output_head_policy(128_000, -1000, -128, head_kind="cluster_is")
    with pytest.raises(SheafError):
        validate_output_head_policy(128_000, True, 128, head_kind="cluster_is")

    # validate_hypergraph_claim rejects self-null (hg is null_hg)
    train_docs = [
        CorpusDocument(f"d_{i}", f"dom_{i % 2}", f"t_{i} s_{i % 3}", f"sha_{i}") for i in range(10)
    ]
    eval_docs = [CorpusDocument(f"e_{i}", "dom_0", f"ev_{i} s_{i % 2}", f"sha_e_{i}") for i in range(2)]
    _, hg = build_decontaminated_hypergraph(train_docs, eval_docs, min_edge_size=2, max_edge_size=8, seed=7)
    with pytest.raises(SheafError):
        validate_hypergraph_claim(hg, hg, hyper_bpb=1.05, null_bpb=1.12)

    # validate_partition_policy with live ClusterPartition across valid large-vocab regime
    big_part = build_balanced_partition(64, 8, 8, [rng.integers(0, 64, size=64, dtype=np.int32)], mode="mass_random")
    res_npu = validate_partition_policy(
        big_part, head_kind="cluster_is", hardware_target="edge_npu", batch_size=1, dtype="int8"
    )
    assert res_npu["ok"] is True and res_npu["bijection_ok"] is True
