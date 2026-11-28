"""Unit, mathematical-invariant, and edge-case tests for ``hadukenn_core.zigzag_topology``."""

from __future__ import annotations

import numpy as np
import pytest

from hadukenn_core.zigzag_topology import (
    ZigzagTopologyAnalyzer,
    build_knn_vr_complex,
    compute_zigzag_persistence,
    diagnose_token_shuffling,
    prune_layers_plateau,
)
from sheaf_verify.hashing import SheafError


def test_knn_vr_complex_dual_modes_and_distance_matrix() -> None:
    rng = np.random.default_rng(101)
    c0 = rng.normal(0.0, 0.08, (6, 3))
    c1 = rng.normal(0.0, 0.08, (6, 3)) + np.array([10.0, 0.0, 0.0])
    c2 = rng.normal(0.0, 0.08, (6, 3)) + np.array([0.0, 10.0, 0.0])
    pts = np.vstack([c0, c1, c2])
    dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)

    res_cap = build_knn_vr_complex(pts, k_nn=5, radius_quantile=0.22, filter_mode="op21_vr_cap")
    res_cap_d = build_knn_vr_complex(dist, k_nn=5, radius_quantile=0.22, filter_mode="op21_vr_cap")
    assert np.array_equal(res_cap["mask"], res_cap_d["mask"])
    assert res_cap["mask"].shape == (18, 18)
    assert np.array_equal(res_cap["mask"], res_cap["mask"].T)
    assert np.all(np.diag(res_cap["mask"]) == 0)
    nz = res_cap["mask"] > 0
    if np.any(nz):
        assert float(np.max(dist[nz])) <= res_cap["radius"] + 1e-12
    assert res_cap["beta_0"] >= 1

    res_short = build_knn_vr_complex(pts, k_nn=5, radius=1.5, filter_mode="app_d_short_filter")
    nz_short = res_short["mask"] > 0
    if np.any(nz_short):
        assert float(np.min(dist[nz_short])) > 1.5 - 1e-12
    assert res_short["beta_0"] > 1

    with pytest.raises(SheafError):
        build_knn_vr_complex(np.array([[1.0, 2.0]]))
    with pytest.raises(SheafError):
        build_knn_vr_complex(np.array([[0.0, float("nan")], [0.0, 1.0]]))
    with pytest.raises(SheafError):
        build_knn_vr_complex(pts, filter_mode="invalid_mode")


def test_zigzag_analyzer_collinear_beta1_zero_and_plateau_pruning() -> None:
    line_pts = np.column_stack([np.linspace(0.0, 10.0, 12), np.zeros((12, 3))])
    collinear_traj = np.stack([line_pts, line_pts.copy(), line_pts.copy()], axis=0)
    res_col = ZigzagTopologyAnalyzer(k_nn=4).analyze(collinear_traj)
    assert np.all(res_col["beta_1"] == 0)
    assert np.allclose(res_col["z1_bar"], 0.0, atol=1e-12)
    assert np.allclose(res_col["b1_freq"], 0.0, atol=1e-12)

    empty_prune = prune_layers_plateau(res_col["z1_bar"], threshold=0.90)
    assert empty_prune["pruned_layers"] == []

    rng = np.random.default_rng(202)
    traj = np.zeros((8, 24, 12))
    traj[0] = rng.normal(0.0, 1.0, (24, 12))
    traj[1] = traj[0] + rng.normal(0.0, 0.7, (24, 12))
    plateau = rng.normal(0.0, 1.0, (24, 12))
    for ell in (2, 3, 4, 5):
        traj[ell] = plateau + rng.normal(0.0, 0.005, (24, 12))
    traj[6] = rng.normal(0.0, 1.0, (24, 12))
    traj[7] = rng.normal(0.0, 1.0, (24, 12))

    res = compute_zigzag_persistence(traj, k_nn=4)
    z1_bar = res["z1_bar"]
    assert z1_bar.shape == (8,)
    assert np.all(np.isfinite(z1_bar))
    assert np.all(np.isfinite(res["b1_freq"]))

    pruned = prune_layers_plateau(z1_bar, threshold=0.90)
    idx = pruned["pruned_layers"]
    assert len(idx) >= 1
    assert 0 not in idx and 6 not in idx and 7 not in idx
    assert any(layer in (2, 3, 4, 5) for layer in idx)


def test_app_f_shuffling_diagnostic_and_input_guards() -> None:
    rng = np.random.default_rng(303)
    base = rng.normal(0.0, 1.0, (24, 12))
    structured = np.stack([base + 0.02 * ell * rng.normal(0.0, 0.1, (24, 12)) for ell in range(6)], axis=0)
    shuffled = structured.copy()
    for ell in range(1, 5):
        shuffled[ell] = structured[ell, rng.permutation(24)]

    diag = diagnose_token_shuffling(structured, shuffled, k_nn=4)
    assert diag["structured_mid_late_z1"] > diag["shuffled_mid_late_z1"]
    assert diag["is_diagnostic_only"] is True
    assert diag["diagnostic_only"] is True

    with pytest.raises(SheafError):
        ZigzagTopologyAnalyzer(k_nn=4).analyze(np.ones((1, 10, 4)))
    with pytest.raises(SheafError):
        prune_layers_plateau(np.array([0.1, 0.5, 0.2]), threshold=1.5)


def test_zigzag_validation_and_non_plateau_decay_fixes() -> None:
    # 1. Negative / bool protect_readout_layers rejected cleanly with SheafError
    with pytest.raises(SheafError):
        prune_layers_plateau(np.array([0.2, 0.3, 0.95, 0.98, 0.96, 0.94, 0.97, 0.0]), protect_readout_layers=-2)
    with pytest.raises(SheafError):
        prune_layers_plateau(np.array([0.2, 0.3, 0.95, 0.98, 0.96, 0.0]), protect_readout_layers=True)

    # 2. Strictly decaying non-plateau profile yields empty pruned_layers (Paper 2 Alg. 2)
    decay_res = prune_layers_plateau([100.0, 0.2, 0.1, 0.05, 0.01, 0.0], threshold=0.90)
    assert decay_res["pruned_layers"] == []

    # 3. Asymmetric distance matrix & bool k_nn rejected in build_knn_vr_complex
    d_asym = np.array([[0.0, 1.0, 2.0], [10.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    with pytest.raises(SheafError):
        build_knn_vr_complex(d_asym, k_nn=1, is_distance_matrix=True)
    with pytest.raises(SheafError):
        build_knn_vr_complex(np.eye(3), k_nn=True)
