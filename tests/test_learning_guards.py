"""Pure-stdlib unit tests for ``sheaf_verify.learning_guards`` (Feature 11)."""

from __future__ import annotations

import math

import pytest

from sheaf_verify.hashing import SheafError
from sheaf_verify.learning_guards import (
    adaboost_exp_bound,
    bic_score,
    gini_impurity,
    hoeffding_bound,
    jl_min_dimension,
    logsumexp,
    rademacher_kernel_bound,
    reject_option_decision,
    tikhonov_optimal_lambda,
    tikhonov_stability_rate,
    validate_split_disjointness,
    verify_elbo_monotonicity,
    zigzag_effective_pi,
    zigzag_persistence_ratio,
)


def test_petersen_bounds_valid():
    h_200 = hoeffding_bound(200, 0.05, h_size=10)
    h_800 = hoeffding_bound(800, 0.05, h_size=10)
    assert h_200 == pytest.approx(math.sqrt((math.log(10) + math.log(2.0 / 0.05)) / 400.0))
    assert h_800 < h_200

    assert rademacher_kernel_bound(2.0, 16.0, 100) == pytest.approx(0.08)
    assert rademacher_kernel_bound(0.0, 16.0, 100, s=1.0) == pytest.approx(0.1)

    assert tikhonov_stability_rate(1.5, 0.03, 100) == pytest.approx(1.5)
    assert tikhonov_optimal_lambda(1.5, 2.0, 100) == pytest.approx(math.sqrt((2 * 2.25) / 400.0))

    assert jl_min_dimension(1000, 0.1) == 13816
    assert adaboost_exp_bound(0.1, 20) == pytest.approx(math.exp(-0.4))


def test_petersen_bounds_reject_invalid_params():
    for fn, args in (
        (hoeffding_bound, (0, 0.05, 1)),
        (hoeffding_bound, (100, 0.05, 0)),
        (hoeffding_bound, (100, 0.0, 1)),
        (hoeffding_bound, (100, 1.0, 1)),
        (hoeffding_bound, (True, 0.05, 1)),
        (rademacher_kernel_bound, (-1.0, 16.0, 100)),
        (rademacher_kernel_bound, (1.0, -1.0, 100)),
        (rademacher_kernel_bound, (1.0, 16.0, 0)),
        (tikhonov_stability_rate, (-1.0, 0.1, 100)),
        (tikhonov_stability_rate, (1.0, 0.0, 100)),
        (tikhonov_stability_rate, (1.0, 0.1, 0)),
        (tikhonov_optimal_lambda, (-1.0, 1.0, 10)),
        (tikhonov_optimal_lambda, (1.0, 0.0, 10)),
        (tikhonov_optimal_lambda, (1.0, 1.0, 0)),
        (jl_min_dimension, (1, 0.1)),
        (jl_min_dimension, (100, 0.0)),
        (jl_min_dimension, (100, 0.5)),
        (adaboost_exp_bound, (0.0, 10)),
        (adaboost_exp_bound, (0.5, 10)),
        (adaboost_exp_bound, (0.1, 0)),
    ):
        with pytest.raises(SheafError):
            fn(*args)
    with pytest.raises(SheafError):
        rademacher_kernel_bound(1.0, 16.0, 100, s=-0.1)


def test_petersen_bounds_extreme_finite_overflow_and_underflow_guards():
    # Large finite inputs that do not overflow when factored without squaring
    assert tikhonov_stability_rate(1e200, 1e200, 10) == pytest.approx(2e199)
    assert tikhonov_optimal_lambda(1e200, 1.0, 10) == pytest.approx(math.sqrt(2.0 / 10.0) * 1e200)
    assert tikhonov_optimal_lambda(1.0, 1e200, 10) == pytest.approx(math.sqrt(2.0 / 10.0) * 1e-200)
    assert jl_min_dimension(2, 1e-150) > 0

    # Extreme underflow/overflow raises SheafError cleanly (no ZeroDivisionError/OverflowError)
    assert hoeffding_bound(1, 1e-320) == pytest.approx(math.sqrt((math.log(2.0) - math.log(1e-320)) / 2.0))
    with pytest.raises(SheafError):
        hoeffding_bound(10**400, 0.05)
    with pytest.raises(SheafError):
        rademacher_kernel_bound(1e308, 1e308, 1)
    with pytest.raises(SheafError):
        rademacher_kernel_bound(1.0, 1.0, 10**400)
    with pytest.raises(SheafError):
        adaboost_exp_bound(0.1, 10**400)
    with pytest.raises(SheafError):
        jl_min_dimension(2, 1e-200)
    with pytest.raises(SheafError):
        tikhonov_stability_rate(1e200, 1.0, 10)
    with pytest.raises(SheafError):
        tikhonov_stability_rate(1.0, 1.0, 10**400)
    with pytest.raises(SheafError):
        tikhonov_optimal_lambda(1e200, 1e-200, 10)
    with pytest.raises(SheafError):
        tikhonov_optimal_lambda(1.0, 1.0, 10**400)


def test_soulmachine_logsumexp_bic_elbo_and_reject_option():
    hi = logsumexp([1000.0, 1001.0])
    lo = logsumexp([-1000.0, -1001.0])
    assert math.isfinite(hi) and hi == pytest.approx(1001.0 + math.log1p(math.exp(-1.0)))
    assert math.isfinite(lo) and lo == pytest.approx(-1000.0 + math.log1p(math.exp(-1.0)))
    for bad_vals in ("123", [], [1.0, float("nan")], [1.0, float("inf")]):
        with pytest.raises(SheafError):
            logsumexp(bad_vals)  # type: ignore[arg-type]

    assert bic_score(-50.0, 5, 100) == pytest.approx(-50.0 - 2.5 * math.log(100.0))
    with pytest.raises(SheafError):
        bic_score(-1e308, 10**308, 10)
    with pytest.raises(SheafError):
        bic_score(-1.0, 10**400, 10)
    with pytest.raises(SheafError):
        bic_score(-10.0, -1, 10)
    with pytest.raises(SheafError):
        bic_score(-10.0, 1, 0)

    elbo_ok = verify_elbo_monotonicity([-12.0, -9.5, -8.1])
    assert elbo_ok["ok"] is True
    assert elbo_ok["min_delta"] == pytest.approx(1.4)
    elbo_drop = verify_elbo_monotonicity([-5.0, -4.0, -7.5])
    assert elbo_drop["ok"] is False
    for bad_elbo in ("ab", [], [-5.0, float("nan"), -4.0]):
        with pytest.raises(SheafError):
            verify_elbo_monotonicity(bad_elbo)  # type: ignore[arg-type]
    with pytest.raises(SheafError):
        verify_elbo_monotonicity([-1.0, -0.5], tol=-1.0)

    acc = reject_option_decision([0.9, 0.1], 0.2, 1.0)
    assert acc["reject"] is False and acc["predicted_class"] == 0
    rej = reject_option_decision([0.6, 0.4], 0.2, 1.0)
    assert rej["reject"] is True and rej["predicted_class"] is None
    # Exact equality boundary lambda_r == lambda_s * (1 - max_p) must NOT reject even under float64 rounding
    for probs, lr, ls in (
        ([0.8, 0.2], 0.2, 1.0),
        ([0.7, 0.3], 0.3, 1.0),
        ([0.6, 0.4], 0.4, 1.0),
        ([0.7, 0.3], 0.6, 2.0),
    ):
        boundary = reject_option_decision(probs, lr, ls)
        assert boundary["reject"] is False and boundary["predicted_class"] == 0
    assert reject_option_decision([0.699999, 0.300001], 0.3, 1.0)["reject"] is True
    assert reject_option_decision([0.5, 0.5], lambda_r=1e-13, lambda_s=1e-12)["reject"] is True
    assert reject_option_decision([1.0 - 1e-15, 1e-15], lambda_r=0.0, lambda_s=1.0)["reject"] is True
    assert reject_option_decision([1.0, 0.0], lambda_r=0.0, lambda_s=1.0)["reject"] is False
    for bad_args in (
        ("01", 0.2, 1.0),
        ([], 0.2, 1.0),
        ([-0.1, 1.1], 0.2, 1.0),
        ([0.6, 0.6], 0.2, 1.0),
        ([0.7, 0.3], 0.2, 0.0),
        ([0.7, 0.3], -0.1, 1.0),
        ([0.7, 0.3], 1.5, 1.0),
    ):
        with pytest.raises(SheafError):
            reject_option_decision(*bad_args)  # type: ignore[arg-type]


def test_zigzag_effective_pi_and_persistence_ratio():
    grid_4 = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 2.0, 0.0],
        [0.0, 3.0, 4.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    pi_hat_4 = zigzag_effective_pi(grid_4)
    assert pi_hat_4[1][1] == pytest.approx(10.0)

    grid_3 = [[0.0, 0.0, 0.0], [0.0, 2.0, 3.0], [0.0, 1.0, 4.0]]
    pi_hat_3 = zigzag_effective_pi(grid_3)
    assert len(pi_hat_3) == 2 and pi_hat_3[1][1] == pytest.approx(10.0)

    assert zigzag_persistence_ratio(pi_hat_4, beta_p=0, l1=0, l2=1) == 0.0
    assert zigzag_persistence_ratio([[1.0, 2.0], [3.0, 4.0]], beta_p=2, l1=0, l2=0) == pytest.approx(1.0)
    assert zigzag_persistence_ratio([[1.0, 2.0], [3.0, 4.0]], beta_p=10**400, l1=0, l2=1) == 0.0
    assert zigzag_persistence_ratio([[0.0, 1e308], [0.0, 0.0]], beta_p=10**310, l1=0, l2=0) == pytest.approx(0.01)

    for bad_grid in (
        "12",
        [],
        ["12", "34"],
        [[1.0, 2.0]],
        [[-0.5, 0.0], [0.0, 0.0]],
        [[1e308] * 4 for _ in range(4)],
    ):
        with pytest.raises(SheafError):
            zigzag_effective_pi(bad_grid)  # type: ignore[arg-type]

    with pytest.raises(SheafError):
        zigzag_persistence_ratio(pi_hat_4, beta_p=-1)
    with pytest.raises(SheafError):
        zigzag_persistence_ratio(pi_hat_4, beta_p=1, l1=0, l2=5)
    with pytest.raises(SheafError):
        zigzag_persistence_ratio([[1e308] * 4 for _ in range(4)], beta_p=1, l1=1, l2=2)


def test_alam_gini_impurity_and_split_disjointness():
    assert gini_impurity([10, 0]) == pytest.approx(0.0)
    assert gini_impurity([5, 5]) == pytest.approx(0.5)
    assert gini_impurity([1, 1, 1, 1]) == pytest.approx(0.75)
    assert gini_impurity([0.7, 0.3]) == pytest.approx(0.42)
    assert gini_impurity([1e308, 1e308]) == pytest.approx(0.5)
    assert gini_impurity([1e-320, 1e-320]) == pytest.approx(0.5)

    for bad_counts in ("12", [], [0, 0], [-1, 5], [float("nan"), 0.5]):
        with pytest.raises(SheafError):
            gini_impurity(bad_counts)  # type: ignore[arg-type]

    clean = validate_split_disjointness({"tr1", "tr2", "syn1"}, {"te1", "te2"}, {"syn1"})
    assert clean["ok"] is True and clean["n_synthetic"] == 1

    for tr, te, syn in (
        ("tr", {"te"}, set()),
        (set(), {"te"}, set()),
        ({"tr"}, set(), set()),
        ({"a", "b"}, {"b", "c"}, set()),
        ({"a", "syn1"}, {"c", "syn2"}, {"syn1", "syn2"}),
    ):
        with pytest.raises(SheafError):
            validate_split_disjointness(tr, te, syn)  # type: ignore[arg-type]
