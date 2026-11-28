"""Pure-stdlib unit tests for ``sheaf_verify.formal_verifiers`` (Feature 10)."""

from __future__ import annotations

from fractions import Fraction

import pytest

from sheaf_verify.formal_verifiers import (
    DIMENSIONLESS,
    GOLDILOCKS_P,
    HASH_RATE,
    MAX_TICK_ABS,
    PLANCK_TIME_S,
    SECOND,
    WATT,
    QFraction,
    SIUnit,
    canonical_receipt_sha256,
    verify_canonical_receipt,
    verify_goldilocks_r1cs_dipole,
    verify_sha256_causal_cone,
    verify_simt_warp_and_banks,
    verify_tick_monotonicity,
)
from sheaf_verify.hashing import SheafError


def test_constants_exact():
    assert PLANCK_TIME_S == 5.391247e-44
    assert MAX_TICK_ABS == (1 << 62) == 4611686018427387904
    assert GOLDILOCKS_P == (1 << 64) - (1 << 32) + 1 == 18446744069414584321


def test_qfraction_arithmetic_reduction_and_equality():
    q1 = QFraction(6, 4)
    q2 = QFraction(-2, -6)
    q3 = QFraction(1, -2)
    assert (q1.num, q1.den) == (3, 2)
    assert (q2.num, q2.den) == (1, 3)
    assert (q3.num, q3.den) == (-1, 2)
    assert repr(QFraction(4, 1)) == "4"
    assert repr(q1) == "3/2"

    assert q1 + q2 == QFraction(11, 6)
    assert 1 + q2 == QFraction(4, 3)
    assert q1 - q2 == QFraction(7, 6)
    assert 1 - q2 == QFraction(2, 3)
    assert q1 * q2 == QFraction(1, 2)
    assert 2 * q2 == QFraction(2, 3)
    assert q1 / q2 == QFraction(9, 2)
    assert 1 / q2 == 3
    assert -q2 == QFraction(-1, 3)
    assert +q2 == q2

    assert QFraction(2, 1) == 2
    assert QFraction(2, 1) != 3
    assert QFraction(3, 2) != 2
    assert (QFraction(1, 1) == True) is False  # noqa: E712
    assert QFraction(3, 2) == Fraction(6, 4)
    assert hash(QFraction(2, 1)) == hash(2) == hash(Fraction(2, 1))
    assert (QFraction(1, 2) + "bad" if False else True)
    with pytest.raises(TypeError):
        _ = QFraction(1, 2) + "x"  # type: ignore[operator]
    with pytest.raises(TypeError):
        _ = "x" - QFraction(1, 2)  # type: ignore[operator]
    with pytest.raises(TypeError):
        _ = QFraction(1, 2) * "x"  # type: ignore[operator]
    with pytest.raises(TypeError):
        _ = "x" / QFraction(1, 2)  # type: ignore[operator]
    with pytest.raises(TypeError):
        _ = QFraction(1, 2) - "x"  # type: ignore[operator]
    with pytest.raises(TypeError):
        _ = QFraction(1, 2) / "x"  # type: ignore[operator]


def test_qfraction_rejects_bool_and_zero_den():
    with pytest.raises(ZeroDivisionError):
        QFraction(1, 0)
    with pytest.raises(ZeroDivisionError):
        _ = QFraction(1, 2) / QFraction(0, 1)
    with pytest.raises(SheafError):
        QFraction(True, 1)
    with pytest.raises(SheafError):
        QFraction(1, False)


def test_siunit_algebra_and_guards():
    energy = WATT * SECOND
    assert energy.dims == (2, 1, -2, 0, 0, 0, 0)
    assert (DIMENSIONLESS / SECOND).same_dims(HASH_RATE) is True
    assert SECOND.same_dims(WATT) is False
    assert (WATT * SECOND) == SIUnit((2, 1, -2, 0, 0, 0, 0), 1.0, "J")
    assert "SIUnit" in repr(SECOND)

    ms = SIUnit((0, 0, 1, 0, 0, 0, 0), scale=1e-3, symbol="")
    combined = ms * DIMENSIONLESS
    assert combined.scale == pytest.approx(1e-3)
    assert (SECOND / ms).scale == pytest.approx(1000.0)

    frac_unit = SIUnit((Fraction(1, 2), QFraction(0), 0, 0, 0, 0, 0), scale=2.0)
    assert frac_unit.dims[0] == QFraction(1, 2)

    zero_unit = SIUnit((0, 0, 1, 0, 0, 0, 0), scale=0.0)
    with pytest.raises(ZeroDivisionError):
        _ = SECOND / zero_unit

    with pytest.raises(SheafError):
        SIUnit((0, 0, 1), scale=1.0)
    with pytest.raises(SheafError):
        SIUnit("0010000")  # type: ignore[arg-type]
    with pytest.raises(SheafError):
        SIUnit((0, 0, True, 0, 0, 0, 0))
    with pytest.raises(SheafError):
        SIUnit((0, 0, 1, 0, 0, 0, 0), scale=float("nan"))
    with pytest.raises(SheafError):
        SIUnit((0, 0, 1, 0, 0, 0, 0), scale=1.0, symbol=123)  # type: ignore[arg-type]
    with pytest.raises(SheafError):
        SECOND.same_dims("s")  # type: ignore[arg-type]
    assert (SECOND == "s") is False
    with pytest.raises(TypeError):
        _ = SECOND * 2  # type: ignore[operator]
    with pytest.raises(TypeError):
        _ = SECOND / 2  # type: ignore[operator]


def test_verify_tick_monotonicity_valid_and_invalid():
    ok_res = verify_tick_monotonicity(
        timestamps=[0.0, 1e-6, 2.5e-6],
        ticks=[-MAX_TICK_ABS, 0, 100, 100, MAX_TICK_ABS],
    )
    assert ok_res["ok"] is True
    assert ok_res["min_dt"] == pytest.approx(1e-6)
    assert ok_res["max_abs_tick"] == MAX_TICK_ABS

    assert verify_tick_monotonicity(timestamps=[0.0, 1.0])["ok"] is True
    assert verify_tick_monotonicity(ticks=[5])["ok"] is True

    assert verify_tick_monotonicity(timestamps=[0.0, 1e-45])["ok"] is False
    assert verify_tick_monotonicity(ticks=[0, MAX_TICK_ABS + 1])["ok"] is False
    assert verify_tick_monotonicity(ticks=[0, -MAX_TICK_ABS - 1])["ok"] is False
    assert verify_tick_monotonicity(ticks=[10, 9])["ok"] is False

    for bad_kwargs in (
        {},
        {"timestamps": "01"},
        {"timestamps": [0.0]},
        {"timestamps": [0.0, float("nan")]},
        {"timestamps": [0.0, float("inf")]},
        {"timestamps": [True, 1.0]},
        {"ticks": "01"},
        {"ticks": []},
        {"ticks": [True, 2]},
        {"ticks": [1.5]},
    ):
        with pytest.raises(SheafError):
            verify_tick_monotonicity(**bad_kwargs)  # type: ignore[arg-type]


def test_verify_sha256_causal_cone_and_receipt():
    res3 = verify_sha256_causal_cone(mut_word_idx=3)
    assert res3["ok"] is True
    assert res3["dead_terminal_reg_slices"] == 1
    assert res3["invariant_w_words"] == 17
    assert res3["eff_rounds"] < 52.0

    res0 = verify_sha256_causal_cone(mut_word_idx=0)
    assert res0["invariant_w_words"] != 17

    for bad_idx in (-1, 16, True, 1.5):
        with pytest.raises(SheafError):
            verify_sha256_causal_cone(mut_word_idx=bad_idx)  # type: ignore[arg-type]

    body = {"step": 1, "invariant_w_words": 17, "dead_terminal_reg_slices": 1}
    digest = canonical_receipt_sha256(body)
    assert len(digest) == 64
    assert verify_canonical_receipt(body, digest)["ok"] is True

    bad_digest = ("0" if digest[0] != "0" else "1") + digest[1:]
    assert verify_canonical_receipt(body, bad_digest)["ok"] is False

    with pytest.raises(SheafError):
        verify_canonical_receipt(body, "not-a-sha256")
    with pytest.raises(SheafError):
        canonical_receipt_sha256({"x": float("nan")})


@pytest.mark.parametrize(
    ("stride", "expected_conflicts", "expected_ok"),
    [
        (33, 0, True),
        (1, 0, True),
        (32, 31, False),
        (2, 16, False),
        (4, 24, False),
        (16, 30, False),
    ],
)
def test_verify_simt_warp_and_banks(stride: int, expected_conflicts: int, expected_ok: bool):
    lanes = [float(i + 1) for i in range(32)]
    res = verify_simt_warp_and_banks(lanes, stride=stride)
    assert res["conflicts"] == expected_conflicts
    assert res["bank_conflicts"] == expected_conflicts
    assert res["reduced_sum"] == 528.0
    assert res["butterfly_err"] == 0.0
    assert len(res["stage_snapshots"]) == 5
    assert res["ok"] is expected_ok


def test_verify_simt_warp_invalid_inputs():
    with pytest.raises(SheafError):
        verify_simt_warp_and_banks("x" * 32, stride=1)  # type: ignore[arg-type]
    with pytest.raises(SheafError):
        verify_simt_warp_and_banks([1.0] * 31, stride=1)
    with pytest.raises(SheafError):
        verify_simt_warp_and_banks([float("nan")] * 32, stride=1)
    with pytest.raises(SheafError):
        verify_simt_warp_and_banks([True] + [1.0] * 31, stride=1)
    with pytest.raises(SheafError):
        verify_simt_warp_and_banks([1.0] * 32, stride=-1)
    with pytest.raises(SheafError):
        verify_simt_warp_and_banks([1.0] * 32, stride=True)
    with pytest.raises(SheafError, match="overflows float64"):
        verify_simt_warp_and_banks([1e308] * 32, stride=33)
    with pytest.raises(SheafError, match="overflows float64"):
        verify_simt_warp_and_banks([1e308] * 8 + [-1e308] * 8 + [1e308] * 8 + [-1e308] * 8, stride=33)


def test_verify_goldilocks_r1cs_dipole():
    good_dipole = [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5, 0.6, -0.6]
    ok_res = verify_goldilocks_r1cs_dipole(0.25, good_dipole)
    assert ok_res["ok"] is True
    assert ok_res["r1cs_gate_residual_mod_p"] == 0
    assert ok_res["dipole_sum_mod_p"] == 0

    tamper_z = verify_goldilocks_r1cs_dipole(0.25, good_dipole, z_witness=12345)
    assert tamper_z["ok"] is False
    assert tamper_z["r1cs_gate_residual_mod_p"] != 0

    tamper_dip = verify_goldilocks_r1cs_dipole(0.25, good_dipole, v_minus_override=1)
    assert tamper_dip["ok"] is False
    assert tamper_dip["dipole_sum_mod_p"] != 0

    scale_fp = 1 << 24
    y_degen_field = (-scale_fp * pow(3, GOLDILOCKS_P - 2, GOLDILOCKS_P)) % GOLDILOCKS_P
    with pytest.raises(SheafError, match="degenerate Goldilocks denominator"):
        verify_goldilocks_r1cs_dipole(0.25, good_dipole, y_field=y_degen_field)

    with pytest.raises(SheafError):
        verify_goldilocks_r1cs_dipole(float("nan"), good_dipole)
    with pytest.raises(SheafError):
        verify_goldilocks_r1cs_dipole(0.25, "0123456789ab")  # type: ignore[arg-type]
    with pytest.raises(SheafError):
        verify_goldilocks_r1cs_dipole(0.25, [0.1, -0.1])
    with pytest.raises(SheafError):
        verify_goldilocks_r1cs_dipole(0.25, [float("nan")] + good_dipole[1:])
    with pytest.raises(SheafError):
        verify_goldilocks_r1cs_dipole(0.25, good_dipole, y_field=True)
    with pytest.raises(SheafError, match="overflows float64"):
        verify_goldilocks_r1cs_dipole(1e308, good_dipole)
    with pytest.raises(SheafError, match="overflows float64"):
        verify_goldilocks_r1cs_dipole(0.0, [1e308] * 12)
