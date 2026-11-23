import math

import pytest

from sheaf_verify.stats import hedges_g, student_t_two_tailed, welch


def test_zero_t_is_certain():
    assert student_t_two_tailed(0.0, 8) == 1.0


def test_student_t_reference_value():
    assert student_t_two_tailed(1.0, 8) == pytest.approx(0.346593507087334, rel=1e-9)


def test_fixture_welch_against_bpe():
    primary = [1.75, 2.0, 2.25, 1.5, 2.5]
    control = [2.25, 2.5, 2.75, 2.5, 2.5]
    result = welch(primary, control)
    assert result.t == pytest.approx(-2.581988897471611, rel=1e-12)
    assert result.df == pytest.approx(5.538461538461538, rel=1e-12)
    assert result.p == pytest.approx(0.04481595032209489, rel=1e-12)
    assert hedges_g(primary, control) == pytest.approx(-1.4749615655468602, rel=1e-12)
    assert math.isfinite(result.p)
