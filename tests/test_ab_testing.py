import math

import pytest

from src.ab_testing import (
    achieved_power,
    analyse_ab_test,
    lift_interval,
    required_sample_size,
    sample_ratio_mismatch,
    two_proportion_ztest,
    wilson_interval,
)


def test_wilson_interval_matches_reference_value():
    # 10 successes in 100 trials: reference Wilson 95% interval is (0.0552, 0.1744)
    lo, hi = wilson_interval(10, 100)
    assert lo == pytest.approx(0.0552, abs=1e-4)
    assert hi == pytest.approx(0.1744, abs=1e-4)


def test_ztest_matches_hand_calculation():
    z, p = two_proportion_ztest(x_c=100, n_c=1000, x_t=130, n_t=1000)
    p_pool = 230 / 2000
    expected_z = 0.03 / math.sqrt(p_pool * (1 - p_pool) * (2 / 1000))
    assert z == pytest.approx(expected_z)
    assert p == pytest.approx(0.0347, abs=1e-3)


def test_lift_interval_contains_point_estimate():
    lo, hi = lift_interval(100, 10000, 130, 10000)
    assert lo < 0.30 < hi


def test_required_sample_size_gives_target_power():
    n = required_sample_size(0.01, 0.012)
    assert achieved_power(0.01, 0.012, int(n), int(n)) == pytest.approx(0.80, abs=0.01)


def test_sample_ratio_mismatch_flags_imbalance():
    assert sample_ratio_mismatch(5000, 5000) == pytest.approx(1.0)
    assert sample_ratio_mismatch(5000, 5600) < 0.001


def test_decision_labels():
    assert analyse_ab_test("x", 100, 10000, 160, 10000, random_state=0).decision == "Treatment wins"
    assert analyse_ab_test("x", 100, 10000, 101, 10000, random_state=0).decision == "No significant difference"
    assert analyse_ab_test("x", 100, 10000, 100, 13000, random_state=0).srm_flag is True


def test_sample_ratio_check_is_off_without_an_intended_split():
    result = analyse_ab_test("x", 100, 10_000, 900, 90_000, expected_split=None, random_state=0)
    assert result.srm_flag is False and math.isnan(result.srm_p_value)
    assert result.control_share == pytest.approx(0.1)
    assert result.decision != "Do not trust: sample ratio mismatch"
    # The same 10/90 traffic does fail against an intended 50/50 split, and passes against 10/90.
    assert analyse_ab_test("x", 100, 10_000, 900, 90_000, expected_split=0.5, random_state=0).srm_flag is True
    assert analyse_ab_test("x", 100, 10_000, 900, 90_000, expected_split=0.1, random_state=0).srm_flag is False
