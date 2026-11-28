"""Sandbox tests for the HaDukeNN CPU slice. Loaded by jovin-harness as the tests file.

These tests import `solution` (the payload module). They are stdlib-only and
must stay under the sandbox input cap. They do not run the host pytest suite
and they do not use a GPU.
"""

from solution import (
    cce_mup_loss_f64,
    evaluate_claim,
    evaluate_guards,
    evaluate_parity,
    evaluate_timings,
)

HAND_H = [[0.2, -0.1, 0.05, 0.3], [0.0, 0.4, -0.2, 0.1]]
HAND_E = [
    [0.1, 0.0, -0.1, 0.2],
    [0.3, -0.2, 0.05, 0.0],
    [-0.1, 0.2, 0.1, -0.05],
]
HAND_Y = [1, 0]
HAND_MUP = 4.0
# Independent float64 recomputation of cce_mup_loss_f64 on HAND_* (muP scale = 1).
HAND_LOSS = [1.0537651865257764, 1.0623660923692526]


def _claim(**overrides):
    document = {
        "format": "sheaf-verify.hal-claim.v1",
        "study_profile": "hal",
        "hypothesis_id": "c11-vs-serial-loop",
        "description": "The compiled C11 tiled CCE kernel is faster than a serial Python loop on the recorded shapes.",
        "performance": {
            "metric": "speedup",
            "direction": "greater",
            "baseline": "serial_python_loop",
            "baselines_reported": ["serial_python_loop", "packed_numpy"],
            "min_speedup": 1.5,
            "subject": "c11_tiled_cce",
        },
        "parity": {"max_abs": 1e-4, "ops": ["cce_mup_fwd"]},
        "guards": {"thermal_stop_celsius": 85},
    }
    document.update(overrides)
    return document


def _parity(loss=None):
    return {
        "format": "sheaf-verify.hal-parity.v1",
        "cases": [
            {
                "name": "cce_tiny",
                "op": "cce_mup_fwd",
                "inputs": {"H": HAND_H, "E": HAND_E, "y": HAND_Y, "mup_base_d": HAND_MUP},
                "kernel_outputs": {"loss": list(HAND_LOSS if loss is None else loss)},
            }
        ],
    }


def _timings(subject_ms=None, serial=None, numpy_ms=None, speedup=None):
    case = {
        "name": "shape_small",
        "subject_ms": subject_ms or [2.0, 2.1, 1.9],
        "baselines_ms": {
            "serial_python_loop": serial or [10.0, 10.2, 9.8],
            "packed_numpy": numpy_ms or [1.0, 1.1, 0.9],
        },
    }
    if speedup is not None:
        case["speedup"] = speedup
    return {"format": "sheaf-verify.hal-timings.v1", "cases": [case]}


def _guards(peak=50, observed=False, draw=None):
    document = {
        "format": "sheaf-verify.hal-guards.v1",
        "thermal": {
            "stop_celsius": 85,
            "unit": "C",
            "samples": [
                {"t": "2026-01-01T00:00:10Z", "celsius": 47},
                {"t": "2026-01-01T00:00:20Z", "celsius": peak},
            ],
        },
        "power": {
            "observed": observed,
            "probe": "nvidia-smi --query-gpu=power.draw --format=csv,noheader",
            "reason": "power.draw not available in the unit-test harness",
        },
    }
    if observed:
        document["power"] = {"observed": True, "draw_watts": draw}
    return document


def _failed(criteria):
    return [item for item in criteria if not item["passed"]]


def test_cce_matches_hand_computed_reference():
    got = cce_mup_loss_f64(HAND_H, HAND_E, HAND_Y, HAND_MUP)
    assert len(got) == 2
    assert all(abs(left - right) < 1e-12 for left, right in zip(got, HAND_LOSS, strict=True))


def test_cce_rejects_ragged_hidden():
    # Under process isolation the tests cannot catch payload exception *classes*
    # (they are not BaseException in the tests process). Assert on the message.
    try:
        cce_mup_loss_f64([[0.1, 0.2], [0.3]], HAND_E, HAND_Y, HAND_MUP)
    except Exception as error:
        assert "ragged" in str(error)
        return
    raise AssertionError("expected an error for a ragged H")


def test_claim_requires_both_baselines():
    claim = _claim()
    claim["performance"]["baselines_reported"] = ["serial_python_loop"]
    failed = _failed(evaluate_claim(claim))
    assert any(item["id"] == "hal.claim.baselines" for item in failed), failed


def test_claim_accepts_honest_serial_baseline():
    failed = _failed(evaluate_claim(_claim()))
    assert failed == []


def test_parity_passes_when_kernel_matches_float64():
    failed = _failed(evaluate_parity(_parity(), _claim()))
    assert failed == []


def test_parity_fails_when_kernel_loss_is_wrong():
    loss = [HAND_LOSS[0] + 0.01, HAND_LOSS[1]]
    failed = _failed(evaluate_parity(_parity(loss=loss), _claim()))
    assert any(item["id"] == "hal.parity.cce_tiny" for item in failed), failed


def test_timings_recompute_from_raw_repeats_and_meet_threshold():
    criteria = evaluate_timings(_timings(), _claim())
    assert _failed(criteria) == []
    threshold = next(item for item in criteria if item["id"] == "hal.perf.threshold")
    assert "5.0000" in threshold["detail"]


def test_timings_fail_when_below_min_speedup():
    # Subject slower than the serial loop, so speedup is well under 1.5.
    failed = _failed(evaluate_timings(_timings(subject_ms=[10.0, 10.2, 9.8], serial=[2.0, 2.1, 1.9]), _claim()))
    assert any(item["id"] == "hal.perf.threshold" for item in failed), failed


def test_timings_reject_self_reported_speedup_mismatch():
    failed = _failed(evaluate_timings(_timings(speedup=999.0), _claim()))
    assert any(item["id"].endswith("speedup_field") for item in failed), failed


def test_guards_pass_below_stop_with_explained_power():
    assert _failed(evaluate_guards(_guards())) == []


def test_guards_fail_when_peak_reaches_stop():
    failed = _failed(evaluate_guards(_guards(peak=85)))
    assert any(item["id"] == "hal.guard.thermal" for item in failed), failed


def test_guards_fail_unobserved_power_without_reason():
    document = _guards()
    document["power"] = {"observed": False}
    failed = _failed(evaluate_guards(document))
    assert any(item["id"] == "hal.guard.power" for item in failed), failed


def test_guards_observed_power_records_draw():
    criteria = evaluate_guards(_guards(observed=True, draw=5.56))
    assert _failed(criteria) == []
    power = next(item for item in criteria if item["id"] == "hal.guard.power")
    assert "5.56" in power["detail"]
