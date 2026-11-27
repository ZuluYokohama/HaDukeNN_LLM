"""Replay labeling, sealed-path refusal, and live T-46 pass / own-criteria fail paths."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sheaf_verify.campaign_live import (
    REPLAY_LABEL,
    SEALED_T46,
    evaluate_campaign,
    evaluate_t46_live_criteria,
    resolve_want_live,
    run_t46_live,
)
from sheaf_verify.hashing import SheafError

ROOT = Path(__file__).resolve().parents[1]
SEALED_T46_REL = ROOT / "iterations" / "t46-tri-machine-hal" / "T46_TRI_MACHINE_RESULTS.json"


def test_t46_json_only_is_labeled_replay():
    payload = evaluate_campaign(mode="t46-tri-machine-hal", path=SEALED_T46_REL, replay=True)
    assert payload["evaluation"] == "replay"
    assert payload["evaluation_label"] == REPLAY_LABEL
    assert payload["ok"] is True
    assert payload["document_sha256"]
    assert all(item["passed"] for item in payload["criteria"])


def test_t49_json_only_is_labeled_replay():
    path = ROOT / "iterations" / "t49-hst-foundation-stage" / "T49_HST_FOUNDATION_RESULTS.json"
    payload = evaluate_campaign(mode="t49-hst-foundation", path=path, replay=True)
    assert payload["evaluation"] == "replay"
    assert REPLAY_LABEL in payload["evaluation_label"]
    assert payload["ok"] is True


def test_live_and_replay_flags_conflict():
    with pytest.raises(SheafError, match="cannot be combined"):
        resolve_want_live(mode="t46-tri-machine-hal", live=True, replay=True)


def test_live_not_implemented_for_t47():
    with pytest.raises(SheafError, match="not implemented"):
        resolve_want_live(mode="t47-cosimilar-trace", live=True, replay=False)


def test_ci_defaults_to_replay(monkeypatch):
    monkeypatch.setenv("CI", "1")
    assert resolve_want_live(mode="t46-tri-machine-hal", live=False, replay=False) is False


def test_default_is_replay_even_when_hardware_is_present():
    assert resolve_want_live(mode="t46-tri-machine-hal", live=False, replay=False) is False


def test_t46_live_refuses_sealed_overwrite(tmp_path):
    with pytest.raises(SheafError, match="sealed"):
        run_t46_live(sealed_path=SEALED_T46_REL, live_out=SEALED_T46)


def test_t46_live_own_criteria_fail_on_bad_speedup_and_parity():
    live_doc = {
        "host_probe": {"cuda_arch_primary": "sm_89"},
        "benchmarks_and_parity": {
            "cce_loss_max_abs_err": 0.5,
            "cce_dh_max_abs_err": 0.0,
            "cce_de_max_abs_err": 0.0,
            "c11_speedup_over_py_loop": 0.2,
            "c11_speedup_over_numpy_gemm": 0.1,
            "sheaf_kv_gate_max_abs_err": 0.0,
            "sheaf_coherent_neighbor_mass": 1.0,
            "causal_span_prefix_invariance": True,
            "zenbook_npu_int8_cosine_sim": 0.5,
            "zenbook_npu_loop_dynamic_allocs": 3,
            "lapbeast_watchdog_rc143_verified": False,
            "slab_overflow_refusal_verified": False,
        },
    }
    evidence = {
        "summaries": {"parity_max_abs_vs_f64": 0.5, "hal_criteria": {"timings": [], "guards": [], "parity": []}}
    }
    criteria = evaluate_t46_live_criteria(live_doc, evidence)
    failed = {item["id"] for item in criteria if not item["passed"]}
    assert "T46_LIVE_CCE_PARITY" in failed
    assert "T46_LIVE_F64_REFERENCE" in failed
    assert "T46_LIVE_C11_SPEEDUP_VS_SERIAL" in failed
    assert "T46_LIVE_HOST_SIM_INT8_LOOP" in failed
    assert "T46_LIVE_WATCHDOG_STATE_MACHINE" in failed
    assert "T46_LIVE_RAW_TIMINGS" in failed


def test_t46_live_inprocess_when_hal_library_present(tmp_path):
    from sheaf_verify.campaign_live import t46_hardware_present

    if not t46_hardware_present():
        pytest.skip("tri-machine HAL library is not available on this host")
    dest = tmp_path / "T46_live.json"
    payload = run_t46_live(sealed_path=SEALED_T46_REL, live_out=dest, register_store=tmp_path / "store")
    assert dest.is_file()
    assert dest.resolve() != SEALED_T46_REL.resolve()
    assert payload["evaluation"] == "live"
    assert payload["replay"]["evaluation"] == "replay"
    assert payload["replay"]["ok"] is True
    # Sealed bytes unchanged
    sealed = json.loads(SEALED_T46_REL.read_text())
    assert sealed.get("measurement_mode") != "live" or "sealed" in SEALED_T46_REL.name
    doc = json.loads(dest.read_text())
    assert "bundle" not in (doc.get("regulatory_compliance") or {})
    assert doc["measurement_mode"] == "live"
    assert payload["ok"] is True, payload["criteria"]
    assert (tmp_path / "store" / "campaign-live-pointer.json").is_file()
