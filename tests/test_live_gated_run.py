"""Helpers and safety checks for the live LapBeast gated-run driver."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from sheaf_verify.campaign import _t46_ready_advisory, verify_t46_tri_machine_hal

ROOT = Path(__file__).resolve().parents[1]


def _load_driver():
    path = ROOT / "scripts" / "live_gated_run.py"
    spec = importlib.util.spec_from_file_location("live_gated_run", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_driver_refuses_sealed_t46_path(tmp_path):
    mod = _load_driver()
    sealed = mod.SEALED_T46
    with pytest.raises(SystemExit, match="sealed"):
        mod.write_live_json('{"schema": "x"}', sealed)


def test_write_live_json_strips_bundle(tmp_path):
    mod = _load_driver()
    dest = tmp_path / "live.json"
    raw = json.dumps({"schema": "sheaf-verify.tri-machine-hal.v1", "regulatory_compliance": {"bundle": {"k": 1}, "scitt_rfc9943_verified": True}})
    mod.write_live_json("note\n" + raw, dest)
    doc = json.loads(dest.read_text())
    assert "bundle" not in doc["regulatory_compliance"]
    assert doc["regulatory_compliance"]["scitt_rfc9943_verified"] is True


def test_ensure_seal_key_creates_mode_600_and_is_idempotent(tmp_path):
    mod = _load_driver()
    path = tmp_path / "keys" / "seal"
    first = mod.ensure_seal_key(path)
    data = first.read_bytes()
    assert first.stat().st_mode & 0o777 == 0o600
    assert len(data) == 32
    second = mod.ensure_seal_key(path)
    assert second.read_bytes() == data


def test_t46_advisory_uses_probed_sm89_not_hardcoded_sm120():
    doc = {
        "host_probe": {
            "cuda_arch_primary": "sm_89",
            "primary_accel": "NVIDIA GeForce RTX 4080 Laptop GPU (12282 MiB, sm_89)",
        },
        "profiles": {"lapbeast": {"cuda_arch_primary": "sm_89"}},
    }
    text = _t46_ready_advisory(doc)
    assert "TRI_MACHINE_HAL_READY" in text
    assert "sm_89" in text
    assert "4080" in text
    assert "sm_120/sm_75" not in text


def test_verify_t46_advisory_follows_document_arch():
    live = ROOT / "iterations" / "t46-tri-machine-hal" / "T46_TRI_MACHINE_RESULTS.lapbeast-live.json"
    if not live.is_file():
        pytest.skip("live T46 JSON is not in the tree")
    doc = json.loads(live.read_text())
    _ok, _criteria, advisories = verify_t46_tri_machine_hal(doc)
    joined = " ".join(advisories)
    assert "TRI_MACHINE_HAL_READY" in joined
    assert "sm_120/sm_75" not in joined
    arch = (doc.get("host_probe") or {}).get("cuda_arch_primary") or doc["profiles"]["lapbeast"]["cuda_arch_primary"]
    assert arch in joined


def test_lapbeast_profile_cuda_flags_are_sm89_only():
    from hadukenn_core.tri_machine import LAPBEAST_PROFILE

    flags = " ".join(LAPBEAST_PROFILE.compiler_flags_cuda)
    assert "sm_89" in flags
    assert "sm_120" not in flags
    assert LAPBEAST_PROFILE.cuda_arch_primary == "sm_89"
    assert "4080" in LAPBEAST_PROFILE.primary_accel
