"""Stdlib tests for the T-49 real-corpus pre-registration and watchdog."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "iterations" / "t49-hst-foundation-stage" / "T49_REALCORPUS_PREREG.json"


def _mod():
    path = ROOT / "scripts" / "t49_realcorpus.py"
    spec = importlib.util.spec_from_file_location("t49_realcorpus", path)
    assert spec is not None and spec.loader is not None
    # Importing the script inserts src/ onto sys.path; that is intentional.
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prereg_has_locked_protocol_fields():
    doc = json.loads(PREREG.read_text(encoding="utf-8"))
    assert doc["schema"] == "sheaf-verify.t49-realcorpus-prereg.v1"
    assert doc["campaign_id"] == "T-49-REAL-WIKI103"
    assert doc["seeds"]["official"] == [13, 21, 34, 55, 89]
    assert doc["statistical_test"]["alpha"] == 0.01
    assert doc["statistical_test"]["method"].startswith("sheaf_verify.stats.paired_t")
    assert doc["corpus"]["raw_bytes"]["wiki.train.raw"] == 539931870
    assert len(doc["corpus"]["raw_sha256"]["wiki.train.raw"]) == 64
    assert doc["model"]["d_model"] == 512
    assert {arm["arm_id"] for arm in doc["arms"]} == {"hst_v1", "byte_bpe"}
    assert (
        "will win" not in doc["hypothesis_to_test"].lower()
        or "unknown" in json.dumps(doc["explicit_non_claims"]).lower()
    )
    assert doc["status"] == "preregistered_not_run"


def test_watchdog_pause_at_83_stop_at_85_and_ac():
    m = _mod()
    assert m.decide_guard(82.9, True, 99, is_running=True)[0] == "run"
    assert m.decide_guard(83.0, True, 99, is_running=True) == (
        "pause",
        "gpu 83.0C >= pause 83C",
    )
    assert m.decide_guard(85.0, True, 99, is_running=True)[0] == "stop"
    assert m.decide_guard(70.0, False, 50, is_running=True)[0] == "pause"
    assert m.decide_guard(70.0, False, 10, is_running=True)[0] == "stop"


def test_budget_lock_quantizes_and_refuses_too_slow():
    m = _mod()
    # 2 MB/s * 0.85 * 9h / 10 runs = 5.508e9, capped at 200e6
    assert m.lock_train_bytes(bytes_per_sec=2_000_000) == 200_000_000
    # 20 kB/s is too slow for the 20 MB minimum
    try:
        m.lock_train_bytes(bytes_per_sec=5_000)
    except RuntimeError as exc:
        assert "minimum" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
    # 80 kB/s * 0.85 * 32400 / 10 = 220,320,000 -> cap 200e6? 80e3*0.85*32400=2.203e9 /10=2.20e8 -> 200e6
    assert m.lock_train_bytes(bytes_per_sec=80_000) == 200_000_000
    # 25 kB/s: 25e3*0.85*32400/10 = 68,850,000 -> 60e6
    assert m.lock_train_bytes(bytes_per_sec=25_000) == 60_000_000
