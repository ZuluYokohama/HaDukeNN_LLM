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


def test_watchdog_resume_hysteresis_matches_prereg():
    m = _mod()
    # Not paused yet: below 83 C runs.
    assert m.decide_guard_hysteresis(81.0, True, 99, is_running=True, paused=False)[0] == "run"
    # Once paused, stay paused until <= 80 C.
    assert m.decide_guard_hysteresis(81.0, True, 99, is_running=True, paused=True)[0] == "pause"
    assert m.decide_guard_hysteresis(80.0, True, 99, is_running=True, paused=True)[0] == "run"
    assert m.decide_guard_hysteresis(85.0, True, 99, is_running=True, paused=True)[0] == "stop"
    assert m.decide_guard_hysteresis(70.0, False, 50, is_running=True, paused=True)[0] == "pause"


class _ByteTok:
    def encode(self, text):
        return list(text.encode("utf-8"))


def test_stream_skip_replays_exactly_like_take(tmp_path):
    m = _mod()
    raw = tmp_path / "train.raw"
    raw.write_bytes(b"".join(f"line {i} with some words\n".encode() for i in range(5000)))
    a = m.ByteWindowStream(raw, _ByteTok(), "byte_bpe", 64, 60_000)
    b = m.ByteWindowStream(raw, _ByteTok(), "byte_bpe", 64, 60_000)
    try:
        for _ in range(37):
            assert a._take(4) is not None
            assert b.skip_batch(4)
        assert a.bytes_charged == b.bytes_charged
        assert a._token_buf == b._token_buf
        assert a._take(4) == b._take(4)
    finally:
        a.close()
        b.close()


def _drain(stream, batch):
    out = []
    while True:
        taken = stream._take(batch)
        if taken is None:
            return out, stream.bytes_charged, stream.eof
        out.append(taken)


def test_parallel_encode_stream_is_identical_to_serial(tmp_path):
    import pytest

    m = _mod()
    raw = tmp_path / "train.raw"
    # Long lines and a missing final newline exercise the pending/eof paths; small reads exercise boundaries.
    body = b"".join(f"row {i} ".encode() * (1 + i % 7) + b"\n" for i in range(4000)) + b"no trailing newline"
    raw.write_bytes(body)
    for budget in (50_000, 123_457, 1_000, 7, len(body) - 5, len(body) + 10):
        serial = m.ByteWindowStream(raw, _ByteTok(), "byte_bpe", 32, budget)
        try:
            parallel = m.ByteWindowStream(raw, _ByteTok(), "byte_bpe", 32, budget, workers=2, mp_context="fork")
        except (OSError, ValueError) as exc:  # pragma: no cover - sandbox without fork
            pytest.skip(f"no process pool: {exc}")
        try:
            assert _drain(serial, 3) == _drain(parallel, 3)
            assert serial.bytes_charged <= budget
        finally:
            serial.close()
            parallel.close()


def test_model_init_is_not_degenerate():
    import pytest

    torch = pytest.importorskip("torch")
    m = _mod()
    doc = json.loads(PREREG.read_text(encoding="utf-8"))
    cfg = dict(doc["model"], n_layers=2)
    torch.manual_seed(0)
    model = m.build_model(cfg, 512, torch.device("cpu"), torch, torch.nn, torch.nn.functional)
    norms = [p for n, p in model.named_parameters() if p.ndim == 1]
    assert norms and all(bool((p == 1).all()) for p in norms)
    x = torch.randint(0, 512, (2, 16))
    logits, loss = model(x, x)
    assert float(logits.abs().max()) > 0.0
    assert abs(float(loss) - __import__("math").log(512)) > 1e-4


def test_watchdog_pauses_when_battery_drains_on_ac():
    m = _mod()
    assert m.decide_guard_hysteresis(60.0, True, 19, is_running=True, paused=False)[0] == "pause"
    assert m.decide_guard_hysteresis(60.0, True, 45, is_running=True, paused=True)[0] == "pause"
    assert m.decide_guard_hysteresis(60.0, True, 50, is_running=True, paused=True)[0] == "run"
    assert m.decide_guard_hysteresis(60.0, True, 30, is_running=True, paused=False)[0] == "run"
    assert m.decide_guard_hysteresis(60.0, True, None, is_running=True, paused=True)[0] == "run"
