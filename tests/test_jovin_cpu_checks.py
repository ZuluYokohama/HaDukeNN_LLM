"""Host tests for the receipted jovin-harness CPU slice."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from sheaf_verify.hal import (
    cce_mup_loss_f64,
    evaluate_hal0,
    evaluate_hal4,
    evaluate_hal5,
    evaluate_hal6,
)
from sheaf_verify.hashing import parse_json_bytes

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIDECARS = Path.home() / "hadukenn" / "logs" / "live-gated-sidecars"
MAX_INPUT_BYTES = 262144


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _payload():
    return _load("jovin_cpu_payload", ROOT / "scripts" / "jovin_cpu_payload.py")


def _driver():
    return _load("run_jovin_cpu_checks", ROOT / "scripts" / "run_jovin_cpu_checks.py")


class _Ctx:
    """Minimal Context stand-in for comparing payload evaluators to sheaf_verify.hal."""

    def __init__(self, **docs):
        self._raw = {}
        for role, document in docs.items():
            if document is None:
                continue
            if isinstance(document, bytes):
                self._raw[role] = document
            else:
                self._raw[role] = json.dumps(document).encode("utf-8")

    def has(self, role: str) -> bool:
        return role in self._raw

    def payload(self, role: str) -> bytes:
        return self._raw[role]

    def parse(self, role: str):
        return parse_json_bytes(self._raw[role], role)


def _ids(criteria) -> list[tuple[str, bool, str]]:
    out = []
    for item in criteria:
        if isinstance(item, dict):
            out.append((item["id"], bool(item["passed"]), item["detail"]))
        else:
            out.append((item.id, bool(item.passed), item.detail))
    return out


def _tiny_case():
    hidden = [[0.2, -0.1, 0.05, 0.3], [0.0, 0.4, -0.2, 0.1]]
    embedding = [
        [0.1, 0.0, -0.1, 0.2],
        [0.3, -0.2, 0.05, 0.0],
        [-0.1, 0.2, 0.1, -0.05],
    ]
    targets = [1, 0]
    mup_base_d = 4.0
    loss = cce_mup_loss_f64(hidden, embedding, targets, mup_base_d)
    return hidden, embedding, targets, mup_base_d, loss


def _fixture_docs():
    hidden, embedding, targets, mup_base_d, loss = _tiny_case()
    claim = {
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
    parity = {
        "format": "sheaf-verify.hal-parity.v1",
        "cases": [
            {
                "name": "cce_tiny",
                "op": "cce_mup_fwd",
                "inputs": {"H": hidden, "E": embedding, "y": targets, "mup_base_d": mup_base_d},
                "kernel_outputs": {"loss": loss},
            }
        ],
    }
    timings = {
        "format": "sheaf-verify.hal-timings.v1",
        "cases": [
            {
                "name": "shape_small",
                "subject_ms": [2.0, 2.1, 1.9],
                "baselines_ms": {
                    "serial_python_loop": [10.0, 10.2, 9.8],
                    "packed_numpy": [1.0, 1.1, 0.9],
                },
            }
        ],
    }
    guards = {
        "format": "sheaf-verify.hal-guards.v1",
        "thermal": {
            "stop_celsius": 85,
            "unit": "C",
            "samples": [
                {"t": "2026-01-01T00:00:10Z", "celsius": 47},
                {"t": "2026-01-01T00:00:20Z", "celsius": 50},
            ],
        },
        "power": {
            "observed": False,
            "probe": "nvidia-smi --query-gpu=power.draw --format=csv,noheader",
            "reason": "power.draw not available in the unit-test harness",
        },
    }
    return claim, parity, timings, guards


def test_payload_cce_matches_sheaf_verify():
    payload = _payload()
    hidden, embedding, targets, mup_base_d, expected = _tiny_case()
    got = payload.cce_mup_loss_f64(hidden, embedding, targets, mup_base_d)
    assert got == expected


def test_payload_evaluators_match_hal_gates_on_fixtures():
    payload = _payload()
    claim, parity, timings, guards = _fixture_docs()
    ctx = _Ctx(claim=claim, **{"log.parity": parity, "log.timings": timings, "log.guards": guards})
    assert _ids(payload.evaluate_claim(claim)) == _ids(evaluate_hal0(ctx))
    assert _ids(payload.evaluate_parity(parity, claim)) == _ids(evaluate_hal4(ctx))
    assert _ids(payload.evaluate_timings(timings, claim)) == _ids(evaluate_hal5(ctx))
    assert _ids(payload.evaluate_guards(guards)) == _ids(evaluate_hal6(ctx))


def test_payload_evaluators_match_hal_gates_on_live_sidecars():
    if not all(
        (DEFAULT_SIDECARS / name).is_file() for name in ("claim.json", "parity.json", "timings.json", "guards.json")
    ):
        pytest.skip("live HAL sidecars are not on this host")
    payload = _payload()
    claim = json.loads((DEFAULT_SIDECARS / "claim.json").read_text())
    parity = json.loads((DEFAULT_SIDECARS / "parity.json").read_text())
    timings = json.loads((DEFAULT_SIDECARS / "timings.json").read_text())
    guards = json.loads((DEFAULT_SIDECARS / "guards.json").read_text())
    ctx = _Ctx(claim=claim, **{"log.parity": parity, "log.timings": timings, "log.guards": guards})
    assert _ids(payload.evaluate_claim(claim)) == _ids(evaluate_hal0(ctx))
    assert _ids(payload.evaluate_parity(parity, claim)) == _ids(evaluate_hal4(ctx))
    assert _ids(payload.evaluate_timings(timings, claim)) == _ids(evaluate_hal5(ctx))
    assert _ids(payload.evaluate_guards(guards)) == _ids(evaluate_hal6(ctx))
    assert payload.all_passed(payload.evaluate_claim(claim))
    assert payload.all_passed(payload.evaluate_parity(parity, claim))
    assert payload.all_passed(payload.evaluate_timings(timings, claim))
    assert payload.all_passed(payload.evaluate_guards(guards))


def test_payload_and_assembled_tests_fit_sandbox_input_cap():
    driver = _driver()
    code = (ROOT / "scripts" / "jovin_cpu_payload.py").read_bytes()
    assert len(code) <= MAX_INPUT_BYTES
    tests, _hashes = driver.assemble_tests(None)
    assert len(tests.encode("utf-8")) <= MAX_INPUT_BYTES
    if all(
        (DEFAULT_SIDECARS / name).is_file() for name in ("claim.json", "parity.json", "timings.json", "guards.json")
    ):
        live, hashes = driver.assemble_tests(DEFAULT_SIDECARS)
        assert len(live.encode("utf-8")) <= MAX_INPUT_BYTES
        assert hashes is not None and "parity.json" in hashes


def test_driver_dry_run_reports_sizes(capsys):
    driver = _driver()
    rc = driver.main(["--dry-run", "--no-live"])
    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "dry_run"
    assert summary["gpu"] is False
    assert summary["live_sidecars"] is False
    assert 0 < summary["code_bytes"] <= MAX_INPUT_BYTES
    assert 0 < summary["tests_bytes"] <= MAX_INPUT_BYTES


def test_driver_refuses_sealed_t46_as_out():
    driver = _driver()
    with pytest.raises(SystemExit, match="sealed"):
        driver.main(["--dry-run", "--out", str(driver.SEALED_T46)])


def test_driver_refuses_gpu_flags():
    driver = _driver()
    with pytest.raises(SystemExit, match="GPU"):
        driver.main(["--gpus", "all", "--dry-run"])
    with pytest.raises(SystemExit, match="GPU"):
        driver.main(["--runtime=nvidia"])


def test_live_embed_is_fail_closed_on_mutated_timings(tmp_path):
    if not all(
        (DEFAULT_SIDECARS / name).is_file() for name in ("claim.json", "parity.json", "timings.json", "guards.json")
    ):
        pytest.skip("live HAL sidecars are not on this host")
    sidecars = tmp_path / "sidecars"
    sidecars.mkdir()
    for name in ("claim.json", "parity.json", "timings.json", "guards.json"):
        (sidecars / name).write_text((DEFAULT_SIDECARS / name).read_text(encoding="utf-8"), encoding="utf-8")
    timings = json.loads((sidecars / "timings.json").read_text())
    for case in timings["cases"]:
        case["subject_ms"] = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
    (sidecars / "timings.json").write_text(json.dumps(timings), encoding="utf-8")

    payload = _payload()
    claim = json.loads((sidecars / "claim.json").read_text())
    failed = [item for item in payload.evaluate_timings(timings, claim) if not item["passed"]]
    assert any(item["id"] == "hal.perf.threshold" for item in failed), failed

    driver = _driver()
    text, hashes = driver.assemble_tests(sidecars)
    assert "test_live_timings_recomputed_from_raw_repeats" in text
    assert hashes is not None
    assert hashes["timings.json"] != _driver().sha256_file(DEFAULT_SIDECARS / "timings.json")


def _doctor_ready(driver) -> bool:
    jovin = driver.default_jovin()
    if not jovin.is_file():
        return False
    try:
        health = driver.run_json([str(jovin), "doctor"])
    except SystemExit:
        return False
    return bool(health.get("ready"))


def test_e2e_jovin_cpu_checks_when_sandbox_is_ready(tmp_path):
    driver = _driver()
    if not _doctor_ready(driver):
        pytest.skip("jovin-harness doctor is not ready")
    out = tmp_path / "jovin-cpu-checks.json"
    argv = [sys.executable, str(ROOT / "scripts" / "run_jovin_cpu_checks.py"), "--out", str(out)]
    if all(
        (DEFAULT_SIDECARS / name).is_file() for name in ("claim.json", "parity.json", "timings.json", "guards.json")
    ):
        argv.extend(["--live-sidecars", str(DEFAULT_SIDECARS)])
    else:
        argv.append("--no-live")
    import subprocess

    proc = subprocess.run(argv, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads(out.read_text())
    assert summary["status"] == "passed"
    assert summary["gpu"] is False
    assert summary["chain_valid"] is True
    assert summary["receipt_id"]
    assert summary["receipt_sha256"]
    assert summary["passed_count"] == summary["total"]
    assert summary["integrity"] == "process_isolated"
    if "--live-sidecars" in argv:
        assert summary["live_sidecars"] is True
        assert summary["total"] >= 16  # 12 deterministic + 4 live
