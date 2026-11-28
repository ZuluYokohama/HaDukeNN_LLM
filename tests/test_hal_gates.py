"""Pass and own-criteria fail paths for HAL0–HAL6, plus the training-profile firewall."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from tests.support import criterion, latest

from sheaf_verify.hal import cce_mup_loss_f64, set_probe
from sheaf_verify.hashing import canonical_bytes
from sheaf_verify.runner import SequenceClock, run
from sheaf_verify.store import Store
from sheaf_verify.synthetic import TEST_KEY

ENV = {
    "python": "3.12.11",
    "cuda": "release 12.4",
    "driver": "595.99.02",
    "gpu": "NVIDIA GeForce RTX 4080 Laptop GPU (12282 MiB, sm_89)",
    "platform": "Linux-test-harness",
}

CLAIM = {
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


def _parity_case() -> dict:
    hidden = [[0.2, -0.1, 0.05, 0.3], [0.0, 0.4, -0.2, 0.1]]
    embedding = [
        [0.1, 0.0, -0.1, 0.2],
        [0.3, -0.2, 0.05, 0.0],
        [-0.1, 0.2, 0.1, -0.05],
    ]
    targets = [1, 0]
    mup_base_d = 4.0
    loss = cce_mup_loss_f64(hidden, embedding, targets, mup_base_d)
    return {
        "name": "cce_tiny",
        "op": "cce_mup_fwd",
        "inputs": {"H": hidden, "E": embedding, "y": targets, "mup_base_d": mup_base_d},
        "kernel_outputs": {"loss": loss},
    }


def _timings() -> dict:
    return {
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


def _guards() -> dict:
    return {
        "format": "sheaf-verify.hal-guards.v1",
        "thermal": {
            "stop_celsius": 85,
            "unit": "C",
            "samples": [{"t": "2026-01-01T00:00:10Z", "celsius": 47}, {"t": "2026-01-01T00:00:20Z", "celsius": 50}],
        },
        "power": {
            "observed": False,
            "probe": "nvidia-smi --query-gpu=power.draw --format=csv,noheader",
            "reason": "power.draw not available in the unit-test harness",
        },
    }


@pytest.fixture(autouse=True)
def _live_probe():
    set_probe(lambda: dict(ENV))
    yield
    set_probe(None)


def _parts(root: Path) -> dict:
    code = root / "code"
    code.mkdir()
    (code / "ref.py").write_text("def numpy_gemm_cce():\n    return None\n", encoding="utf-8")
    host = root / "host_ref.py"
    host.write_text("def numpy_gemm_cce():\n    return None\n", encoding="utf-8")
    kernel = root / "kernel.c"
    kernel.write_text("void hadukenn_tiled_cce_mup_fwd_bwd_f32(void) {}\n", encoding="utf-8")
    return {
        "code": code,
        "host": host,
        "kernel": kernel,
        "env": deepcopy(ENV),
        "claim": deepcopy(CLAIM),
        "config": {"format": "sheaf-verify.config.v1", "study": "hal-unit"},
        "parity": {"format": "sheaf-verify.hal-parity.v1", "cases": [_parity_case()]},
        "timings": _timings(),
        "guards": _guards(),
        "metrics": {"format": "sheaf-verify.metrics.note.v1", "note": "HAL fixture; not a G0 record"},
        "kernel_meta": {
            "backend": "c",
            "entry": "hadukenn_tiled_cce_mup_fwd_bwd_f32",
        },
        "include_metrics": True,
    }


def build_hal(tmp_path: Path, name: str = "store", mutate=None, profile: str = "hal") -> Store:
    work = tmp_path / name
    work.mkdir()
    parts = _parts(work)
    if mutate is not None:
        mutate(parts)
    store = Store(work / "sheaf")
    store.init()

    def register_json(kind: str, role: str, document: dict, when: str, inputs: list[str] | None = None):
        return store.register(
            kind=kind,
            role=role,
            registered_at=when,
            payload=canonical_bytes(document),
            inputs=inputs or [],
        )

    code_sha, _ = store.register_tree(role="code", path=parts["code"], registered_at="2026-01-01T00:00:02Z")
    store.register_environment(role="environment", registered_at="2026-01-01T00:00:03Z", **parts["env"])
    host_sha, host_man = store.register(
        kind="kernel",
        role="kernel.host",
        registered_at="2026-01-01T00:00:04Z",
        payload=parts["host"].read_bytes(),
        metadata={"backend": "cpu", "entry": "numpy_gemm_cce"},
    )
    meta = dict(parts["kernel_meta"])
    if "host_reference_sha256" not in meta and meta.get("backend") in {"c", "cuda", "opencl"}:
        meta["host_reference_sha256"] = host_man["payload_sha256"]
    inputs = [host_sha] if meta.get("host_reference_sha256") else []
    store.register(
        kind="kernel",
        role="kernel",
        registered_at="2026-01-01T00:00:05Z",
        payload=parts["kernel"].read_bytes(),
        inputs=inputs,
        metadata=meta,
    )
    register_json("config", "config", parts["config"], "2026-01-01T00:00:06Z")
    claim_sha, _ = register_json("claim", "claim", parts["claim"], "2026-01-01T00:00:07Z")
    if parts.get("include_metrics"):
        metrics_sha, _ = register_json("metrics", "metrics", parts["metrics"], "2026-01-01T00:00:08Z", inputs=[code_sha])
    else:
        metrics_sha = None
    register_json("log", "log.parity", parts["parity"], "2026-01-01T00:00:09Z", inputs=[claim_sha])
    register_json("log", "log.timings", parts["timings"], "2026-01-01T00:00:10Z", inputs=[claim_sha])
    register_json("log", "log.guards", parts["guards"], "2026-01-01T00:00:11Z")
    del metrics_sha
    run(store, key=TEST_KEY, clock=SequenceClock(), profile=profile)
    return store


def test_hal_passing_study(tmp_path):
    rows = latest(build_hal(tmp_path))
    assert {row["gate_id"] for row in rows.values()} == {"HAL0", "HAL1", "HAL2", "HAL3", "HAL4", "HAL5", "HAL6"}
    assert all(row["status"] == "pass" for row in rows.values())
    assert all(row.get("profile") == "hal" for row in rows.values())


def test_training_profile_does_not_skip_training_gates(tmp_path):
    store = build_hal(tmp_path, profile="training")
    rows = latest(store)
    assert "P0" in rows and "G0" in rows
    assert "HAL0" not in rows
    assert all(row.get("profile") == "training" for row in rows.values())
    assert any(row["status"] == "fail" for row in rows.values())


def test_hal0_fails_when_baselines_are_omitted(tmp_path):
    def mutate(parts):
        parts["claim"]["performance"]["baselines_reported"] = ["serial_python_loop"]

    rows = latest(build_hal(tmp_path, "hal0", mutate))
    assert rows["HAL0"]["status"] == "fail"
    assert criterion(rows["HAL0"], "hal.claim.baselines")["passed"] is False
    assert rows["HAL5"]["status"] == "fail"
    assert any(item["id"] == "upstream.HAL0" and item["passed"] is False for item in rows["HAL5"]["criteria"])


def test_hal1_fails_when_the_live_probe_disagrees(tmp_path):
    def mutate(parts):
        parts["env"]["gpu"] = "NVIDIA A100 (40960 MiB, sm_80)"

    rows = latest(build_hal(tmp_path, "hal1", mutate))
    assert rows["HAL0"]["status"] == "pass"
    assert rows["HAL1"]["status"] == "fail"
    assert criterion(rows["HAL1"], "environment.live.gpu")["passed"] is False


def test_hal2_fails_when_the_code_listing_is_empty(tmp_path):
    store = build_hal(tmp_path, "hal2-pre")
    # Rebuild with an empty code payload after the helper's tree snapshot would be valid:
    def mutate(parts):
        del parts
        # handled below — placeholder

    del mutate
    work = tmp_path / "hal2"
    work.mkdir()
    parts = _parts(work)
    store = Store(work / "sheaf")
    store.init()
    store.register(
        kind="code_tree",
        role="code",
        registered_at="2026-01-01T00:00:02Z",
        payload=canonical_bytes([]),
        metadata={"content_sha256": "00", "file_count": 0},
    )
    store.register_environment(role="environment", registered_at="2026-01-01T00:00:03Z", **parts["env"])
    host_sha, host_man = store.register(
        kind="kernel",
        role="kernel.host",
        registered_at="2026-01-01T00:00:04Z",
        payload=parts["host"].read_bytes(),
        metadata={"backend": "cpu", "entry": "numpy_gemm_cce"},
    )
    store.register(
        kind="kernel",
        role="kernel",
        registered_at="2026-01-01T00:00:05Z",
        payload=parts["kernel"].read_bytes(),
        inputs=[host_sha],
        metadata={
            "backend": "c",
            "entry": "hadukenn_tiled_cce_mup_fwd_bwd_f32",
            "host_reference_sha256": host_man["payload_sha256"],
        },
    )
    store.register(kind="config", role="config", registered_at="2026-01-01T00:00:06Z", payload=canonical_bytes(parts["config"]))
    store.register(kind="claim", role="claim", registered_at="2026-01-01T00:00:07Z", payload=canonical_bytes(parts["claim"]))
    store.register(kind="log", role="log.parity", registered_at="2026-01-01T00:00:09Z", payload=canonical_bytes(parts["parity"]))
    store.register(kind="log", role="log.timings", registered_at="2026-01-01T00:00:10Z", payload=canonical_bytes(parts["timings"]))
    store.register(kind="log", role="log.guards", registered_at="2026-01-01T00:00:11Z", payload=canonical_bytes(parts["guards"]))
    run(store, key=TEST_KEY, clock=SequenceClock(), profile="hal")
    rows = latest(store)
    assert rows["HAL0"]["status"] == "pass"
    assert rows["HAL2"]["status"] == "fail"
    assert criterion(rows["HAL2"], "code.listing")["passed"] is False or criterion(rows["HAL2"], "code.identity")["passed"] is False


def test_hal3_fails_when_a_c_kernel_lacks_host_reference(tmp_path):
    def mutate(parts):
        parts["kernel_meta"] = {"backend": "c", "entry": "hadukenn_tiled_cce_mup_fwd_bwd_f32", "host_reference_sha256": None}

    rows = latest(build_hal(tmp_path, "hal3", mutate))
    assert rows["HAL2"]["status"] == "pass"
    assert rows["HAL3"]["status"] == "fail"
    assert criterion(rows["HAL3"], "kernel.host_reference")["passed"] is False


def test_hal4_fails_when_outputs_diverge_from_the_float64_reference(tmp_path):
    def mutate(parts):
        parts["parity"]["cases"][0]["kernel_outputs"]["loss"] = [12.0, 12.0]

    rows = latest(build_hal(tmp_path, "hal4", mutate))
    assert rows["HAL3"]["status"] == "pass"
    assert rows["HAL4"]["status"] == "fail"
    assert criterion(rows["HAL4"], "hal.parity.cce_tiny")["passed"] is False


def test_hal5_fails_when_the_claim_baseline_is_not_met(tmp_path):
    def mutate(parts):
        # Honest numbers: C11 is slower than packed NumPy. Claiming that baseline must fail.
        parts["claim"]["performance"]["baseline"] = "packed_numpy"
        parts["claim"]["description"] = "The compiled C11 tiled CCE kernel is faster than packed NumPy on this host."

    rows = latest(build_hal(tmp_path, "hal5", mutate))
    assert rows["HAL0"]["status"] == "pass"
    assert rows["HAL5"]["status"] == "fail"
    assert criterion(rows["HAL5"], "hal.perf.threshold")["passed"] is False


def test_hal6_fails_when_temperature_reaches_the_stop(tmp_path):
    def mutate(parts):
        parts["guards"]["thermal"]["samples"][1]["celsius"] = 85

    rows = latest(build_hal(tmp_path, "hal6", mutate))
    assert rows["HAL1"]["status"] == "pass"
    assert rows["HAL6"]["status"] == "fail"
    assert criterion(rows["HAL6"], "hal.guard.thermal")["passed"] is False


def test_hal4_rejects_mup_overflow_non_finite_loss(tmp_path):
    from sheaf_verify.hashing import SheafError

    with pytest.raises(SheafError, match="overflow|finite"):
        cce_mup_loss_f64([[4.0, 4.0]], [[4.0, 4.0], [0.5, 0.5]], [0], 1e308)

    with pytest.raises(SheafError, match="finite"):
        cce_mup_loss_f64([[1.0, 1.0]], [[1.0, 1.0]], [0], float("nan"))

    def mutate(parts):
        parts["parity"]["cases"][0]["inputs"]["H"] = [[4.0, 4.0, 4.0, 4.0], [1.0, 1.0, 1.0, 1.0]]
        parts["parity"]["cases"][0]["inputs"]["E"][0] = [4.0, 4.0, 4.0, 4.0]
        parts["parity"]["cases"][0]["inputs"]["mup_base_d"] = 1e308
        parts["parity"]["cases"][0]["kernel_outputs"]["loss"] = [0.0, 0.0]

    rows = latest(build_hal(tmp_path, "hal4-overflow", mutate))
    assert rows["HAL3"]["status"] == "pass"
    assert rows["HAL4"]["status"] == "fail"
    assert criterion(rows["HAL4"], "hal.parity.cce_tiny")["passed"] is False


def test_hal6_enforces_claim_thermal_threshold_and_allows_missing_optional_guards(tmp_path):
    def mutate_spoof(parts):
        parts["guards"]["thermal"]["stop_celsius"] = 150
        parts["guards"]["thermal"]["samples"][1]["celsius"] = 90

    rows_spoof = latest(build_hal(tmp_path, "hal6-spoof", mutate_spoof))
    assert rows_spoof["HAL1"]["status"] == "pass"
    assert rows_spoof["HAL6"]["status"] == "fail"
    assert criterion(rows_spoof["HAL6"], "hal.guard.stop")["passed"] is False

    def mutate_optional(parts):
        parts["claim"].pop("guards", None)

    rows_opt = latest(build_hal(tmp_path, "hal6-opt", mutate_optional))
    assert rows_opt["HAL6"]["status"] == "pass"


def test_hal6_rejects_overflow_stop_and_non_finite_thermal_samples():
    import json

    from sheaf_verify.hal import evaluate_hal6
    from sheaf_verify.hashing import parse_json_bytes

    class _RawCtx:
        def __init__(self, claim_doc, guards_doc):
            self._docs = {"claim": claim_doc, "log.guards": guards_doc}

        def has(self, role: str) -> bool:
            return role in self._docs

        def parse(self, role: str):
            raw = json.dumps(self._docs[role], allow_nan=True).encode("utf-8")
            return parse_json_bytes(raw, role)

    base_guards = _guards()

    # 10**400 stop_celsius must fail cleanly without OverflowError
    big_stop_guards = deepcopy(base_guards)
    big_stop_guards["thermal"]["stop_celsius"] = 10**400
    res_big = evaluate_hal6(_RawCtx(deepcopy(CLAIM), big_stop_guards))
    assert any(c.id == "hal.guard.stop" and not c.passed for c in res_big)

    # 10**400 claim thermal_stop_celsius must fail cleanly without OverflowError
    big_claim = deepcopy(CLAIM)
    big_claim["guards"]["thermal_stop_celsius"] = 10**400
    res_big_claim = evaluate_hal6(_RawCtx(big_claim, deepcopy(base_guards)))
    assert any(c.id == "hal.guard.stop" and not c.passed for c in res_big_claim)

    # NaN / Inf sample alongside >=2 valid samples must still fail hal.guard.thermal
    for bad_c in (float("nan"), float("inf")):
        nan_guards = deepcopy(base_guards)
        nan_guards["thermal"]["samples"].append({"t": "2026-01-01T00:00:30Z", "celsius": bad_c})
        res_nan = evaluate_hal6(_RawCtx(deepcopy(CLAIM), nan_guards))
        assert any(c.id == "hal.guard.thermal" and not c.passed for c in res_nan)
