"""Tier 1 isolated happy-path governance, PR #13, and pure-stdlib formal guard E2E tests (Features 1-12).

Pre-M2 collection safety: zero top-level imports of `sheaf_verify.*`, `hadukenn_core.*`,
`scripts.*`, `numpy`, `scipy`, or `torch`. All target symbols are resolved inside test bodies.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from tests_e2e.conftest import (
    no_gpu_env,
    resolve_any_public_symbol,
    resolve_public_symbol,
    run_sheaf_cli,
    scaffold_mini_selftest_repo,
)

CANONICAL_ANCHOR = "307711a4d841c2c8fa0bbd954d15b2a585012608528c9f53b9ea08ad717bd7c4"


def _load_script_module(repo_root: Path, script_rel: str, module_name: str) -> Any:
    script_path = repo_root / script_rel
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None, f"Cannot load {script_path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_hal_parts(work: Path) -> dict[str, Any]:
    cce_fn = resolve_public_symbol("sheaf_verify", "cce_mup_loss_f64", ("sheaf_verify.hal",))
    code = work / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "ref.py").write_text("def numpy_gemm_cce():\n    return None\n", encoding="utf-8")
    host = work / "host_ref.py"
    host.write_text("def numpy_gemm_cce():\n    return None\n", encoding="utf-8")
    kernel = work / "kernel.c"
    kernel.write_text("void hadukenn_tiled_cce_mup_fwd_bwd_f32(void) {}\n", encoding="utf-8")

    hidden = [[0.2, -0.1, 0.05, 0.3], [0.0, 0.4, -0.2, 0.1]]
    embedding = [
        [0.1, 0.0, -0.1, 0.2],
        [0.3, -0.2, 0.05, 0.0],
        [-0.1, 0.2, 0.1, -0.05],
    ]
    targets = [1, 0]
    mup_base_d = 4.0
    loss = cce_fn(hidden, embedding, targets, mup_base_d)
    env = {
        "python": "3.12.11",
        "cuda": "release 12.4",
        "driver": "595.99.02",
        "gpu": "NVIDIA GeForce RTX 4080 Laptop GPU (12282 MiB, sm_89)",
        "platform": "Linux-test-harness",
    }
    claim = {
        "format": "sheaf-verify.hal-claim.v1",
        "study_profile": "hal",
        "hypothesis_id": "e2e-hal-t1",
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
                {"t": "2026-01-01T00:00:10Z", "celsius": 45},
                {"t": "2026-01-01T00:00:20Z", "celsius": 62},
            ],
        },
        "power": {
            "observed": True,
            "draw_watts": 140.5,
        },
    }
    return {
        "code": code,
        "host": host,
        "kernel": kernel,
        "env": env,
        "claim": claim,
        "config": {"format": "sheaf-verify.config.v1", "study": "hal-e2e"},
        "metrics": {"format": "sheaf-verify.metrics.note.v1", "note": "HAL fixture; not a G0 record"},
        "parity": parity,
        "timings": timings,
        "guards": guards,
    }


def _register_hal_store(store_dir: Path, parts: dict[str, Any]) -> Any:
    store_cls = resolve_public_symbol("sheaf_verify", "Store", ("sheaf_verify.store",))
    canonical_bytes = resolve_public_symbol("sheaf_verify", "canonical_bytes", ("sheaf_verify.hashing",))
    store = store_cls(store_dir)
    store.init()
    code_sha, _ = store.register_tree(role="code", path=parts["code"], registered_at="2026-01-01T00:00:02Z")
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
    store.register(
        kind="config",
        role="config",
        registered_at="2026-01-01T00:00:06Z",
        payload=canonical_bytes(parts["config"]),
    )
    claim_sha, _ = store.register(
        kind="claim",
        role="claim",
        registered_at="2026-01-01T00:00:07Z",
        payload=canonical_bytes(parts["claim"]),
    )
    store.register(
        kind="metrics",
        role="metrics",
        registered_at="2026-01-01T00:00:08Z",
        payload=canonical_bytes(parts["metrics"]),
        inputs=[code_sha],
    )
    store.register(
        kind="log",
        role="log.parity",
        registered_at="2026-01-01T00:00:09Z",
        payload=canonical_bytes(parts["parity"]),
        inputs=[claim_sha],
    )
    store.register(
        kind="log",
        role="log.timings",
        registered_at="2026-01-01T00:00:10Z",
        payload=canonical_bytes(parts["timings"]),
        inputs=[claim_sha],
    )
    store.register(
        kind="log",
        role="log.guards",
        registered_at="2026-01-01T00:00:11Z",
        payload=canonical_bytes(parts["guards"]),
    )
    return store


def _make_inline_ctx(docs: dict[str, Any]) -> Any:
    parse_json_bytes = resolve_public_symbol("sheaf_verify", "parse_json_bytes", ("sheaf_verify.hashing",))

    class _InlineContext:
        def __init__(self, items: dict[str, Any]) -> None:
            self._raw = {k: json.dumps(v).encode("utf-8") for k, v in items.items() if v is not None}

        def has(self, role: str) -> bool:
            return role in self._raw

        def payload(self, role: str) -> bytes:
            return self._raw[role]

        def parse(self, role: str) -> Any:
            return parse_json_bytes(self._raw[role], role)

    return _InlineContext(docs)


def _verdict_passed(result: Any) -> bool:
    if isinstance(result, bool):
        return result
    for attr in ("passed", "ok", "accepted", "approved", "allowed"):
        if hasattr(result, attr) and isinstance(getattr(result, attr), bool):
            return bool(getattr(result, attr))
    if isinstance(result, tuple) and result and isinstance(result[0], bool):
        return result[0]
    if isinstance(result, dict):
        for key in ("ok", "passed", "accepted", "approved", "allowed"):
            if key in result and isinstance(result[key], bool):
                return bool(result[key])
    if isinstance(result, list) and result:
        if all(hasattr(c, "passed") for c in result):
            return all(bool(c.passed) for c in result)
        if all(isinstance(c, dict) and "passed" in c for c in result):
            return all(bool(c["passed"]) for c in result)
    raise AssertionError(f"Unrecognized verifier return shape: {result!r}")


def _call_with_aliases(
    fn: Any,
    alias_values: list[tuple[tuple[str, ...], Any]],
    positional_fallback: tuple[Any, ...],
) -> Any:
    sig = inspect.signature(fn)
    params = sig.parameters
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    kwargs: dict[str, Any] = {}
    for aliases, val in alias_values:
        matched = False
        for name in aliases:
            if name in params:
                kwargs[name] = val
                matched = True
                break
        if not matched and has_varkw:
            kwargs[aliases[0]] = val
    if len(kwargs) == len(alias_values) or all(
        p.default is not inspect.Parameter.empty
        or p.name in kwargs
        or p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for p in params.values()
    ):
        return fn(**kwargs)
    return fn(*positional_fallback)


# ==============================================================================
# Feature 1: Immutable Anchor & runner.py Profile Flag (6 Tier 1 tests)
# ==============================================================================


@pytest.mark.tier1
def test_f1_cli_anchor_fixtures_passing_canonical(repo_root: Path, readonly_snapshot: dict[str, str]) -> None:
    del readonly_snapshot
    proc = run_sheaf_cli(["anchor"], store=repo_root / "fixtures" / "passing")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"sheaf-verify.anchor.v1 {CANONICAL_ANCHOR}"


@pytest.mark.tier1
def test_f1_cli_verify_fixtures_passing_with_test_seal_key(
    repo_root: Path, readonly_snapshot: dict[str, str]
) -> None:
    del readonly_snapshot
    proc = run_sheaf_cli(
        ["verify", "--json", "--seal-key-file", str(repo_root / "fixtures" / "test_seal.key")],
        store=repo_root / "fixtures" / "passing",
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["head"] == CANONICAL_ANCHOR
    assert payload["problems"] == []


@pytest.mark.tier1
def test_f1_runner_default_omits_profile_field_and_matches_canonical(tmp_path: Path) -> None:
    synthetic_build = resolve_public_symbol("sheaf_verify", "build", ("sheaf_verify.synthetic",))
    store_dir = tmp_path / "synthetic_store"
    synthetic_build(store_dir)

    ledger_lines = (store_dir / "ledger" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(ledger_lines) == 22
    for raw_line in ledger_lines:
        rec = json.loads(raw_line)
        assert "profile" not in rec

    head = json.loads((store_dir / "ledger" / "head.json").read_text(encoding="utf-8"))
    assert head["head_sha256"] == CANONICAL_ANCHOR


@pytest.mark.tier1
def test_f1_runner_hal_profile_preserves_profile_hal(tmp_path: Path) -> None:
    set_probe = resolve_public_symbol("sheaf_verify", "set_probe", ("sheaf_verify.hal",))
    run_fn = resolve_public_symbol("sheaf_verify", "run", ("sheaf_verify.runner",))
    seq_clock = resolve_public_symbol("sheaf_verify", "SequenceClock", ("sheaf_verify.runner",))
    verify_store = resolve_public_symbol("sheaf_verify", "verify_store", ("sheaf_verify.verify",))
    test_key = resolve_public_symbol("sheaf_verify", "TEST_KEY", ("sheaf_verify.synthetic",))

    parts = _make_hal_parts(tmp_path / "hal_work")
    store = _register_hal_store(tmp_path / "hal_store", parts)
    set_probe(lambda: dict(parts["env"]))
    try:
        records = run_fn(store, key=test_key, clock=seq_clock(), profile="hal")
        report = verify_store(store, key=test_key)
    finally:
        set_probe(None)

    assert [r["gate_id"] for r in records] == ["HAL0", "HAL1", "HAL2", "HAL3", "HAL4", "HAL5", "HAL6"]
    assert all(r.get("profile") == "hal" for r in records)
    assert all(r["status"] == "pass" for r in records)
    assert report.ok is True


@pytest.mark.tier1
def test_f1_cli_run_training_and_anchor_out_outside_store(repo_root: Path, tmp_path: Path) -> None:
    synthetic_build = resolve_public_symbol("sheaf_verify", "build", ("sheaf_verify.synthetic",))
    store_dir = tmp_path / "store"
    synthetic_build(store_dir)
    key_file = repo_root / "fixtures" / "test_seal.key"

    run_proc = run_sheaf_cli(
        ["run", "--profile", "training", "--json", "--seal-key-file", str(key_file)],
        store=store_dir,
    )
    assert run_proc.returncode == 0, run_proc.stderr
    run_payload = json.loads(run_proc.stdout)
    assert run_payload["failed"] == []

    anchor_out = tmp_path / "anchors" / "head.txt"
    anc_proc = run_sheaf_cli(["anchor", "--out", str(anchor_out)], store=store_dir)
    assert anc_proc.returncode == 0, anc_proc.stderr
    assert anchor_out.read_text(encoding="utf-8") == anc_proc.stdout
    assert anc_proc.stdout.startswith("sheaf-verify.anchor.v1 ")


@pytest.mark.tier1
def test_f1_cli_verify_and_bundle_on_synthetic_store(repo_root: Path, tmp_path: Path) -> None:
    synthetic_build = resolve_public_symbol("sheaf_verify", "build", ("sheaf_verify.synthetic",))
    store_dir = tmp_path / "store"
    synthetic_build(store_dir)
    key_file = repo_root / "fixtures" / "test_seal.key"

    ver_proc = run_sheaf_cli(["verify", "--json", "--seal-key-file", str(key_file)], store=store_dir)
    assert ver_proc.returncode == 0, ver_proc.stderr
    assert json.loads(ver_proc.stdout)["ok"] is True

    bundle_dir = tmp_path / "bundle"
    bun_proc = run_sheaf_cli(
        ["bundle", "--out", str(bundle_dir), "--json", "--seal-key-file", str(key_file)],
        store=store_dir,
    )
    assert bun_proc.returncode == 0, bun_proc.stderr
    assert (bundle_dir / "REPORT.md").is_file()
    assert not (bundle_dir / "ledger" / ".lock").exists()


# ==============================================================================
# Feature 2: sheaf_verify.selftest Parametrize, Fixtures & Pytest Non-Shadowing (6 Tier 1 tests)
# ==============================================================================


@pytest.mark.tier1
def test_f2_selftest_fresh_process_cli_passes(repo_root: Path, tmp_path: Path) -> None:
    report_out = tmp_path / "selftest_report.json"
    proc = subprocess.run(
        [sys.executable, "-m", "sheaf_verify.selftest", "--report-out", str(report_out)],
        cwd=repo_root,
        env=no_gpu_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=900.0,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(report_out.read_text(encoding="utf-8"))
    assert report["schema"] == "sheaf-verify.antigravity-selftest.v1"
    assert report["ok"] is True
    assert report["pytest_suite"]["failed"] == 0
    assert report["pytest_suite"]["sheaf_verify_line_coverage_pct"] >= 85.0


@pytest.mark.tier1
def test_f2_selftest_canonical_anchor_matches_307711a4() -> None:
    anchor = resolve_public_symbol("sheaf_verify", "CANONICAL_ANCHOR", ("sheaf_verify.selftest",))
    assert anchor == CANONICAL_ANCHOR


@pytest.mark.tier1
def test_f2_selftest_does_not_shadow_real_pytest(repo_root: Path, tmp_path: Path) -> None:
    mini = scaffold_mini_selftest_repo(repo_root, tmp_path / "mini_real_pytest", create_default_test=True)
    probe_test = tmp_path / "test_real_pytest_probe.py"
    probe_test.write_text(
        f"""from pathlib import Path

import pytest

from sheaf_verify.selftest import run_full_verification


def test_real_pytest_remains_unshadowed() -> None:
    mini = Path({str(mini)!r})
    _ok, report = run_full_verification(mini)
    import pytest as pytest_after

    assert hasattr(pytest_after, "__file__")
    assert getattr(pytest_after, "_sheaf_shim", False) is False
    assert report["pytest_suite"]["passed"] == 1
    assert report["pytest_suite"]["failed"] == 0
""",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(probe_test)],
        cwd=repo_root,
        env=no_gpu_env(),
        capture_output=True,
        text=True,
        check=False,
        timeout=60.0,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.tier1
def test_f2_selftest_shim_executes_single_and_stacked_parametrize(repo_root: Path, tmp_path: Path) -> None:
    mini = scaffold_mini_selftest_repo(repo_root, tmp_path / "mini_param")
    (mini / "tests" / "test_param.py").write_text(
        """import pytest

SEEN = []


@pytest.mark.parametrize("x", [1, 2, 3])
def test_single(x):
    SEEN.append(("single", x))
    assert x in (1, 2, 3)


@pytest.mark.parametrize("outer", ["A", "B"])
@pytest.mark.parametrize("inner", [10, 20])
def test_stacked(outer, inner):
    SEEN.append((outer, inner))
    assert outer in ("A", "B") and inner in (10, 20)


@pytest.mark.parametrize("u, v", [(1, 2), (3, 4)])
def test_multi_arg(u, v):
    assert v == u + 1


def test_z_order_recorded():
    assert SEEN[:3] == [("single", 1), ("single", 2), ("single", 3)]
    assert SEEN[3:] == [("A", 10), ("A", 20), ("B", 10), ("B", 20)]
""",
        encoding="utf-8",
    )
    rep_out = tmp_path / "param_report.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "sheaf_verify.selftest",
            "--repo-root",
            str(mini),
            "--report-out",
            str(rep_out),
        ],
        cwd=repo_root,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(repo_root / "src")},
        capture_output=True,
        text=True,
        check=False,
        timeout=60.0,
    )
    assert proc.returncode in (0, 1), proc.stderr + proc.stdout
    report = json.loads(rep_out.read_text(encoding="utf-8"))
    assert report["lint"]["ok"] is True, report["lint"]
    assert report["canonical_fixture"]["ok"] is True, report["canonical_fixture"]
    assert report["pytest_suite"]["failed"] == 0, report["pytest_suite"]["failures"]
    assert report["pytest_suite"]["passed"] == 10


@pytest.mark.tier1
def test_f2_selftest_shim_executes_autouse_fixture_and_yield_teardown(repo_root: Path, tmp_path: Path) -> None:
    mini = scaffold_mini_selftest_repo(repo_root, tmp_path / "mini_auto")
    (mini / "tests" / "test_auto.py").write_text(
        """import pytest

LOG = []


@pytest.fixture(autouse=True)
def _fixture():
    LOG.append("enter")
    yield
    LOG.append("exit")


def test_one():
    LOG.append("one")


def test_two():
    assert LOG == ["enter", "one", "exit", "enter"]
""",
        encoding="utf-8",
    )
    rep_out = tmp_path / "auto_report.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "sheaf_verify.selftest",
            "--repo-root",
            str(mini),
            "--report-out",
            str(rep_out),
        ],
        cwd=repo_root,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(repo_root / "src")},
        capture_output=True,
        text=True,
        check=False,
        timeout=60.0,
    )
    assert proc.returncode in (0, 1), proc.stderr + proc.stdout
    report = json.loads(rep_out.read_text(encoding="utf-8"))
    assert report["lint"]["ok"] is True, report["lint"]
    assert report["canonical_fixture"]["ok"] is True, report["canonical_fixture"]
    assert report["pytest_suite"]["failed"] == 0, report["pytest_suite"]["failures"]
    assert report["pytest_suite"]["passed"] == 2


@pytest.mark.tier1
def test_f2_selftest_shim_handles_importorskip_and_skip(repo_root: Path, tmp_path: Path) -> None:
    mini = scaffold_mini_selftest_repo(repo_root, tmp_path / "mini_skip")
    (mini / "tests" / "test_skips.py").write_text(
        """import pytest


def test_skip_call():
    pytest.skip("skip intentionally")


def test_importorskip_missing():
    pytest.importorskip("definitely_missing_pkg_e2e_xyz")


@pytest.mark.skip(reason="marked")
def test_mark_skip():
    raise AssertionError("must not run")


def test_passes():
    assert 1 + 1 == 2
""",
        encoding="utf-8",
    )
    rep_out = tmp_path / "skip_report.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "sheaf_verify.selftest",
            "--repo-root",
            str(mini),
            "--report-out",
            str(rep_out),
        ],
        cwd=repo_root,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(repo_root / "src")},
        capture_output=True,
        text=True,
        check=False,
        timeout=60.0,
    )
    assert proc.returncode in (0, 1), proc.stderr + proc.stdout
    report = json.loads(rep_out.read_text(encoding="utf-8"))
    assert report["lint"]["ok"] is True, report["lint"]
    assert report["canonical_fixture"]["ok"] is True, report["canonical_fixture"]
    assert report["pytest_suite"]["failed"] == 0, report["pytest_suite"]["failures"]
    assert report["pytest_suite"]["passed"] == 1


# ==============================================================================
# Feature 3: sheaf_verify.compliance Exact Fraction Witness Consensus (6 Tier 1 tests)
# ==============================================================================


@pytest.mark.tier1
def test_f3_compliance_exact_unit_lattice_arithmetic() -> None:
    lattice = resolve_public_symbol("sheaf_verify", "ExactUnitLattice", ("sheaf_verify.compliance",))
    tot, sym = lattice.add("0.1", "USD", "0.2", "USD")
    assert tot == Fraction(3, 10)
    assert sym == "USD"

    diff, sym_u = lattice.sub("2.5", "kUSD", "50000", "cents")
    assert diff == Fraction(2000, 1)
    assert sym_u == "USD"

    val, sym_e, dim_e = lattice.normalize("1", "kWh")
    assert val == Fraction(3_600_000, 1)
    assert sym_e == "J"
    assert dim_e == "energy"


@pytest.mark.tier1
def test_f3_compliance_unanimous_multi_unit_witnesses_valid() -> None:
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))
    ont = ont_cls("e2e.f3.unanimous", [rule_cls(action_name="wire", dimension="currency")])

    verdict = ont.evaluate_transfer_action(
        action_name="wire",
        actor_role="operator",
        src_entity="AcctA",
        dst_entity="AcctB",
        src_witnesses={"w1": ("1.5", "kUSD"), "w2": ("150000", "cents"), "w3": (1500, "USD")},
        dst_initial=("100", "USD"),
        transfer_amount=("250", "USD"),
    )
    assert verdict.verdict == "VALID"
    assert verdict.approved is True
    assert verdict.resolved_initial_src == "1500"
    assert verdict.post_src_balance == "1250"
    assert verdict.post_dst_balance == "350"


@pytest.mark.tier1
def test_f3_compliance_single_outlier_recoverable_fault() -> None:
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))
    ont = ont_cls("e2e.f3.outlier", [rule_cls(action_name="wire", dimension="currency")])

    verdict = ont.evaluate_transfer_action(
        action_name="wire",
        actor_role="operator",
        src_entity="AcctA",
        dst_entity="AcctB",
        src_witnesses={"w1": ("1000", "USD"), "w2": ("1000", "USD"), "w3": ("9999", "USD")},
        dst_initial=("200", "USD"),
        transfer_amount=("300", "USD"),
    )
    assert verdict.verdict == "RECOVERABLE_FAULT"
    assert verdict.approved is True
    assert verdict.isolated_fault_sources == ("w3",)
    assert verdict.resolved_initial_src == "1000"


@pytest.mark.tier1
def test_f3_compliance_fraction_exact_rational_nonterminating() -> None:
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))
    ont = ont_cls("e2e.f3.rational", [rule_cls(action_name="pump", dimension="volume")])

    verdict = ont.evaluate_transfer_action(
        action_name="pump",
        actor_role="operator",
        src_entity="Tank1",
        dst_entity="Tank2",
        src_witnesses={"w1": (Fraction(1, 3), "L"), "w2": (Fraction(1, 3), "L")},
        dst_initial=(Fraction(2, 3), "L"),
        transfer_amount=(Fraction(1, 6), "L"),
    )
    assert verdict.verdict == "VALID"
    assert verdict.post_src_balance == "1/6"
    assert verdict.post_dst_balance == "5/6"


@pytest.mark.tier1
def test_f3_compliance_policy_sha256_deterministic() -> None:
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))

    a = ont_cls("pol.v1", [rule_cls(action_name="wire", dimension="currency", max_amount=Fraction(100, 1))])
    b = ont_cls("pol.v1", [rule_cls(action_name="wire", dimension="currency", max_amount=Fraction(100, 1))])
    c = ont_cls("pol.v1", [rule_cls(action_name="wire", dimension="currency", max_amount=Fraction(200, 1))])

    assert a.policy_sha256 == b.policy_sha256
    assert a.policy_sha256 != c.policy_sha256
    assert len(a.policy_sha256) == 64


@pytest.mark.tier1
def test_f3_compliance_formal_verdicts_and_as_dict_keys() -> None:
    verdicts = resolve_public_symbol("sheaf_verify", "FORMAL_VERDICTS", ("sheaf_verify.compliance",))
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))

    assert verdicts == (
        "VALID",
        "INVALID",
        "IMPOSSIBLE",
        "TRANSLATION_AMBIGUOUS",
        "ABSTAIN_TIE",
        "RECOVERABLE_FAULT",
    )
    ont = ont_cls("pol.keys", [rule_cls(action_name="wire", dimension="currency")])
    res = ont.evaluate_transfer_action(
        action_name="wire",
        actor_role="operator",
        src_entity="A",
        dst_entity="B",
        src_witnesses={"w1": ("500", "USD")},
        dst_initial=("100", "USD"),
        transfer_amount=("50", "USD"),
    )
    d = res.as_dict()
    expected_keys = {
        "approved",
        "canonical_unit",
        "human_escalation_required",
        "isolated_fault_sources",
        "policy_sha256",
        "post_dst_balance",
        "post_src_balance",
        "reason",
        "resolved_initial_src",
        "verdict",
    }
    assert set(d) == expected_keys
    json.dumps(d)


# ==============================================================================
# Feature 4: sheaf_verify.compliance Post-State Dimension Validation (6 Tier 1 tests)
# ==============================================================================


@pytest.mark.tier1
def test_f4_compliance_matching_post_src_and_dst_same_unit_valid() -> None:
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))
    ont = ont_cls("f4.same", [rule_cls(action_name="wire", dimension="currency")])

    v = ont.evaluate_transfer_action(
        action_name="wire",
        actor_role="operator",
        src_entity="A",
        dst_entity="B",
        src_witnesses={"w1": ("1000", "USD")},
        dst_initial=("200", "USD"),
        transfer_amount=("100", "USD"),
        claimed_post_src=("900", "USD"),
        claimed_post_dst=("300", "USD"),
    )
    assert v.verdict == "VALID"
    assert v.approved is True


@pytest.mark.tier1
def test_f4_compliance_matching_post_state_scaled_compatible_units() -> None:
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))
    ont = ont_cls("f4.scaled", [rule_cls(action_name="wire", dimension="currency")])

    v = ont.evaluate_transfer_action(
        action_name="wire",
        actor_role="operator",
        src_entity="A",
        dst_entity="B",
        src_witnesses={"w1": ("1000", "USD")},
        dst_initial=("200", "USD"),
        transfer_amount=("100", "USD"),
        claimed_post_src=("0.9", "kUSD"),
        claimed_post_dst=("30000", "cents"),
    )
    assert v.verdict == "VALID"
    assert v.post_src_balance == "900"
    assert v.post_dst_balance == "300"


@pytest.mark.tier1
def test_f4_compliance_partial_claimed_post_src_only_valid() -> None:
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))
    ont = ont_cls("f4.src_only", [rule_cls(action_name="pump", dimension="volume")])

    v = ont.evaluate_transfer_action(
        action_name="pump",
        actor_role="operator",
        src_entity="T1",
        dst_entity="T2",
        src_witnesses={"w1": ("1", "L")},
        dst_initial=("2", "L"),
        transfer_amount=("0.5", "L"),
        claimed_post_src=("500", "mL"),
        claimed_post_dst=None,
    )
    assert v.verdict == "VALID"
    assert v.approved is True


@pytest.mark.tier1
def test_f4_compliance_partial_claimed_post_dst_only_valid() -> None:
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))
    ont = ont_cls("f4.dst_only", [rule_cls(action_name="weigh", dimension="mass")])

    v = ont.evaluate_transfer_action(
        action_name="weigh",
        actor_role="operator",
        src_entity="S1",
        dst_entity="S2",
        src_witnesses={"w1": ("5", "kg")},
        dst_initial=("1", "kg"),
        transfer_amount=("1.5", "kg"),
        claimed_post_src=None,
        claimed_post_dst=("2500", "g"),
    )
    assert v.verdict == "VALID"
    assert v.approved is True


@pytest.mark.tier1
def test_f4_compliance_both_claimed_post_none_valid() -> None:
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))
    ont = ont_cls("f4.none", [rule_cls(action_name="charge", dimension="energy")])

    v = ont.evaluate_transfer_action(
        action_name="charge",
        actor_role="operator",
        src_entity="Bat1",
        dst_entity="Bat2",
        src_witnesses={"w1": ("10", "kJ")},
        dst_initial=("2", "kJ"),
        transfer_amount=("3", "kJ"),
        claimed_post_src=None,
        claimed_post_dst=None,
    )
    assert v.verdict == "VALID"
    assert v.approved is True


@pytest.mark.tier1
def test_f4_compliance_recoverable_fault_with_valid_post_state() -> None:
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))
    ont = ont_cls("f4.rec", [rule_cls(action_name="wire", dimension="currency")])

    v = ont.evaluate_transfer_action(
        action_name="wire",
        actor_role="operator",
        src_entity="A",
        dst_entity="B",
        src_witnesses={"w1": ("1000", "USD"), "w2": ("1000", "USD"), "w3": ("42", "USD")},
        dst_initial=("200", "USD"),
        transfer_amount=("100", "USD"),
        claimed_post_src=("900", "USD"),
        claimed_post_dst=("300", "USD"),
    )
    assert v.verdict == "RECOVERABLE_FAULT"
    assert v.approved is True
    assert v.isolated_fault_sources == ("w3",)


# ==============================================================================
# Feature 5: Regulatory Bundle Predicate & Metadata Binding (6 Tier 1 tests)
# ==============================================================================


@pytest.mark.tier1
def test_f5_bundle_emit_and_verify_roundtrip() -> None:
    emit_fn = resolve_public_symbol("sheaf_verify", "emit_regulatory_bundle", ("sheaf_verify.compliance",))
    verify_fn = resolve_public_symbol("sheaf_verify", "verify_regulatory_bundle", ("sheaf_verify.compliance",))
    key = b"e2e-reg-bundle-key"
    events = [
        {"verdict": "VALID", "approved": True},
        {"verdict": "INVALID", "approved": False},
        {"verdict": "ABSTAIN_TIE", "approved": False, "human_escalation_required": True},
    ]
    bundle = emit_fn(
        system_id="hadukenn-e2e-sys",
        trace_head_sha256="a" * 64,
        state_digest_sha256="b" * 64,
        policy_sha256="c" * 64,
        events=events,
        created_at="2026-11-28T00:00:00Z",
        signing_key=key,
    )
    assert bundle["bundle_schema"] == "sheaf-verify.regulatory-bundle.v1"
    ok_flag, errs = verify_fn(bundle, signing_key=key)
    assert ok_flag is True
    assert errs == []


@pytest.mark.tier1
def test_f5_bundle_dsse_subjects_have_system_id_prefix() -> None:
    emit_fn = resolve_public_symbol("sheaf_verify", "emit_regulatory_bundle", ("sheaf_verify.compliance",))
    bundle = emit_fn(
        system_id="sys-prefix-test",
        trace_head_sha256="1" * 64,
        state_digest_sha256="2" * 64,
        policy_sha256="3" * 64,
        events=[{"verdict": "VALID", "approved": True}],
        created_at="2026-11-28T00:00:00Z",
        signing_key=b"e2e-key-prefix-0123456",
    )
    stmt = json.loads(base64.b64decode(bundle["dsse_statement_envelope"]["payload"]).decode("utf-8"))
    names = [s["name"] for s in stmt["subject"]]
    assert names == [
        "sys-prefix-test/events.json",
        "sys-prefix-test/trace_head",
        "sys-prefix-test/state_digest",
    ]


@pytest.mark.tier1
def test_f5_bundle_predicate_binds_finra_euai_scitt_fields() -> None:
    emit_fn = resolve_public_symbol("sheaf_verify", "emit_regulatory_bundle", ("sheaf_verify.compliance",))
    pol_sha = "4" * 64
    bundle = emit_fn(
        system_id="sys-pred-bind",
        trace_head_sha256="5" * 64,
        state_digest_sha256="6" * 64,
        policy_sha256=pol_sha,
        events=[{"verdict": "VALID", "approved": True}],
        created_at="2026-11-28T01:02:03Z",
        signing_key=b"e2e-key-pred-01234567",
    )
    stmt = json.loads(base64.b64decode(bundle["dsse_statement_envelope"]["payload"]).decode("utf-8"))
    pred = stmt["predicate"]
    assert pred["system_id"] == "sys-pred-bind"
    assert pred["created_at"] == "2026-11-28T01:02:03Z"
    assert pred["finra_rule_3110_worm"]["policy_snapshot_sha256"] == pol_sha
    assert pred["finra_rule_3110_worm"]["worm_integrity_sha256"] == bundle["events_sha256"]
    assert pred["eu_ai_act_article_12"]["total_events_logged"] == 1
    assert pred["ietf_scitt_rfc9943"]["merkle_tree_alg"] == "RFC9162-SHA256"


@pytest.mark.tier1
def test_f5_bundle_tracks_escalation_and_refusal_indices() -> None:
    emit_fn = resolve_public_symbol("sheaf_verify", "emit_regulatory_bundle", ("sheaf_verify.compliance",))
    events = [
        {"verdict": "VALID", "approved": True},
        {"verdict": "ABSTAIN_TIE", "approved": False},
        {"verdict": "INVALID", "approved": False},
        {"verdict": "TRANSLATION_AMBIGUOUS", "approved": False},
    ]
    bundle = emit_fn(
        system_id="sys-idx",
        trace_head_sha256="a" * 64,
        state_digest_sha256="b" * 64,
        policy_sha256="c" * 64,
        events=events,
        created_at="2026-11-28T00:00:00Z",
        signing_key=b"e2e-key-idx-012345678",
    )
    stmt = json.loads(base64.b64decode(bundle["dsse_statement_envelope"]["payload"]).decode("utf-8"))
    pred = stmt["predicate"]
    assert pred["eu_ai_act_article_12"]["escalated_event_indices"] == [1, 3]
    assert pred["eu_ai_act_article_12"]["refused_event_indices"] == [1, 2, 3]
    assert pred["finra_rule_3110_worm"]["supervisory_escalation_count"] == 2


@pytest.mark.tier1
def test_f5_bundle_minimum_retention_6_months_accepted() -> None:
    emit_fn = resolve_public_symbol("sheaf_verify", "emit_regulatory_bundle", ("sheaf_verify.compliance",))
    verify_fn = resolve_public_symbol("sheaf_verify", "verify_regulatory_bundle", ("sheaf_verify.compliance",))
    key = b"e2e-key-ret6-01234567"
    bundle = emit_fn(
        system_id="sys-ret6",
        trace_head_sha256="a" * 64,
        state_digest_sha256="b" * 64,
        policy_sha256="c" * 64,
        events=[{"verdict": "VALID", "approved": True}],
        created_at="2026-11-28T00:00:00Z",
        signing_key=key,
        retention_months=6,
    )
    ok_flag, errs = verify_fn(bundle, signing_key=key)
    assert ok_flag is True
    assert errs == []
    stmt = json.loads(base64.b64decode(bundle["dsse_statement_envelope"]["payload"]).decode("utf-8"))
    assert stmt["predicate"]["finra_rule_3110_worm"]["retention_years_floor"] == 0


@pytest.mark.tier1
def test_f5_bundle_t49_sealed_bundle_verifies_structure(
    repo_root: Path, readonly_snapshot: dict[str, str]
) -> None:
    del readonly_snapshot
    t49_path = repo_root / "iterations" / "t49-hst-foundation-stage" / "T49_HST_FOUNDATION_RESULTS.json"
    doc = json.loads(t49_path.read_text(encoding="utf-8"))
    bundle = doc["scitt_rfc9943_bundle"]
    assert bundle["bundle_schema"] == "sheaf-verify.regulatory-bundle.v1"
    stmt = json.loads(base64.b64decode(bundle["dsse_statement_envelope"]["payload"]).decode("utf-8"))
    names = [s["name"] for s in stmt["subject"]]
    assert names == [
        "hadukenn-hst-v1-foundation-lm/events.json",
        "hadukenn-hst-v1-foundation-lm/trace_head",
        "hadukenn-hst-v1-foundation-lm/state_digest",
    ]
    assert stmt["predicate"]["finra_rule_3110_worm"]["worm_integrity_sha256"] == bundle["events_sha256"]


# ==============================================================================
# Feature 6: HAL4 Finite Loss/Scale Guard (6 Tier 1 tests)
# ==============================================================================


@pytest.mark.tier1
def test_f6_hal4_exact_float64_parity_passes(repo_root: Path, tmp_path: Path) -> None:
    eval_hal4 = resolve_public_symbol("sheaf_verify", "evaluate_hal4", ("sheaf_verify.hal",))
    jovin = _load_script_module(repo_root, "scripts/jovin_cpu_payload.py", "e2e_jovin_f6_1")
    parts = _make_hal_parts(tmp_path / "f6_1")
    ctx = _make_inline_ctx({"claim": parts["claim"], "log.parity": parts["parity"]})

    assert all(c.passed for c in eval_hal4(ctx))
    assert jovin.all_passed(jovin.evaluate_parity(parts["parity"], parts["claim"]))


@pytest.mark.tier1
def test_f6_hal4_within_claim_max_abs_passes(repo_root: Path, tmp_path: Path) -> None:
    eval_hal4 = resolve_public_symbol("sheaf_verify", "evaluate_hal4", ("sheaf_verify.hal",))
    jovin = _load_script_module(repo_root, "scripts/jovin_cpu_payload.py", "e2e_jovin_f6_2")
    parts = _make_hal_parts(tmp_path / "f6_2")
    parts["parity"]["cases"][0]["kernel_outputs"]["loss"][0] += 5e-5
    ctx = _make_inline_ctx({"claim": parts["claim"], "log.parity": parts["parity"]})

    assert all(c.passed for c in eval_hal4(ctx))
    assert jovin.all_passed(jovin.evaluate_parity(parts["parity"], parts["claim"]))


@pytest.mark.tier1
def test_f6_hal4_tighter_claim_max_abs_honored(repo_root: Path, tmp_path: Path) -> None:
    eval_hal4 = resolve_public_symbol("sheaf_verify", "evaluate_hal4", ("sheaf_verify.hal",))
    jovin = _load_script_module(repo_root, "scripts/jovin_cpu_payload.py", "e2e_jovin_f6_3")
    parts = _make_hal_parts(tmp_path / "f6_3")
    parts["claim"]["parity"]["max_abs"] = 1e-6

    pass_parity = json.loads(json.dumps(parts["parity"]))
    pass_parity["cases"][0]["kernel_outputs"]["loss"][0] += 5e-7
    ctx_ok = _make_inline_ctx({"claim": parts["claim"], "log.parity": pass_parity})
    assert all(c.passed for c in eval_hal4(ctx_ok))
    assert jovin.all_passed(jovin.evaluate_parity(pass_parity, parts["claim"]))

    fail_parity = json.loads(json.dumps(parts["parity"]))
    fail_parity["cases"][0]["kernel_outputs"]["loss"][0] += 5e-6
    ctx_bad = _make_inline_ctx({"claim": parts["claim"], "log.parity": fail_parity})
    assert not all(c.passed for c in eval_hal4(ctx_bad))
    assert not jovin.all_passed(jovin.evaluate_parity(fail_parity, parts["claim"]))


@pytest.mark.tier1
def test_f6_hal4_multiple_cases_all_evaluated(repo_root: Path, tmp_path: Path) -> None:
    eval_hal4 = resolve_public_symbol("sheaf_verify", "evaluate_hal4", ("sheaf_verify.hal",))
    jovin = _load_script_module(repo_root, "scripts/jovin_cpu_payload.py", "e2e_jovin_f6_4")
    parts = _make_hal_parts(tmp_path / "f6_4")
    second = json.loads(json.dumps(parts["parity"]["cases"][0]))
    second["name"] = "cce_second"
    parts["parity"]["cases"].append(second)
    ctx = _make_inline_ctx({"claim": parts["claim"], "log.parity": parts["parity"]})

    crit_hal = eval_hal4(ctx)
    crit_jov = jovin.evaluate_parity(parts["parity"], parts["claim"])
    assert len(crit_hal) == 3
    assert all(c.passed for c in crit_hal)
    assert jovin.all_passed(crit_jov)


@pytest.mark.tier1
def test_f6_hal4_matches_jovin_cpu_payload_criterion_ids(repo_root: Path, tmp_path: Path) -> None:
    eval_hal4 = resolve_public_symbol("sheaf_verify", "evaluate_hal4", ("sheaf_verify.hal",))
    jovin = _load_script_module(repo_root, "scripts/jovin_cpu_payload.py", "e2e_jovin_f6_5")
    parts = _make_hal_parts(tmp_path / "f6_5")
    ctx = _make_inline_ctx({"claim": parts["claim"], "log.parity": parts["parity"]})

    hal_ids = [(c.id, c.passed) for c in eval_hal4(ctx)]
    jov_ids = [(c["id"], c["passed"]) for c in jovin.evaluate_parity(parts["parity"], parts["claim"])]
    assert hal_ids == jov_ids


@pytest.mark.tier1
def test_f6_hal4_schema_fields_match_hal_parity_schema(repo_root: Path, tmp_path: Path) -> None:
    parity_schema = json.loads((repo_root / "schemas" / "hal-parity.schema.json").read_text(encoding="utf-8"))
    claim_schema = json.loads((repo_root / "schemas" / "hal-claim.schema.json").read_text(encoding="utf-8"))
    parts = _make_hal_parts(tmp_path / "f6_6")

    for req in parity_schema["required"]:
        assert req in parts["parity"]
    assert parts["parity"]["format"] == parity_schema["properties"]["format"]["const"]
    assert len(parts["parity"]["cases"]) >= parity_schema["properties"]["cases"]["minItems"]

    for req in claim_schema["required"]:
        assert req in parts["claim"]
    assert parts["claim"]["format"] == claim_schema["properties"]["format"]["const"]
    assert parts["claim"]["study_profile"] == claim_schema["properties"]["study_profile"]["const"]


# ==============================================================================
# Feature 7: HAL6 Claim Thermal Threshold Enforcement (6 Tier 1 tests)
# ==============================================================================


@pytest.mark.tier1
def test_f7_hal6_passes_below_claim_thermal_stop_with_observed_power(repo_root: Path, tmp_path: Path) -> None:
    eval_hal6 = resolve_public_symbol("sheaf_verify", "evaluate_hal6", ("sheaf_verify.hal",))
    jovin = _load_script_module(repo_root, "scripts/jovin_cpu_payload.py", "e2e_jovin_f7_1")
    parts = _make_hal_parts(tmp_path / "f7_1")
    ctx = _make_inline_ctx({"claim": parts["claim"], "log.guards": parts["guards"]})

    assert all(c.passed for c in eval_hal6(ctx))
    assert jovin.all_passed(jovin.evaluate_guards(parts["guards"], parts["claim"]))


@pytest.mark.tier1
def test_f7_hal6_passes_with_unobserved_power_probe_and_reason(repo_root: Path, tmp_path: Path) -> None:
    eval_hal6 = resolve_public_symbol("sheaf_verify", "evaluate_hal6", ("sheaf_verify.hal",))
    jovin = _load_script_module(repo_root, "scripts/jovin_cpu_payload.py", "e2e_jovin_f7_2")
    parts = _make_hal_parts(tmp_path / "f7_2")
    parts["guards"]["power"] = {
        "observed": False,
        "probe": "nvidia-smi --query-gpu=power.draw --format=csv,noheader",
        "reason": "unavailable in CPU container",
    }
    ctx = _make_inline_ctx({"claim": parts["claim"], "log.guards": parts["guards"]})

    hal_res = eval_hal6(ctx)
    jov_res = jovin.evaluate_guards(parts["guards"], parts["claim"])
    assert all(c.passed for c in hal_res)
    assert jovin.all_passed(jov_res)


@pytest.mark.tier1
def test_f7_hal6_tighter_log_stop_below_claim_stop_passes_below_both(repo_root: Path, tmp_path: Path) -> None:
    eval_hal6 = resolve_public_symbol("sheaf_verify", "evaluate_hal6", ("sheaf_verify.hal",))
    jovin = _load_script_module(repo_root, "scripts/jovin_cpu_payload.py", "e2e_jovin_f7_3")
    parts = _make_hal_parts(tmp_path / "f7_3")
    parts["guards"]["thermal"]["stop_celsius"] = 80
    parts["guards"]["thermal"]["samples"][1]["celsius"] = 75
    ctx = _make_inline_ctx({"claim": parts["claim"], "log.guards": parts["guards"]})

    assert all(c.passed for c in eval_hal6(ctx))
    assert jovin.all_passed(jovin.evaluate_guards(parts["guards"], parts["claim"]))


@pytest.mark.tier1
def test_f7_hal6_full_hal_profile_run_passes_with_claim_guard(tmp_path: Path) -> None:
    set_probe = resolve_public_symbol("sheaf_verify", "set_probe", ("sheaf_verify.hal",))
    run_fn = resolve_public_symbol("sheaf_verify", "run", ("sheaf_verify.runner",))
    seq_clock = resolve_public_symbol("sheaf_verify", "SequenceClock", ("sheaf_verify.runner",))
    test_key = resolve_public_symbol("sheaf_verify", "TEST_KEY", ("sheaf_verify.synthetic",))

    parts = _make_hal_parts(tmp_path / "f7_4")
    store = _register_hal_store(tmp_path / "f7_4_store", parts)
    set_probe(lambda: dict(parts["env"]))
    try:
        records = run_fn(store, key=test_key, clock=seq_clock(), profile="hal")
    finally:
        set_probe(None)
    assert all(r["status"] == "pass" for r in records)


@pytest.mark.tier1
def test_f7_hal6_jovin_one_arg_backward_compat_passes(repo_root: Path, tmp_path: Path) -> None:
    jovin = _load_script_module(repo_root, "scripts/jovin_cpu_payload.py", "e2e_jovin_f7_5")
    parts = _make_hal_parts(tmp_path / "f7_5")
    assert jovin.all_passed(jovin.evaluate_guards(parts["guards"]))


@pytest.mark.tier1
def test_f7_hal6_schema_matches_hal_guards_schema(repo_root: Path, tmp_path: Path) -> None:
    guards_schema = json.loads((repo_root / "schemas" / "hal-guards.schema.json").read_text(encoding="utf-8"))
    parts = _make_hal_parts(tmp_path / "f7_6")
    for req in guards_schema["required"]:
        assert req in parts["guards"]
    assert parts["guards"]["format"] == guards_schema["properties"]["format"]["const"]
    assert isinstance(parts["guards"]["thermal"], dict)


# ==============================================================================
# Feature 8: Cluster-IS Gumbel Top-k Without-Replacement Weights (6 Tier 1 tests)
# ==============================================================================


@pytest.mark.tier1
def test_f8_numpy_cluster_is_k_eff_equals_nc_minus_1_matches_exact_ce() -> None:
    import numpy as np

    build_part = resolve_public_symbol("hadukenn_core", "build_balanced_partition", ("hadukenn_core.partitions",))
    cluster_is = resolve_public_symbol("hadukenn_core", "cluster_is_fwd_bwd", ("hadukenn_core.heads",))
    flat_tied = resolve_public_symbol("hadukenn_core", "flat_tied_fwd_bwd", ("hadukenn_core.heads",))

    rng = np.random.default_rng(7)
    nc, s_sz, d, n = 4, 4, 8, 12
    v = nc * s_sz
    part = build_part(v, nc, s_sz, [rng.integers(0, v, size=64, dtype=np.int32)], mode="mass_random", seed=7)
    H = rng.normal(0, 0.5, size=(n, d)).astype(np.float64)
    E = rng.normal(0, 0.5, size=(v, d)).astype(np.float64)
    delta_root = rng.normal(0, 0.2, size=(nc, d)).astype(np.float64)
    targets = rng.integers(0, v, size=n, dtype=np.int64)

    loss_is, root_nll, dH_is, dE_is, dd_is = cluster_is(
        H, E, delta_root, part, targets, k_neg=nc - 1, rng=np.random.default_rng(11)
    )
    loss_ex, dH_ex, dE_ex = flat_tied(H, E, targets)

    assert isinstance(root_nll, float) and math.isfinite(root_nll)
    assert np.allclose(loss_is, loss_ex, atol=1e-5)
    assert dH_is is not None and np.allclose(dH_is, dH_ex, atol=1e-5)
    assert dE_is is not None and np.allclose(dE_is, dE_ex, atol=1e-5)
    assert dd_is is not None and dd_is.shape == (nc, d)


@pytest.mark.tier1
def test_f8_numpy_cluster_is_k1_exact_and_grad_shapes() -> None:
    import numpy as np

    build_part = resolve_public_symbol("hadukenn_core", "build_balanced_partition", ("hadukenn_core.partitions",))
    cluster_is = resolve_public_symbol("hadukenn_core", "cluster_is_fwd_bwd", ("hadukenn_core.heads",))

    rng = np.random.default_rng(19)
    nc, s_sz, d, n = 8, 8, 12, 16
    v = nc * s_sz
    part = build_part(v, nc, s_sz, [rng.integers(0, v, size=128, dtype=np.int32)], mode="mass_random", seed=19)
    H = rng.normal(0, 0.3, size=(n, d)).astype(np.float32)
    E = rng.normal(0, 0.3, size=(v, d)).astype(np.float32)
    delta_root = np.zeros((nc, d), dtype=np.float32)
    targets = rng.integers(0, v, size=n, dtype=np.int64)

    loss, root_nll, dH, dE, dd = cluster_is(H, E, delta_root, part, targets, k_neg=1, rng=rng)
    assert loss.shape == (n,)
    assert np.all(np.isfinite(loss))
    assert math.isfinite(root_nll)
    assert dH is not None and dH.shape == (n, d) and np.all(np.isfinite(dH))
    assert dE is not None and dE.shape == (v, d) and np.all(np.isfinite(dE))
    assert dd is not None and dd.shape == (nc, d) and np.all(np.isfinite(dd))


@pytest.mark.tier1
def test_f8_numpy_cluster_is_finite_diff_gradient_parity() -> None:
    import numpy as np

    build_part = resolve_public_symbol("hadukenn_core", "build_balanced_partition", ("hadukenn_core.partitions",))
    cluster_is = resolve_public_symbol("hadukenn_core", "cluster_is_fwd_bwd", ("hadukenn_core.heads",))

    rng = np.random.default_rng(23)
    nc, s_sz, d, n = 4, 4, 6, 4
    v = nc * s_sz
    part = build_part(v, nc, s_sz, [rng.integers(0, v, size=64, dtype=np.int32)], mode="mass_random", seed=23)
    H = rng.normal(0, 0.4, size=(n, d)).astype(np.float64)
    E = rng.normal(0, 0.4, size=(v, d)).astype(np.float64)
    delta_root = rng.normal(0, 0.1, size=(nc, d)).astype(np.float64)
    targets = np.array([0, 5, 9, 14], dtype=np.int64)

    _, _, dH, dE, _ = cluster_is(H, E, delta_root, part, targets, k_neg=3, rng=np.random.default_rng(1))
    assert dH is not None and dE is not None

    eps = 1e-5
    fd_H = np.zeros_like(H)
    for i in range(n):
        for j in range(d):
            Hp = H.copy()
            Hm = H.copy()
            Hp[i, j] += eps
            Hm[i, j] -= eps
            lp, *_ = cluster_is(
                Hp, E, delta_root, part, targets, k_neg=3, rng=np.random.default_rng(1), compute_grad=False
            )
            lm, *_ = cluster_is(
                Hm, E, delta_root, part, targets, k_neg=3, rng=np.random.default_rng(1), compute_grad=False
            )
            fd_H[i, j] = float(np.mean(lp) - np.mean(lm)) / (2.0 * eps)
    assert np.allclose(dH, fd_H, atol=1e-5)

    fd_E = np.zeros_like(E)
    for i in range(v):
        for j in range(d):
            Ep = E.copy()
            Em = E.copy()
            Ep[i, j] += eps
            Em[i, j] -= eps
            lp, *_ = cluster_is(
                H, Ep, delta_root, part, targets, k_neg=3, rng=np.random.default_rng(1), compute_grad=False
            )
            lm, *_ = cluster_is(
                H, Em, delta_root, part, targets, k_neg=3, rng=np.random.default_rng(1), compute_grad=False
            )
            fd_E[i, j] = float(np.mean(lp) - np.mean(lm)) / (2.0 * eps)
    assert np.allclose(dE, fd_E, atol=1e-5)


@pytest.mark.tier1
def test_f8_numpy_cluster_is_without_replacement_monte_carlo_unbiased() -> None:
    import numpy as np

    build_part = resolve_public_symbol("hadukenn_core", "build_balanced_partition", ("hadukenn_core.partitions",))
    cluster_is = resolve_public_symbol("hadukenn_core", "cluster_is_fwd_bwd", ("hadukenn_core.heads",))
    flat_tied = resolve_public_symbol("hadukenn_core", "flat_tied_fwd_bwd", ("hadukenn_core.heads",))

    rng = np.random.default_rng(31)
    nc, s_sz, d = 4, 4, 4
    v = nc * s_sz
    part = build_part(v, nc, s_sz, [rng.integers(0, v, size=64, dtype=np.int32)], mode="mass_random", seed=31)
    H = np.array([[0.6, -0.3, 0.2, 0.4]], dtype=np.float64)
    E = rng.normal(0, 0.3, size=(v, d)).astype(np.float64)
    delta_root = np.array(
        [[1.2, 0.0, 0.0, 0.0], [-0.5, 0.2, 0.0, 0.0], [0.0, 0.8, 0.0, 0.0], [0.1, -0.4, 0.3, 0.0]],
        dtype=np.float64,
    )
    targets = np.array([int(part.cluster_units[0, 0])], dtype=np.int64)
    exact_loss = float(flat_tied(H, E, targets, compute_grad=False)[0][0])

    sample_rng = np.random.default_rng(1234)
    losses = []
    for _ in range(500):
        l_val, *_ = cluster_is(H, E, delta_root, part, targets, k_neg=2, rng=sample_rng, compute_grad=False)
        losses.append(float(l_val[0]))
    mean_loss = sum(losses) / len(losses)
    assert all(math.isfinite(x) for x in losses)
    assert abs(mean_loss - exact_loss) < 0.15


@pytest.mark.tier1
def test_f8_torch_bridge_availability_guard_and_parity_if_installed(repo_root: Path) -> None:
    tb = importlib.import_module("hadukenn_core.torch_bridge")
    assert isinstance(tb.TORCH_AVAILABLE, bool)
    if tb.TORCH_AVAILABLE:
        import torch
        import torch.nn.functional as F

        torch.manual_seed(0)
        nc, s_sz, d, n = 4, 4, 8, 10
        v = nc * s_sz
        cu = torch.arange(v, dtype=torch.long).view(nc, s_sz)
        head = tb.ParadigmaticClusterISHeadTorch(cu, d_model=d, k_neg=nc - 1, chunk_tokens=16)
        base_w = torch.randn(v, d, dtype=torch.float32, requires_grad=True)
        H = torch.randn(n, d, dtype=torch.float32, requires_grad=True)
        targets = torch.randint(0, v, (n,), dtype=torch.long)

        lm_loss, root_nll = head(H, targets, base_w)
        exact = F.cross_entropy(H @ base_w.t(), targets)
        assert torch.allclose(lm_loss, exact, atol=1e-5)
        assert torch.isfinite(root_nll)
        (lm_loss + root_nll).backward()
        assert H.grad is not None and torch.all(torch.isfinite(H.grad))
    else:
        assert importlib.util.find_spec("torch") is None
        assert len(tb.NF4_VALUES) == 16
        audit_fn = resolve_public_symbol("sheaf_verify", "audit_kernel_source", ("sheaf_verify.anticheat",))
        src_text = (repo_root / "src" / "hadukenn_core" / "torch_bridge.py").read_text(encoding="utf-8")
        rep = audit_fn(src_text)
        assert rep.ok is True
        assert rep.violations == ()


@pytest.mark.tier1
def test_f8_torch_bridge_passes_audit_kernel_and_nf4_constants(repo_root: Path) -> None:
    audit_fn = resolve_public_symbol("sheaf_verify", "audit_kernel_source", ("sheaf_verify.anticheat",))
    nf4_vals = resolve_public_symbol("hadukenn_core", "NF4_VALUES", ("hadukenn_core.torch_bridge",))

    src_text = (repo_root / "src" / "hadukenn_core" / "torch_bridge.py").read_text(encoding="utf-8")
    rep = audit_fn(src_text)
    assert rep.ok is True
    assert rep.violations == ()

    assert len(nf4_vals) == 16
    assert nf4_vals[0] == -1.0
    assert nf4_vals[-1] == 1.0
    assert 0.0 in nf4_vals
    assert all(nf4_vals[i] < nf4_vals[i + 1] for i in range(15))


# ==============================================================================
# Feature 9: HST-v1 Merkle Leaf Count Inclusion (6 Tier 1 tests)
# ==============================================================================


@pytest.mark.tier1
def test_f9_hst_reference_manifest_passes_integrity_and_schema(
    repo_root: Path, readonly_snapshot: dict[str, str]
) -> None:
    del readonly_snapshot
    verify_hst = resolve_public_symbol("sheaf_verify", "verify_hst_manifest_integrity", ("sheaf_verify.hst_standard",))
    manifest_path = repo_root / "iterations" / "t49-hst-foundation-stage" / "HST_V1_REFERENCE_MANIFEST.json"
    schema_path = repo_root / "schemas" / "hst-v1.schema.json"
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    ok_flag, probs = verify_hst(doc)
    assert ok_flag is True
    assert probs == []
    assert set(schema["required"]).issubset(set(doc))
    assert doc["schema"] == schema["properties"]["schema"]["const"]
    assert doc["digit_atomic_policy"] is True


@pytest.mark.tier1
def test_f9_hst_export_from_manifest_merkle_roundtrip() -> None:
    tok_cls = resolve_public_symbol("sheaf_verify", "HSTTokenizer", ("sheaf_verify.hst_standard",))
    verify_hst = resolve_public_symbol("sheaf_verify", "verify_hst_manifest_integrity", ("sheaf_verify.hst_standard",))

    corpus = [
        "hypergraph sheaf cohomology verification kernel parity 2026 " * 8,
        "merkle tree root inclusion proof causal span invariance " * 8,
    ]
    tok = tok_cls.train(corpus, vocab_size=300, stage_a_fraction=0.65, min_pair_count=2, min_superspan_pmi=0.5)
    manifest = tok.export_manifest()
    reloaded = tok_cls.from_manifest(manifest)

    assert reloaded.merkle_root_sha256() == manifest["merkle_root_sha256"]
    ok_flag, probs = verify_hst(manifest)
    assert ok_flag is True
    assert probs == []


@pytest.mark.tier1
def test_f9_hst_merkle_root_changes_when_rule_count_changes() -> None:
    tok_cls = resolve_public_symbol("sheaf_verify", "HSTTokenizer", ("sheaf_verify.hst_standard",))
    rule_cls = resolve_public_symbol("sheaf_verify", "HSTMergedRule", ("sheaf_verify.hst_standard",))

    ctrl = tok_cls.universal_control_map()
    ctrl_bytes = [k.encode("utf-8") for k, _ in sorted(ctrl.items(), key=lambda kv: kv[1])]
    base_bytes = [bytes([b]) for b in range(256)] + ctrl_bytes
    tid = len(base_bytes)
    merged_bytes = base_bytes + [b"ab"]

    tok_a = tok_cls(
        vocab_size=tid + 1,
        id_to_bytes=merged_bytes,
        rules=[rule_cls(token_id=tid, left_id=97, right_id=98, stage="A_lexical", pmi_bits=1.2, count=10, byte_len=2)],
        control_map=ctrl,
    )
    tok_b = tok_cls(
        vocab_size=tid + 1,
        id_to_bytes=merged_bytes,
        rules=[rule_cls(token_id=tid, left_id=97, right_id=98, stage="A_lexical", pmi_bits=1.2, count=11, byte_len=2)],
        control_map=ctrl,
    )
    assert tok_a.merkle_root_sha256() != tok_b.merkle_root_sha256()


@pytest.mark.tier1
def test_f9_hst_three_encode_modes_lossless_and_digit_atomic() -> None:
    tok_cls = resolve_public_symbol("sheaf_verify", "HSTTokenizer", ("sheaf_verify.hst_standard",))
    corpus = ["alpha beta gamma delta alpha beta 1024 2048 " * 10]
    tok = tok_cls.train(corpus, vocab_size=290, min_pair_count=2, min_superspan_pmi=0.2)

    sample = b"\x00\xff alpha beta 2048 \xf0\x9f\xa7\xaa"
    lengths = {}
    for mode in ("causal_local", "staged_bpe", "dp_min_tokens"):
        ids = tok.encode_bytes(sample, mode=mode)
        assert tok.decode_bytes(ids) == sample
        lengths[mode] = len(ids)
    assert lengths["dp_min_tokens"] <= lengths["causal_local"]

    digits = tok.encode_bytes(b"2048", mode="causal_local")
    assert [tok.id_to_bytes[i] for i in digits] == [b"2", b"0", b"4", b"8"]


@pytest.mark.tier1
def test_f9_hst_universal_control_tokens_and_prefix_invariance() -> None:
    tok_cls = resolve_public_symbol("sheaf_verify", "HSTTokenizer", ("sheaf_verify.hst_standard",))
    corpus = ["prefix invariance across super spans in HST standard " * 10]
    tok = tok_cls.train(corpus, vocab_size=290, min_pair_count=2, min_superspan_pmi=0.2)

    tagged = "<|hst:bos|> check <|hst:unit|> L <|hst:receipt|>"
    ids = tok.encode(tagged, mode="causal_local")
    assert ids[0] == 256
    assert 262 in ids
    assert ids[-1] == 263
    assert tok.decode(ids) == tagged

    ok_pref, rep = tok.verify_causal_prefix_invariance(
        b"prefix invariance across super spans in HST standard ",
        [b"one", b"two_123", b"three!"],
    )
    assert ok_pref is True
    assert rep["violations"] == 0


@pytest.mark.tier1
def test_f9_hst_cli_t49_and_t46_sealed_campaigns_pass(
    repo_root: Path, readonly_snapshot: dict[str, str]
) -> None:
    del readonly_snapshot
    t49_path = repo_root / "iterations" / "t49-hst-foundation-stage" / "T49_HST_FOUNDATION_RESULTS.json"
    t46_path = repo_root / "iterations" / "t46-tri-machine-hal" / "T46_TRI_MACHINE_RESULTS.json"

    p49 = run_sheaf_cli(["campaign", "--mode", "t49-hst-foundation", "--path", str(t49_path), "--json"])
    assert p49.returncode == 0, p49.stderr
    assert json.loads(p49.stdout)["ok"] is True

    p46 = run_sheaf_cli(["campaign", "--mode", "t46-tri-machine-hal", "--path", str(t46_path), "--json"])
    assert p46.returncode == 0, p46.stderr
    assert json.loads(p46.stdout)["ok"] is True


# ==============================================================================
# Feature 10: Pure-Stdlib Algebraic, Dimensional, Tick, SIMT, Crypto & R1CS Verifiers (7 Tier 1 tests)
# ==============================================================================


@pytest.mark.tier1
def test_t1_f10_qfraction_and_siunit_algebra() -> None:
    qf_cls = resolve_public_symbol("sheaf_verify", "QFraction")
    si_cls = resolve_public_symbol("sheaf_verify", "SIUnit")
    second = resolve_public_symbol("sheaf_verify", "SECOND")
    dimensionless = resolve_public_symbol("sheaf_verify", "DIMENSIONLESS")
    watt = resolve_public_symbol("sheaf_verify", "WATT")
    hash_rate = resolve_public_symbol("sheaf_verify", "HASH_RATE")

    q1 = qf_cls(6, 4)
    q2 = qf_cls(-2, -6)
    assert q1 == qf_cls(3, 2)
    assert q1.num == 3 and q1.den == 2
    assert q2 == qf_cls(1, 3)
    assert q2.den > 0
    assert qf_cls(1, 2) + qf_cls(1, 3) == qf_cls(5, 6)
    assert qf_cls(1, 2) - qf_cls(1, 3) == qf_cls(1, 6)
    assert qf_cls(2, 3) * qf_cls(3, 4) == qf_cls(1, 2)

    energy = watt * second
    assert isinstance(energy, si_cls)
    assert tuple(int(x.num) // int(x.den) for x in energy.dims) == (2, 1, -2, 0, 0, 0, 0)
    assert (dimensionless / second).same_dims(hash_rate) is True


@pytest.mark.tier1
def test_t1_f10_constants_exact() -> None:
    planck = resolve_public_symbol("sheaf_verify", "PLANCK_TIME_S")
    max_tick = resolve_public_symbol("sheaf_verify", "MAX_TICK_ABS")
    goldilocks = resolve_public_symbol("sheaf_verify", "GOLDILOCKS_P")

    assert planck == 5.391247e-44
    assert max_tick == (1 << 62) == 4611686018427387904
    assert goldilocks == (1 << 64) - (1 << 32) + 1 == 18446744069414584321


@pytest.mark.tier1
def test_t1_f10_tick_monotonicity_and_planck_floor() -> None:
    max_tick = resolve_public_symbol("sheaf_verify", "MAX_TICK_ABS")
    tick_fn = resolve_any_public_symbol(
        "sheaf_verify",
        ("verify_tick_monotonicity", "verify_ticks", "check_tick_monotonicity", "op_04_verify_ticks"),
    )
    timestamps = [0.0, 1e-6, 2.5e-6]
    ticks = [-max_tick, 0, 100, 100, max_tick]
    sig = inspect.signature(tick_fn)
    if len(sig.parameters) == 1:
        res_t = tick_fn(timestamps)
        res_q = tick_fn(ticks)
        assert _verdict_passed(res_t) is True
        assert _verdict_passed(res_q) is True
    else:
        res = _call_with_aliases(
            tick_fn,
            [
                (("timestamps", "t", "times"), timestamps),
                (("ticks", "q_ticks", "q"), ticks),
            ],
            (timestamps, ticks),
        )
        assert _verdict_passed(res) is True


@pytest.mark.tier1
def test_t1_f10_op10_sha256_cone_ker_and_receipt() -> None:
    canonical_bytes = resolve_public_symbol("sheaf_verify", "canonical_bytes", ("sheaf_verify.hashing",))
    cone_fn = resolve_any_public_symbol(
        "sheaf_verify",
        ("verify_sha256_causal_cone", "op_10_crypto_audit", "verify_crypto_causal_cone", "sha256_causal_cone_metrics"),
    )
    receipt_fn = resolve_any_public_symbol(
        "sheaf_verify",
        ("verify_canonical_receipt", "compute_canonical_receipt", "canonical_receipt_sha256"),
    )

    cone_res = cone_fn() if len(inspect.signature(cone_fn).parameters) == 0 else cone_fn(mut_word_idx=3)
    if isinstance(cone_res, dict):
        assert cone_res["dead_terminal_reg_slices"] == 1
        assert cone_res["invariant_w_words"] == 17
    else:
        assert getattr(cone_res, "dead_terminal_reg_slices") == 1
        assert getattr(cone_res, "invariant_w_words") == 17

    body = {"step": 1, "invariant_w_words": 17, "dead_terminal_reg_slices": 1}
    expected_sha = hashlib.sha256(canonical_bytes(body)).hexdigest()
    sig_r = inspect.signature(receipt_fn)
    if len(sig_r.parameters) >= 2:
        assert _verdict_passed(receipt_fn(body, expected_sha)) is True
    else:
        got_sha = receipt_fn(body)
        assert got_sha == expected_sha
        assert len(got_sha) == 64


@pytest.mark.tier1
def test_t1_f10_op14_simt_warp_butterfly_and_bank_padding() -> None:
    simt_fn = resolve_any_public_symbol(
        "sheaf_verify",
        ("verify_simt_warp_and_banks", "verify_warp_butterfly", "simt_bank_conflicts", "count_bank_conflicts"),
    )
    lanes = [float(i + 1) for i in range(32)]
    sig = inspect.signature(simt_fn)
    if "stride" in sig.parameters and len(sig.parameters) == 1:
        assert simt_fn(stride=32) == 31
        assert simt_fn(stride=33) == 0
        assert simt_fn(stride=1) == 0
    else:
        res_32 = _call_with_aliases(simt_fn, [(("lanes", "values"), lanes), (("stride",), 32)], (lanes, 32))
        res_33 = _call_with_aliases(simt_fn, [(("lanes", "values"), lanes), (("stride",), 33)], (lanes, 33))
        if isinstance(res_32, dict):
            c_key = "conflicts" if "conflicts" in res_32 else "bank_conflicts"
            assert c_key in res_32 and res_32[c_key] == 31
            assert c_key in res_33 and res_33[c_key] == 0
            sum_key = next((k for k in ("reduced_sum", "butterfly_sum", "warp_sum") if k in res_33), None)
            assert sum_key is not None, f"Missing butterfly sum key in {res_33!r}"
            assert float(res_33[sum_key]) == 528.0
            if "butterfly_err" in res_33:
                assert float(res_33["butterfly_err"]) == 0.0
        elif hasattr(res_32, "conflicts") or hasattr(res_32, "bank_conflicts"):
            c_attr = "conflicts" if hasattr(res_32, "conflicts") else "bank_conflicts"
            assert getattr(res_32, c_attr) == 31
            assert getattr(res_33, c_attr) == 0
            sum_attr = next((k for k in ("reduced_sum", "butterfly_sum", "warp_sum") if hasattr(res_33, k)), None)
            assert sum_attr is not None, f"Missing butterfly sum attribute on {res_33!r}"
            assert float(getattr(res_33, sum_attr)) == 528.0
            if hasattr(res_33, "butterfly_err"):
                assert float(res_33.butterfly_err) == 0.0
        else:
            raise AssertionError(f"Unrecognized SIMT verifier result: {res_32!r}")


@pytest.mark.tier1
def test_t1_f10_op16_goldilocks_r1cs_and_12neuron_dipole() -> None:
    goldilocks_p = resolve_public_symbol("sheaf_verify", "GOLDILOCKS_P")
    r1cs_fn = resolve_any_public_symbol(
        "sheaf_verify",
        ("verify_goldilocks_r1cs_dipole", "verify_goldilocks_r1cs", "goldilocks_r1cs_dipole_verify"),
    )
    y_float = 0.25
    dipole_acts = [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5, 0.6, -0.6]
    res = _call_with_aliases(
        r1cs_fn,
        [(("y_float", "y"), y_float), (("dipole_acts", "witness", "acts"), dipole_acts)],
        (y_float, dipole_acts),
    )
    if isinstance(res, dict):
        r_key = next(
            (k for k in ("r1cs_gate_residual", "r1cs_gate_residual_mod_p", "residual") if k in res),
            None,
        )
        assert r_key is not None, f"Missing R1CS residual key in {res!r}"
        assert res[r_key] == 0
        d_key = next((k for k in ("dipole_sum_mod_p", "mod_sum") if k in res), None)
        assert d_key is not None, f"Missing dipole sum key in {res!r}"
        assert res[d_key] == 0
        if "ok" in res or "passed" in res:
            assert _verdict_passed(res) is True
    else:
        assert _verdict_passed(res) is True
        r_attr = next(
            (k for k in ("r1cs_gate_residual", "r1cs_gate_residual_mod_p", "residual") if hasattr(res, k)),
            None,
        )
        assert r_attr is not None, f"Missing R1CS residual attribute on {res!r}"
        assert getattr(res, r_attr) == 0
        d_attr = next((k for k in ("dipole_sum_mod_p", "mod_sum") if hasattr(res, k)), None)
        assert d_attr is not None, f"Missing dipole sum attribute on {res!r}"
        assert getattr(res, d_attr) == 0
    assert goldilocks_p == (1 << 64) - (1 << 32) + 1


@pytest.mark.tier1
def test_t1_f10_sheaf_verify_100pct_pure_stdlib_ast(repo_root: Path) -> None:
    sheaf_dir = repo_root / "src" / "sheaf_verify"
    py_files = sorted(p for p in sheaf_dir.rglob("*.py") if "__pycache__" not in p.parts)
    assert len(py_files) >= 38
    stdlib = sys.stdlib_module_names
    forbidden_seen: list[str] = []
    for py_file in py_files:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top not in stdlib and top != "sheaf_verify":
                        forbidden_seen.append(f"{py_file.name}:{node.lineno}:{top}")
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                top = node.module.split(".")[0]
                if top not in stdlib and top != "sheaf_verify":
                    forbidden_seen.append(f"{py_file.name}:{node.lineno}:{top}")
    assert forbidden_seen == []


# ==============================================================================
# Feature 11: Pure-Stdlib Generalization, Info, Zigzag PI_hat, Gini & Split-Integrity Guards (7 Tier 1 tests)
# ==============================================================================


@pytest.mark.tier1
def test_t1_f11_petersen_hoeffding_and_rademacher_bounds() -> None:
    hoeffding = resolve_public_symbol("sheaf_verify", "hoeffding_bound")
    rademacher = resolve_public_symbol("sheaf_verify", "rademacher_kernel_bound")

    h_200 = _call_with_aliases(
        hoeffding,
        [(("m", "n_samples"), 200), (("delta",), 0.05), (("h_size", "num_hypotheses", "hypothesis_count"), 10)],
        (200, 0.05, 10),
    )
    h_800 = _call_with_aliases(
        hoeffding,
        [(("m", "n_samples"), 800), (("delta",), 0.05), (("h_size", "num_hypotheses", "hypothesis_count"), 10)],
        (800, 0.05, 10),
    )
    expected_200 = math.sqrt((math.log(10) + math.log(2.0 / 0.05)) / (2 * 200))
    assert math.isclose(float(h_200), expected_200, rel_tol=0.0, abs_tol=1e-12)
    assert float(h_800) < float(h_200)

    r_val = _call_with_aliases(
        rademacher,
        [(("lam", "bound_lambda", "radius_lambda"), 2.0), (("trace_k", "trace_K"), 16.0), (("m", "n_samples"), 100)],
        (2.0, 16.0, 100),
    )
    assert math.isclose(float(r_val), 0.08, rel_tol=0.0, abs_tol=1e-12)


@pytest.mark.tier1
def test_t1_f11_petersen_tikhonov_jl_and_adaboost_bounds() -> None:
    tikhonov = resolve_public_symbol("sheaf_verify", "tikhonov_stability_rate")
    jl_dim = resolve_public_symbol("sheaf_verify", "jl_min_dimension")
    adaboost = resolve_public_symbol("sheaf_verify", "adaboost_exp_bound")

    t_rate = _call_with_aliases(
        tikhonov,
        [(("rho", "lipschitz"), 1.5), (("lam", "reg"), 0.03), (("m", "n_samples"), 100)],
        (1.5, 0.03, 100),
    )
    assert math.isclose(float(t_rate), 1.5, rel_tol=0.0, abs_tol=1e-12)

    p_min = _call_with_aliases(
        jl_dim,
        [(("n", "n_points"), 1000), (("eps", "epsilon"), 0.1)],
        (1000, 0.1),
    )
    assert int(p_min) == math.ceil(20.0 * math.log(1000) / (0.1**2)) == 13816

    b_val = _call_with_aliases(
        adaboost,
        [(("gamma", "margin"), 0.1), (("t_rounds", "T", "rounds"), 20)],
        (0.1, 20),
    )
    assert math.isclose(float(b_val), math.exp(-2.0 * (0.1**2) * 20), rel_tol=1e-12, abs_tol=1e-12)


@pytest.mark.tier1
def test_t1_f11_soulmachine_logsumexp_extreme_and_bic() -> None:
    lse = resolve_public_symbol("sheaf_verify", "logsumexp")
    bic = resolve_public_symbol("sheaf_verify", "bic_score")

    hi = lse([1000.0, 1001.0])
    lo = lse([-1000.0, -1001.0])
    assert math.isfinite(hi) and math.isclose(hi, 1001.0 + math.log1p(math.exp(-1.0)), rel_tol=1e-12)
    assert math.isfinite(lo) and math.isclose(lo, -1000.0 + math.log1p(math.exp(-1.0)), rel_tol=1e-12)

    score = _call_with_aliases(
        bic,
        [(("log_likelihood", "log_lik"), -50.0), (("dof", "k"), 5), (("n", "n_samples"), 100)],
        (-50.0, 5, 100),
    )
    assert math.isclose(float(score), -50.0 - 2.5 * math.log(100.0), rel_tol=0.0, abs_tol=1e-12)


@pytest.mark.tier1
def test_t1_f11_elbo_monotonicity_and_reject_option_cost() -> None:
    elbo_fn = resolve_any_public_symbol(
        "sheaf_verify",
        ("verify_elbo_monotonicity", "check_elbo_monotonicity", "elbo_is_monotone"),
    )
    reject_fn = resolve_any_public_symbol(
        "sheaf_verify",
        ("reject_option_decision", "reject_option_cost", "bayes_reject_decision"),
    )

    assert _verdict_passed(elbo_fn([-12.0, -9.5, -8.1])) is True

    d_accept = _call_with_aliases(
        reject_fn,
        [(("probs", "posteriors"), [0.9, 0.1]), (("lambda_r",), 0.2), (("lambda_s",), 1.0)],
        ([0.9, 0.1], 0.2, 1.0),
    )
    d_reject = _call_with_aliases(
        reject_fn,
        [(("probs", "posteriors"), [0.6, 0.4]), (("lambda_r",), 0.2), (("lambda_s",), 1.0)],
        ([0.6, 0.4], 0.2, 1.0),
    )
    if isinstance(d_accept, dict):
        assert d_accept["reject"] is False
        assert d_reject["reject"] is True
    elif hasattr(d_accept, "reject"):
        assert d_accept.reject is False
        assert d_reject.reject is True
    else:
        assert bool(d_accept) is False
        assert bool(d_reject) is True


@pytest.mark.tier1
def test_t1_f11_zigzag_pi_hat_shift_and_zp_ratio() -> None:
    pi_shift = resolve_any_public_symbol(
        "sheaf_verify",
        ("zigzag_effective_pi", "shift_persistence_image", "verify_zigzag_pi_hat"),
    )
    zp_fn = resolve_any_public_symbol(
        "sheaf_verify",
        ("zigzag_persistence_ratio", "zigzag_interlayer_persistence", "compute_zigzag_zp"),
    )

    grid = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 2.0, 0.0],
        [0.0, 3.0, 4.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    pi_hat = pi_shift(grid)
    assert math.isclose(float(pi_hat[1][1]), 1.0 + 2.0 + 3.0 + 4.0, abs_tol=1e-12)

    z_pos = _call_with_aliases(
        zp_fn, [(("pi_hat", "pi_eff"), pi_hat), (("beta_p",), 4), (("l1",), 0), (("l2",), 1)], (pi_hat, 4)
    )
    z_zero = _call_with_aliases(
        zp_fn, [(("pi_hat", "pi_eff"), pi_hat), (("beta_p",), 0), (("l1",), 0), (("l2",), 1)], (pi_hat, 0)
    )
    assert math.isfinite(float(z_pos))
    assert float(z_zero) == 0.0


@pytest.mark.tier1
def test_t1_f11_alam_gini_impurity() -> None:
    gini = resolve_public_symbol("sheaf_verify", "gini_impurity")
    assert math.isclose(float(gini([10, 0])), 0.0, abs_tol=1e-12)
    assert math.isclose(float(gini([5, 5])), 0.5, abs_tol=1e-12)
    assert math.isclose(float(gini([1, 1, 1, 1])), 0.75, abs_tol=1e-12)
    assert math.isclose(float(gini([0.7, 0.3])), 0.42, abs_tol=1e-12)


@pytest.mark.tier1
def test_t1_f11_split_disjointness_clean_train_only_smote() -> None:
    split_fn = resolve_public_symbol("sheaf_verify", "validate_split_disjointness")
    res = _call_with_aliases(
        split_fn,
        [
            (("train_ids", "train"), {"tr_1", "tr_2", "smote_1"}),
            (("test_ids", "test"), {"te_1", "te_2"}),
            (("synthetic_ids", "synthetic"), {"smote_1"}),
        ],
        ({"tr_1", "tr_2", "smote_1"}, {"te_1", "te_2"}, {"smote_1"}),
    )
    assert _verdict_passed(res) is True


# ==============================================================================
# Feature 12: Pure-Stdlib SIIP Dual-Mode Gate & RLM Appendix A Pitfall Guards (7 Tier 1 tests)
# ==============================================================================


@pytest.mark.tier1
def test_t1_f12_siip_strict_decrease_commit_and_no_damp() -> None:
    gate_cls = resolve_any_public_symbol("sheaf_verify", ("SiipVerifyGate", "SIIPVerifyGate", "siip_verify_gate"))
    gate = gate_cls()
    step_fn = getattr(gate, "damp_direction", getattr(gate, "apply_hesitation_damping", None))
    assert step_fn is not None, "SiipVerifyGate must expose damp_direction / apply_hesitation_damping"
    damped = step_fn([0.9, 0.1], [1.0, 0.0])
    damped_vec = damped[0] if isinstance(damped, tuple) and isinstance(damped[0], (list, tuple)) else damped
    assert math.isclose(float(damped_vec[0]), 0.9, abs_tol=1e-12)
    assert math.isclose(float(damped_vec[1]), 0.1, abs_tol=1e-12)

    verify_fn = getattr(gate, "verify_commit", getattr(gate, "verify_candidate", None))
    assert verify_fn is not None, "SiipVerifyGate must expose verify_commit / verify_candidate"
    res = verify_fn(f_k=1.0, f_cand=0.75, x_cand=[0.5, 0.5])
    assert _verdict_passed(res) is True


@pytest.mark.tier1
def test_t1_f12_siip_hesitation_damping_triggered() -> None:
    gate_cls = resolve_any_public_symbol("sheaf_verify", ("SiipVerifyGate", "SIIPVerifyGate", "siip_verify_gate"))
    gate = gate_cls()
    step_fn = getattr(gate, "damp_direction", getattr(gate, "apply_hesitation_damping", None))
    assert step_fn is not None
    damped = step_fn([-0.8, 0.1], [1.0, 0.0])
    damped_vec = damped[0] if isinstance(damped, tuple) and isinstance(damped[0], (list, tuple)) else damped
    assert math.isclose(float(damped_vec[0]), -0.4, abs_tol=1e-12)
    assert math.isclose(float(damped_vec[1]), 0.05, abs_tol=1e-12)
    if hasattr(gate, "hesitation_damping_events"):
        assert gate.hesitation_damping_events >= 1


@pytest.mark.tier1
def test_t1_f12_siip_dual_mode_switch_sandbox_vs_formal() -> None:
    gate_cls = resolve_any_public_symbol("sheaf_verify", ("SiipVerifyGate", "SIIPVerifyGate", "siip_verify_gate"))
    sheaf_err = resolve_public_symbol("sheaf_verify", "SheafError", ("sheaf_verify.hashing",))
    gate = gate_cls()
    set_mode = getattr(gate, "set_mode", None)
    if set_mode is not None:
        set_mode("speculative")
        spec_mode = getattr(gate, "mode")
        assert spec_mode in ("speculative", "yolo", "sandbox")
        verify_fn = getattr(gate, "verify_commit", getattr(gate, "verify_candidate", None))
        assert verify_fn is not None, "SiipVerifyGate must expose verify_commit / verify_candidate"
        assert _verdict_passed(verify_fn(f_k=1.0, f_cand=1.2, x_cand=[0.5, 0.5])) is True

        set_mode("formal")
        form_mode = getattr(gate, "mode")
        assert form_mode in ("formal", "verify")
        assert form_mode != spec_mode
        try:
            form_bad = verify_fn(f_k=1.0, f_cand=1.2, x_cand=[0.5, 0.5])
        except (sheaf_err, ValueError):
            pass
        else:
            assert _verdict_passed(form_bad) is False
        assert _verdict_passed(verify_fn(f_k=1.0, f_cand=0.75, x_cand=[0.5, 0.5])) is True
    else:
        spec_gate = gate_cls(mode="speculative")
        form_gate = gate_cls(mode="formal")
        spec_mode = getattr(spec_gate, "mode")
        form_mode = getattr(form_gate, "mode")
        assert spec_mode in ("speculative", "yolo", "sandbox")
        assert form_mode in ("formal", "verify")
        assert spec_mode != form_mode
        spec_verify = getattr(spec_gate, "verify_commit", getattr(spec_gate, "verify_candidate", None))
        form_verify = getattr(form_gate, "verify_commit", getattr(form_gate, "verify_candidate", None))
        assert spec_verify is not None and form_verify is not None
        assert _verdict_passed(spec_verify(f_k=1.0, f_cand=1.2, x_cand=[0.5, 0.5])) is True
        try:
            form_bad = form_verify(f_k=1.0, f_cand=1.2, x_cand=[0.5, 0.5])
        except (sheaf_err, ValueError):
            pass
        else:
            assert _verdict_passed(form_bad) is False
        assert _verdict_passed(form_verify(f_k=1.0, f_cand=0.75, x_cand=[0.5, 0.5])) is True


@pytest.mark.tier1
def test_t1_f12_rlm_budget_policy_allows_valid_batch() -> None:
    policy_cls = resolve_public_symbol("sheaf_verify", "RLMBudgetPolicy")
    state_cls = resolve_public_symbol("sheaf_verify", "RLMBudgetState")

    policy = policy_cls(
        max_depth=1,
        max_subcalls=16,
        max_total_tokens=100_000,
        min_batch_chars=10_000,
        thinking_reserve=1024,
    )
    state = state_cls()
    if hasattr(policy, "check_subcall"):
        check_fn = policy.check_subcall
        first_alias: tuple[tuple[str, ...], Any] = (("state",), state)
        positional: tuple[Any, ...] = (state, 1, 25_000, 5_000)
    else:
        check_fn = getattr(state, "record_subcall", None)
        assert check_fn is not None, "RLMBudgetPolicy/State must expose check_subcall or record_subcall"
        first_alias = (("policy",), policy)
        positional = (policy, 1, 25_000, 5_000)
    res = _call_with_aliases(
        check_fn,
        [
            first_alias,
            (("depth",), 1),
            (("chars", "prompt_chars", "subcall_chars"), 25_000),
            (("tokens", "delta_tokens"), 5_000),
            (("max_output_tokens",), 4096),
            (("thinking_budget",), 2048),
        ],
        positional,
    )
    assert _verdict_passed(res) is True


@pytest.mark.tier1
def test_t1_f12_parse_final_direct_text_outside_code() -> None:
    parse_fn = resolve_public_symbol("sheaf_verify", "parse_final_directive")
    res = parse_fn("Analysis complete.\nFINAL(42 verified pairs)", {})
    if isinstance(res, dict):
        assert res.get("value") == "42 verified pairs"
    elif hasattr(res, "value"):
        assert res.value == "42 verified pairs"
    elif isinstance(res, tuple):
        assert "42 verified pairs" in res
    else:
        assert res == "42 verified pairs"


@pytest.mark.tier1
def test_t1_f12_parse_final_var_valid_nonempty_in_namespace() -> None:
    parse_fn = resolve_public_symbol("sheaf_verify", "parse_final_directive")
    ns = {"result_pairs": [("a", "b"), ("c", "d")]}
    res = parse_fn("Done.\nFINAL_VAR(result_pairs)", ns)
    val = res["value"] if isinstance(res, dict) else getattr(res, "value", res[1] if isinstance(res, tuple) else res)
    assert val == [("a", "b"), ("c", "d")]


@pytest.mark.tier1
def test_t1_f12_anti_repetition_tracker_progress_allowed() -> None:
    tracker_cls = resolve_public_symbol("sheaf_verify", "AntiRepetitionTracker")
    tracker = tracker_cls(max_consec_no_new_var=3)
    step_fn = getattr(tracker, "record_turn", getattr(tracker, "step", getattr(tracker, "update", None)))
    assert step_fn is not None, "AntiRepetitionTracker must expose record_turn/step/update"

    r1 = _call_with_aliases(step_fn, [(("output", "text"), "step1"), (("var_names", "vars"), {"a"})], ("step1", {"a"}))
    r2 = _call_with_aliases(
        step_fn, [(("output", "text"), "step2"), (("var_names", "vars"), {"a", "b"})], ("step2", {"a", "b"})
    )
    for r in (r1, r2):
        if isinstance(r, bool):
            assert r is False
        elif hasattr(r, "should_stop"):
            assert r.should_stop is False
        elif hasattr(r, "stagnant"):
            assert r.stagnant is False
        elif hasattr(r, "halt"):
            assert r.halt is False
        elif isinstance(r, dict):
            key = next((k for k in ("should_stop", "stagnant", "halt") if k in r), None)
            assert key is not None, f"Missing stop flag key in {r!r}"
            assert r[key] is False
        else:
            raise AssertionError(f"Unrecognized AntiRepetitionTracker result: {r!r}")
