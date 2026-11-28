"""Tier 2 boundary, negative, and exclusion E2E tests for Features 1-12.

Pre-M2 collection safety: zero top-level imports of `sheaf_verify.*`, `hadukenn_core.*`,
`scripts.*`, `numpy`, `scipy`, or `torch`. All target symbols are resolved inside test bodies.
"""

from __future__ import annotations

import base64
import copy
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
    resolve_any_public_symbol,
    resolve_public_symbol,
    run_sheaf_cli,
    scaffold_mini_selftest_repo,
)


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
    return {
        "code": code,
        "host": host,
        "kernel": kernel,
        "env": {
            "python": "3.12.11",
            "cuda": "release 12.4",
            "driver": "595.99.02",
            "gpu": "NVIDIA GeForce RTX 4080 Laptop GPU (12282 MiB, sm_89)",
            "platform": "Linux-test-harness",
        },
        "claim": {
            "format": "sheaf-verify.hal-claim.v1",
            "study_profile": "hal",
            "hypothesis_id": "e2e-hal-t2",
            "description": (
                "The compiled C11 tiled CCE kernel is faster than a serial Python loop on the recorded shapes."
            ),
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
        },
        "config": {"format": "sheaf-verify.config.v1", "study": "hal-e2e-t2"},
        "metrics": {"format": "sheaf-verify.metrics.note.v1", "note": "HAL fixture; not a G0 record"},
        "parity": {
            "format": "sheaf-verify.hal-parity.v1",
            "cases": [
                {
                    "name": "cce_tiny",
                    "op": "cce_mup_fwd",
                    "inputs": {"H": hidden, "E": embedding, "y": targets, "mup_base_d": mup_base_d},
                    "kernel_outputs": {"loss": loss},
                }
            ],
        },
        "timings": {
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
        },
        "guards": {
            "format": "sheaf-verify.hal-guards.v1",
            "thermal": {
                "stop_celsius": 85,
                "unit": "C",
                "samples": [
                    {"t": "2026-01-01T00:00:10Z", "celsius": 45},
                    {"t": "2026-01-01T00:00:20Z", "celsius": 62},
                ],
            },
            "power": {"observed": True, "draw_watts": 140.5},
        },
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


def _make_inline_ctx(docs: dict[str, Any], *, allow_nonstandard_json: bool = False) -> Any:
    parse_json_bytes = resolve_public_symbol("sheaf_verify", "parse_json_bytes", ("sheaf_verify.hashing",))

    class _InlineContext:
        def __init__(self, items: dict[str, Any]) -> None:
            self._raw = {
                k: json.dumps(v, allow_nan=allow_nonstandard_json).encode("utf-8")
                for k, v in items.items()
                if v is not None
            }

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


def _assert_rejected(fn: Any, *args: Any, **kwargs: Any) -> None:
    sheaf_err = resolve_public_symbol("sheaf_verify", "SheafError", ("sheaf_verify.hashing",))
    try:
        res = fn(*args, **kwargs)
    except (sheaf_err, ValueError, ZeroDivisionError):
        return
    assert _verdict_passed(res) is False, f"Expected rejection, got passing result {res!r}"


def _bind_with_aliases(
    fn: Any,
    alias_values: list[tuple[tuple[str, ...], Any]],
    positional_fallback: tuple[Any, ...],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
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
        sig.bind(**kwargs)
        return (), kwargs
    sig.bind(*positional_fallback)
    return positional_fallback, {}


def _call_with_aliases(
    fn: Any,
    alias_values: list[tuple[tuple[str, ...], Any]],
    positional_fallback: tuple[Any, ...],
) -> Any:
    args, kwargs = _bind_with_aliases(fn, alias_values, positional_fallback)
    return fn(*args, **kwargs)


# ==============================================================================
# Feature 1: Immutable Anchor & runner.py Profile Flag (6 Tier 2 tests)
# ==============================================================================


@pytest.mark.tier2
def test_f1_cli_anchor_refuses_out_inside_store(tmp_path: Path) -> None:
    synthetic_build = resolve_public_symbol("sheaf_verify", "build", ("sheaf_verify.synthetic",))
    store_dir = tmp_path / "store"
    synthetic_build(store_dir)
    proc = run_sheaf_cli(["anchor", "--out", str(store_dir / "ledger" / "anchor.txt")], store=store_dir)
    assert proc.returncode == 2
    assert "outside the store" in proc.stderr


@pytest.mark.tier2
def test_f1_cli_anchor_missing_or_corrupt_head_exits_2(tmp_path: Path) -> None:
    empty_store = tmp_path / "empty"
    empty_store.mkdir()
    p_missing = run_sheaf_cli(["anchor"], store=empty_store)
    assert p_missing.returncode == 2
    assert p_missing.stderr.strip()

    corrupt_store = tmp_path / "corrupt"
    (corrupt_store / "ledger").mkdir(parents=True)
    (corrupt_store / "ledger" / "head.json").write_text('{"head_sha256": "not_a_sha"}', encoding="utf-8")
    p_corrupt = run_sheaf_cli(["anchor"], store=corrupt_store)
    assert p_corrupt.returncode == 2
    assert p_corrupt.stderr.strip()


@pytest.mark.tier2
def test_f1_cli_run_invalid_profile_rejected(repo_root: Path, tmp_path: Path) -> None:
    key_file = repo_root / "fixtures" / "test_seal.key"
    proc = run_sheaf_cli(
        ["run", "--profile", "invalid_profile", "--seal-key-file", str(key_file)],
        store=tmp_path / "s",
    )
    assert proc.returncode == 2


@pytest.mark.tier2
def test_f1_runner_unknown_profile_raises_sheaf_error(tmp_path: Path) -> None:
    sheaf_err = resolve_public_symbol("sheaf_verify", "SheafError", ("sheaf_verify.hashing",))
    run_fn = resolve_public_symbol("sheaf_verify", "run", ("sheaf_verify.runner",))
    synthetic_build = resolve_public_symbol("sheaf_verify", "build", ("sheaf_verify.synthetic",))
    test_key = resolve_public_symbol("sheaf_verify", "TEST_KEY", ("sheaf_verify.synthetic",))
    store = synthetic_build(tmp_path / "s")

    with pytest.raises(sheaf_err, match="unknown study profile"):
        run_fn(store, key=test_key, profile="bogus")


@pytest.mark.tier2
def test_f1_cli_verify_fails_on_tampered_ledger_record(repo_root: Path, tmp_path: Path) -> None:
    synthetic_build = resolve_public_symbol("sheaf_verify", "build", ("sheaf_verify.synthetic",))
    store_dir = tmp_path / "tampered"
    synthetic_build(store_dir)
    ledger_path = store_dir / "ledger" / "ledger.jsonl"
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["status"] = "fail"
    lines[0] = json.dumps(rec, separators=(",", ":"), sort_keys=True)
    ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    proc = run_sheaf_cli(
        ["verify", "--json", "--seal-key-file", str(repo_root / "fixtures" / "test_seal.key")],
        store=store_dir,
    )
    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    assert len(payload["problems"]) >= 1


@pytest.mark.tier2
def test_f1_cli_verify_requires_key_without_allow_unsigned(tmp_path: Path) -> None:
    synthetic_build = resolve_public_symbol("sheaf_verify", "build", ("sheaf_verify.synthetic",))
    store_dir = tmp_path / "s"
    synthetic_build(store_dir)
    proc = run_sheaf_cli(["verify"], store=store_dir)
    assert proc.returncode == 2
    assert "seal key" in proc.stderr


# ==============================================================================
# Feature 2: sheaf_verify.selftest Parametrize, Fixtures & Pytest Non-Shadowing (6 Tier 2 tests)
# ==============================================================================


@pytest.mark.tier2
def test_f2_selftest_shim_fails_on_parametrized_assertion_failure(repo_root: Path, tmp_path: Path) -> None:
    mini = scaffold_mini_selftest_repo(repo_root, tmp_path / "mini_fail")
    (mini / "tests" / "test_pfail.py").write_text(
        """import pytest


@pytest.mark.parametrize("x", [1, 2])
def test_only_one(x):
    assert x == 1
""",
        encoding="utf-8",
    )
    rep_out = tmp_path / "pfail_report.json"
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
    assert proc.returncode == 1
    report = json.loads(rep_out.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["lint"]["ok"] is True, report["lint"]
    assert report["pytest_suite"]["passed"] == 1
    assert report["pytest_suite"]["failed"] == 1


@pytest.mark.tier2
def test_f2_selftest_audit_flags_non_stdlib_import_in_sheaf_verify(tmp_path: Path) -> None:
    audit_fn = resolve_public_symbol("sheaf_verify", "audit_python_style_stdlib", ("sheaf_verify.selftest",))
    mini = tmp_path / "mini_nonstd"
    (mini / "src" / "sheaf_verify").mkdir(parents=True)
    (mini / "src" / "sheaf_verify" / "bad_mod.py").write_text("import numpy as np\n", encoding="utf-8")
    issues = audit_fn(mini)
    assert len(issues) >= 1
    assert any("numpy" in msg for msg in issues)


@pytest.mark.tier2
def test_f2_selftest_audit_flags_tab_indentation_and_syntax_error(tmp_path: Path) -> None:
    audit_fn = resolve_public_symbol("sheaf_verify", "audit_python_style_stdlib", ("sheaf_verify.selftest",))
    mini = tmp_path / "mini_lint"
    (mini / "src" / "sheaf_verify").mkdir(parents=True)
    (mini / "tests").mkdir(parents=True)
    (mini / "src" / "sheaf_verify" / "tabbed.py").write_text("def f():\n\treturn 1\n", encoding="utf-8")
    (mini / "tests" / "bad_syn.py").write_text("def broken(:\n    pass\n", encoding="utf-8")
    issues = audit_fn(mini)
    joined = "\n".join(issues)
    assert "E101" in joined
    assert "E999" in joined


@pytest.mark.tier2
def test_f2_selftest_fails_when_fixture_anchor_drifts(repo_root: Path, tmp_path: Path) -> None:
    run_full = resolve_public_symbol("sheaf_verify", "run_full_verification", ("sheaf_verify.selftest",))
    mini = scaffold_mini_selftest_repo(repo_root, tmp_path / "mini_anchor_drift", create_default_test=True)

    head_path = mini / "fixtures" / "passing" / "ledger" / "head.json"
    head = json.loads(head_path.read_text(encoding="utf-8"))
    head["head_sha256"] = "0" * 64
    head_path.write_text(json.dumps(head), encoding="utf-8")

    ok_flag, report = run_full(mini)
    assert ok_flag is False
    assert report["canonical_fixture"]["ok"] is False


@pytest.mark.tier2
def test_f2_selftest_shim_raises_mismatch_and_broken_importorskip_handled(repo_root: Path, tmp_path: Path) -> None:
    mini = scaffold_mini_selftest_repo(repo_root, tmp_path / "mini_raises")
    (mini / "tests" / "broken_pkg.py").write_text("raise ImportError('broken optional dep')\n", encoding="utf-8")
    (mini / "tests" / "test_raises.py").write_text(
        """import pytest


def test_raises_did_not_raise():
    with pytest.raises(ValueError):
        _ = 1 + 1


def test_importorskip_broken_module():
    pytest.importorskip("broken_pkg")
    raise AssertionError("should have skipped")
""",
        encoding="utf-8",
    )
    rep_out = tmp_path / "raises_report.json"
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
    assert proc.returncode == 1
    report = json.loads(rep_out.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["lint"]["ok"] is True, report["lint"]
    assert report["pytest_suite"]["failed"] == 1
    assert any("test_raises_did_not_raise" in f for f in report["pytest_suite"]["failures"])


@pytest.mark.tier2
def test_f2_selftest_cli_report_out_creates_nested_parent_and_nonzero_on_fail(
    repo_root: Path, tmp_path: Path
) -> None:
    broken = scaffold_mini_selftest_repo(repo_root, tmp_path / "broken_repo", create_default_test=True)
    (broken / "tests" / "test_fail.py").write_text("def test_fail():\n    assert False\n", encoding="utf-8")
    rep_out = tmp_path / "nested" / "dir" / "rep.json"

    proc = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "sheaf_verify.selftest",
            "--repo-root",
            str(broken),
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
    assert proc.returncode == 1
    assert rep_out.is_file()
    report = json.loads(rep_out.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["pytest_suite"]["failed"] == 1
    assert "test_fail" in "\n".join(report["pytest_suite"]["failures"])


# ==============================================================================
# Feature 3: sheaf_verify.compliance Exact Fraction Witness Consensus (6 Tier 2 tests)
# ==============================================================================


@pytest.mark.tier2
def test_f3_compliance_sub_tolerance_1e7_difference_abstains() -> None:
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))
    ont = ont_cls("f3.subtol", [rule_cls(action_name="wire", dimension="currency")])

    v = ont.evaluate_transfer_action(
        action_name="wire",
        actor_role="operator",
        src_entity="A",
        dst_entity="B",
        src_witnesses={
            "w1": ("100000000", "USD"),
            "w2": (Fraction(100_000_000) + Fraction(1, 10**7), "USD"),
        },
        dst_initial=("0", "USD"),
        transfer_amount=("10", "USD"),
    )
    assert v.verdict == "ABSTAIN_TIE"
    assert v.approved is False
    assert v.human_escalation_required is True
    assert v.resolved_initial_src is None


@pytest.mark.tier2
def test_f3_compliance_huge_float64_collapse_1e17_plus_1_abstains() -> None:
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))
    ont = ont_cls("f3.huge", [rule_cls(action_name="wire", dimension="currency")])

    v = ont.evaluate_transfer_action(
        action_name="wire",
        actor_role="operator",
        src_entity="A",
        dst_entity="B",
        src_witnesses={"w1": (Fraction(10**17), "USD"), "w2": (Fraction(10**17 + 1), "USD")},
        dst_initial=("0", "USD"),
        transfer_amount=("1", "USD"),
    )
    assert v.verdict == "ABSTAIN_TIE"
    assert v.approved is False
    assert v.resolved_initial_src is None


@pytest.mark.tier2
def test_f3_compliance_2v2_and_3way_split_abstain() -> None:
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))
    ont = ont_cls("f3.splits", [rule_cls(action_name="wire", dimension="currency")])

    v_2v2 = ont.evaluate_transfer_action(
        action_name="wire",
        actor_role="operator",
        src_entity="A",
        dst_entity="B",
        src_witnesses={"w1": ("100", "USD"), "w2": ("100", "USD"), "w3": ("200", "USD"), "w4": ("200", "USD")},
        dst_initial=("0", "USD"),
        transfer_amount=("10", "USD"),
    )
    assert v_2v2.verdict == "ABSTAIN_TIE"
    assert v_2v2.human_escalation_required is True

    v_3way = ont.evaluate_transfer_action(
        action_name="wire",
        actor_role="operator",
        src_entity="A",
        dst_entity="B",
        src_witnesses={"w1": ("100", "USD"), "w2": ("200", "USD"), "w3": ("300", "USD")},
        dst_initial=("0", "USD"),
        transfer_amount=("10", "USD"),
    )
    assert v_3way.verdict == "ABSTAIN_TIE"
    assert v_3way.human_escalation_required is True


@pytest.mark.tier2
def test_f3_compliance_lattice_rejects_float_and_bool() -> None:
    lattice = resolve_public_symbol("sheaf_verify", "ExactUnitLattice", ("sheaf_verify.compliance",))
    sheaf_err = resolve_public_symbol("sheaf_verify", "SheafError", ("sheaf_verify.hashing",))

    with pytest.raises(sheaf_err):
        lattice.normalize(0.1, "USD")
    with pytest.raises(sheaf_err):
        lattice.normalize(True, "USD")


@pytest.mark.tier2
def test_f3_compliance_lattice_rejects_malformed_and_cross_dim() -> None:
    lattice = resolve_public_symbol("sheaf_verify", "ExactUnitLattice", ("sheaf_verify.compliance",))
    sheaf_err = resolve_public_symbol("sheaf_verify", "SheafError", ("sheaf_verify.hashing",))

    for bad_val in ("nan", "1/0", "not_a_number", ""):
        with pytest.raises((sheaf_err, ValueError, ZeroDivisionError)):
            lattice.normalize(bad_val, "USD")
    with pytest.raises(sheaf_err, match="dimensional mismatch"):
        lattice.add("1", "USD", "1", "L")
    with pytest.raises(sheaf_err, match="dimensional mismatch"):
        lattice.sub("1", "USD", "1", "kg")


@pytest.mark.tier2
def test_f3_compliance_undeclared_role_ambiguous_overdraft() -> None:
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))
    ont = ont_cls("f3.guards", [rule_cls(action_name="wire", dimension="currency")])

    v_role = ont.evaluate_transfer_action(
        action_name="wire",
        actor_role="guest",
        src_entity="A",
        dst_entity="B",
        src_witnesses={"w1": ("500", "USD")},
        dst_initial=("0", "USD"),
        transfer_amount=("10", "USD"),
    )
    assert v_role.verdict == "IMPOSSIBLE"

    v_over = ont.evaluate_transfer_action(
        action_name="wire",
        actor_role="operator",
        src_entity="A",
        dst_entity="B",
        src_witnesses={"w1": ("10", "USD")},
        dst_initial=("0", "USD"),
        transfer_amount=("50", "USD"),
    )
    assert v_over.verdict == "IMPOSSIBLE"

    v_amb = ont.evaluate_transfer_action(
        action_name="wire",
        actor_role="operator",
        src_entity="A",
        dst_entity="?",
        src_witnesses={"w1": ("500", "USD")},
        dst_initial=("0", "USD"),
        transfer_amount=("10", "USD"),
    )
    assert v_amb.verdict == "TRANSLATION_AMBIGUOUS"

    v_self = ont.evaluate_transfer_action(
        action_name="wire",
        actor_role="operator",
        src_entity="A",
        dst_entity="A",
        src_witnesses={"w1": ("500", "USD")},
        dst_initial=("0", "USD"),
        transfer_amount=("10", "USD"),
    )
    assert v_self.verdict == "TRANSLATION_AMBIGUOUS"


# ==============================================================================
# Feature 4: sheaf_verify.compliance Post-State Dimension Validation (6 Tier 2 tests)
# ==============================================================================


@pytest.mark.tier2
def test_f4_compliance_claimed_post_src_wrong_dimension_invalid() -> None:
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))
    ont = ont_cls("f4.badsrc", [rule_cls(action_name="wire", dimension="currency")])

    v = ont.evaluate_transfer_action(
        action_name="wire",
        actor_role="operator",
        src_entity="A",
        dst_entity="B",
        src_witnesses={"w1": ("1000", "USD")},
        dst_initial=("200", "USD"),
        transfer_amount=("100", "USD"),
        claimed_post_src=("900", "L"),
        claimed_post_dst=("300", "USD"),
    )
    assert v.verdict == "INVALID"
    assert v.approved is False


@pytest.mark.tier2
def test_f4_compliance_claimed_post_dst_wrong_dimension_invalid() -> None:
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))
    ont = ont_cls("f4.baddst", [rule_cls(action_name="wire", dimension="currency")])

    v = ont.evaluate_transfer_action(
        action_name="wire",
        actor_role="operator",
        src_entity="A",
        dst_entity="B",
        src_witnesses={"w1": ("1000", "USD")},
        dst_initial=("200", "USD"),
        transfer_amount=("100", "USD"),
        claimed_post_src=("900", "USD"),
        claimed_post_dst=("300", "kg"),
    )
    assert v.verdict == "INVALID"
    assert v.approved is False


@pytest.mark.tier2
def test_f4_compliance_claimed_post_ratio_unit_on_currency_invalid() -> None:
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))
    ont = ont_cls("f4.ratio", [rule_cls(action_name="wire", dimension="currency")])

    for ratio_unit in ("1", "%"):
        v = ont.evaluate_transfer_action(
            action_name="wire",
            actor_role="operator",
            src_entity="A",
            dst_entity="B",
            src_witnesses={"w1": ("1000", "USD")},
            dst_initial=("200", "USD"),
            transfer_amount=("100", "USD"),
            claimed_post_src=("900", ratio_unit),
        )
        assert v.verdict == "INVALID"
        assert v.approved is False


@pytest.mark.tier2
def test_f4_compliance_claimed_post_unknown_unit_invalid() -> None:
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))
    ont = ont_cls("f4.unk", [rule_cls(action_name="wire", dimension="currency")])

    v = ont.evaluate_transfer_action(
        action_name="wire",
        actor_role="operator",
        src_entity="A",
        dst_entity="B",
        src_witnesses={"w1": ("1000", "USD")},
        dst_initial=("200", "USD"),
        transfer_amount=("100", "USD"),
        claimed_post_src=("900", "parsecs"),
    )
    assert v.verdict == "INVALID"
    assert v.approved is False


@pytest.mark.tier2
def test_f4_compliance_claimed_post_float_or_bool_invalid() -> None:
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))
    ont = ont_cls("f4.floatbool", [rule_cls(action_name="wire", dimension="currency")])

    v_float = ont.evaluate_transfer_action(
        action_name="wire",
        actor_role="operator",
        src_entity="A",
        dst_entity="B",
        src_witnesses={"w1": ("1000", "USD")},
        dst_initial=("200", "USD"),
        transfer_amount=("100", "USD"),
        claimed_post_src=(900.0, "USD"),  # type: ignore[arg-type]
    )
    assert v_float.verdict == "INVALID"
    assert v_float.approved is False

    v_bool = ont.evaluate_transfer_action(
        action_name="wire",
        actor_role="operator",
        src_entity="A",
        dst_entity="B",
        src_witnesses={"w1": ("1000", "USD")},
        dst_initial=("200", "USD"),
        transfer_amount=("100", "USD"),
        claimed_post_dst=(True, "USD"),  # type: ignore[arg-type]
    )
    assert v_bool.verdict == "INVALID"
    assert v_bool.approved is False


@pytest.mark.tier2
def test_f4_compliance_claimed_post_conservation_and_split_mismatch_invalid() -> None:
    rule_cls = resolve_public_symbol("sheaf_verify", "ActionRuleSpec", ("sheaf_verify.compliance",))
    ont_cls = resolve_public_symbol("sheaf_verify", "LocalActionOntology", ("sheaf_verify.compliance",))
    ont = ont_cls("f4.cons", [rule_cls(action_name="wire", dimension="currency")])

    v_sum = ont.evaluate_transfer_action(
        action_name="wire",
        actor_role="operator",
        src_entity="A",
        dst_entity="B",
        src_witnesses={"w1": ("1000", "USD")},
        dst_initial=("200", "USD"),
        transfer_amount=("100", "USD"),
        claimed_post_src=("900", "USD"),
        claimed_post_dst=("400", "USD"),
    )
    assert v_sum.verdict == "INVALID"
    assert v_sum.approved is False

    v_split = ont.evaluate_transfer_action(
        action_name="wire",
        actor_role="operator",
        src_entity="A",
        dst_entity="B",
        src_witnesses={"w1": ("1000", "USD")},
        dst_initial=("200", "USD"),
        transfer_amount=("100", "USD"),
        claimed_post_src=("850", "USD"),
        claimed_post_dst=("350", "USD"),
    )
    assert v_split.verdict == "INVALID"
    assert v_split.approved is False


# ==============================================================================
# Feature 5: Regulatory Bundle Predicate & Metadata Binding (6 Tier 2 tests)
# ==============================================================================


@pytest.mark.tier2
def test_f5_bundle_rejects_spoofed_system_id_prefix_in_subject() -> None:
    emit_fn = resolve_public_symbol("sheaf_verify", "emit_regulatory_bundle", ("sheaf_verify.compliance",))
    verify_fn = resolve_public_symbol("sheaf_verify", "verify_regulatory_bundle", ("sheaf_verify.compliance",))
    dsse_sign = resolve_public_symbol("sheaf_verify", "dsse_sign", ("sheaf_verify.attestation",))
    key = b"e2e-key-spoof-subj-0123"

    bundle = emit_fn(
        system_id="sys-a",
        trace_head_sha256="a" * 64,
        state_digest_sha256="b" * 64,
        policy_sha256="c" * 64,
        events=[{"verdict": "VALID", "approved": True}],
        created_at="2026-11-28T00:00:00Z",
        signing_key=key,
    )
    stmt = json.loads(base64.b64decode(bundle["dsse_statement_envelope"]["payload"]).decode("utf-8"))
    stmt["subject"][0]["name"] = "other-sys/events.json"
    tampered = copy.deepcopy(bundle)
    tampered["dsse_statement_envelope"] = dsse_sign(stmt, key=key)

    ok_flag, errs = verify_fn(tampered, signing_key=key)
    assert ok_flag is False
    assert len(errs) >= 1


@pytest.mark.tier2
def test_f5_bundle_rejects_predicate_system_id_or_created_at_mismatch() -> None:
    emit_fn = resolve_public_symbol("sheaf_verify", "emit_regulatory_bundle", ("sheaf_verify.compliance",))
    verify_fn = resolve_public_symbol("sheaf_verify", "verify_regulatory_bundle", ("sheaf_verify.compliance",))
    dsse_sign = resolve_public_symbol("sheaf_verify", "dsse_sign", ("sheaf_verify.attestation",))
    key = b"key-pred-mismatch"

    bundle = emit_fn(
        system_id="sys-a",
        trace_head_sha256="a" * 64,
        state_digest_sha256="b" * 64,
        policy_sha256="c" * 64,
        events=[{"verdict": "VALID", "approved": True}],
        created_at="2026-11-28T00:00:00Z",
        signing_key=key,
    )
    stmt = json.loads(base64.b64decode(bundle["dsse_statement_envelope"]["payload"]).decode("utf-8"))
    stmt["predicate"]["system_id"] = "sys-other"
    tampered_sys = copy.deepcopy(bundle)
    tampered_sys["dsse_statement_envelope"] = dsse_sign(stmt, key=key)
    ok_s, errs_s = verify_fn(tampered_sys, signing_key=key)
    assert ok_s is False and len(errs_s) >= 1

    tampered_ts = copy.deepcopy(bundle)
    tampered_ts["created_at"] = "2099-01-01T00:00:00Z"
    ok_t, errs_t = verify_fn(tampered_ts, signing_key=key)
    assert ok_t is False and len(errs_t) >= 1


@pytest.mark.tier2
def test_f5_bundle_rejects_predicate_policy_sha256_mismatch() -> None:
    emit_fn = resolve_public_symbol("sheaf_verify", "emit_regulatory_bundle", ("sheaf_verify.compliance",))
    verify_fn = resolve_public_symbol("sheaf_verify", "verify_regulatory_bundle", ("sheaf_verify.compliance",))
    key = b"key-pol-mismatch"

    bundle = emit_fn(
        system_id="sys-a",
        trace_head_sha256="a" * 64,
        state_digest_sha256="b" * 64,
        policy_sha256="c" * 64,
        events=[{"verdict": "VALID", "approved": True}],
        created_at="2026-11-28T00:00:00Z",
        signing_key=key,
    )
    tampered = copy.deepcopy(bundle)
    tampered["policy_sha256"] = "f" * 64
    ok_flag, errs = verify_fn(tampered, signing_key=key)
    assert ok_flag is False and len(errs) >= 1


@pytest.mark.tier2
def test_f5_bundle_rejects_tampered_worm_integrity_sha256() -> None:
    emit_fn = resolve_public_symbol("sheaf_verify", "emit_regulatory_bundle", ("sheaf_verify.compliance",))
    verify_fn = resolve_public_symbol("sheaf_verify", "verify_regulatory_bundle", ("sheaf_verify.compliance",))
    dsse_sign = resolve_public_symbol("sheaf_verify", "dsse_sign", ("sheaf_verify.attestation",))
    key = b"e2e-key-worm-01234567"

    bundle = emit_fn(
        system_id="sys-a",
        trace_head_sha256="a" * 64,
        state_digest_sha256="b" * 64,
        policy_sha256="c" * 64,
        events=[{"verdict": "VALID", "approved": True}],
        created_at="2026-11-28T00:00:00Z",
        signing_key=key,
    )
    stmt = json.loads(base64.b64decode(bundle["dsse_statement_envelope"]["payload"]).decode("utf-8"))
    stmt["predicate"]["finra_rule_3110_worm"]["worm_integrity_sha256"] = "0" * 64
    tampered = copy.deepcopy(bundle)
    tampered["dsse_statement_envelope"] = dsse_sign(stmt, key=key)

    ok_flag, errs = verify_fn(tampered, signing_key=key)
    assert ok_flag is False and len(errs) >= 1


@pytest.mark.tier2
def test_f5_bundle_rejects_tampered_events_wrong_key_and_bad_receipt() -> None:
    emit_fn = resolve_public_symbol("sheaf_verify", "emit_regulatory_bundle", ("sheaf_verify.compliance",))
    verify_fn = resolve_public_symbol("sheaf_verify", "verify_regulatory_bundle", ("sheaf_verify.compliance",))
    key = b"e2e-key-events-rcpt-01"

    bundle = emit_fn(
        system_id="sys-a",
        trace_head_sha256="a" * 64,
        state_digest_sha256="b" * 64,
        policy_sha256="c" * 64,
        events=[{"verdict": "VALID", "approved": True}, {"verdict": "VALID", "approved": True}],
        created_at="2026-11-28T00:00:00Z",
        signing_key=key,
    )

    bad_ev = copy.deepcopy(bundle)
    bad_ev["events"][0]["approved"] = False
    ok_e, errs_e = verify_fn(bad_ev, signing_key=key)
    assert ok_e is False and len(errs_e) >= 1

    ok_k, errs_k = verify_fn(bundle, signing_key=b"wrong-key")
    assert ok_k is False and len(errs_k) >= 1

    bad_rcpt = copy.deepcopy(bundle)
    bad_rcpt["scitt_merkle_receipt"]["audit_path"] = ["0" * 64]
    ok_r, errs_r = verify_fn(bad_rcpt, signing_key=key)
    assert ok_r is False and len(errs_r) >= 1


@pytest.mark.tier2
def test_f5_bundle_emit_rejects_retention_under_6_and_empty_events() -> None:
    emit_fn = resolve_public_symbol("sheaf_verify", "emit_regulatory_bundle", ("sheaf_verify.compliance",))
    sheaf_err = resolve_public_symbol("sheaf_verify", "SheafError", ("sheaf_verify.hashing",))

    with pytest.raises(sheaf_err):
        emit_fn(
            system_id="sys-a",
            trace_head_sha256="a" * 64,
            state_digest_sha256="b" * 64,
            policy_sha256="c" * 64,
            events=[{"verdict": "VALID"}],
            created_at="2026-11-28T00:00:00Z",
            signing_key=b"k",
            retention_months=5,
        )

    with pytest.raises(sheaf_err):
        emit_fn(
            system_id="sys-a",
            trace_head_sha256="a" * 64,
            state_digest_sha256="b" * 64,
            policy_sha256="c" * 64,
            events=[],
            created_at="2026-11-28T00:00:00Z",
            signing_key=b"k",
        )


# ==============================================================================
# Feature 6: HAL4 Finite Loss/Scale Guard (6 Tier 2 tests)
# ==============================================================================


@pytest.mark.tier2
def test_f6_hal4_rejects_nan_posinf_neginf_reported_loss(repo_root: Path, tmp_path: Path) -> None:
    eval_hal4 = resolve_public_symbol("sheaf_verify", "evaluate_hal4", ("sheaf_verify.hal",))
    jovin = _load_script_module(repo_root, "scripts/jovin_cpu_payload.py", "e2e_jovin_f6_t2_1")
    base = _make_hal_parts(tmp_path / "f6_t2_1")

    for bad_val in (float("nan"), float("inf"), float("-inf")):
        bad_p = copy.deepcopy(base["parity"])
        bad_p["cases"][0]["kernel_outputs"]["loss"][0] = bad_val
        ctx = _make_inline_ctx({"claim": base["claim"], "log.parity": bad_p}, allow_nonstandard_json=True)
        assert not all(c.passed for c in eval_hal4(ctx))
        assert not jovin.all_passed(jovin.evaluate_parity(bad_p, base["claim"]))


@pytest.mark.tier2
def test_f6_hal4_rejects_nonfinite_reference_overflow_nan_inf(repo_root: Path, tmp_path: Path) -> None:
    eval_hal4 = resolve_public_symbol("sheaf_verify", "evaluate_hal4", ("sheaf_verify.hal",))
    jovin = _load_script_module(repo_root, "scripts/jovin_cpu_payload.py", "e2e_jovin_f6_t2_2")
    base = _make_hal_parts(tmp_path / "f6_t2_2")

    base["parity"]["cases"][0]["inputs"]["H"] = [[1e200, 1e200]]
    base["parity"]["cases"][0]["inputs"]["E"] = [[1e200, 1e200], [0.0, 0.0]]
    base["parity"]["cases"][0]["inputs"]["y"] = [0]
    base["parity"]["cases"][0]["inputs"]["mup_base_d"] = 2.0
    base["parity"]["cases"][0]["kernel_outputs"]["loss"] = [0.0]

    ctx = _make_inline_ctx({"claim": base["claim"], "log.parity": base["parity"]})
    assert not all(c.passed for c in eval_hal4(ctx))
    assert not jovin.all_passed(jovin.evaluate_parity(base["parity"], base["claim"]))


@pytest.mark.tier2
def test_f6_hal4_rejects_nonfinite_mup_base_d_scale(repo_root: Path, tmp_path: Path) -> None:
    eval_hal4 = resolve_public_symbol("sheaf_verify", "evaluate_hal4", ("sheaf_verify.hal",))
    jovin = _load_script_module(repo_root, "scripts/jovin_cpu_payload.py", "e2e_jovin_f6_t2_3")
    base = _make_hal_parts(tmp_path / "f6_t2_3")

    for bad_scale in (float("nan"), float("inf"), float("-inf"), True):
        bad_p = copy.deepcopy(base["parity"])
        bad_p["cases"][0]["inputs"]["mup_base_d"] = bad_scale
        ctx = _make_inline_ctx({"claim": base["claim"], "log.parity": bad_p}, allow_nonstandard_json=True)
        assert not all(c.passed for c in eval_hal4(ctx))
        assert not jovin.all_passed(jovin.evaluate_parity(bad_p, base["claim"]))


@pytest.mark.tier2
def test_f6_hal4_rejects_nonfinite_inputs_in_h_or_e(repo_root: Path, tmp_path: Path) -> None:
    eval_hal4 = resolve_public_symbol("sheaf_verify", "evaluate_hal4", ("sheaf_verify.hal",))
    jovin = _load_script_module(repo_root, "scripts/jovin_cpu_payload.py", "e2e_jovin_f6_t2_4")
    base = _make_hal_parts(tmp_path / "f6_t2_4")

    bad_h = copy.deepcopy(base["parity"])
    bad_h["cases"][0]["inputs"]["H"][0][0] = float("nan")
    ctx_h = _make_inline_ctx({"claim": base["claim"], "log.parity": bad_h}, allow_nonstandard_json=True)
    assert not all(c.passed for c in eval_hal4(ctx_h))
    assert not jovin.all_passed(jovin.evaluate_parity(bad_h, base["claim"]))

    bad_e = copy.deepcopy(base["parity"])
    bad_e["cases"][0]["inputs"]["E"][0][0] = float("inf")
    ctx_e = _make_inline_ctx({"claim": base["claim"], "log.parity": bad_e}, allow_nonstandard_json=True)
    assert not all(c.passed for c in eval_hal4(ctx_e))
    assert not jovin.all_passed(jovin.evaluate_parity(bad_e, base["claim"]))


@pytest.mark.tier2
def test_f6_hal4_rejects_loss_length_mismatch_and_unsupported_op(repo_root: Path, tmp_path: Path) -> None:
    eval_hal4 = resolve_public_symbol("sheaf_verify", "evaluate_hal4", ("sheaf_verify.hal",))
    jovin = _load_script_module(repo_root, "scripts/jovin_cpu_payload.py", "e2e_jovin_f6_t2_5")
    base = _make_hal_parts(tmp_path / "f6_t2_5")

    bad_len = copy.deepcopy(base["parity"])
    bad_len["cases"][0]["kernel_outputs"]["loss"] = [0.5]
    assert not all(c.passed for c in eval_hal4(_make_inline_ctx({"claim": base["claim"], "log.parity": bad_len})))
    assert not jovin.all_passed(jovin.evaluate_parity(bad_len, base["claim"]))

    bad_op = copy.deepcopy(base["parity"])
    bad_op["cases"][0]["op"] = "unsupported_op"
    assert not all(c.passed for c in eval_hal4(_make_inline_ctx({"claim": base["claim"], "log.parity": bad_op})))
    assert not jovin.all_passed(jovin.evaluate_parity(bad_op, base["claim"]))

    empty_cases = {"format": "sheaf-verify.hal-parity.v1", "cases": []}
    assert not all(c.passed for c in eval_hal4(_make_inline_ctx({"claim": base["claim"], "log.parity": empty_cases})))
    assert not jovin.all_passed(jovin.evaluate_parity(empty_cases, base["claim"]))


@pytest.mark.tier2
def test_f6_hal4_enforces_hal_max_abs_ceiling_against_loose_claim(repo_root: Path, tmp_path: Path) -> None:
    eval_hal4 = resolve_public_symbol("sheaf_verify", "evaluate_hal4", ("sheaf_verify.hal",))
    jovin = _load_script_module(repo_root, "scripts/jovin_cpu_payload.py", "e2e_jovin_f6_t2_6")
    base = _make_hal_parts(tmp_path / "f6_t2_6")

    base["claim"]["parity"]["max_abs"] = 1.0
    base["parity"]["cases"][0]["kernel_outputs"]["loss"][0] += 1e-2
    ctx = _make_inline_ctx({"claim": base["claim"], "log.parity": base["parity"]})

    assert not all(c.passed for c in eval_hal4(ctx))
    assert not jovin.all_passed(jovin.evaluate_parity(base["parity"], base["claim"]))


# ==============================================================================
# Feature 7: HAL6 Claim Thermal Threshold Enforcement (6 Tier 2 tests)
# ==============================================================================


@pytest.mark.tier2
def test_f7_hal6_rejects_spoofed_higher_log_stop_exceeding_claim(repo_root: Path, tmp_path: Path) -> None:
    eval_hal6 = resolve_public_symbol("sheaf_verify", "evaluate_hal6", ("sheaf_verify.hal",))
    jovin = _load_script_module(repo_root, "scripts/jovin_cpu_payload.py", "e2e_jovin_f7_t2_1")
    base = _make_hal_parts(tmp_path / "f7_t2_1")

    base["guards"]["thermal"]["stop_celsius"] = 120
    base["guards"]["thermal"]["samples"][1]["celsius"] = 88
    ctx = _make_inline_ctx({"claim": base["claim"], "log.guards": base["guards"]})

    assert not all(c.passed for c in eval_hal6(ctx))
    assert not jovin.all_passed(jovin.evaluate_guards(base["guards"], base["claim"]))


@pytest.mark.tier2
def test_f7_hal6_rejects_spoofed_higher_log_stop_even_when_samples_cool(repo_root: Path, tmp_path: Path) -> None:
    eval_hal6 = resolve_public_symbol("sheaf_verify", "evaluate_hal6", ("sheaf_verify.hal",))
    jovin = _load_script_module(repo_root, "scripts/jovin_cpu_payload.py", "e2e_jovin_f7_t2_2")
    base = _make_hal_parts(tmp_path / "f7_t2_2")

    base["guards"]["thermal"]["stop_celsius"] = 105
    base["guards"]["thermal"]["samples"] = [
        {"t": "2026-01-01T00:00:10Z", "celsius": 40},
        {"t": "2026-01-01T00:00:20Z", "celsius": 45},
    ]
    ctx = _make_inline_ctx({"claim": base["claim"], "log.guards": base["guards"]})

    assert not all(c.passed for c in eval_hal6(ctx))
    assert not jovin.all_passed(jovin.evaluate_guards(base["guards"], base["claim"]))


@pytest.mark.tier2
def test_f7_hal6_rejects_boundary_equal_to_claim_stop(repo_root: Path, tmp_path: Path) -> None:
    eval_hal6 = resolve_public_symbol("sheaf_verify", "evaluate_hal6", ("sheaf_verify.hal",))
    jovin = _load_script_module(repo_root, "scripts/jovin_cpu_payload.py", "e2e_jovin_f7_t2_3")
    base = _make_hal_parts(tmp_path / "f7_t2_3")

    base["guards"]["thermal"]["samples"][1]["celsius"] = 85
    ctx = _make_inline_ctx({"claim": base["claim"], "log.guards": base["guards"]})

    assert not all(c.passed for c in eval_hal6(ctx))
    assert not jovin.all_passed(jovin.evaluate_guards(base["guards"], base["claim"]))


@pytest.mark.tier2
def test_f7_hal6_rejects_fewer_than_two_samples_or_bool_celsius(repo_root: Path, tmp_path: Path) -> None:
    eval_hal6 = resolve_public_symbol("sheaf_verify", "evaluate_hal6", ("sheaf_verify.hal",))
    jovin = _load_script_module(repo_root, "scripts/jovin_cpu_payload.py", "e2e_jovin_f7_t2_4")
    base = _make_hal_parts(tmp_path / "f7_t2_4")

    one_sample = copy.deepcopy(base["guards"])
    one_sample["thermal"]["samples"] = [{"t": "2026-01-01T00:00:10Z", "celsius": 45}]
    assert not all(c.passed for c in eval_hal6(_make_inline_ctx({"claim": base["claim"], "log.guards": one_sample})))
    assert not jovin.all_passed(jovin.evaluate_guards(one_sample, base["claim"]))

    bool_temp = copy.deepcopy(base["guards"])
    bool_temp["thermal"]["samples"][1]["celsius"] = True
    assert not all(c.passed for c in eval_hal6(_make_inline_ctx({"claim": base["claim"], "log.guards": bool_temp})))
    assert not jovin.all_passed(jovin.evaluate_guards(bool_temp, base["claim"]))


@pytest.mark.tier2
def test_f7_hal6_rejects_missing_power_draw_or_missing_probe_reason(repo_root: Path, tmp_path: Path) -> None:
    eval_hal6 = resolve_public_symbol("sheaf_verify", "evaluate_hal6", ("sheaf_verify.hal",))
    jovin = _load_script_module(repo_root, "scripts/jovin_cpu_payload.py", "e2e_jovin_f7_t2_5")
    base = _make_hal_parts(tmp_path / "f7_t2_5")

    no_draw = copy.deepcopy(base["guards"])
    no_draw["power"] = {"observed": True}
    assert not all(c.passed for c in eval_hal6(_make_inline_ctx({"claim": base["claim"], "log.guards": no_draw})))
    assert not jovin.all_passed(jovin.evaluate_guards(no_draw, base["claim"]))

    no_reason = copy.deepcopy(base["guards"])
    no_reason["power"] = {"observed": False, "probe": "nvidia-smi", "reason": "  "}
    assert not all(c.passed for c in eval_hal6(_make_inline_ctx({"claim": base["claim"], "log.guards": no_reason})))
    assert not jovin.all_passed(jovin.evaluate_guards(no_reason, base["claim"]))


@pytest.mark.tier2
def test_f7_hal6_full_cli_run_hal_exits_1_on_claim_thermal_breach(repo_root: Path, tmp_path: Path) -> None:
    eval_hal6 = resolve_public_symbol("sheaf_verify", "evaluate_hal6", ("sheaf_verify.hal",))
    ctx_cls = resolve_public_symbol("sheaf_verify", "Context", ("sheaf_verify.context",))

    parts = _make_hal_parts(tmp_path / "f7_t2_6")
    parts["guards"]["thermal"]["stop_celsius"] = 95
    parts["guards"]["thermal"]["samples"][1]["celsius"] = 86
    store = _register_hal_store(tmp_path / "f7_t2_6_store", parts)

    hal6_criteria = eval_hal6(ctx_cls(store))
    assert any(c.id == "hal.guard.stop" and not c.passed for c in hal6_criteria)

    proc = run_sheaf_cli(
        ["run", "--profile", "hal", "--json", "--seal-key-file", str(repo_root / "fixtures" / "test_seal.key")],
        store=store.root,
    )
    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert "HAL6" in payload["failed"]


# ==============================================================================
# Feature 8: Cluster-IS Gumbel Top-k Without-Replacement Weights (6 Tier 2 tests)
# ==============================================================================


@pytest.mark.tier2
def test_f8_numpy_cluster_is_k_neg_exceeding_nc_minus_1_clamped() -> None:
    import numpy as np

    build_part = resolve_public_symbol("hadukenn_core", "build_balanced_partition", ("hadukenn_core.partitions",))
    cluster_is = resolve_public_symbol("hadukenn_core", "cluster_is_fwd_bwd", ("hadukenn_core.heads",))
    flat_tied = resolve_public_symbol("hadukenn_core", "flat_tied_fwd_bwd", ("hadukenn_core.heads",))

    rng = np.random.default_rng(41)
    nc, s_sz, d, n = 4, 4, 8, 8
    v = nc * s_sz
    part = build_part(v, nc, s_sz, [rng.integers(0, v, size=64, dtype=np.int32)], mode="mass_random", seed=41)
    H = rng.normal(0, 0.4, size=(n, d)).astype(np.float64)
    E = rng.normal(0, 0.4, size=(v, d)).astype(np.float64)
    delta_root = np.zeros((nc, d), dtype=np.float64)
    targets = rng.integers(0, v, size=n, dtype=np.int64)

    loss_is, root_nll, dH, dE, dd = cluster_is(H, E, delta_root, part, targets, k_neg=10, rng=rng)
    loss_ex, _, _ = flat_tied(H, E, targets, compute_grad=False)
    assert math.isfinite(root_nll)
    assert np.allclose(loss_is, loss_ex, atol=1e-5)
    assert dH is not None and dE is not None and dd is not None


@pytest.mark.tier2
def test_f8_numpy_cluster_is_minimal_nc2_single_token_boundary() -> None:
    import numpy as np

    build_part = resolve_public_symbol("hadukenn_core", "build_balanced_partition", ("hadukenn_core.partitions",))
    cluster_is = resolve_public_symbol("hadukenn_core", "cluster_is_fwd_bwd", ("hadukenn_core.heads",))
    flat_tied = resolve_public_symbol("hadukenn_core", "flat_tied_fwd_bwd", ("hadukenn_core.heads",))

    rng = np.random.default_rng(43)
    nc, s_sz, d, n = 2, 4, 4, 1
    v = nc * s_sz
    part = build_part(v, nc, s_sz, [rng.integers(0, v, size=32, dtype=np.int32)], mode="mass_random", seed=43)
    H = rng.normal(0, 0.5, size=(n, d)).astype(np.float64)
    E = rng.normal(0, 0.5, size=(v, d)).astype(np.float64)
    delta_root = np.zeros((nc, d), dtype=np.float64)
    targets = np.array([3], dtype=np.int64)

    loss_is, root_nll, dH, dE, dd = cluster_is(H, E, delta_root, part, targets, k_neg=1, rng=rng)
    loss_ex, _, _ = flat_tied(H, E, targets, compute_grad=False)
    assert loss_is.shape == (1,)
    assert math.isfinite(root_nll)
    assert np.allclose(loss_is, loss_ex, atol=1e-5)
    assert dH is not None and np.all(np.isfinite(dH))
    assert dE is not None and np.all(np.isfinite(dE))
    assert dd is not None and np.all(np.isfinite(dd))


@pytest.mark.tier2
def test_f8_numpy_cluster_is_extreme_logits_no_nan_inf() -> None:
    import numpy as np

    build_part = resolve_public_symbol("hadukenn_core", "build_balanced_partition", ("hadukenn_core.partitions",))
    cluster_is = resolve_public_symbol("hadukenn_core", "cluster_is_fwd_bwd", ("hadukenn_core.heads",))

    rng = np.random.default_rng(47)
    nc, s_sz, d, n = 4, 4, 8, 8
    v = nc * s_sz
    part = build_part(v, nc, s_sz, [rng.integers(0, v, size=64, dtype=np.int32)], mode="mass_random", seed=47)
    H = rng.normal(0, 20.0, size=(n, d)).astype(np.float32)
    E = rng.normal(0, 20.0, size=(v, d)).astype(np.float32)
    delta_root = rng.normal(0, 10.0, size=(nc, d)).astype(np.float32)
    targets = rng.integers(0, v, size=n, dtype=np.int64)

    loss, root_nll, dH, dE, dd = cluster_is(H, E, delta_root, part, targets, k_neg=2, rng=rng)
    assert np.all(np.isfinite(loss))
    assert math.isfinite(root_nll)
    assert dH is not None and np.all(np.isfinite(dH))
    assert dE is not None and np.all(np.isfinite(dE))
    assert dd is not None and np.all(np.isfinite(dd))


@pytest.mark.tier2
def test_f8_torch_head_invalid_constructor_args_rejected() -> None:
    tb = importlib.import_module("hadukenn_core.torch_bridge")
    assert isinstance(tb.TORCH_AVAILABLE, bool)
    if not tb.TORCH_AVAILABLE:
        assert not hasattr(tb, "ParadigmaticClusterISHeadTorch")
        return

    import torch

    cu = torch.arange(16, dtype=torch.long).view(4, 4)
    for kwargs in (
        {"d_model": 0},
        {"d_model": True},
        {"d_model": 8, "k_neg": 0},
        {"d_model": 8, "k_neg": 4},
        {"d_model": 8, "chunk_tokens": 0},
        {"d_model": 8, "chunk_tokens": False},
    ):
        with pytest.raises(ValueError):
            tb.ParadigmaticClusterISHeadTorch(cu, **kwargs)
    with pytest.raises(ValueError):
        tb.ParadigmaticClusterISHeadTorch(torch.arange(4, dtype=torch.long).view(1, 4), d_model=8, k_neg=1)


@pytest.mark.tier2
def test_f8_torch_head_empty_token_batch_preserves_grad_graph() -> None:
    tb = importlib.import_module("hadukenn_core.torch_bridge")
    assert isinstance(tb.TORCH_AVAILABLE, bool)
    if not tb.TORCH_AVAILABLE:
        assert isinstance(tb.NF4_VALUES, tuple) and len(tb.NF4_VALUES) == 16
        return

    import torch

    cu = torch.arange(16, dtype=torch.long).view(4, 4)
    head = tb.ParadigmaticClusterISHeadTorch(cu, d_model=8, k_neg=2)
    base_w = torch.randn(16, 8, requires_grad=True)
    hidden = torch.zeros(0, 8, requires_grad=True)
    targets = torch.zeros(0, dtype=torch.long)
    lm_loss, root_nll = head(hidden, targets, base_w)
    assert float(lm_loss) == 0.0
    assert float(root_nll) == 0.0
    (lm_loss + root_nll).backward()


@pytest.mark.tier2
def test_f8_cluster_partition_non_bijection_rejected() -> None:
    import numpy as np

    build_part = resolve_public_symbol("hadukenn_core", "build_balanced_partition", ("hadukenn_core.partitions",))
    with pytest.raises(ValueError, match="bijection"):
        build_part(64, 7, 9, [np.arange(64, dtype=np.int32)], mode="mass_random", seed=1)


# ==============================================================================
# Feature 9: HST-v1 Merkle Leaf Count Inclusion (6 Tier 2 tests)
# ==============================================================================


@pytest.mark.tier2
def test_f9_hst_tampered_merge_count_rejected_by_merkle_check(repo_root: Path) -> None:
    tok_cls = resolve_public_symbol("sheaf_verify", "HSTTokenizer", ("sheaf_verify.hst_standard",))
    verify_hst = resolve_public_symbol("sheaf_verify", "verify_hst_manifest_integrity", ("sheaf_verify.hst_standard",))
    doc = json.loads(
        (repo_root / "iterations" / "t49-hst-foundation-stage" / "HST_V1_REFERENCE_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    tampered = copy.deepcopy(doc)
    tampered["merges"][0]["count"] += 1

    ok_flag, probs = verify_hst(tampered)
    assert ok_flag is False
    assert any("Merkle root mismatch" in msg for msg in probs)
    with pytest.raises(ValueError, match="Merkle root mismatch"):
        tok_cls.from_manifest(tampered)


@pytest.mark.tier2
def test_f9_hst_tampered_merkle_root_and_byte_len_rejected(repo_root: Path) -> None:
    tok_cls = resolve_public_symbol("sheaf_verify", "HSTTokenizer", ("sheaf_verify.hst_standard",))
    verify_hst = resolve_public_symbol("sheaf_verify", "verify_hst_manifest_integrity", ("sheaf_verify.hst_standard",))
    doc = json.loads(
        (repo_root / "iterations" / "t49-hst-foundation-stage" / "HST_V1_REFERENCE_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )

    for mutator in (
        lambda d: d.__setitem__("merkle_root_sha256", "0" * 64),
        lambda d: d["merges"][0].__setitem__("byte_len", 99),
        lambda d: d["merges"][0].__setitem__("token_b64", "AAAA"),
    ):
        bad = copy.deepcopy(doc)
        mutator(bad)
        ok_flag, probs = verify_hst(bad)
        assert ok_flag is False and len(probs) >= 1
        with pytest.raises(ValueError):
            tok_cls.from_manifest(bad)


@pytest.mark.tier2
def test_f9_hst_rejects_bool_negative_oob_ids_and_bad_stage(repo_root: Path) -> None:
    tok_cls = resolve_public_symbol("sheaf_verify", "HSTTokenizer", ("sheaf_verify.hst_standard",))
    doc = json.loads(
        (repo_root / "iterations" / "t49-hst-foundation-stage" / "HST_V1_REFERENCE_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )

    for key, bad_val in (
        ("left_id", -1),
        ("left_id", True),
        ("right_id", 999_999),
        ("stage", "C_bad"),
        ("count", -1),
    ):
        bad = copy.deepcopy(doc)
        bad["merges"][0][key] = bad_val
        with pytest.raises(ValueError):
            tok_cls.from_manifest(bad)


@pytest.mark.tier2
def test_f9_hst_rejects_bad_vocab_size_and_stage_a_fraction() -> None:
    tok_cls = resolve_public_symbol("sheaf_verify", "HSTTokenizer", ("sheaf_verify.hst_standard",))
    with pytest.raises(ValueError):
        tok_cls.train(["short corpus"], vocab_size=267)
    with pytest.raises(ValueError):
        tok_cls.train(["short corpus"], vocab_size=300, stage_a_fraction=0.2)
    with pytest.raises(ValueError):
        tok_cls.train(["short corpus"], vocab_size=300, stage_a_fraction=0.99)


@pytest.mark.tier2
def test_f9_hst_rejects_unknown_encode_mode_and_unknown_decode_id() -> None:
    tok_cls = resolve_public_symbol("sheaf_verify", "HSTTokenizer", ("sheaf_verify.hst_standard",))
    tok = tok_cls.train(["alpha beta gamma " * 8], vocab_size=280, min_pair_count=2, min_superspan_pmi=0.2)
    with pytest.raises(ValueError):
        tok.encode_bytes(b"abc", mode="bogus")
    with pytest.raises(ValueError):
        tok.decode_bytes([-1])
    with pytest.raises(ValueError):
        tok.decode_bytes([999_999])


@pytest.mark.tier2
def test_f9_hst_t49_campaign_rejects_corrupted_pillar_and_non_dict(repo_root: Path, tmp_path: Path) -> None:
    t49_path = repo_root / "iterations" / "t49-hst-foundation-stage" / "T49_HST_FOUNDATION_RESULTS.json"
    doc = json.loads(t49_path.read_text(encoding="utf-8"))
    doc["pillar_2_equal_bytes_foundation_lm"]["wins_all_5_seeds"] = False
    bad_path = tmp_path / "bad_t49.json"
    bad_path.write_text(json.dumps(doc), encoding="utf-8")

    proc_bad = run_sheaf_cli(["campaign", "--mode", "t49-hst-foundation", "--path", str(bad_path), "--json"])
    assert proc_bad.returncode == 1
    assert json.loads(proc_bad.stdout)["ok"] is False

    list_path = tmp_path / "list_t49.json"
    list_path.write_text("[]", encoding="utf-8")
    proc_list = run_sheaf_cli(["campaign", "--mode", "t49-hst-foundation", "--path", str(list_path), "--json"])
    assert proc_list.returncode != 0


# ==============================================================================
# Feature 10: Pure-Stdlib Algebraic, Dimensional, Tick, SIMT, Crypto & R1CS Verifiers (7 Tier 2 tests)
# ==============================================================================


@pytest.mark.tier2
def test_t2_f10_qfraction_zero_den_and_siunit_mismatch() -> None:
    qf_cls = resolve_public_symbol("sheaf_verify", "QFraction")
    si_cls = resolve_public_symbol("sheaf_verify", "SIUnit")
    second = resolve_public_symbol("sheaf_verify", "SECOND")
    watt = resolve_public_symbol("sheaf_verify", "WATT")

    with pytest.raises((ZeroDivisionError, ValueError)):
        qf_cls(1, 0)
    assert second.same_dims(watt) is False

    zero_unit = si_cls((0, 0, 1, 0, 0, 0, 0), scale=0.0)
    with pytest.raises((ZeroDivisionError, ValueError)):
        _ = second / zero_unit


@pytest.mark.tier2
def test_t2_f10_tick_planck_floor_and_nan_inf_rejected() -> None:
    tick_fn = resolve_any_public_symbol(
        "sheaf_verify",
        ("verify_tick_monotonicity", "verify_ticks", "check_tick_monotonicity", "op_04_verify_ticks"),
    )
    sig = inspect.signature(tick_fn)
    if len(sig.parameters) == 1:
        assert _verdict_passed(tick_fn([0.0, 1e-6])) is True
        for bad_ts in ([0.0, 1e-45], [0.0, float("nan")], [0.0, float("inf")]):
            _assert_rejected(tick_fn, bad_ts)
    else:
        base_args, base_kwargs = _bind_with_aliases(
            tick_fn,
            [(("timestamps", "t", "times"), [0.0, 1e-6]), (("ticks", "q_ticks", "q"), [0, 1])],
            ([0.0, 1e-6], [0, 1]),
        )
        assert _verdict_passed(tick_fn(*base_args, **base_kwargs)) is True
        for bad_ts in ([0.0, 1e-45], [0.0, float("nan")], [0.0, float("inf")]):
            args, kwargs = _bind_with_aliases(
                tick_fn,
                [(("timestamps", "t", "times"), bad_ts), (("ticks", "q_ticks", "q"), [0, 1])],
                (bad_ts, [0, 1]),
            )
            _assert_rejected(tick_fn, *args, **kwargs)


@pytest.mark.tier2
def test_t2_f10_tick_max_abs_overflow_and_reversal() -> None:
    max_tick = resolve_public_symbol("sheaf_verify", "MAX_TICK_ABS")
    tick_fn = resolve_any_public_symbol(
        "sheaf_verify",
        ("verify_tick_monotonicity", "verify_ticks", "check_tick_monotonicity", "op_04_verify_ticks"),
    )
    sig = inspect.signature(tick_fn)
    if len(sig.parameters) == 1:
        assert _verdict_passed(tick_fn([0, 100])) is True
        for bad_q in ([0, max_tick + 1], [0, -max_tick - 1], [10, 9], [True, 2]):
            _assert_rejected(tick_fn, bad_q)
    else:
        base_args, base_kwargs = _bind_with_aliases(
            tick_fn,
            [(("timestamps", "t", "times"), [0.0, 1.0]), (("ticks", "q_ticks", "q"), [0, 100])],
            ([0.0, 1.0], [0, 100]),
        )
        assert _verdict_passed(tick_fn(*base_args, **base_kwargs)) is True
        for bad_q in ([0, max_tick + 1], [0, -max_tick - 1], [10, 9], [True, 2]):
            args, kwargs = _bind_with_aliases(
                tick_fn,
                [(("timestamps", "t", "times"), [0.0, 1.0]), (("ticks", "q_ticks", "q"), bad_q)],
                ([0.0, 1.0], bad_q),
            )
            _assert_rejected(tick_fn, *args, **kwargs)


@pytest.mark.tier2
def test_t2_f10_op10_tampered_receipt_and_perturbed_schedule() -> None:
    canonical_bytes = resolve_public_symbol("sheaf_verify", "canonical_bytes", ("sheaf_verify.hashing",))
    cone_fn = resolve_any_public_symbol(
        "sheaf_verify",
        ("verify_sha256_causal_cone", "op_10_crypto_audit", "verify_crypto_causal_cone", "sha256_causal_cone_metrics"),
    )
    receipt_fn = resolve_any_public_symbol(
        "sheaf_verify",
        ("verify_canonical_receipt", "compute_canonical_receipt", "canonical_receipt_sha256"),
    )

    body = {"step": 1, "invariant_w_words": 17}
    good_sha = hashlib.sha256(canonical_bytes(body)).hexdigest()
    bad_sha = ("0" if good_sha[0] != "0" else "1") + good_sha[1:]
    sig_r = inspect.signature(receipt_fn)
    if len(sig_r.parameters) >= 2:
        assert _verdict_passed(receipt_fn(body, good_sha)) is True
        _assert_rejected(receipt_fn, body, bad_sha)
    else:
        assert receipt_fn(body) == good_sha
        assert receipt_fn(body) != bad_sha
        assert receipt_fn({"step": 2, "invariant_w_words": 17}) != good_sha

    if "mut_word_idx" in inspect.signature(cone_fn).parameters:
        res_w3 = cone_fn(mut_word_idx=3)
        res_w0 = cone_fn(mut_word_idx=0)
        inv_3 = res_w3["invariant_w_words"] if isinstance(res_w3, dict) else res_w3.invariant_w_words
        inv_0 = res_w0["invariant_w_words"] if isinstance(res_w0, dict) else res_w0.invariant_w_words
        assert inv_3 == 17
        assert inv_0 != inv_3


@pytest.mark.tier2
def test_t2_f10_op14_bank_conflict_strided_even_and_bad_warp() -> None:
    simt_fn = resolve_any_public_symbol(
        "sheaf_verify",
        ("verify_simt_warp_and_banks", "verify_warp_butterfly", "simt_bank_conflicts", "count_bank_conflicts"),
    )
    sig = inspect.signature(simt_fn)
    if "stride" in sig.parameters and len(sig.parameters) == 1:
        assert simt_fn(stride=2) == 16
        assert simt_fn(stride=4) == 24
        assert simt_fn(stride=16) == 30
        _assert_rejected(simt_fn, stride=-1)
    else:
        lanes_32 = [float(i + 1) for i in range(32)]
        for stride, expected in ((2, 16), (4, 24), (16, 30)):
            res = _call_with_aliases(
                simt_fn, [(("lanes", "values"), lanes_32), (("stride",), stride)], (lanes_32, stride)
            )
            if isinstance(res, dict):
                c_key = "conflicts" if "conflicts" in res else "bank_conflicts"
                assert c_key in res, f"Missing conflicts key in {res!r}"
                assert res[c_key] == expected
            elif hasattr(res, "conflicts"):
                assert res.conflicts == expected
            elif hasattr(res, "bank_conflicts"):
                assert res.bank_conflicts == expected
            else:
                raise AssertionError(f"Unrecognized SIMT verifier result: {res!r}")
        for bad_lanes in ([1.0] * 31, [float("nan")] * 32):
            pos_fb = (bad_lanes,) if len(sig.parameters) == 1 else (bad_lanes, 1)
            args, kwargs = _bind_with_aliases(
                simt_fn,
                [(("lanes", "values"), bad_lanes), (("stride",), 1)],
                pos_fb,
            )
            _assert_rejected(simt_fn, *args, **kwargs)


@pytest.mark.tier2
def test_t2_f10_op16_r1cs_tampered_witness_and_degenerate_denom() -> None:
    goldilocks_p = resolve_public_symbol("sheaf_verify", "GOLDILOCKS_P")
    r1cs_fn = resolve_any_public_symbol(
        "sheaf_verify",
        ("verify_goldilocks_r1cs_dipole", "verify_goldilocks_r1cs", "goldilocks_r1cs_dipole_verify"),
    )
    dipole_good = [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5, 0.6, -0.6]

    baseline = _call_with_aliases(
        r1cs_fn,
        [(("y_float", "y"), 0.25), (("dipole_acts", "witness", "acts"), dipole_good)],
        (0.25, dipole_good),
    )
    assert _verdict_passed(baseline) is True

    scale_fp = 1 << 24
    y_degen_field = (-scale_fp * pow(3, goldilocks_p - 2, goldilocks_p)) % goldilocks_p
    assert (scale_fp + 3 * y_degen_field) % goldilocks_p == 0

    sig = inspect.signature(r1cs_fn)
    params = sig.parameters

    z_param = next((k for k in ("z_witness", "z", "z_override") if k in params), None)
    if z_param is not None:
        z_args, z_kwargs = _bind_with_aliases(
            r1cs_fn,
            [
                (("y_float", "y"), 0.25),
                (("dipole_acts", "witness", "acts"), dipole_good),
                ((z_param,), 12345),
            ],
            (0.25, dipole_good),
        )
        _assert_rejected(r1cs_fn, *z_args, **z_kwargs)

    v_minus_param = next(
        (k for k in ("v_minus_override", "dipole_v_minus", "v_minus", "dipole_tamper") if k in params),
        None,
    )
    if v_minus_param is not None:
        vm_args, vm_kwargs = _bind_with_aliases(
            r1cs_fn,
            [
                (("y_float", "y"), 0.25),
                (("dipole_acts", "witness", "acts"), dipole_good),
                ((v_minus_param,), 1),
            ],
            (0.25, dipole_good),
        )
        _assert_rejected(r1cs_fn, *vm_args, **vm_kwargs)

    y_field_param = next((k for k in ("y_field", "y_quantized", "y_fp") if k in params), None)
    assert y_field_param is not None, (
        f"{r1cs_fn.__name__} must expose y_field/y_quantized/y_fp to test exact B % p == 0 "
        f"(float64 loses 10 bits on {y_degen_field})"
    )
    degen_aliases: list[tuple[tuple[str, ...], Any]] = [
        ((y_field_param,), y_degen_field),
        (("dipole_acts", "witness", "acts"), dipole_good),
    ]
    y_float_param = next((k for k in ("y_float", "y") if k in params), None)
    if y_float_param is not None and params[y_float_param].default is inspect.Parameter.empty:
        degen_aliases.insert(0, ((y_float_param,), 0.25))
    d_args, d_kwargs = _bind_with_aliases(r1cs_fn, degen_aliases, (0.25, dipole_good))
    _assert_rejected(r1cs_fn, *d_args, **d_kwargs)

    for bad_y in (float("nan"), float("inf")):
        y_args, y_kwargs = _bind_with_aliases(
            r1cs_fn,
            [(("y_float", "y"), bad_y), (("dipole_acts", "witness", "acts"), dipole_good)],
            (bad_y, dipole_good),
        )
        _assert_rejected(r1cs_fn, *y_args, **y_kwargs)

    for bad_dipole in ([0.1, -0.1], [float("nan")] + dipole_good[1:]):
        dp_args, dp_kwargs = _bind_with_aliases(
            r1cs_fn,
            [(("y_float", "y"), 0.25), (("dipole_acts", "witness", "acts"), bad_dipole)],
            (0.25, bad_dipole),
        )
        _assert_rejected(r1cs_fn, *dp_args, **dp_kwargs)


@pytest.mark.tier2
def test_t2_f10_ast_audit_rejects_third_party_import(tmp_path: Path) -> None:
    audit_fn = resolve_public_symbol("sheaf_verify", "audit_python_style_stdlib", ("sheaf_verify.selftest",))
    mini = tmp_path / "mini_t2_f10"
    (mini / "src" / "sheaf_verify").mkdir(parents=True)
    (mini / "src" / "sheaf_verify" / "evil.py").write_text("import torch\n", encoding="utf-8")
    issues = audit_fn(mini)
    assert len(issues) >= 1
    assert any("torch" in issue for issue in issues)


# ==============================================================================
# Feature 11: Pure-Stdlib Generalization, Info, Zigzag PI_hat, Gini & Split-Integrity Guards (7 Tier 2 tests)
# ==============================================================================


@pytest.mark.tier2
def test_t2_f11_petersen_bounds_invalid_params() -> None:
    hoeffding = resolve_public_symbol("sheaf_verify", "hoeffding_bound")
    tikhonov = resolve_public_symbol("sheaf_verify", "tikhonov_stability_rate")
    jl_dim = resolve_public_symbol("sheaf_verify", "jl_min_dimension")
    adaboost = resolve_public_symbol("sheaf_verify", "adaboost_exp_bound")

    _assert_rejected(hoeffding, 0, 0.05, 1)
    _assert_rejected(hoeffding, 100, 1.0, 1)
    _assert_rejected(hoeffding, 100, 0.0, 1)
    _assert_rejected(tikhonov, 1.0, 0.0, 100)
    _assert_rejected(jl_dim, 100, 0.0)
    _assert_rejected(jl_dim, 100, 0.5)
    _assert_rejected(jl_dim, 100, 0.6)
    _assert_rejected(adaboost, 0.5, 10)
    _assert_rejected(adaboost, -0.1, 10)


@pytest.mark.tier2
def test_t2_f11_logsumexp_empty_nan_inf_and_bic_invalid() -> None:
    lse = resolve_public_symbol("sheaf_verify", "logsumexp")
    bic = resolve_public_symbol("sheaf_verify", "bic_score")

    _assert_rejected(lse, [])
    _assert_rejected(lse, [1.0, float("nan")])
    _assert_rejected(lse, [1.0, float("inf")])
    _assert_rejected(bic, -10.0, 2, 0)
    _assert_rejected(bic, -10.0, -1, 10)


@pytest.mark.tier2
def test_t2_f11_elbo_drop_and_nan_rejected() -> None:
    elbo_fn = resolve_any_public_symbol(
        "sheaf_verify",
        ("verify_elbo_monotonicity", "check_elbo_monotonicity", "elbo_is_monotone"),
    )
    _assert_rejected(elbo_fn, [-5.0, -4.0, -7.5])
    _assert_rejected(elbo_fn, [-5.0, float("nan"), -4.0])
    _assert_rejected(elbo_fn, [])


@pytest.mark.tier2
def test_t2_f11_reject_option_invalid_costs_and_probs() -> None:
    reject_fn = resolve_any_public_symbol(
        "sheaf_verify",
        ("reject_option_decision", "reject_option_cost", "bayes_reject_decision"),
    )
    _assert_rejected(reject_fn, [0.7, 0.3], 1.5, 1.0)
    _assert_rejected(reject_fn, [0.7, 0.3], -0.1, 1.0)
    _assert_rejected(reject_fn, [0.6, 0.6], 0.2, 1.0)
    _assert_rejected(reject_fn, [float("nan"), 1.0], 0.2, 1.0)


@pytest.mark.tier2
def test_t2_f11_zigzag_beta0_and_negative_pi_rejected() -> None:
    pi_shift = resolve_any_public_symbol(
        "sheaf_verify",
        ("zigzag_effective_pi", "shift_persistence_image", "verify_zigzag_pi_hat"),
    )
    zp_fn = resolve_any_public_symbol(
        "sheaf_verify",
        ("zigzag_persistence_ratio", "zigzag_interlayer_persistence", "compute_zigzag_zp"),
    )

    good_grid = [[0.0, 0.0], [0.0, 1.0]]
    z_zero = _call_with_aliases(zp_fn, [(("pi_hat", "pi_eff"), good_grid), (("beta_p",), 0)], (good_grid, 0))
    assert float(z_zero) == 0.0
    z_args, z_kwargs = _bind_with_aliases(
        zp_fn, [(("pi_hat", "pi_eff"), good_grid), (("beta_p",), -1)], (good_grid, -1)
    )
    _assert_rejected(zp_fn, *z_args, **z_kwargs)
    _assert_rejected(pi_shift, [[-0.5, 0.0], [0.0, 0.0]])


@pytest.mark.tier2
def test_t2_f11_gini_invalid_counts_and_probs() -> None:
    gini = resolve_public_symbol("sheaf_verify", "gini_impurity")
    _assert_rejected(gini, [])
    _assert_rejected(gini, [0, 0])
    _assert_rejected(gini, [-1, 5])
    _assert_rejected(gini, [float("nan"), 0.5])


@pytest.mark.tier2
def test_t2_f11_split_disjointness_catches_presplit_smote_leakage() -> None:
    split_fn = resolve_public_symbol("sheaf_verify", "validate_split_disjointness")
    for train_ids, test_ids, syn_ids in (
        ({"a", "b"}, {"b", "c"}, set()),
        ({"a", "smote_1"}, {"c", "smote_2"}, {"smote_1", "smote_2"}),
    ):
        args, kwargs = _bind_with_aliases(
            split_fn,
            [
                (("train_ids", "train"), train_ids),
                (("test_ids", "test"), test_ids),
                (("synthetic_ids", "synthetic"), syn_ids),
            ],
            (train_ids, test_ids, syn_ids),
        )
        _assert_rejected(split_fn, *args, **kwargs)


# ==============================================================================
# Feature 12: Pure-Stdlib SIIP Dual-Mode Gate & RLM Appendix A Pitfall Guards (7 Tier 2 tests)
# ==============================================================================


@pytest.mark.tier2
def test_t2_f12_siip_rejects_nan_inf_and_non_decreasing() -> None:
    gate_cls = resolve_any_public_symbol("sheaf_verify", ("SiipVerifyGate", "SIIPVerifyGate", "siip_verify_gate"))
    gate = gate_cls()
    verify_fn = getattr(gate, "verify_commit", getattr(gate, "verify_candidate", None))
    assert verify_fn is not None

    assert _verdict_passed(verify_fn(f_k=1.0, f_cand=0.5, x_cand=[0.5, 0.5])) is True
    for f_cand, x_cand in (
        (1.0, [0.5, 0.5]),
        (1.0001, [0.5, 0.5]),
        (float("nan"), [0.5, 0.5]),
        (float("-inf"), [0.5, 0.5]),
        (0.5, [1.0, float("nan")]),
    ):
        _assert_rejected(verify_fn, f_k=1.0, f_cand=f_cand, x_cand=x_cand)


@pytest.mark.tier2
def test_t2_f12_siip_hesitation_boundary_exact_minus_025() -> None:
    gate_cls = resolve_any_public_symbol("sheaf_verify", ("SiipVerifyGate", "SIIPVerifyGate", "siip_verify_gate"))
    gate = gate_cls()
    step_fn = getattr(gate, "damp_direction", getattr(gate, "apply_hesitation_damping", None))
    assert step_fn is not None

    zero_out = step_fn([1.0, 0.0], [0.0, 0.0])
    zero_vec = zero_out[0] if isinstance(zero_out, tuple) and isinstance(zero_out[0], (list, tuple)) else zero_out
    assert math.isclose(float(zero_vec[0]), 1.0, abs_tol=1e-12)

    p_exact = [-0.25, math.sqrt(1.0 - 0.25**2)]
    exact_out = step_fn(p_exact, [1.0, 0.0])
    exact_vec = exact_out[0] if isinstance(exact_out, tuple) and isinstance(exact_out[0], (list, tuple)) else exact_out
    assert math.isclose(float(exact_vec[0]), -0.25, abs_tol=1e-12)

    p_below = [-0.3, math.sqrt(1.0 - 0.3**2)]
    below_out = step_fn(p_below, [1.0, 0.0])
    below_vec = below_out[0] if isinstance(below_out, tuple) and isinstance(below_out[0], (list, tuple)) else below_out
    assert math.isclose(float(below_vec[0]), -0.15, abs_tol=1e-12)


def _bind_rlm_check_call(
    policy: Any,
    state: Any,
    *,
    depth: int,
    chars: int,
    tokens: int,
    max_output_tokens: int | None = None,
    thinking_budget: int | None = None,
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    if hasattr(policy, "check_subcall"):
        check_fn = policy.check_subcall
        aliases: list[tuple[tuple[str, ...], Any]] = [
            (("state",), state),
            (("depth",), depth),
            (("chars", "prompt_chars", "subcall_chars"), chars),
            (("tokens", "delta_tokens"), tokens),
        ]
        positional: tuple[Any, ...] = (state, depth, chars, tokens)
    else:
        check_fn = getattr(state, "record_subcall", None)
        assert check_fn is not None, "RLMBudgetPolicy/State must expose check_subcall or record_subcall"
        aliases = [
            (("policy",), policy),
            (("depth",), depth),
            (("chars", "prompt_chars", "subcall_chars"), chars),
            (("tokens", "delta_tokens"), tokens),
        ]
        positional = (policy, depth, chars, tokens)
    if max_output_tokens is not None:
        aliases.append((("max_output_tokens",), max_output_tokens))
    if thinking_budget is not None:
        aliases.append((("thinking_budget",), thinking_budget))
    args, kwargs = _bind_with_aliases(check_fn, aliases, positional)
    return check_fn, args, kwargs


@pytest.mark.tier2
def test_t2_f12_rlm_budget_exhaustion_depth_subcalls_tokens() -> None:
    policy_cls = resolve_public_symbol("sheaf_verify", "RLMBudgetPolicy")
    state_cls = resolve_public_symbol("sheaf_verify", "RLMBudgetState")

    policy = policy_cls(
        max_depth=1,
        max_subcalls=2,
        max_total_tokens=1_000,
        min_batch_chars=100,
        thinking_reserve=256,
    )

    state_base = state_cls()
    fn_b, args_b, kwargs_b = _bind_rlm_check_call(policy, state_base, depth=1, chars=500, tokens=100)
    assert _verdict_passed(fn_b(*args_b, **kwargs_b)) is True

    state_depth = state_cls()
    fn_d, args_d, kwargs_d = _bind_rlm_check_call(policy, state_depth, depth=2, chars=500, tokens=100)
    _assert_rejected(fn_d, *args_d, **kwargs_d)

    state_tok = state_cls()
    fn_t, args_t, kwargs_t = _bind_rlm_check_call(policy, state_tok, depth=1, chars=500, tokens=2_000)
    _assert_rejected(fn_t, *args_t, **kwargs_t)

    state_cnt = state_cls()
    for _ in range(2):
        fn_c, args_c, kwargs_c = _bind_rlm_check_call(policy, state_cnt, depth=1, chars=500, tokens=100)
        assert _verdict_passed(fn_c(*args_c, **kwargs_c)) is True
    fn_c3, args_c3, kwargs_c3 = _bind_rlm_check_call(policy, state_cnt, depth=1, chars=500, tokens=100)
    _assert_rejected(fn_c3, *args_c3, **kwargs_c3)


@pytest.mark.tier2
def test_t2_f12_rlm_micro_batch_and_thinking_exhaustion() -> None:
    policy_cls = resolve_public_symbol("sheaf_verify", "RLMBudgetPolicy")
    state_cls = resolve_public_symbol("sheaf_verify", "RLMBudgetState")

    policy = policy_cls(
        max_depth=1,
        max_subcalls=16,
        max_total_tokens=100_000,
        min_batch_chars=10_000,
        thinking_reserve=1024,
    )

    state_ok = state_cls()
    fn_ok, args_ok, kwargs_ok = _bind_rlm_check_call(
        policy,
        state_ok,
        depth=1,
        chars=20_000,
        tokens=500,
        max_output_tokens=4096,
        thinking_budget=1024,
    )
    assert _verdict_passed(fn_ok(*args_ok, **kwargs_ok)) is True

    state_micro = state_cls()
    fn_m, args_m, kwargs_m = _bind_rlm_check_call(
        policy,
        state_micro,
        depth=1,
        chars=120,
        tokens=500,
        max_output_tokens=4096,
        thinking_budget=1024,
    )
    _assert_rejected(fn_m, *args_m, **kwargs_m)

    state_think = state_cls()
    fn_th, args_th, kwargs_th = _bind_rlm_check_call(
        policy,
        state_think,
        depth=1,
        chars=20_000,
        tokens=500,
        max_output_tokens=4096,
        thinking_budget=3900,
    )
    _assert_rejected(fn_th, *args_th, **kwargs_th)


@pytest.mark.tier2
def test_t2_f12_parse_final_ignores_fenced_code_blocks() -> None:
    parse_fn = resolve_public_symbol("sheaf_verify", "parse_final_directive")
    text = "```repl\nprint('FINAL(fake)')\n# FINAL_VAR(fake)\n```\nStill working..."
    ns = {"fake": "should_not_return"}
    try:
        res = parse_fn(text, ns)
    except ValueError:
        return
    if isinstance(res, dict):
        rejected_flag = (
            ("found" in res and res["found"] is False)
            or ("accepted" in res and res["accepted"] is False)
            or ("ok" in res and res["ok"] is False)
            or ("value" in res and res["value"] is None)
        )
        assert rejected_flag, f"Expected rejected/empty parse result, got {res!r}"
    else:
        assert res is None


@pytest.mark.tier2
def test_t2_f12_parse_final_var_missing_empty_and_plan_text() -> None:
    parse_fn = resolve_public_symbol("sheaf_verify", "parse_final_directive")
    for text, ns in (
        ("FINAL_VAR(missing_var)", {}),
        ("FINAL_VAR(empty_str)", {"empty_str": ""}),
        ("FINAL_VAR(empty_list)", {"empty_list": []}),
        ("FINAL(I will now write a Python script to inspect the context in step 1)", {}),
    ):
        try:
            res = parse_fn(text, ns)
        except ValueError:
            continue
        if isinstance(res, dict):
            rejected_flag = (
                ("found" in res and res["found"] is False)
                or ("accepted" in res and res["accepted"] is False)
                or ("ok" in res and res["ok"] is False)
                or ("value" in res and res["value"] is None)
            )
            assert rejected_flag, f"Expected rejected/empty parse result, got {res!r}"
        else:
            assert res is None


@pytest.mark.tier2
def test_t2_f12_anti_repetition_tracker_halts_stagnant_loop() -> None:
    tracker_cls = resolve_public_symbol("sheaf_verify", "AntiRepetitionTracker")
    tracker = tracker_cls(max_consec_no_new_var=3)
    step_fn = getattr(tracker, "record_turn", getattr(tracker, "step", getattr(tracker, "update", None)))
    assert step_fn is not None

    _call_with_aliases(
        step_fn,
        [
            (("output", "text"), "verified step 1"),
            (("var_names", "vars"), {"verified_buf"}),
            (("verified_value",), "good"),
        ],
        ("verified step 1", {"verified_buf"}),
    )
    last_res = None
    for _ in range(4):
        last_res = _call_with_aliases(
            step_fn,
            [(("output", "text"), "verify again"), (("var_names", "vars"), {"verified_buf"})],
            ("verify again", {"verified_buf"}),
        )

    if isinstance(last_res, bool):
        assert last_res is True
    elif hasattr(last_res, "should_stop"):
        assert last_res.should_stop is True
    elif hasattr(last_res, "stagnant"):
        assert last_res.stagnant is True
    elif hasattr(last_res, "halt"):
        assert last_res.halt is True
    elif isinstance(last_res, dict):
        key = next((k for k in ("should_stop", "stagnant", "halt") if k in last_res), None)
        assert key is not None, f"Missing stop flag key in {last_res!r}"
        assert last_res[key] is True
    else:
        raise AssertionError(f"Unrecognized AntiRepetitionTracker result: {last_res!r}")
