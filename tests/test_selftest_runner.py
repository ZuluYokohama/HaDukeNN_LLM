"""Regression tests for pure-stdlib selftest runner (parametrize, autouse fixtures, skips, no shadowing)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _write_mini_repo(base: Path) -> Path:
    (base / "src" / "sheaf_verify").mkdir(parents=True)
    (base / "tests").mkdir(parents=True)
    (base / "tests" / "test_sample.py").write_text(
        '''import inspect
import pytest

EVENTS = []


@pytest.fixture(autouse=True)
def _auto_setup_teardown():
    EVENTS.append("enter")
    yield
    EVENTS.append("exit")


@pytest.mark.parametrize("outer", ["A", "B"])
@pytest.mark.parametrize("inner", [1, 2])
def test_stacked_parametrize_and_autouse(outer, inner):
    assert EVENTS[-1] == "enter"
    EVENTS.append(f"{outer}:{inner}")


@pytest.mark.parametrize(argnames="k", argvalues=[10, 20])
def test_kwargs_parametrize(k):
    EVENTS.append(f"kw:{k}")


@pytest.mark.parametrize("v", [1, pytest.param(99, marks=pytest.mark.skip(reason="skip param"))])
def test_param_skip_mark(v):
    assert v != 99
    EVENTS.append(f"pv:{v}")


@pytest.mark.custom_bare
@pytest.mark.xfail()
@pytest.mark.custom_kw(a=1)
def test_custom_and_zero_arg_marks():
    EVENTS.append("custom_ok")


def test_param_wrapper_and_skip():
    p = pytest.param("x", 42)
    assert p.values == ("x", 42)
    pytest.skip("intentional skip should not fail")


def test_importorskip_missing_package():
    pytest.importorskip("definitely_nonexistent_pkg_for_selftest_12345")
    raise AssertionError("should have skipped")


def test_importorskip_minversion_skip():
    pytest.importorskip("json", minversion="99.0.0")
    raise AssertionError("should have skipped on minversion")


def test_importorskip_minversion_pass():
    mod = pytest.importorskip("json", minversion="1.0")
    assert hasattr(mod, "loads")
    EVENTS.append("minver_ok")


@pytest.mark.skip(reason="marked skip")
def test_marked_skip():
    raise AssertionError("marked skip should not execute")


@pytest.mark.skip
def test_bare_marked_skip():
    raise AssertionError("bare marked skip should not execute")


def test_z_events_sequence():
    assert inspect.isfunction(test_kwargs_parametrize)
    assert inspect.isfunction(test_custom_and_zero_arg_marks)
    assert inspect.isfunction(test_marked_skip)
    assert inspect.isfunction(test_bare_marked_skip)
    pairs = [e for e in EVENTS if ":" in e]
    assert pairs == ["kw:10", "kw:20", "pv:1", "A:1", "A:2", "B:1", "B:2"]
    assert "custom_ok" in EVENTS
    assert "minver_ok" in EVENTS
    assert EVENTS.count("enter") == 13
    assert EVENTS.count("exit") == 12
''',
        encoding="utf-8",
    )
    return base


def test_selftest_stdlib_shim_supports_parametrize_autouse_and_skips(tmp_path: Path):
    mini = _write_mini_repo(tmp_path / "mini")
    script = (
        "import inspect, json, sys\n"
        f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
        "import sheaf_verify.selftest as st\n"
        "st._install_pytest_shim()\n"
        "st._install_pytest_shim()\n"
        "import pytest\n"
        "@pytest.mark.skip(reason='r')\n"
        "def _f1(): pass\n"
        "@pytest.mark.skip\n"
        "def _f2(): pass\n"
        "@pytest.mark.parametrize(argnames='x', argvalues=[1, 2])\n"
        "def _f3(x): pass\n"
        "assert inspect.isfunction(_f1) and inspect.isfunction(_f2) and inspect.isfunction(_f3)\n"
        f"res1 = st._run_pytest_suite_stdlib(__import__('pathlib').Path({str(mini)!r}))\n"
        f"res2 = st._run_pytest_suite_stdlib(__import__('pathlib').Path({str(mini)!r}))\n"
        "print(json.dumps({'passed': res1[0], 'failed': res1[1], 'failures': res1[2], 'passed2': res2[0], 'failed2': res2[1]}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["failed"] == 0, out["failures"]
    assert out["passed"] == 10
    assert out["failed2"] == 0
    assert out["passed2"] == 10


def test_selftest_multi_autouse_teardown_isolation(tmp_path: Path):
    mini = tmp_path / "mini_teardown"
    (mini / "src" / "sheaf_verify").mkdir(parents=True)
    (mini / "tests").mkdir(parents=True)
    (mini / "tests" / "test_teardown.py").write_text(
        '''import pytest

LOG = []


@pytest.fixture(autouse=True)
def _fix1():
    LOG.append("fix1_enter")
    yield
    LOG.append("fix1_exit")


@pytest.fixture(autouse=True)
def _fix2():
    LOG.append("fix2_enter")
    yield
    raise RuntimeError("fix2 teardown boom")


def test_body_ok():
    LOG.append("body")
''',
        encoding="utf-8",
    )
    script = (
        "import json, sys; "
        f"sys.path.insert(0, {str(ROOT / 'src')!r}); "
        "import sheaf_verify.selftest as st; "
        f"res = st._run_pytest_suite_stdlib(__import__('pathlib').Path({str(mini)!r})); "
        "import _selftest_test_teardown as mod; "
        "print(json.dumps({'passed': res[0], 'failed': res[1], 'failures': res[2], 'log': mod.LOG}))"
    )
    proc = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["passed"] == 0
    assert out["failed"] == 1
    assert out["log"] == ["fix1_enter", "fix2_enter", "body", "fix1_exit"]


def test_selftest_does_not_shadow_real_pytest_when_installed(tmp_path: Path):
    # Check in normal Python (with site-packages); skip only if pytest is genuinely not installed.
    probe = subprocess.run(
        [sys.executable, "-c", "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pytest') else 2)"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return
    mini = _write_mini_repo(tmp_path / "mini_real")
    script = (
        "import json, sys; "
        f"sys.path.insert(0, {str(ROOT / 'src')!r}); "
        "import sheaf_verify.selftest as st; "
        "st._install_pytest_shim(); "
        "import pytest; "
        "is_shim = bool(getattr(pytest, '_sheaf_shim', False)); "
        f"res = st._run_pytest_suite_stdlib(__import__('pathlib').Path({str(mini)!r})); "
        "print(json.dumps({'is_shim': is_shim, 'passed': res[0], 'failed': res[1], 'failures': res[2]}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["is_shim"] is False
    assert out["failed"] == 0, out["failures"]
    assert out["passed"] == 10
