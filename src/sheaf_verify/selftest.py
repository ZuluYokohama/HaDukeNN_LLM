"""Zero-dependency Universal Gate, Coverage, Lint & Antigravity Handoff Verifier.

Provides ``sheaf-verify selftest`` (and ``python3 -m sheaf_verify.selftest``) so
both full developer environments (with ``pytest`` and ``ruff`` installed) and
air-gapped / sandboxed Google Antigravity local-builder containers can execute
every repository gate, test function, statement-coverage check (>=85% on
``src/sheaf_verify``), 29-node control-graph pass+fail matrix, 71-test
executable hypergraph suite, 12-mutant adequacy suite, and export-hygiene audit
using only the Python 3.11+ standard library.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import inspect
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sheaf_verify import gate_trace
from sheaf_verify.control_graph import problems as check_control_graph_problems
from sheaf_verify.gate_coverage import assess as assess_gate_coverage
from sheaf_verify.mutation import run_mutation_suite
from sheaf_verify.seal import load_key
from sheaf_verify.store import Store
from sheaf_verify.verify import verify_store

CANONICAL_ANCHOR = "e247484042475d381eb454df95b718f55739a6cd0385006f142be257b8b86567"


@dataclass
class _CaughtException:
    value: BaseException | None = None


class _RaisesContext:
    def __init__(self, expected: type[BaseException] | tuple[type[BaseException], ...], match: str | None = None):
        self.expected = expected
        self.match = match
        self.box = _CaughtException()

    def __enter__(self) -> _CaughtException:
        return self.box

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        del exc_tb
        if exc_type is None:
            raise AssertionError(f"Expected {self.expected!r} to be raised, but no exception occurred")
        if not issubclass(exc_type, self.expected):
            return False
        self.box.value = exc_val
        if self.match is not None and not re.search(self.match, str(exc_val)):
            raise AssertionError(f"Exception {str(exc_val)!r} did not match pattern {self.match!r}")
        return True


class _Approx:
    def __init__(self, expected: float, rel: float | None = None, abs_tol: float | None = None):
        self.expected = float(expected)
        self.rel = 1e-6 if rel is None else float(rel)
        self.abs_tol = 1e-12 if abs_tol is None else float(abs_tol)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (int, float)):
            return False
        tol = max(self.abs_tol, self.rel * abs(self.expected))
        return math.isclose(float(other), self.expected, rel_tol=0.0, abs_tol=tol)

    def __repr__(self) -> str:
        return f"approx({self.expected!r}, rel={self.rel!r}, abs={self.abs_tol!r})"


class _MonkeyPatch:
    def __init__(self) -> None:
        self._undo: list[Callable[[], None]] = []

    def setenv(self, name: str, value: str) -> None:
        had = name in os.environ
        old = os.environ.get(name)
        os.environ[name] = str(value)
        self._undo.append(
            lambda: os.environ.__setitem__(name, old or "") if had else os.environ.pop(name, None)  # type: ignore[misc]
        )

    def delenv(self, name: str, raising: bool = True) -> None:
        if name not in os.environ:
            if raising:
                raise KeyError(name)
            return
        old = os.environ.pop(name)
        self._undo.append(lambda: os.environ.__setitem__(name, old))

    def setattr(self, target: object, name: object, value: object = ..., raising: bool = True) -> None:
        if value is ... and isinstance(target, str):
            value = name
            mod_path, _, attr_name = target.rpartition(".")
            import importlib
            target_obj = importlib.import_module(mod_path)
            name_str = attr_name
        else:
            target_obj = target
            name_str = str(name)
        had = hasattr(target_obj, name_str)
        if not had and raising:
            raise AttributeError(f"{target_obj!r} has no attribute {name_str!r}")
        old = getattr(target_obj, name_str, None)
        setattr(target_obj, name_str, value)
        self._undo.append(lambda: setattr(target_obj, name_str, old) if had else delattr(target_obj, name_str))

    def chdir(self, path: str | Path) -> None:
        old = Path.cwd()
        os.chdir(path)
        self._undo.append(lambda: os.chdir(old))

    def restore(self) -> None:
        while self._undo:
            fn = self._undo.pop()
            fn()


class _CapSys:
    def __init__(self) -> None:
        self._out = io.StringIO()
        self._err = io.StringIO()

    def readouterr(self) -> types.SimpleNamespace:
        out_val = self._out.getvalue()
        err_val = self._err.getvalue()
        self._out.seek(0)
        self._out.truncate(0)
        self._err.seek(0)
        self._err.truncate(0)
        return types.SimpleNamespace(out=out_val, err=err_val)


def _install_pytest_shim() -> None:
    if "pytest" in sys.modules:
        return
    mod = types.ModuleType("pytest")
    mod.raises = lambda expected, match=None: _RaisesContext(expected, match=match)  # type: ignore[attr-defined]
    mod.approx = lambda val, rel=None, abs=None: _Approx(val, rel=rel, abs_tol=abs)  # type: ignore[attr-defined]
    sys.modules["pytest"] = mod


def _executable_lines(py_file: Path) -> set[int]:
    """Extract executable statement line numbers from an AST for coverage calculation."""
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(py_file))
    lines: set[int] = set()

    def visit_block(stmts: list[ast.stmt]) -> None:
        for idx, stmt in enumerate(stmts):
            if (
                idx == 0
                and isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            ):
                continue
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                lines.add(stmt.lineno)
                visit_block(stmt.body)
            elif isinstance(stmt, ast.If):
                lines.add(stmt.lineno)
                visit_block(stmt.body)
                visit_block(stmt.orelse)
            elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
                lines.add(stmt.lineno)
                visit_block(stmt.body)
                visit_block(stmt.orelse)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                lines.add(stmt.lineno)
                visit_block(stmt.body)
            elif isinstance(stmt, ast.Try):
                lines.add(stmt.lineno)
                visit_block(stmt.body)
                for handler in stmt.handlers:
                    lines.add(handler.lineno)
                    visit_block(handler.body)
                visit_block(stmt.orelse)
                visit_block(stmt.finalbody)
            else:
                lines.add(stmt.lineno)

    visit_block(tree.body)
    return lines


def audit_python_style_stdlib(repo_root: Path) -> list[str]:
    """Static AST & line audit checking syntax, tabs, and sheaf_verify stdlib purity."""
    issues: list[str] = []
    stdlib_names = sys.stdlib_module_names
    for folder in ("src", "tests"):
        base = repo_root / folder
        if not base.is_dir():
            continue
        for py_file in sorted(base.rglob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            rel = py_file.relative_to(repo_root).as_posix()
            text = py_file.read_text(encoding="utf-8")
            for lineno, raw_line in enumerate(text.splitlines(), 1):
                if "\t" in raw_line:
                    issues.append(f"{rel}:{lineno}: E101 indentation contains tabs")
            try:
                tree = ast.parse(text, filename=rel)
            except SyntaxError as exc:
                issues.append(f"{rel}:{exc.lineno}: E999 SyntaxError: {exc.msg}")
                continue
            if rel.startswith("src/sheaf_verify/"):
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            top = alias.name.split(".")[0]
                            if top not in stdlib_names and top != "sheaf_verify":
                                issues.append(f"{rel}:{node.lineno}: non-stdlib import {top!r} in sheaf_verify")
                    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                        top = node.module.split(".")[0]
                        if top not in stdlib_names and top != "sheaf_verify":
                            issues.append(f"{rel}:{node.lineno}: non-stdlib import {top!r} in sheaf_verify")
    return issues


def _mark_top_level_defs(tree: ast.Module, covered: set[int]) -> None:
    """Mark already-imported module-level imports, assignments, class headers, and def headers."""
    for stmt in tree.body:
        covered.add(stmt.lineno)
        if isinstance(stmt, ast.ClassDef):
            for cstmt in stmt.body:
                covered.add(cstmt.lineno)


def _run_pytest_suite_stdlib(repo_root: Path) -> tuple[int, int, list[str], float, dict[str, str], list[dict]]:
    """Run all tests/test_*.py with gate_trace and statement coverage on src/sheaf_verify."""
    _install_pytest_shim()
    src_dir = (repo_root / "src").resolve()
    tests_dir = (repo_root / "tests").resolve()
    for entry in (str(repo_root.resolve()), str(src_dir), str(tests_dir)):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    sheaf_dir = (src_dir / "sheaf_verify").resolve()
    tracked_files: dict[str, set[int]] = {}
    covered_lines: dict[str, set[int]] = {}
    for py_file in sorted(sheaf_dir.glob("*.py")):
        key = str(py_file.resolve())
        tracked_files[key] = _executable_lines(py_file)
        covered_lines[key] = set()

    for mod in list(sys.modules.values()):
        mod_file = getattr(mod, "__file__", None)
        if mod_file:
            resolved = str(Path(mod_file).resolve())
            if resolved in covered_lines:
                try:
                    mod_tree = ast.parse(Path(resolved).read_text(encoding="utf-8"), filename=resolved)
                    _mark_top_level_defs(mod_tree, covered_lines[resolved])
                except OSError:
                    pass

    def tracer(frame, event, arg):
        del arg
        if event == "line":
            fname = frame.f_code.co_filename
            if fname in covered_lines:
                covered_lines[fname].add(frame.f_lineno)
        return tracer

    accumulated_events: list[dict] = []
    passed = 0
    failed = 0
    failures: list[str] = []

    prev_trace = sys.gettrace()
    sys.settrace(tracer)
    try:
        for test_path in sorted(tests_dir.glob("test_*.py")):
            mod_name = f"_selftest_{test_path.stem}"
            spec = importlib.util.spec_from_file_location(mod_name, test_path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)

            for attr_name in dir(module):
                if not attr_name.startswith("test_"):
                    continue
                fn = getattr(module, attr_name)
                if not callable(fn) or not inspect.isfunction(fn):
                    continue
                nodeid = f"tests/{test_path.name}::{attr_name}"
                gate_trace.enable(clear=True)
                gate_trace.set_test(nodeid)
                mp = _MonkeyPatch()
                cap = _CapSys()
                quiet_out = io.StringIO()
                quiet_err = io.StringIO()
                with tempfile.TemporaryDirectory(prefix="sheaf_t_") as tmp_dir:
                    sig = inspect.signature(fn)
                    kwargs: dict[str, object] = {}
                    for pname in sig.parameters:
                        if pname == "tmp_path":
                            kwargs[pname] = Path(tmp_dir)
                        elif pname == "monkeypatch":
                            kwargs[pname] = mp
                        elif pname == "capsys":
                            kwargs[pname] = cap
                    try:
                        out_sink = cap._out if "capsys" in kwargs else quiet_out
                        err_sink = cap._err if "capsys" in kwargs else quiet_err
                        with contextlib.redirect_stdout(out_sink), contextlib.redirect_stderr(err_sink):
                            fn(**kwargs)
                        passed += 1
                    except BaseException as exc:  # noqa: BLE001
                        failed += 1
                        failures.append(f"{nodeid}: {type(exc).__name__}: {exc}")
                    finally:
                        mp.restore()
                        accumulated_events.extend(gate_trace.events())
    finally:
        sys.settrace(prev_trace)

    events = accumulated_events
    total_exec = 0
    total_hit = 0
    per_file_cov: dict[str, str] = {}
    for fpath, exec_set in sorted(tracked_files.items()):
        if Path(fpath).name == "selftest.py":
            continue
        hits = exec_set & covered_lines[fpath]
        total_exec += len(exec_set)
        total_hit += len(hits)
        pct = 100.0 if not exec_set else (100.0 * len(hits) / len(exec_set))
        per_file_cov[Path(fpath).name] = f"{len(hits)}/{len(exec_set)} ({pct:.1f}%)"

    overall_cov = 100.0 if total_exec == 0 else (100.0 * total_hit / total_exec)
    return passed, failed, failures, overall_cov, per_file_cov, events


def run_full_verification(repo_root: Path | None = None) -> tuple[bool, dict]:
    """Run all 8 verification gates across the repository and return (ok, report_dict)."""
    t0 = time.perf_counter()
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()

    style_issues = audit_python_style_stdlib(root)
    ruff_bin = shutil.which("ruff")
    ruff_mode = "stdlib-ast"
    if ruff_bin is not None:
        proc = subprocess.run([ruff_bin, "check", "."], cwd=root, capture_output=True, text=True, check=False)
        ruff_mode = "ruff-cli"
        if proc.returncode != 0:
            style_issues.append(proc.stdout.strip() or proc.stderr.strip())

    cg_problems = check_control_graph_problems(root)

    docs_proc = subprocess.run(
        [sys.executable, str(root / ".github" / "scripts" / "docs_budget.py")],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    docs_ok = docs_proc.returncode == 0

    store = Store(root / "fixtures" / "passing")
    seal_key = load_key(root / "fixtures" / "test_seal.key")
    vrep = verify_store(store, key=seal_key)
    anchor_head = vrep.head
    fixture_ok = (
        vrep.ok
        and anchor_head == CANONICAL_ANCHOR
        and all(g["status"] == "pass" for g in vrep.gate_rows)
        and all(c["answered"] for c in vrep.checklist)
    )

    passed_tests, failed_tests, test_failures, cov_pct, per_file_cov, trace_events = _run_pytest_suite_stdlib(root)
    gc_rep = assess_gate_coverage(trace_events)

    exec_iter_dir = root / "iterations" / "exec-hypergraph-v01"
    ut_proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=exec_iter_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    m_ran = re.search(r"Ran (\d+) tests?", ut_proc.stderr + ut_proc.stdout)
    ut_count = int(m_ran.group(1)) if m_ran else 0
    exec_ut_ok = ut_proc.returncode == 0 and ut_count >= 70

    with tempfile.TemporaryDirectory(prefix="sheaf_mut_self_") as mut_tmp:
        mut_rep = run_mutation_suite(Path(mut_tmp))
    mut_ok = mut_rep.survived == 0 and mut_rep.mutation_score == 1.0

    ag_files = [
        root / "AGENTS.md",
        root / "ANTIGRAVITY_HANDOFF.md",
        root / ".agents" / "rules" / "00-repository-invariants.md",
        root / ".agents" / "skills" / "verify-all-gates" / "SKILL.md",
        root / ".agents" / "skills" / "run-sheaf-bridge" / "SKILL.md",
    ]
    ag_ok = all(p.is_file() for p in ag_files)

    all_ok = (
        len(style_issues) == 0
        and len(cg_problems) == 0
        and docs_ok
        and fixture_ok
        and failed_tests == 0
        and cov_pct >= 85.0
        and gc_rep.ok
        and exec_ut_ok
        and mut_ok
        and ag_ok
    )

    report = {
        "schema": "sheaf-verify.antigravity-selftest.v1",
        "ok": all_ok,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "lint": {"mode": ruff_mode, "ok": len(style_issues) == 0, "issues": style_issues},
        "control_graph": {"ok": len(cg_problems) == 0, "problems": cg_problems},
        "docs_budget": {"ok": docs_ok, "detail": docs_proc.stdout.strip() or docs_proc.stderr.strip()},
        "canonical_fixture": {
            "ok": fixture_ok,
            "anchor_sha256": anchor_head,
            "gates_passed": sum(1 for g in vrep.gate_rows if g["status"] == "pass"),
            "gates_total": len(vrep.gate_rows),
        },
        "pytest_suite": {
            "passed": passed_tests,
            "failed": failed_tests,
            "failures": test_failures,
            "sheaf_verify_line_coverage_pct": round(cov_pct, 2),
            "coverage_floor_pct": 85.0,
            "per_module_coverage": per_file_cov,
        },
        "gate_trace_coverage": {
            "ok": gc_rep.ok,
            "events_recorded": len(trace_events),
            "problems": gc_rep.problems,
        },
        "exec_hypergraph_unittest": {
            "ok": exec_ut_ok,
            "tests_run": ut_count,
            "failures": 0 if ut_proc.returncode == 0 else 1,
        },
        "mutation_adequacy": {
            "ok": mut_ok,
            "generated": mut_rep.generated,
            "killed": mut_rep.killed,
            "survived": mut_rep.survived,
            "mutation_score": mut_rep.mutation_score,
        },
        "antigravity_handoff_files": {
            "ok": ag_ok,
            "checked": [p.relative_to(root).as_posix() for p in ag_files],
        },
    }
    return all_ok, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sheaf_verify.selftest")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--report-out", type=Path, default=None)
    args = parser.parse_args(argv)

    ok_flag, report = run_full_verification(args.repo_root)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0 if ok_flag else 1


if __name__ == "__main__":
    raise SystemExit(main())
