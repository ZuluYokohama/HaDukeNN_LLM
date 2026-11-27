"""Include the isolated executable-state iteration in the existing CI gate."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ITERATION = Path(__file__).resolve().parents[1] / "iterations" / "exec-hypergraph-v01"


def test_executable_iteration_contracts():
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
        cwd=ITERATION,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_executable_iteration_pilot(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "hadukenn_exec", "evaluate", "--no-timing", "--out", str(tmp_path)],
        cwd=ITERATION,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "pass"
    assert summary["tenfold_capability_established"] is False


def test_recorded_manifest_matches_iteration_sources():
    manifest = json.loads((ITERATION / "results" / "confirmation" / "run_manifest.json").read_text())
    for relative, expected in {**manifest["code_sha256"], **manifest["spec_sha256"]}.items():
        assert hashlib.sha256((ITERATION / relative).read_bytes()).hexdigest() == expected, relative
