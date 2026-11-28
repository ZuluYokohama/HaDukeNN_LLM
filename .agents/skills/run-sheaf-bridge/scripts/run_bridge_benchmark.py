#!/usr/bin/env python3
"""Run T-45 Oracle-to-Small-LLM Sheaf-Bridge suite and verify via sheaf-verify campaign."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
for entry in (str(ROOT), str(ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from hadukenn_core.sheaf_bridge import main as run_bridge_main  # noqa: E402


def main() -> int:
    rc = run_bridge_main()
    if rc != 0:
        return rc
    results_path = ROOT / "iterations" / "t45-oracle-sheaf-bridge" / "T45_SHEAF_BRIDGE_RESULTS.json"
    env = {"PYTHONPATH": str(ROOT / "src")}
    proc = subprocess.run(
        [sys.executable, "-m", "sheaf_verify", "campaign", "--mode", "t45-sheaf-bridge", "--path", str(results_path)],
        cwd=ROOT,
        env={**dict(__import__("os").environ), **env},
        check=False,
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
