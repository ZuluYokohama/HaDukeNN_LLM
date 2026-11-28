#!/usr/bin/env python3
"""Antigravity skill entrypoint executing the full 8-gate repository verification."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
for entry in (str(ROOT), str(ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from sheaf_verify.selftest import main  # noqa: E402

if __name__ == "__main__":
    report_path = ROOT / "ANTIGRAVITY_VERIFICATION_REPORT.json"
    raise SystemExit(main(["--repo-root", str(ROOT), "--report-out", str(report_path)]))
