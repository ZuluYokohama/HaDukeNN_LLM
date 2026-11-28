#!/usr/bin/env python3
"""Plugin skill script running sheaf_verify.selftest against the active workspace."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    cwd = Path.cwd()
    env = dict(os.environ)
    src_dir = cwd / "src"
    if src_dir.is_dir():
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{cwd}:{src_dir}" + (f":{existing}" if existing else "")
    proc = subprocess.run(
        [sys.executable, "-m", "sheaf_verify.selftest", "--repo-root", str(cwd)],
        cwd=cwd,
        env=env,
        check=False,
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
