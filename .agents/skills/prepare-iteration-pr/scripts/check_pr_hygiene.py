#!/usr/bin/env python3
"""Read-only export-hygiene, binary-weight, and docs-budget preflight for iteration PRs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin"}
SKIP_DIRS = {".git", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__"}


def main() -> int:
    binaries: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or any(p in SKIP_DIRS or p.endswith(".egg-info") for p in path.parts):
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            binaries.append(path.relative_to(ROOT).as_posix())

    docs_proc = subprocess.run(
        [sys.executable, str(ROOT / ".github" / "scripts" / "docs_budget.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    anchor_proc = subprocess.run(
        [sys.executable, "-m", "sheaf_verify", "anchor", "--store", "fixtures/passing"],
        cwd=ROOT,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    anchor_line = anchor_proc.stdout.strip()
    expected = "sheaf-verify.anchor.v1 e247484042475d381eb454df95b718f55739a6cd0385006f142be257b8b86567"
    ok = (not binaries) and (docs_proc.returncode == 0) and (anchor_line == expected)
    summary = {
        "ok": ok,
        "forbidden_binaries": binaries,
        "docs_budget": docs_proc.stdout.strip() or docs_proc.stderr.strip(),
        "canonical_anchor": anchor_line,
        "anchor_unchanged": anchor_line == expected,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
