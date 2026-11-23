#!/usr/bin/env python3
"""Fail when docs/ is over 1 MB or an HTML page loads an external script."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "docs"
TOTAL_LIMIT = 1_000_000
PAGE_LIMIT = 300 * 1024
SCRIPT_TAG = re.compile(r"<script\b([^>]*)>", re.IGNORECASE)
SCRIPT_SRC = re.compile(r"""\bsrc\s*=\s*(['"]?)([^'"\s>]+)\1""", re.IGNORECASE)


def main() -> int:
    if not ROOT.is_dir():
        print("docs/ is missing", file=sys.stderr)
        return 1
    files = [path for path in ROOT.rglob("*") if path.is_file()]
    total = sum(path.stat().st_size for path in files)
    if total > TOTAL_LIMIT:
        print(f"docs/ is {total} bytes, over {TOTAL_LIMIT}", file=sys.stderr)
        return 1
    for page in sorted(ROOT.rglob("*.html")):
        size = page.stat().st_size
        if size > PAGE_LIMIT:
            print(f"{page.relative_to(ROOT.parent)} is {size} bytes, over {PAGE_LIMIT}", file=sys.stderr)
            return 1
        text = page.read_text(encoding="utf-8")
        for match in SCRIPT_TAG.finditer(text):
            source = SCRIPT_SRC.search(match.group(1))
            if source is None:
                continue
            url = source.group(2)
            external = url.startswith(("http://", "https://", "//")) or "://" in url
            if external:
                print(f"{page.relative_to(ROOT.parent)} loads an external script: {url}", file=sys.stderr)
                return 1
    print(f"docs/ {total} bytes across {len(files)} files; no external scripts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
