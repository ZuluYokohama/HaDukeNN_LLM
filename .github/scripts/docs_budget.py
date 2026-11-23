#!/usr/bin/env python3
"""Fail when docs/ is over 1 MB or an HTML page loads an external script."""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "docs"
TOTAL_LIMIT = 1_000_000
PAGE_LIMIT = 300 * 1024


class _ScriptSrcParser(HTMLParser):
    """Collect real ``src`` attributes from script tags, including void tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "script":
            return
        for name, value in attrs:
            if name.lower() == "src" and value is not None:
                self.sources.append(value.strip())

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)


def script_sources(text: str) -> list[str]:
    parser = _ScriptSrcParser()
    parser.feed(text)
    parser.close()
    return parser.sources


def is_external_script(url: str) -> bool:
    normalized = url.strip().lower()
    return normalized.startswith(("http://", "https://", "//")) or "://" in normalized


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
        for url in script_sources(page.read_text(encoding="utf-8")):
            if is_external_script(url):
                print(f"{page.relative_to(ROOT.parent)} loads an external script: {url}", file=sys.stderr)
                return 1
    print(f"docs/ {total} bytes across {len(files)} files; no external scripts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
