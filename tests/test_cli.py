import json
import os
import subprocess
import sys
from pathlib import Path

from sheaf_verify.synthetic import TEST_KEY, build


def _run(args, cwd):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").resolve())
    return subprocess.run(
        [sys.executable, "-m", "sheaf_verify", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_verify_and_bundle(tmp_path: Path):
    store = build(tmp_path / "store")
    key = tmp_path / "key"
    key.write_bytes(TEST_KEY)
    verified = _run(["verify", "--json", "--store", str(store.root), "--seal-key-file", str(key)], tmp_path)
    assert verified.returncode == 0, verified.stderr
    payload = json.loads(verified.stdout)
    assert payload["ok"] is True
    bundled = _run(
        ["bundle", "--store", str(store.root), "--seal-key-file", str(key), "--out", str(tmp_path / "bundle")],
        tmp_path,
    )
    assert bundled.returncode == 0, bundled.stderr
    assert (tmp_path / "bundle" / "REPORT.md").is_file()
    assert (tmp_path / "bundle" / "ledger" / "ledger.jsonl").is_file()
    assert not (tmp_path / "bundle" / "ledger" / ".lock").exists()


def test_anchor_prints_the_head_outside_the_store(tmp_path: Path):
    store = build(tmp_path / "store")
    head = json.loads((store.root / "ledger" / "head.json").read_text(encoding="utf-8"))
    printed = _run(
        ["anchor", "--store", str(store.root), "--out", str(tmp_path / "anchor.txt")],
        tmp_path,
    )
    assert printed.returncode == 0, printed.stderr
    assert printed.stdout.strip() == f"sheaf-verify.anchor.v1 {head['head_sha256']}"
    assert (tmp_path / "anchor.txt").read_text(encoding="utf-8").strip() == printed.stdout.strip()
    refused = _run(
        ["anchor", "--store", str(store.root), "--out", str(store.root / "ledger" / "anchor.txt")],
        tmp_path,
    )
    assert refused.returncode == 2
