"""Drive the command functions in this process so coverage sees them."""

import json
from pathlib import Path

import pytest

from sheaf_verify.cli import main
from sheaf_verify.hashing import sha256_hex
from sheaf_verify.synthetic import TEST_KEY, build


def _key(tmp_path: Path) -> Path:
    path = tmp_path / "key"
    path.write_bytes(TEST_KEY)
    return path


def test_verify_checklist_bundle_anchor_and_run(tmp_path, capsys):
    store = build(tmp_path / "store")
    key = _key(tmp_path)
    root = str(store.root)
    assert main(["verify", "--store", root, "--seal-key-file", str(key)]) == 0
    assert "Review report" in capsys.readouterr().out
    assert main(["verify", "--json", "--store", root, "--seal-key-file", str(key)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert main(["checklist", "--json", "--store", root, "--seal-key-file", str(key)]) == 0
    capsys.readouterr()
    assert main(["anchor", "--store", root, "--out", str(tmp_path / "anchor.txt")]) == 0
    assert capsys.readouterr().out.startswith("sheaf-verify.anchor.v1 ")
    assert main(["bundle", "--store", root, "--seal-key-file", str(key), "--out", str(tmp_path / "bundle")]) == 0
    assert (tmp_path / "bundle" / "REPORT.md").is_file()
    capsys.readouterr()
    assert main(["run", "--json", "--store", root, "--seal-key-file", str(key)]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["failed"] == []


def test_register_commands_and_refused_inputs(tmp_path, capsys):
    root = tmp_path / "empty"
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"unit-bytes")
    code = main([
        "register",
        "--store", str(root),
        "--kind", "raw_data",
        "--role", "corpus.train",
        "--path", str(blob),
        "--registered-at", "2026-01-01T00:00:00Z",
        "--metadata", '{"document_ids":["doc-a"],"order":0}',
        "--json",
    ])
    assert code == 0
    registered = json.loads(capsys.readouterr().out)
    assert registered["manifest"]["role"] == "corpus.train"

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "model.py").write_text("class Model:\n    pass\n", encoding="utf-8")
    assert main([
        "register-tree",
        "--store", str(root),
        "--path", str(tree),
        "--registered-at", "2026-01-01T00:00:01Z",
    ]) == 0
    assert "file_count" in capsys.readouterr().out

    assert main([
        "register-env",
        "--store", str(root),
        "--python", "3.12",
        "--cuda", "none",
        "--driver", "none",
        "--gpu", "none",
        "--platform", "linux",
        "--registered-at", "2026-01-01T00:00:02Z",
        "--json",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["manifest"]["kind"] == "environment"

    digest = sha256_hex(b"external-checkpoint-not-a-weight")
    assert main([
        "register-external",
        "--store", str(root),
        "--role", "checkpoint.hypergraph",
        "--payload-sha256", digest,
        "--size-bytes", "4096",
        "--locator", "external://trainer/checkpoints/hypergraph",
        "--registered-at", "2026-01-01T00:00:03Z",
    ]) == 0
    capsys.readouterr()

    assert main(["register", "--store", str(root), "--kind", "raw_data", "--role", "corpus.valid", "--path", str(blob), "--metadata", "[]"]) == 2
    assert "metadata must be a JSON object" in capsys.readouterr().err
    assert main(["verify", "--allow-unsigned", "--store", str(root)]) in (0, 1)
    capsys.readouterr()
    loose = tmp_path / "loose.bin"
    loose.write_bytes(b"clock")
    assert main(["register", "--store", str(root), "--kind", "raw_data", "--role", "corpus.clock", "--path", str(loose)]) == 0
    capsys.readouterr()
    assert main(["verify", "--store", str(root)]) == 2
    assert "seal key" in capsys.readouterr().err
    assert main(["anchor", "--store", str(tmp_path / "missing")]) == 2
    built = build(tmp_path / "sealed")
    assert main(["bundle", "--store", str(built.root), "--seal-key-file", str(_key(tmp_path)), "--out", str(built.root)]) == 2
    capsys.readouterr()
    assert main(["anchor", "--store", str(built.root), "--out", str(built.root / "anchor.txt")]) == 2
    head = built.root / "ledger" / "head.json"
    head.write_text('{"head_sha256":"nope"}\n', encoding="utf-8")
    assert main(["anchor", "--store", str(built.root)]) == 2


def test_module_entry_point_exits(monkeypatch):
    import runpy

    monkeypatch.setattr("sheaf_verify.cli.main", lambda argv=None: 0)
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("sheaf_verify.__main__", run_name="__main__")
    assert caught.value.code == 0
