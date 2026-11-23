from pathlib import Path

import pytest

from sheaf_verify.hashing import SheafError, canonical_bytes, sha256_hex
from sheaf_verify.store import Store


def test_canonical_json_is_sorted_and_compact():
    assert canonical_bytes({"b": 1, "a": [2, 3]}) == b'{"a":[2,3],"b":1}'


def test_checkpoint_bytes_are_refused(tmp_path: Path):
    store = Store(tmp_path / "store")
    with pytest.raises(SheafError, match="outside the store"):
        store.register(
            kind="checkpoint",
            role="checkpoint.hypergraph",
            registered_at="2026-01-01T00:00:01Z",
            payload=b"weights",
        )


def test_symlink_file_is_refused(tmp_path: Path):
    target = tmp_path / "real.bin"
    target.write_bytes(b"abc")
    link = tmp_path / "link.bin"
    link.symlink_to(target)
    store = Store(tmp_path / "store")
    with pytest.raises(SheafError, match="symlink"):
        store.register_file(
            kind="raw_data",
            role="corpus.train",
            path=link,
            registered_at="2026-01-01T00:00:01Z",
        )


def test_manifest_name_is_its_hash(tmp_path: Path):
    store = Store(tmp_path / "store")
    digest, manifest = store.register(
        kind="raw_data",
        role="corpus.train",
        registered_at="2026-01-01T00:00:01Z",
        payload=b"train-bytes-0001",
        metadata={"document_ids": ["doc-train"], "order": 0},
    )
    raw = store.manifest_path(digest).read_bytes()
    assert sha256_hex(raw) == digest
    assert manifest["payload_sha256"] == sha256_hex(b"train-bytes-0001")
