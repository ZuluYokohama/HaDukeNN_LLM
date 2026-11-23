"""Content-addressed artifact store.

Each registration is a canonical manifest. Its file name is the SHA-256 of
that manifest. Inline bytes live at ``payloads/<payload sha256>``. Checkpoint
bytes are refused: a checkpoint manifest records the hash, the size, and a
locator on the trainer machine.

Manifest inputs are other manifest hashes. Registration only accepts inputs
that already exist, which keeps the provenance graph a DAG.
"""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path

from sheaf_verify.git_tree import git_tree_sha1
from sheaf_verify.hashing import (
    SheafError,
    canonical_bytes,
    is_sha256,
    parse_json_bytes,
    reject_symlink,
    sha256_hex,
)
from sheaf_verify.policy import MAX_INLINE_BYTES, MAX_TREE_BYTES

MANIFEST_FORMAT = "sheaf-verify.manifest.v1"
KINDS = frozenset({
    "raw_data",
    "hypergraph",
    "vocabulary",
    "code_tree",
    "environment",
    "config",
    "seeds",
    "checkpoint",
    "log",
    "metrics",
    "figure",
    "kernel",
    "result",
    "claim",
    "trace",
})
SKIP_DIR_NAMES = frozenset({".git", "__pycache__", ".sheaf", ".venv", ".pytest_cache"})
ROLE_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")
TIME_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)

    def manifests_dir(self) -> Path:
        return self.root / "manifests"

    def payloads_dir(self) -> Path:
        return self.root / "payloads"

    def init(self) -> None:
        self.manifests_dir().mkdir(parents=True, exist_ok=True)
        self.payloads_dir().mkdir(parents=True, exist_ok=True)
        (self.root / "ledger").mkdir(parents=True, exist_ok=True)
        (self.root / "snapshots").mkdir(parents=True, exist_ok=True)

    def register(
        self,
        *,
        kind: str,
        role: str,
        registered_at: str,
        payload: bytes | None = None,
        payload_sha256: str | None = None,
        size_bytes: int | None = None,
        storage: str = "inline",
        locator: str | None = None,
        inputs: list[str] | None = None,
        metadata: dict | None = None,
    ) -> tuple[str, dict]:
        if kind not in KINDS:
            raise SheafError(f"unknown artifact kind {kind}")
        if not isinstance(role, str) or ROLE_RE.fullmatch(role) is None:
            raise SheafError(f"invalid role {role!r}")
        if not isinstance(registered_at, str) or TIME_RE.fullmatch(registered_at) is None:
            raise SheafError("registered_at must be YYYY-MM-DDTHH:MM:SSZ")
        metadata = {} if metadata is None else metadata
        if not isinstance(metadata, dict):
            raise SheafError("metadata must be an object")
        inputs = [] if inputs is None else list(inputs)
        self._check_inputs(inputs)
        if storage == "inline":
            if payload is None:
                raise SheafError("inline artifact is missing its bytes")
            if kind == "checkpoint":
                raise SheafError("checkpoint bytes must stay outside the store")
            if len(payload) > MAX_INLINE_BYTES:
                raise SheafError("inline artifact exceeds the store byte cap")
            digest = sha256_hex(payload)
            size = len(payload)
            locator = f"payloads/{digest}"
        elif storage == "external":
            if kind != "checkpoint":
                raise SheafError("only a checkpoint may be stored by hash alone")
            if payload is not None:
                raise SheafError("refusing to store checkpoint bytes")
            if not is_sha256(payload_sha256):
                raise SheafError("checkpoint payload hash must be SHA-256 hex")
            if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
                raise SheafError("checkpoint size_bytes must be a non-negative integer")
            if not isinstance(locator, str) or not locator:
                raise SheafError("checkpoint locator is required")
            digest = payload_sha256
            size = size_bytes
            if self.payload_path(digest).exists():
                raise SheafError("checkpoint hash collides with a stored payload")
        else:
            raise SheafError(f"unknown storage {storage}")
        manifest = {
            "format": MANIFEST_FORMAT,
            "kind": kind,
            "role": role,
            "payload_sha256": digest,
            "size_bytes": size,
            "storage": storage,
            "locator": locator,
            "inputs": sorted(inputs),
            "metadata": metadata,
            "registered_at": registered_at,
        }
        raw = canonical_bytes(manifest)
        manifest_sha = sha256_hex(raw)
        self.init()
        path = self.manifest_path(manifest_sha)
        if path.exists():
            return manifest_sha, manifest
        if storage == "inline":
            self._write_new(self.payload_path(digest), payload)
        self._write_new(path, raw)
        return manifest_sha, manifest

    def register_file(
        self,
        *,
        kind: str,
        role: str,
        path: Path,
        registered_at: str,
        inputs: list[str] | None = None,
        metadata: dict | None = None,
    ) -> tuple[str, dict]:
        path = reject_symlink(path, "artifact file")
        if not path.is_file():
            raise SheafError("artifact path is not a file")
        if self._inside_store(path):
            raise SheafError("refusing to register a file from inside the store")
        return self.register(
            kind=kind,
            role=role,
            registered_at=registered_at,
            payload=path.read_bytes(),
            inputs=inputs,
            metadata=metadata,
        )

    def register_tree(self, *, role: str, path: Path, registered_at: str, inputs: list[str] | None = None) -> tuple[str, dict]:
        path = reject_symlink(path, "code tree")
        if not path.is_dir():
            raise SheafError("code tree path is not a directory")
        if self._inside_store(path):
            raise SheafError("refusing to register a tree from inside the store")
        files = _collect_tree(path)
        total = sum(len(data) for _, _, data in files)
        if total > MAX_TREE_BYTES:
            raise SheafError("code tree exceeds the snapshot cap")
        listing = [
            {"mode": mode, "path": rel, "sha256": sha256_hex(data), "size": len(data)}
            for rel, mode, data in files
        ]
        payload = canonical_bytes(listing)
        digest = sha256_hex(payload)
        metadata = {
            "content_sha256": digest,
            "file_count": len(files),
            "git_tree_sha1": git_tree_sha1(files),
        }
        commit = _git_head(path)
        if commit is not None:
            metadata["git_commit"] = commit
        manifest_sha, manifest = self.register(
            kind="code_tree",
            role=role,
            registered_at=registered_at,
            payload=payload,
            inputs=inputs,
            metadata=metadata,
        )
        self._write_snapshot(digest, files)
        return manifest_sha, manifest

    def register_environment(
        self,
        *,
        role: str,
        registered_at: str,
        python: str,
        cuda: str,
        driver: str,
        gpu: str,
        platform: str,
    ) -> tuple[str, dict]:
        lock = {
            "format": "sheaf-verify.environment.v1",
            "python": python,
            "cuda": cuda,
            "driver": driver,
            "gpu": gpu,
            "platform": platform,
        }
        return self.register(
            kind="environment",
            role=role,
            registered_at=registered_at,
            payload=canonical_bytes(lock),
            metadata={},
        )

    def manifest_path(self, manifest_sha: str) -> Path:
        if not is_sha256(manifest_sha):
            raise SheafError("manifest hash must be SHA-256 hex")
        return self.manifests_dir() / f"{manifest_sha}.json"

    def payload_path(self, payload_sha: str) -> Path:
        if not is_sha256(payload_sha):
            raise SheafError("payload hash must be SHA-256 hex")
        return self.payloads_dir() / payload_sha

    def read_manifest(self, manifest_sha: str) -> dict:
        path = self.manifest_path(manifest_sha)
        if not path.is_file():
            raise SheafError(f"missing manifest {manifest_sha}")
        return parse_json_bytes(path.read_bytes(), f"manifest {manifest_sha}")

    def read_payload(self, payload_sha: str) -> bytes:
        path = self.payload_path(payload_sha)
        if not path.is_file():
            raise SheafError(f"missing payload {payload_sha}")
        return path.read_bytes()

    def iter_manifests(self) -> list[tuple[str, dict]]:
        directory = self.manifests_dir()
        if not directory.is_dir():
            return []
        found = []
        for path in sorted(directory.glob("*.json")):
            if path.is_symlink():
                raise SheafError("manifest path must not be a symlink")
            found.append((path.stem, parse_json_bytes(path.read_bytes(), path.name)))
        return found

    def binding(self) -> dict[str, tuple[str, dict]]:
        """Current role map. The later ``registered_at``, then the hash, wins."""
        chosen: dict[str, tuple[str, dict]] = {}
        for manifest_sha, manifest in self.iter_manifests():
            role = manifest.get("role")
            if not isinstance(role, str):
                continue
            current = chosen.get(role)
            key = (str(manifest.get("registered_at")), manifest_sha)
            if current is None or key > (str(current[1].get("registered_at")), current[0]):
                chosen[role] = (manifest_sha, manifest)
        return chosen

    def _check_inputs(self, inputs: list[str]) -> None:
        seen = set()
        for item in inputs:
            if not is_sha256(item):
                raise SheafError("manifest input is not a SHA-256 hash")
            if item in seen:
                raise SheafError("manifest input is repeated")
            seen.add(item)
            if not self.manifest_path(item).is_file():
                raise SheafError(f"manifest input is not registered: {item}")

    def _inside_store(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.root.resolve())
        except ValueError:
            return False
        return True

    def _write_new(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = path.read_bytes()
            if existing != data:
                raise SheafError(f"hash collision or divergent bytes at {path.name}")
            return
        temporary = path.with_name(path.name + ".partial")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(temporary, flags, 0o644)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        os.replace(temporary, path)

    def _write_snapshot(self, content_sha: str, files: list[tuple[str, str, bytes]]) -> None:
        root = self.root / "snapshots" / content_sha
        root.mkdir(parents=True, exist_ok=True)
        for rel, _mode, data in files:
            target = root.joinpath(*rel.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != data:
                raise SheafError("code snapshot diverges from the listing")
            if not target.exists():
                self._write_new(target, data)


def _collect_tree(root: Path) -> list[tuple[str, str, bytes]]:
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        if current.is_symlink():
            raise SheafError("code tree contains a symlinked directory")
        kept = []
        for name in sorted(dirnames):
            if name in SKIP_DIR_NAMES:
                continue
            if (current / name).is_symlink():
                raise SheafError(f"code tree contains a symlinked directory: {name}")
            kept.append(name)
        dirnames[:] = kept
        for name in sorted(filenames):
            path = current / name
            if path.is_symlink():
                raise SheafError(f"code tree contains a symlink: {name}")
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            mode = "100755" if path.stat().st_mode & stat.S_IXUSR else "100644"
            files.append((relative, mode, path.read_bytes()))
    return files


def _git_head(path: Path) -> str | None:
    git_dir = path / ".git"
    if not git_dir.exists():
        return None
    import subprocess

    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        return None
    return commit


def load_json_object(data: bytes, label: str) -> dict:
    value = parse_json_bytes(data, label)
    if not isinstance(value, dict):
        raise SheafError(f"{label} must be a JSON object")
    return value


def dump_pretty(value) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"
