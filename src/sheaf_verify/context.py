"""Read-only view of the current role binding."""

from __future__ import annotations

from pathlib import Path

from sheaf_verify.hashing import SheafError, parse_json_bytes
from sheaf_verify.store import Store


class Context:
    def __init__(self, store: Store):
        self.store = store
        self.binding = store.binding()

    def has(self, role: str) -> bool:
        return role in self.binding

    def manifest_sha(self, role: str) -> str:
        return self.binding[role][0]

    def manifest(self, role: str) -> dict:
        return self.binding[role][1]

    def payload(self, role: str) -> bytes:
        manifest = self.manifest(role)
        if manifest.get("storage") != "inline":
            raise SheafError(f"{role} has no inline bytes")
        return self.store.read_payload(manifest["payload_sha256"])

    def parse(self, role: str):
        return parse_json_bytes(self.payload(role), role)

    def bound(self, roles: tuple[str, ...]) -> dict:
        found = {}
        for role in roles:
            if role not in self.binding:
                continue
            manifest_sha, manifest = self.binding[role]
            found[role] = {
                "manifest_sha256": manifest_sha,
                "payload_sha256": manifest["payload_sha256"],
            }
        return found

    def snapshot_bytes(self, content_sha: str, relative: str) -> bytes:
        if not relative or relative.startswith("/") or "\\" in relative:
            raise SheafError("snapshot path is unsafe")
        parts = relative.split("/")
        if any(part in ("", ".", "..") for part in parts):
            raise SheafError("snapshot path is unsafe")
        root = (self.store.root / "snapshots" / content_sha).resolve()
        path = root.joinpath(*parts)
        if path.is_symlink():
            raise SheafError("snapshot file must not be a symlink")
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise SheafError("snapshot path escapes the snapshot") from error
        if not path.is_file():
            raise SheafError(f"snapshot file is missing: {relative}")
        return path.read_bytes()


def safe_join(root: Path, relative: str) -> Path:
    path = root.joinpath(*relative.split("/"))
    if path.is_symlink():
        raise SheafError("refusing a symlink")
    return path
