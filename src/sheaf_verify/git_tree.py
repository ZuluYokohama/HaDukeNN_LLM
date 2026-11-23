"""Git tree identifiers for a file listing.

Git addresses a tree by SHA-1 of the ``tree`` object. The harness also stores
a SHA-256 of the canonical file listing, and verification recomputes both
from the snapshotted bytes. SHA-1 here is Git's identifier, not the harness
digest.
"""

from __future__ import annotations

import hashlib

from sheaf_verify.hashing import SheafError


def git_tree_sha1(files: list[tuple[str, str, bytes]]) -> str:
    """Return the hex SHA-1 Git would store for this listing.

    ``files`` is ``(posix relative path, mode, bytes)``. Modes are ``100644``
    or ``100755``. Empty directories are not representable in a Git tree.
    """
    root: dict = {}
    for path, mode, data in files:
        if mode not in ("100644", "100755"):
            raise SheafError(f"unsupported file mode {mode}")
        parts = _parts(path)
        node = root
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise SheafError(f"path prefix collides with a file: {path}")
            node = child
        leaf = parts[-1]
        if leaf in node:
            raise SheafError(f"duplicate or prefix-colliding path: {path}")
        node[leaf] = (mode, data)
    return hashlib.sha1(_tree_object(root)).hexdigest()


def _parts(path: str) -> list[str]:
    if not path or path.startswith("/") or "\\" in path:
        raise SheafError(f"unsafe relative path: {path}")
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise SheafError(f"unsafe relative path: {path}")
    return parts


def _tree_object(node: dict) -> bytes:
    entries = []
    for name, value in node.items():
        if isinstance(value, dict):
            digest = hashlib.sha1(_tree_object(value)).digest()
            entries.append((name + "/", f"40000 {name}".encode("utf-8"), digest))
        else:
            mode, data = value
            blob = f"blob {len(data)}\0".encode("ascii") + data
            entries.append((name, f"{mode} {name}".encode("utf-8"), hashlib.sha1(blob).digest()))
    entries.sort(key=lambda item: item[0].encode("utf-8"))
    body = b"".join(header + b"\0" + digest for _, header, digest in entries)
    return f"tree {len(body)}\0".encode("ascii") + body
