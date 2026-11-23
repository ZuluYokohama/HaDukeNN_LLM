"""Canonical JSON and SHA-256.

A stored record is the exact UTF-8 from ``canonical_bytes``. Verification
parses that JSON and rejects the file when the bytes are not that canonical
form, so a whitespace-only edit still fails.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


class SheafError(ValueError):
    """A caller, store, or record violated a harness invariant."""


def canonical_bytes(value) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise SheafError("value is not canonical JSON") from error
    return text.encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def parse_json_bytes(data: bytes, label: str):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SheafError(f"{label} is not UTF-8") from error
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise SheafError(f"{label} is not JSON") from error


def fmt_float(value: float) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SheafError("expected a real number")
    number = float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return format(number, ".12g")


def as_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SheafError(f"{label} must be an integer")
    return value


def as_real(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SheafError(f"{label} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise SheafError(f"{label} must be finite")
    return number


def reject_symlink(path: Path, label: str) -> Path:
    path = Path(path)
    if path.is_symlink():
        raise SheafError(f"{label} must not be a symlink")
    return path
