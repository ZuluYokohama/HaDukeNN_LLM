"""HMAC-SHA256 seals.

The MAC shows that a holder of the key produced the record. It is not a
public-key signature: anyone who has the key can build a new chain that
verifies. Keep the production key off the trainer and off this repository,
and copy the ledger head somewhere that key cannot rewrite.
"""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path

from sheaf_verify.hashing import SheafError, canonical_bytes, reject_symlink

KEY_ID_LABEL = b"sheaf-verify.key-id.v1"
MIN_KEY_BYTES = 16
MAX_KEY_BYTES = 4096
ALGORITHM = "hmac-sha256"


def key_id(key: bytes) -> str:
    key = _check_key(key)
    return hmac.new(key, KEY_ID_LABEL, hashlib.sha256).hexdigest()


def load_key(path: Path) -> bytes:
    path = reject_symlink(path, "seal key file")
    if not path.is_file():
        raise SheafError("seal key file is missing")
    data = path.read_bytes()
    return _check_key(data)


def sign(body: dict, key: bytes) -> dict:
    """Return ``body`` plus ``signature`` and ``record_sha256``.

    The MAC covers the canonical body with the signature and the record hash
    removed. The record hash then covers the body including the signature, so
    stripping the MAC changes the hash.
    """
    key = _check_key(key)
    unsigned = {field: value for field, value in body.items() if field not in ("signature", "record_sha256")}
    signature = {
        "algorithm": ALGORITHM,
        "key_id": key_id(key),
        "value": hmac.new(key, canonical_bytes(unsigned), hashlib.sha256).hexdigest(),
    }
    signed = dict(unsigned)
    signed["signature"] = signature
    signed["record_sha256"] = hashlib.sha256(canonical_bytes(signed)).hexdigest()
    return signed


def mac_matches(body: dict, key: bytes) -> bool:
    signature = body.get("signature")
    if not isinstance(signature, dict) or signature.get("algorithm") != ALGORITHM:
        return False
    if signature.get("key_id") != key_id(key):
        return False
    value = signature.get("value")
    if not isinstance(value, str):
        return False
    unsigned = {field: item for field, item in body.items() if field not in ("signature", "record_sha256")}
    expected = hmac.new(_check_key(key), canonical_bytes(unsigned), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, value)


def _check_key(key: bytes) -> bytes:
    if not isinstance(key, bytes) or not MIN_KEY_BYTES <= len(key) <= MAX_KEY_BYTES:
        raise SheafError("seal key must be 16..4096 bytes")
    return key
