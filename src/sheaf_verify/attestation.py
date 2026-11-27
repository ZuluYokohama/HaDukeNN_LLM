"""Standards-native provenance: strict RFC 7493/8785 JSON, in-toto v1 + DSSE v1.0, and RFC 9162 Merkle receipts.

Bridges ``sheaf-verify`` stores to interoperable supply-chain transparency
standards without adding runtime dependencies:
1. Strict JSON audit parser rejecting duplicate member names, nesting > 128,
   NaN/Infinity, and integers outside IEEE-754 / RFC 7493 safe bounds.
2. ``in-toto`` Statement v1 (``https://in-toto.io/Statement/v1``) wrapped inside
   Dead Simple Signing Envelope (DSSE v1.0) Pre-Authentication Encoding (PAE).
3. RFC 9162 / SCITT-compatible append-only binary Merkle Transparency Tree over
   ``ledger.jsonl``, producing compact ``O(log N)`` inclusion receipts and
   append-only consistency proofs so third-party auditors can verify a single
   gate verdict offline without reading confidential sibling artifacts.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import Any

from sheaf_verify.hashing import SheafError, canonical_bytes, is_sha256, sha256_hex
from sheaf_verify.seal import ALGORITHM, key_id

INTOTO_STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
INTOTO_PAYLOAD_TYPE = "application/vnd.in-toto+json"
SHEAF_PREDICATE_V2 = "https://hadukenn.org/sheaf-verify/gate-verdict/v2"
RECEIPT_FORMAT = "sheaf-verify.scitt-merkle-receipt.v1"
MAX_SAFE_INT = (1 << 53) - 1
MAX_JSON_DEPTH = 128


def validate_strict_json_bytes(raw: bytes, *, max_depth: int = MAX_JSON_DEPTH) -> Any:
    """Parse UTF-8 JSON enforcing RFC 7493 I-JSON safety, unique keys, and bounded nesting."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SheafError("strict JSON payload is not valid UTF-8") from error

    def _no_dup_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        obj: dict[str, Any] = {}
        for k, v in pairs:
            if k in obj:
                raise SheafError(f"duplicate JSON object member {k!r}")
            obj[k] = v
        return obj

    def _bad_const(token: str) -> Any:
        raise SheafError(f"non-finite JSON constant {token!r} is forbidden")

    try:
        parsed = json.loads(text, object_pairs_hook=_no_dup_pairs, parse_constant=_bad_const)
    except json.JSONDecodeError as error:
        raise SheafError(f"invalid JSON syntax: {error.msg}") from error

    def _walk(node: Any, depth: int) -> None:
        if depth > max_depth:
            raise SheafError(f"JSON nesting exceeds maximum depth {max_depth}")
        if isinstance(node, bool) or node is None or isinstance(node, str):
            return
        if isinstance(node, int):
            if abs(node) > MAX_SAFE_INT:
                raise SheafError(f"integer {node} exceeds RFC 7493 safe range +/-{MAX_SAFE_INT}")
            return
        if isinstance(node, float):
            if not (-1e308 <= node <= 1e308):
                raise SheafError("floating-point number out of range")
            return
        if isinstance(node, list):
            for child in node:
                _walk(child, depth + 1)
            return
        if isinstance(node, dict):
            for k, child in node.items():
                if not isinstance(k, str):
                    raise SheafError("JSON object key must be a string")
                _walk(child, depth + 1)

    _walk(parsed, 0)
    return parsed


def dsse_pae(payload_type: str, payload: bytes) -> bytes:
    """Compute DSSE v1.0 Pre-Authentication Encoding: ``DSSEv1 <len> <type> <len> <body>``."""
    if not payload_type or not isinstance(payload_type, str):
        raise SheafError("payload_type must be a non-empty string")
    pt_bytes = payload_type.encode("utf-8")
    header = f"DSSEv1 {len(pt_bytes)} ".encode("ascii") + pt_bytes + f" {len(payload)} ".encode("ascii")
    return header + payload


def intoto_statement_v1(
    *,
    subjects: list[dict[str, Any]],
    predicate_type: str = SHEAF_PREDICATE_V2,
    predicate: dict[str, Any],
) -> dict[str, Any]:
    """Construct a canonical in-toto v1 Statement binding SHA-256 subject digests."""
    if not subjects:
        raise SheafError("in-toto statement requires at least one subject")
    clean_subjects = []
    for sub in subjects:
        name = sub.get("name")
        digest_map = sub.get("digest")
        if not isinstance(name, str) or not name.strip():
            raise SheafError("subject name must be a non-empty string")
        if not isinstance(digest_map, dict) or not is_sha256(digest_map.get("sha256")):
            raise SheafError(f"subject {name!r} requires a valid sha256 digest")
        clean_subjects.append({"digest": {"sha256": digest_map["sha256"]}, "name": name.strip()})
    if not isinstance(predicate, dict):
        raise SheafError("in-toto predicate must be an object")
    statement = {
        "_type": INTOTO_STATEMENT_TYPE,
        "predicate": predicate,
        "predicateType": predicate_type,
        "subject": clean_subjects,
    }
    validate_strict_json_bytes(canonical_bytes(statement))
    return statement


def dsse_sign(statement: dict[str, Any], *, key: bytes, payload_type: str = INTOTO_PAYLOAD_TYPE) -> dict[str, Any]:
    """Sign a canonical JSON statement inside a DSSE v1.0 envelope."""
    raw_payload = canonical_bytes(statement)
    validate_strict_json_bytes(raw_payload)
    kid = key_id(key)
    pae = dsse_pae(payload_type, raw_payload)
    mac = hmac.new(key, pae, hashlib.sha256).digest()
    return {
        "payload": base64.b64encode(raw_payload).decode("ascii"),
        "payloadType": payload_type,
        "signatures": [
            {
                "alg": ALGORITHM,
                "keyid": kid,
                "sig": base64.b64encode(mac).decode("ascii"),
            }
        ],
    }


def dsse_verify(envelope: dict[str, Any], *, key: bytes) -> dict[str, Any]:
    """Verify DSSE v1.0 PAE signature and strict RFC 8785 / in-toto v1 payload invariants."""
    if not isinstance(envelope, dict):
        raise SheafError("DSSE envelope must be an object")
    payload_type = envelope.get("payloadType")
    b64_payload = envelope.get("payload")
    sigs = envelope.get("signatures")
    if payload_type != INTOTO_PAYLOAD_TYPE:
        raise SheafError("unexpected DSSE payloadType")
    if not isinstance(b64_payload, str) or not isinstance(sigs, list) or not sigs:
        raise SheafError("DSSE envelope is missing payload or signatures")
    try:
        raw_payload = base64.b64decode(b64_payload.encode("ascii"), validate=True)
    except Exception as error:
        raise SheafError("DSSE payload is not valid base64") from error
    statement = validate_strict_json_bytes(raw_payload)
    if canonical_bytes(statement) != raw_payload:
        raise SheafError("DSSE payload bytes are not RFC 8785 canonical JSON")
    if not isinstance(statement, dict) or statement.get("_type") != INTOTO_STATEMENT_TYPE:
        raise SheafError("decoded DSSE payload is not an in-toto v1 Statement")
    expected_kid = key_id(key)
    pae = dsse_pae(payload_type, raw_payload)
    expected_mac = hmac.new(key, pae, hashlib.sha256).digest()
    matched = False
    for entry in sigs:
        if not isinstance(entry, dict) or entry.get("keyid") != expected_kid:
            continue
        sig_b64 = entry.get("sig")
        if not isinstance(sig_b64, str):
            continue
        try:
            sig_bytes = base64.b64decode(sig_b64.encode("ascii"), validate=True)
        except Exception:
            continue
        if hmac.compare_digest(expected_mac, sig_bytes):
            matched = True
            break
    if not matched:
        raise SheafError("DSSE envelope has no valid signature for this key")
    return statement


def _leaf_hash(data: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + data).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _largest_power_of_two_less_than(n: int) -> int:
    k = 1
    while (k << 1) < n:
        k <<= 1
    return k


def merkle_tree_head(leaves: list[bytes]) -> str:
    """RFC 9162 / RFC 6962 Merkle Tree Hash (MTH) hex digest."""
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    return _mth_bytes(leaves).hex()


def _mth_bytes(leaves: list[bytes]) -> bytes:
    n = len(leaves)
    if n == 1:
        return _leaf_hash(leaves[0])
    k = _largest_power_of_two_less_than(n)
    return _node_hash(_mth_bytes(leaves[:k]), _mth_bytes(leaves[k:]))


def merkle_audit_path(index: int, leaves: list[bytes]) -> list[str]:
    """RFC 9162 O(log N) Merkle audit path (inclusion proof) for leaf ``index``."""
    n = len(leaves)
    if not 0 <= index < n:
        raise SheafError(f"leaf index {index} out of range 0..{n - 1}")
    return [digest.hex() for digest in _path_bytes(index, leaves)]


def _path_bytes(index: int, leaves: list[bytes]) -> list[bytes]:
    n = len(leaves)
    if n == 1:
        return []
    k = _largest_power_of_two_less_than(n)
    if index < k:
        return [*_path_bytes(index, leaves[:k]), _mth_bytes(leaves[k:])]
    return [*_path_bytes(index - k, leaves[k:]), _mth_bytes(leaves[:k])]


def verify_merkle_inclusion(
    *,
    leaf_bytes: bytes,
    index: int,
    tree_size: int,
    audit_path: list[str],
    expected_root_hex: str,
) -> bool:
    """Recompute RFC 9162 root from ``leaf_bytes``, ``index``, ``tree_size``, and ``audit_path``."""
    if tree_size < 1 or not 0 <= index < tree_size or not is_sha256(expected_root_hex):
        return False
    if any(not is_sha256(node) for node in audit_path):
        return False
    fn = index
    sn = tree_size - 1
    current = _leaf_hash(leaf_bytes)
    for sibling_hex in audit_path:
        if sn == 0:
            return False
        sibling = bytes.fromhex(sibling_hex)
        if (fn & 1) == 1 or fn == sn:
            current = _node_hash(sibling, current)
            while (fn & 1) == 0 and fn != 0:
                fn >>= 1
                sn >>= 1
        else:
            current = _node_hash(current, sibling)
        fn >>= 1
        sn >>= 1
    return sn == 0 and hmac.compare_digest(current.hex(), expected_root_hex)


def merkle_consistency_proof(first_size: int, leaves: list[bytes]) -> list[str]:
    """RFC 9162 SUBPROOF append-only consistency proof between ``first_size`` and ``len(leaves)``."""
    n = len(leaves)
    if not 1 <= first_size <= n:
        raise SheafError("first_size must satisfy 1 <= first_size <= tree_size")
    return [h.hex() for h in _subproof(first_size, leaves, True)]


def _subproof(m: int, leaves: list[bytes], complete: bool) -> list[bytes]:
    n = len(leaves)
    if m == n:
        return [] if complete else [_mth_bytes(leaves)]
    k = _largest_power_of_two_less_than(n)
    if m <= k:
        return [*_subproof(m, leaves[:k], complete), _mth_bytes(leaves[k:])]
    return [*_subproof(m - k, leaves[k:], False), _mth_bytes(leaves[:k])]


def build_ledger_receipt(records: list[dict[str, Any]], seq: int, *, key: bytes | None = None) -> dict[str, Any]:
    """Create a standalone SCITT-style Merkle inclusion receipt for ledger record ``seq``."""
    if not records or not 0 <= seq < len(records):
        raise SheafError(f"ledger sequence {seq} is out of range")
    leaves = [canonical_bytes(rec) for rec in records]
    root = merkle_tree_head(leaves)
    target = records[seq]
    path = merkle_audit_path(seq, leaves)
    receipt: dict[str, Any] = {
        "audit_path": path,
        "format": RECEIPT_FORMAT,
        "gate_id": target["gate_id"],
        "leaf_index": seq,
        "merkle_root_sha256": root,
        "record": target,
        "record_sha256": target["record_sha256"],
        "tree_size": len(records),
    }
    if key is not None:
        statement = intoto_statement_v1(
            subjects=[{"name": f"ledger/seq/{seq}", "digest": {"sha256": target["record_sha256"]}}],
            predicate={
                "gate_id": target["gate_id"],
                "merkle_root_sha256": root,
                "status": target["status"],
                "tree_size": len(records),
            },
        )
        receipt["dsse"] = dsse_sign(statement, key=key)
    return receipt


def verify_ledger_receipt(receipt: dict[str, Any], *, key: bytes | None = None) -> tuple[bool, list[str]]:
    """Offline verification of a single gate inclusion receipt."""
    problems: list[str] = []
    if not isinstance(receipt, dict) or receipt.get("format") != RECEIPT_FORMAT:
        return False, ["receipt format is not sheaf-verify.scitt-merkle-receipt.v1"]
    record = receipt.get("record")
    seq = receipt.get("leaf_index")
    size = receipt.get("tree_size")
    root = receipt.get("merkle_root_sha256")
    path = receipt.get("audit_path")
    if not isinstance(record, dict) or isinstance(seq, bool) or not isinstance(seq, int) or isinstance(size, bool) or not isinstance(size, int):
        return False, ["receipt fields are malformed"]
    if not isinstance(path, list) or not isinstance(root, str):
        return False, ["receipt audit_path or merkle_root_sha256 is missing"]
    unsigned_hash = sha256_hex(canonical_bytes({k: v for k, v in record.items() if k != "record_sha256"}))
    if record.get("record_sha256") != unsigned_hash or receipt.get("record_sha256") != unsigned_hash:
        problems.append("record_sha256 does not match canonical record body")
    leaf_bytes = canonical_bytes(record)
    if not verify_merkle_inclusion(
        leaf_bytes=leaf_bytes,
        index=seq,
        tree_size=size,
        audit_path=path,
        expected_root_hex=root,
    ):
        problems.append("Merkle audit path does not recompute merkle_root_sha256")
    dsse_env = receipt.get("dsse")
    if key is not None:
        if not isinstance(dsse_env, dict):
            problems.append("DSSE signature envelope is missing")
        else:
            try:
                stmt = dsse_verify(dsse_env, key=key)
                pred = stmt.get("predicate", {})
                if pred.get("merkle_root_sha256") != root or pred.get("tree_size") != size:
                    problems.append("DSSE statement predicate disagrees with receipt Merkle root")
            except SheafError as error:
                problems.append(f"DSSE verification failed: {error}")
    return not problems, problems
