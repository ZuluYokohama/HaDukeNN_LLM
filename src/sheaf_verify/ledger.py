"""Append-only hash-chained gate ledger.

Each line is one canonical record. ``prev_sha256`` is the previous line's
``record_sha256``, and that link sits inside the hashed body. Editing,
deleting, or reordering a line breaks the chain. An HMAC, when a seal key is
configured, covers the same body.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path

from sheaf_verify.hashing import SheafError, canonical_bytes, is_sha256, sha256_hex
from sheaf_verify.seal import ALGORITHM, mac_matches, sign

if sys.platform == "win32":
    import msvcrt

    def _lock(handle) -> None:
        # Lock one byte at offset 0. Windows allows a lock past end of file, so
        # the lock file stays empty, as verify requires. LK_LOCK gives up after
        # about ten seconds, so keep retrying until the holder releases it.
        handle.seek(0)
        while True:
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                return
            except OSError:
                continue

    def _unlock(handle) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

    def _unlock(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

LEDGER_FORMAT = "sheaf-verify.ledger.v1"
HEAD_FORMAT = "sheaf-verify.head.v1"
_TIME_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")


class Ledger:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.directory = self.root / "ledger"
        self.path = self.directory / "ledger.jsonl"
        self.head_path = self.directory / "head.json"
        self.lock_path = self.directory / ".lock"

    @contextlib.contextmanager
    def exclusive(self) -> Iterator[None]:
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as handle:
            _lock(handle)
            try:
                yield
            finally:
                _unlock(handle)

    def records(self) -> list[dict]:
        if not self.path.is_file():
            return []
        if self.path.is_symlink():
            raise SheafError("ledger path must not be a symlink")
        found = []
        for number, line in enumerate(self.path.read_bytes().splitlines(), 1):
            found.append(_parse_line(line, number))
        return found

    def append(self, body: dict, *, key: bytes | None) -> dict:
        reserved = {"format", "seq", "prev_sha256", "signature", "record_sha256"}
        overlap = reserved & set(body)
        if overlap:
            raise SheafError("ledger body must not set " + ", ".join(sorted(overlap)))
        if "created_at" not in body or "gate_id" not in body:
            raise SheafError("ledger body is missing created_at or gate_id")
        if not isinstance(body["created_at"], str) or _TIME_RE.fullmatch(body["created_at"]) is None:
            raise SheafError("created_at must be YYYY-MM-DDTHH:MM:SSZ")
        with self.exclusive():
            existing = self.records()
            seq = len(existing)
            previous = existing[-1]["record_sha256"] if existing else None
            record_body = {
                "format": LEDGER_FORMAT,
                "seq": seq,
                "prev_sha256": previous,
                **body,
                "created_at": body["created_at"],
            }
            # ``created_at`` is supplied by the caller so a fixture can be replayed.
            if key is None:
                record_body["signature"] = None
                record_body["record_sha256"] = sha256_hex(canonical_bytes(
                    {field: value for field, value in record_body.items() if field != "record_sha256"}
                ))
                record = record_body
            else:
                record = sign(record_body, key)
            line = canonical_bytes(record) + b"\n"
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            with os.fdopen(descriptor, "ab") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            self._write_head(record, seq + 1)
            return record

    def _write_head(self, record: dict, count: int) -> None:
        payload = {
            "format": HEAD_FORMAT,
            "count": count,
            "head_sha256": record["record_sha256"],
        }
        raw = canonical_bytes(payload)
        temporary = self.head_path.with_name("head.json.partial")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.head_path)

    def verify(self, *, key: bytes | None, allow_unsigned: bool = False) -> tuple[list[str], list[str]]:
        problems: list[str] = []
        notes: list[str] = []
        if self.path.is_symlink() or self.head_path.is_symlink():
            return ["ledger path must not be a symlink"], notes
        raw = self.path.read_bytes() if self.path.is_file() else b""
        if raw and not raw.endswith(b"\n"):
            problems.append("ledger does not end with a newline")
        raw_lines = raw.splitlines()
        records = []
        for number, line in enumerate(raw_lines, 1):
            try:
                record = _parse_line(line, number)
            except SheafError as error:
                problems.append(str(error))
                continue
            records.append(record)
            expect = sha256_hex(canonical_bytes(
                {field: value for field, value in record.items() if field != "record_sha256"}
            ))
            if record.get("record_sha256") != expect:
                problems.append(f"ledger line {number}: record hash does not match its body")
            if record.get("format") != LEDGER_FORMAT:
                problems.append(f"ledger line {number}: unknown format")
            if record.get("seq") != number - 1:
                problems.append(f"ledger line {number}: sequence is {record.get('seq')}")
        previous = None
        signed = 0
        unsigned = 0
        for record in records:
            label = f"ledger seq {record.get('seq')}"
            if record.get("prev_sha256") != previous:
                problems.append(f"{label}: prev_sha256 does not match the preceding record")
            signature = record.get("signature")
            if signature is None:
                unsigned += 1
                if not allow_unsigned:
                    problems.append(f"{label}: record is unsigned")
            else:
                signed += 1
                if not isinstance(signature, dict) or signature.get("algorithm") != ALGORITHM:
                    problems.append(f"{label}: unsupported signature")
                elif key is None:
                    problems.append(f"{label}: seal key is not available")
                elif not mac_matches(record, key):
                    problems.append(f"{label}: signature does not match this seal key")
            previous = record.get("record_sha256")
        if signed and unsigned:
            problems.append("ledger mixes signed and unsigned records")
        if not self.head_path.is_file():
            problems.append("ledger head is missing")
        else:
            try:
                head = json.loads(self.head_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                problems.append("ledger head is not JSON")
            else:
                if self.head_path.read_bytes() != canonical_bytes(head):
                    problems.append("ledger head is not canonical")
                expected_count = len(records)
                expected_head = records[-1]["record_sha256"] if records else None
                if head.get("format") != HEAD_FORMAT or head.get("count") != expected_count:
                    problems.append("ledger head count does not match the chain")
                if head.get("head_sha256") != expected_head:
                    problems.append("ledger head hash does not match the chain")
        return problems, notes


def _parse_line(line: bytes, number: int) -> dict:
    if not line:
        raise SheafError(f"ledger line {number} is empty")
    try:
        text = line.decode("utf-8")
        record = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SheafError(f"ledger line {number} is not JSON") from error
    if not isinstance(record, dict):
        raise SheafError(f"ledger line {number} is not an object")
    if canonical_bytes(record) != line:
        raise SheafError(f"ledger line {number} is not canonical JSON")
    if not is_sha256(record.get("record_sha256")):
        raise SheafError(f"ledger line {number}: missing record hash")
    return record
