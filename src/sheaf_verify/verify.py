"""Recompute a store.

Verification checks bytes, the ledger, gate replay, and the result record.
It does not turn a failed scientific gate into a hash failure: a study that
honestly failed H7 still verifies. The bundle command is what refuses an
open objection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from sheaf_verify.checklist import evaluate_checklist
from sheaf_verify.checks import code_tree_criteria
from sheaf_verify.gates import GATES
from sheaf_verify.hashing import SheafError, canonical_bytes, is_sha256, parse_json_bytes, sha256_file, sha256_hex
from sheaf_verify.ledger import Ledger
from sheaf_verify.runner import RESULT_FORMAT, replay
from sheaf_verify.store import KINDS, ROLE_RE, TIME_RE, Store

ALLOWED_TOP = frozenset({"manifests", "payloads", "snapshots", "ledger"})


@dataclass
class VerifyReport:
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    checklist: list[dict] = field(default_factory=list)
    gate_rows: list[dict] = field(default_factory=list)
    head: str | None = None

    @property
    def ok(self) -> bool:
        return not self.problems


def verify_store(store: Store, *, key: bytes | None, allow_unsigned: bool = False, strict: bool = False) -> VerifyReport:
    report = VerifyReport()
    report.problems.extend(_layout_problems(store.root))
    manifests = _load_manifests(store, report)
    _check_payloads(store, manifests, report, strict=strict)
    _check_code_trees(store, manifests, report)
    report.problems.extend(_provenance_problems(manifests))
    ledger = Ledger(store.root)
    ledger_problems, ledger_notes = ledger.verify(key=key, allow_unsigned=allow_unsigned)
    report.problems.extend(ledger_problems)
    report.notes.extend(ledger_notes)
    records = []
    if ledger.path.is_file() and not ledger.path.is_symlink():
        try:
            records = ledger.records()
        except SheafError as error:
            report.problems.append(f"ledger: {error}")
    try:
        report.problems.extend(replay(store, records))
    except SheafError as error:
        report.problems.append(f"replay: {error}")
    _check_snapshot_dirs(store, manifests, report)
    _check_results(store, manifests, records, report)
    report.head = records[-1]["record_sha256"] if records else None
    report.gate_rows = _gate_rows(records)
    status = {row["gate_id"]: row["status"] for row in report.gate_rows}
    report.checklist = evaluate_checklist(status, report.problems)
    return report


def _layout_problems(root: Path) -> list[str]:
    problems = []
    if not root.is_dir():
        return ["manifest: store directory is missing"]
    if root.is_symlink():
        return ["manifest: store path is a symlink"]
    for path in sorted(root.iterdir()):
        if path.name not in ALLOWED_TOP:
            problems.append(f"manifest: unexpected path {path.name}")
    lock = root / "ledger" / ".lock"
    if lock.exists():
        if lock.is_symlink():
            problems.append("ledger: lock file is a symlink")
        elif lock.stat().st_size != 0:
            problems.append("ledger: lock file is not empty")
    for directory, pattern in (
        (root / "manifests", "*.json"),
        (root / "payloads", "*"),
        (root / "snapshots", "*"),
    ):
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_symlink():
                problems.append(f"manifest: symlink in store: {path.relative_to(root).as_posix()}")
            elif path.is_file() and path.name.endswith(".partial"):
                problems.append(f"manifest: partial file left in store: {path.name}")
    return problems


def _load_manifests(store: Store, report: VerifyReport) -> dict[str, dict]:
    found = {}
    directory = store.manifests_dir()
    if not directory.is_dir():
        report.problems.append("manifest: manifests directory is missing")
        return found
    for path in sorted(directory.iterdir()):
        if path.name.endswith(".partial"):
            continue
        if not path.is_file():
            report.problems.append(f"manifest: unexpected entry {path.name}")
            continue
        stem = path.stem if path.suffix == ".json" else ""
        if path.suffix != ".json" or not is_sha256(stem):
            report.problems.append(f"manifest: unexpected file {path.name}")
            continue
        raw = path.read_bytes()
        try:
            manifest = parse_json_bytes(raw, path.name)
        except SheafError as error:
            report.problems.append(f"manifest: {error}")
            continue
        if not isinstance(manifest, dict) or canonical_bytes(manifest) != raw or sha256_hex(raw) != stem:
            report.problems.append(f"manifest: {stem} is not the canonical hash of its bytes")
            continue
        if manifest.get("format") != "sheaf-verify.manifest.v1":
            report.problems.append(f"manifest: {stem} has an unknown format")
        if manifest.get("kind") not in KINDS:
            report.problems.append(f"manifest: {stem} has an unknown kind")
        if not isinstance(manifest.get("role"), str) or ROLE_RE.fullmatch(manifest["role"]) is None:
            report.problems.append(f"manifest: {stem} has an invalid role")
        if not isinstance(manifest.get("registered_at"), str) or TIME_RE.fullmatch(manifest["registered_at"]) is None:
            report.problems.append(f"manifest: {stem} has an invalid timestamp")
        inputs = manifest.get("inputs")
        if not isinstance(inputs, list) or inputs != sorted(inputs) or any(not is_sha256(item) for item in inputs):
            report.problems.append(f"manifest: {stem} inputs are not a sorted list of hashes")
        found[stem] = manifest
    return found


def _check_payloads(store: Store, manifests: dict[str, dict], report: VerifyReport, *, strict: bool) -> None:
    referenced = set()
    for manifest in manifests.values():
        digest = manifest.get("payload_sha256")
        size = manifest.get("size_bytes")
        storage = manifest.get("storage")
        kind = manifest.get("kind")
        role = manifest.get("role")
        if not is_sha256(digest) or isinstance(size, bool) or not isinstance(size, int) or size < 0:
            report.problems.append(f"manifest: {role} payload hash or size is invalid")
            continue
        path = store.payload_path(digest)
        if storage == "inline":
            referenced.add(digest)
            if kind == "checkpoint":
                report.problems.append(f"manifest: {role} checkpoint is inline")
                continue
            if not path.is_file():
                report.problems.append(f"manifest: missing payload for {role}")
                continue
            data = path.read_bytes()
            if sha256_hex(data) != digest or len(data) != size:
                report.problems.append(f"manifest: payload bytes for {role} do not match the manifest")
        elif storage == "external":
            if kind != "checkpoint":
                report.problems.append(f"manifest: {role} is external but is not a checkpoint")
            if path.exists():
                report.problems.append(f"manifest: checkpoint payload for {role} is stored in the harness")
            _check_locator(manifest, report, strict=strict)
        else:
            report.problems.append(f"manifest: {role} has unknown storage")
    payloads = store.payloads_dir()
    if not payloads.is_dir():
        return
    for path in sorted(payloads.iterdir()):
        if path.name.endswith(".partial"):
            continue
        if not path.is_file() or not is_sha256(path.name):
            report.problems.append(f"manifest: unexpected payload file {path.name}")
            continue
        if path.name not in referenced:
            report.problems.append(f"manifest: payload {path.name} is not referenced by an inline manifest")


def _check_locator(manifest: dict, report: VerifyReport, *, strict: bool) -> None:
    locator = manifest.get("locator")
    role = manifest.get("role")
    if not isinstance(locator, str) or not locator:
        report.problems.append(f"manifest: {role} checkpoint locator is missing")
        return
    if locator.startswith("external://"):
        report.notes.append(f"checkpoint {role} bytes are outside the store at {locator}")
        return
    path = Path(locator)
    if path.is_symlink():
        report.problems.append(f"manifest: checkpoint locator for {role} is a symlink")
        return
    if not path.is_file():
        if strict:
            report.problems.append(f"manifest: checkpoint locator for {role} is not on this machine")
        else:
            report.notes.append(f"checkpoint {role} locator is not on this machine")
        return
    if sha256_file(path) != manifest["payload_sha256"] or path.stat().st_size != manifest["size_bytes"]:
        report.problems.append(f"manifest: checkpoint bytes for {role} do not match the manifest")


def _check_code_trees(store: Store, manifests: dict[str, dict], report: VerifyReport) -> None:
    for manifest in manifests.values():
        if manifest.get("kind") != "code_tree":
            continue
        role = manifest["role"]
        try:
            listing = parse_json_bytes(store.read_payload(manifest["payload_sha256"]), role)
        except SheafError as error:
            report.problems.append(f"manifest: {error}")
            continue
        digest = manifest["payload_sha256"]
        base = store.root / "snapshots" / digest
        if not base.is_dir():
            report.problems.append(f"manifest: code snapshot for {role} is missing")
            continue
        walked = {}
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames.sort()
            for name in sorted(filenames):
                path = Path(dirpath) / name
                if path.is_symlink() or not path.is_file():
                    report.problems.append(f"manifest: code snapshot for {role} contains a non-file")
                    continue
                walked[path.relative_to(base).as_posix()] = path.read_bytes()
        files = []
        seen = set()
        if isinstance(listing, list):
            for entry in listing:
                if isinstance(entry, dict) and entry.get("path") in walked:
                    files.append((entry["path"], entry.get("mode"), walked[entry["path"]]))
                    seen.add(entry["path"])
        for relative, data in sorted(walked.items()):
            if relative not in seen:
                files.append((relative, "100644", data))
        for criterion in code_tree_criteria(manifest, listing, files):
            if not criterion.passed:
                report.problems.append(f"manifest: {role} {criterion.id}: {criterion.detail}")
        metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
        if metadata.get("git_commit"):
            report.notes.append(f"code role {role} records git commit {metadata['git_commit']}; verify recomputes the tree id, not that commit")


def _check_snapshot_dirs(store: Store, manifests: dict[str, dict], report: VerifyReport) -> None:
    expected = {
        manifest["payload_sha256"]
        for manifest in manifests.values()
        if manifest.get("kind") == "code_tree" and is_sha256(manifest.get("payload_sha256"))
    }
    snap = store.root / "snapshots"
    if not snap.is_dir():
        return
    for path in sorted(snap.iterdir()):
        if not path.is_dir() or not is_sha256(path.name) or path.name not in expected:
            report.problems.append(f"manifest: unexpected snapshot {path.name}")


def _provenance_problems(manifests: dict[str, dict]) -> list[str]:
    problems = []
    visiting = set()
    visited = set()

    def walk(digest: str) -> None:
        if digest in visiting:
            problems.append(f"provenance: cycle through {digest}")
            return
        if digest in visited:
            return
        visiting.add(digest)
        manifest = manifests.get(digest)
        if manifest is None:
            problems.append(f"provenance: missing manifest {digest}")
        else:
            for item in manifest.get("inputs", []):
                if item not in manifests:
                    problems.append(f"provenance: {manifest.get('role')} cites missing manifest {item}")
                else:
                    walk(item)
        visiting.discard(digest)
        visited.add(digest)

    for digest in list(manifests):
        walk(digest)
    return problems


def _check_results(store: Store, manifests: dict[str, dict], records: list[dict], report: VerifyReport) -> None:
    chain = {record.get("record_sha256") for record in records}
    latest: dict[str, dict] = {}
    for record in records:
        gate_id = record.get("gate_id")
        if isinstance(gate_id, str):
            latest[gate_id] = record
    results = [(digest, manifest) for digest, manifest in manifests.items() if manifest.get("kind") == "result"]
    if not results:
        report.problems.append("result: no result manifest is registered")
        return
    for digest, manifest in results:
        try:
            document = parse_json_bytes(store.read_payload(manifest["payload_sha256"]), "result")
        except SheafError as error:
            report.problems.append(f"result: {error}")
            continue
        if not isinstance(document, dict) or document.get("format") != RESULT_FORMAT:
            report.problems.append(f"result: {digest} is not a result record")
            continue
        head = document.get("ledger_head_sha256")
        if head not in chain:
            report.problems.append(f"result: {manifest.get('role')} cites a ledger head that is not in the chain")
    binding = store.binding()
    if "result" not in binding:
        report.problems.append("result: result role is missing")
        return
    result_sha, result = binding["result"]
    try:
        document = parse_json_bytes(store.read_payload(result["payload_sha256"]), "result")
    except SheafError as error:
        report.problems.append(f"result: {error}")
        return
    expected_inputs = sorted(binding[role][0] for role in binding if role != "result")
    if result.get("inputs") != expected_inputs:
        report.problems.append("result: latest result inputs do not match the current artifact binding")
    if records and document.get("ledger_head_sha256") != records[-1]["record_sha256"]:
        report.problems.append("result: latest result does not cite the current ledger head")
    gates = document.get("gates")
    expected = [
        {"gate_id": gate.id, "record_sha256": latest[gate.id]["record_sha256"], "status": latest[gate.id]["status"]}
        for gate in GATES
        if gate.id in latest
    ]
    if gates != expected or len(expected) != len(GATES):
        report.problems.append("result: latest result gate list does not match the latest ledger records")
    if result_sha not in manifests:
        report.problems.append("result: latest result manifest is not in the store")


def _gate_rows(records: list[dict]) -> list[dict]:
    latest = {}
    for record in records:
        gate_id = record.get("gate_id")
        if isinstance(gate_id, str):
            latest[gate_id] = record
    rows = []
    for gate in GATES:
        record = latest.get(gate.id)
        rows.append({
            "gate_id": gate.id,
            "record_sha256": None if record is None else record.get("record_sha256"),
            "status": "missing" if record is None else record.get("status"),
            "title": gate.title,
        })
    return rows
