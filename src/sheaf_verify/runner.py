"""Run the gate graph and replay it against a ledger.

``run`` always walks the full graph. A failed gate is written and does not
stop the walk; downstream gates then fail their upstream check. ``replay``
recomputes the latest record of every gate from the artifacts and the whole
ledger. It does not compare ``created_at``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from sheaf_verify.context import Context
from sheaf_verify.gate_trace import emit, trace_outcome
from sheaf_verify.gates import Gate, gates_for_profile, normalize_profile, profile_from_records
from sheaf_verify.hashing import SheafError, canonical_bytes
from sheaf_verify.ledger import Ledger
from sheaf_verify.store import Store

RESULT_FORMAT = "sheaf-verify.result.v1"


class UtcClock:
    def now(self) -> str:
        return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class SequenceClock:
    """Deterministic timestamps for fixtures. Each call advances one second."""

    def __init__(self, start: str = "2026-01-01T00:00:01Z"):
        self._stamp = start

    def now(self) -> str:
        current = self._stamp
        parsed = datetime.strptime(current, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        self._stamp = (parsed + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return current


def evaluate_one(gate: Gate, ctx: Context, records: list[dict]) -> dict:
    criteria = []
    upstream = []
    blocked = False
    for dep in gate.depends_on:
        prior = _latest_match(records, dep, ctx)
        if prior is None or prior.get("status") != "pass":
            blocked = True
            criteria.append(bad_row(f"upstream.{dep}", "no passing record is bound to the current artifact hashes"))
        else:
            criteria.append(ok_row(f"upstream.{dep}", prior["record_sha256"]))
            upstream.append({"gate_id": dep, "record_sha256": prior["record_sha256"]})
    if not blocked:
        missing = [role for role in gate.binds if not ctx.has(role)]
        if missing:
            criteria.append(bad_row("roles.present", "missing roles: " + ", ".join(missing)))
        else:
            criteria.append(ok_row("roles.present", "required roles are registered"))
            try:
                extra = gate.evaluate(ctx)
            except (SheafError, json.JSONDecodeError, OSError, KeyError) as error:
                extra = [_error_criterion(error)]
            criteria.extend(item.as_dict() for item in extra)
    bound = ctx.bound(gate.binds)
    passed = bool(criteria) and all(item["passed"] for item in criteria) and set(bound) == set(gate.binds)
    decision = {
        "bound": bound,
        "criteria": criteria,
        "gate_id": gate.id,
        "status": "pass" if passed else "fail",
        "upstream": upstream,
    }
    emit(_trace_event(decision))
    return decision


def _trace_event(decision: dict) -> dict:
    own = [item for item in decision["criteria"] if not str(item.get("id", "")).startswith("upstream.")]
    upstream = {}
    for item in decision["criteria"]:
        ident = str(item.get("id", ""))
        if ident.startswith("upstream."):
            upstream[ident.removeprefix("upstream.")] = bool(item.get("passed"))
    own_failed = any(not item.get("passed") for item in own)
    return {
        "gate_id": decision["gate_id"],
        "outcome": trace_outcome(decision["status"], bool(own), own_failed, upstream),
        "own_evaluated": bool(own),
        "own_failed": own_failed,
        "status": decision["status"],
        "upstream": upstream,
    }



def run(store: Store, *, key: bytes | None, clock=None, profile: str = "training") -> list[dict]:
    profile = normalize_profile(profile)
    clock = UtcClock() if clock is None else clock
    ledger = Ledger(store.root)
    ctx = Context(store)
    written = []
    for gate in gates_for_profile(profile):
        decision = evaluate_one(gate, ctx, ledger.records())
        decision["created_at"] = clock.now()
        decision["profile"] = profile
        written.append(ledger.append(decision, key=key))
    head = ledger.records()[-1]["record_sha256"]
    outcome = "pass" if all(item["status"] == "pass" for item in written) else "fail"
    payload = {
        "format": RESULT_FORMAT,
        "gates": [
            {"gate_id": item["gate_id"], "record_sha256": item["record_sha256"], "status": item["status"]}
            for item in written
        ],
        "ledger_head_sha256": head,
        "outcome": outcome,
        "profile": profile,
    }
    binding = store.binding()
    inputs = [binding[role][0] for role in sorted(binding) if role != "result"]
    store.register(
        kind="result",
        role="result",
        registered_at=clock.now(),
        payload=canonical_bytes(payload),
        inputs=inputs,
        metadata={"outcome": outcome},
    )
    return written


def replay(store: Store, records: list[dict]) -> list[str]:
    ctx = Context(store)
    latest: dict[str, dict] = {}
    for record in records:
        gate_id = record.get("gate_id")
        if isinstance(gate_id, str):
            latest[gate_id] = record
    problems = []
    profile = profile_from_records(records)
    for gate in gates_for_profile(profile):
        stored = latest.get(gate.id)
        decision = evaluate_one(gate, ctx, records)
        if stored is None:
            problems.append(f"replay: {gate.id} has no ledger record")
            continue
        recorded_profile = stored.get("profile")
        if recorded_profile is not None and recorded_profile != profile:
            problems.append(f"replay: {gate.id} profile does not match the ledger")
        for field in ("status", "bound", "upstream", "criteria"):
            if stored.get(field) != decision[field]:
                problems.append(f"replay: {gate.id} {field} does not match the ledger")
    return problems


def _latest_match(records: list[dict], gate_id: str, ctx: Context):
    found = None
    for record in records:
        if record.get("gate_id") != gate_id or record.get("status") != "pass":
            continue
        if _bound_matches(record.get("bound"), ctx):
            found = record
    return found


def _bound_matches(bound, ctx: Context) -> bool:
    if not isinstance(bound, dict) or not bound:
        return False
    for role, hashes in bound.items():
        if not isinstance(hashes, dict) or not ctx.has(role):
            return False
        if ctx.manifest_sha(role) != hashes.get("manifest_sha256"):
            return False
        if ctx.manifest(role).get("payload_sha256") != hashes.get("payload_sha256"):
            return False
    return True


def ok_row(identifier: str, detail: str) -> dict:
    return {"detail": detail, "id": identifier, "passed": True}


def bad_row(identifier: str, detail: str) -> dict:
    return {"detail": detail, "id": identifier, "passed": False}


class _error_criterion:
    def __init__(self, error: Exception):
        self._row = bad_row("evaluator.error", f"{type(error).__name__}: {error}")

    def as_dict(self) -> dict:
        return self._row
