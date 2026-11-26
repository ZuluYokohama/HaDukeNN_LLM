"""Exact, scoped executable relations for the HaDukeNN research prototype.

This is a symbolic reference runtime, not a trained neural model. Numeric
operations use exact fractions. Learned proposers may later submit the same
commands, but a syntactically accepted definition is not a verified world fact.

``max_nodes`` bounds current bindings plus immutable sources; ``max_steps``
bounds uncached binding evaluations per query. Separate finite caps bound
scopes, revision count, receipt count, source bytes and command bytes. These
are resource limits, not a guaranteed total process-RAM bound. Source text and revisions
are preserved by the replayable command log. Hashes provide integrity relative
to a trusted head; they are not signatures or proof of authorship.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

SCHEMA = "hadukenn.exec.receipt.v1"
ZERO_HASH = "0" * 64
MAX_VALUE_BITS = 4096
MAX_NUMBER_CHARS = 2048
MAX_SOURCE_BYTES = 1_048_576
MAX_JSON_DEPTH = 64
LIMIT_DEFAULTS = {
    "max_nodes": 10_000,
    "max_steps": 10_000,
    "max_scopes": 256,
    "max_revisions": 100_000,
    "max_receipts": 100_000,
    "max_source_bytes": 16_777_216,
    "max_command_bytes": 1_100_000,
}
_NUMBER = re.compile(r"[+-]?[0-9]+(?:/[+-]?[0-9]+)?\Z")
_ROLES = {
    "add": ("left", "right"),
    "sub": ("left", "right"),
    "mul": ("left", "right"),
    "div": ("left", "right"),
    "neg": ("value",),
    "copy": ("value",),
}


def canonical(value: Any) -> bytes:
    """Canonical UTF-8 JSON used by receipts and state commitments."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


class KernelError(Exception):
    """A declared refusal in the executable command language."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _text(value: Any, field: str, *, limit: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > limit:
        raise KernelError("invalid_command", f"{field} must be a nonempty string of at most {limit} characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise KernelError("invalid_command", f"{field} must be valid Unicode") from exc
    return value


def _bounded(value: Fraction) -> Fraction:
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_VALUE_BITS:
        raise KernelError("resource_limit", "exact value exceeds the configured bit bound")
    return value


def _number(value: Any) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise KernelError("invalid_command", "value must be an integer or an exact fraction string")
    if isinstance(value, str):
        if len(value) > MAX_NUMBER_CHARS:
            raise KernelError("resource_limit", "numeric literal is too long")
        if not _NUMBER.fullmatch(value):
            raise KernelError("invalid_command", "numeric string must be a signed integer or fraction")
    try:
        # Fraction does not accept a signed denominator directly.
        if isinstance(value, str) and "/" in value:
            numerator, denominator = value.split("/")
            result = Fraction(int(numerator), int(denominator))
        else:
            result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise KernelError("invalid_command", "invalid exact numeric literal") from exc
    return _bounded(result)


@dataclass(frozen=True)
class Binding:
    revision: int
    kind: str
    source: str | None
    value: Fraction | None = None
    unit: str | None = None
    op: str | None = None
    args: tuple[tuple[str, str], ...] = ()

    def record(self) -> dict[str, Any]:
        data: dict[str, Any] = {"revision": self.revision, "kind": self.kind, "source": self.source}
        if self.kind == "literal":
            data.update(value=str(self.value), unit=self.unit)
        else:
            data.update(op=self.op, args=dict(self.args))
        return data


@dataclass(frozen=True)
class Cached:
    value: Fraction
    unit: str
    revision: int
    computed_at: int


class Kernel:
    """A typed relation store with scoped snapshots and actual execution receipts.

    Commands use ``action`` as the discriminator. Unknown metadata fields are
    retained in receipts but have no semantic effect. ``set`` is an explicit
    authoritative revision in this formal input language, not a confidence-
    based acceptance rule for natural language. Derived references resolve in
    their own scope; a fork snapshots definitions and rebinds those references
    to the new scope. No cross-scope reference syntax is provided.
    """

    def __init__(
        self,
        max_nodes: int = 10_000,
        max_steps: int = 10_000,
        max_scopes: int = 256,
        max_revisions: int = 100_000,
        max_receipts: int = 100_000,
        max_source_bytes: int = 16_777_216,
        max_command_bytes: int = 1_100_000,
    ):
        limits = {
            "max_nodes": max_nodes, "max_steps": max_steps,
            "max_scopes": max_scopes, "max_revisions": max_revisions,
            "max_receipts": max_receipts, "max_source_bytes": max_source_bytes,
            "max_command_bytes": max_command_bytes,
        }
        for name, value in limits.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, value)
        self._scopes: dict[str, dict[str, Binding]] = {"main": {}}
        self._reverse: dict[str, dict[str, set[str]]] = {"main": {}}
        self._cache: dict[tuple[str, str], Cached] = {}
        self._sources: dict[str, dict[str, Any]] = {}
        self._clock = 0
        self._trace: list[dict[str, Any]] = []
        self._active_nodes = 0
        self._source_bytes = 0

    @property
    def config(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in LIMIT_DEFAULTS}

    @property
    def head(self) -> str:
        return self._trace[-1]["hash"] if self._trace else ZERO_HASH

    def _state(self) -> dict[str, Any]:
        return {
            "schema": "hadukenn.exec.state.v1",
            "revision_clock": self._clock,
            "sources": self._sources,
            "scopes": {
                scope: {name: binding.record() for name, binding in bindings.items()}
                for scope, bindings in self._scopes.items()
            },
        }

    def state_digest(self) -> str:
        """Commit the semantic state; caches and the audit log are separate."""
        return digest(self._state())

    def export_trace(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._trace)

    def graph(self) -> dict[str, Any]:
        """Export source text and named-role hyperedges without mutable aliases."""
        nodes = []
        edges = []
        for scope, bindings in sorted(self._scopes.items()):
            for name, binding in sorted(bindings.items()):
                node_id = digest([scope, name])
                nodes.append({"id": node_id, "scope": scope, "name": name, **binding.record()})
                if binding.kind == "derived":
                    edges.append({
                        "id": f"revision-{binding.revision}",
                        "scope": scope,
                        "relation": binding.op,
                        "output": node_id,
                        "ports": {
                            role: {"name": ref, "node": digest([scope, ref]), "resolved": ref in bindings}
                            for role, ref in binding.args
                        },
                        "source": binding.source,
                    })
        return {
            "schema": "hadukenn.exec.graph.v1",
            "sources": [copy.deepcopy(value) for _, value in sorted(self._sources.items())],
            "scopes": sorted(self._scopes),
            "nodes": nodes,
            "hyperedges": edges,
            "state_digest": self.state_digest(),
            "trace_head": self.head,
            "active_nodes": self._active_nodes,
            "limits": self.config,
        }

    def _scope(self, command: dict[str, Any]) -> tuple[str, dict[str, Binding]]:
        scope = _text(command.get("scope", "main"), "scope")
        if scope not in self._scopes:
            raise KernelError("missing_reference", f"unknown scope: {scope}")
        return scope, self._scopes[scope]

    def _source(self, command: dict[str, Any]) -> str | None:
        if "source" not in command:
            return None
        source = _text(command["source"], "source")
        if source not in self._sources:
            raise KernelError("missing_reference", f"unknown source: {source}")
        return source

    def _room(self, additional: int) -> None:
        if self._active_nodes + additional > self.max_nodes:
            raise KernelError("resource_limit", "active bindings and sources exceed max_nodes")

    def _revision_room(self, additional: int) -> None:
        if self._clock + additional > self.max_revisions:
            raise KernelError("resource_limit", "retained revision count exceeds max_revisions")

    def _invalidate(self, scope: str, name: str) -> list[str]:
        pending = [name]
        visited: set[str] = set()
        removed = []
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            if self._cache.pop((scope, current), None) is not None:
                removed.append(current)
            pending.extend(sorted(self._reverse[scope].get(current, ()), reverse=True))
        return sorted(removed)

    def _replace(self, scope: str, name: str, binding: Binding, evidence: list[dict[str, Any]]) -> None:
        bindings = self._scopes[scope]
        old = bindings.get(name)
        invalidated = self._invalidate(scope, name)
        if old:
            for _, ref in old.args:
                dependents = self._reverse[scope].get(ref)
                if dependents is not None:
                    dependents.discard(name)
                    if not dependents:
                        self._reverse[scope].pop(ref, None)
        else:
            self._active_nodes += 1
        self._clock = binding.revision
        bindings[name] = binding
        for _, ref in binding.args:
            self._reverse[scope].setdefault(ref, set()).add(name)
        evidence.append({
            "event": "write", "scope": scope, "name": name,
            "previous_revision": old.revision if old else None,
            "binding": binding.record(), "invalidated": invalidated,
        })

    def _arithmetic(self, binding: Binding, values: dict[str, Cached]) -> tuple[Fraction, str]:
        args = dict(binding.args)
        if binding.op in ("neg", "copy"):
            arg = values[args["value"]]
            return _bounded(-arg.value if binding.op == "neg" else arg.value), arg.unit
        left, right = values[args["left"]], values[args["right"]]
        if binding.op in ("add", "sub"):
            if left.unit != right.unit:
                raise KernelError("unit_mismatch", "addition and subtraction require matching units")
            result = left.value + right.value if binding.op == "add" else left.value - right.value
            return _bounded(result), left.unit
        if binding.op == "mul":
            if left.unit == "1":
                unit = right.unit
            elif right.unit == "1":
                unit = left.unit
            else:
                raise KernelError("unit_mismatch", "multiplication requires a dimensionless operand")
            return _bounded(left.value * right.value), unit
        if binding.op == "div":
            if right.unit == "1":
                unit = left.unit
            elif right.unit == left.unit:
                unit = "1"
            else:
                raise KernelError("unit_mismatch", "division requires a scalar divisor or matching units")
            if right.value == 0:
                raise KernelError("division_by_zero", "division by zero is undefined")
            return _bounded(left.value / right.value), unit
        raise KernelError("invalid_operator", "operator is not in the executable language")

    def _evaluate(self, scope: str, name: str, evidence: list[dict[str, Any]]) -> Cached:
        bindings = self._scopes[scope]
        local: dict[str, Cached] = {}
        active: set[str] = set()
        stack = [(name, False)]
        steps = 0
        while stack:
            current, finishing = stack.pop()
            if current in local:
                continue
            binding = bindings.get(current)
            if binding is None:
                raise KernelError("missing_reference", f"unresolved binding: {scope}/{current}")
            if not finishing:
                cached = self._cache.get((scope, current))
                if cached is not None:
                    local[current] = cached
                    evidence.append({
                        "event": "cache_reuse", "scope": scope, "name": current,
                        "revision": cached.revision, "computed_at": cached.computed_at,
                        "value": str(cached.value), "unit": cached.unit,
                    })
                    continue
                if current in active:
                    raise KernelError("cycle", f"cyclic dependency at {scope}/{current}")
                steps += 1
                if steps > self.max_steps:
                    raise KernelError("resource_limit", "query exceeds max_steps uncached evaluations")
                active.add(current)
                stack.append((current, True))
                for _, ref in reversed(binding.args):
                    if ref not in local:
                        stack.append((ref, False))
                continue
            if binding.kind == "literal":
                assert binding.value is not None and binding.unit is not None
                value, unit = binding.value, binding.unit
            else:
                value, unit = self._arithmetic(binding, local)
            result = Cached(value, unit, binding.revision, len(self._trace) + 1)
            self._cache[(scope, current)] = result
            local[current] = result
            active.remove(current)
            evidence.append({
                "event": "evaluate", "scope": scope, "name": current,
                "revision": binding.revision, "kind": binding.kind,
                "operation": binding.op, "source": binding.source,
                "reads": {
                    role: {"name": ref, "revision": bindings[ref].revision}
                    for role, ref in binding.args
                },
                "value": str(value), "unit": unit,
            })
        return local[name]

    def _apply(self, command: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
        action = command.get("action")
        if not isinstance(action, str):
            raise KernelError("invalid_command", "action must be a string")
        if action == "__oversize_command__":
            raise KernelError("resource_limit", "command exceeds max_command_bytes; descriptor retained")
        if action == "source":
            source_id = _text(command.get("id"), "id")
            text = command.get("text")
            if not isinstance(text, str):
                raise KernelError("invalid_command", "source text must be a string")
            try:
                raw = text.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise KernelError("invalid_command", "source text must be valid Unicode") from exc
            if len(raw) > MAX_SOURCE_BYTES:
                raise KernelError("resource_limit", "source exceeds the byte bound")
            previous = self._sources.get(source_id)
            if previous:
                if previous["text"] != text:
                    raise KernelError("conflict", "source identifiers are immutable")
                evidence.append({"event": "source_reuse", "id": source_id})
                return {"status": "ok"}
            self._room(1)
            if self._source_bytes + len(raw) > self.max_source_bytes:
                raise KernelError("resource_limit", "retained source bytes exceed max_source_bytes")
            record = {"id": source_id, "text": text, "sha256": hashlib.sha256(raw).hexdigest(), "byte_length": len(raw)}
            self._sources[source_id] = record
            self._active_nodes += 1
            self._source_bytes += len(raw)
            evidence.append({"event": "source", **record})
            return {"status": "ok"}
        if action not in {"set", "derive", "retract", "fork", "query", "assert"}:
            raise KernelError("invalid_command", "unknown action")
        scope, bindings = self._scope(command)
        if action == "fork":
            new_scope = _text(command.get("new_scope"), "new_scope")
            if new_scope in self._scopes:
                raise KernelError("conflict", "new scope already exists")
            if len(self._scopes) >= self.max_scopes:
                raise KernelError("resource_limit", "scope count exceeds max_scopes")
            self._room(len(bindings))
            self._revision_room(len(bindings))
            copied = {}
            revisions = {}
            reverse: dict[str, set[str]] = {}
            for name, binding in sorted(bindings.items()):
                self._clock += 1
                cloned = Binding(self._clock, binding.kind, binding.source, binding.value, binding.unit, binding.op, binding.args)
                copied[name] = cloned
                revisions[name] = {"from": binding.revision, "to": cloned.revision}
                for _, ref in cloned.args:
                    reverse.setdefault(ref, set()).add(name)
            self._scopes[new_scope] = copied
            self._reverse[new_scope] = reverse
            self._active_nodes += len(copied)
            evidence.append({"event": "fork", "from": scope, "to": new_scope, "revisions": revisions})
            return {"status": "ok"}
        name = _text(command.get("name"), "name")
        if action == "set":
            value = _number(command.get("value"))
            unit = _text(command.get("unit", "1"), "unit", limit=64)
            source = self._source(command)
            self._room(0 if name in bindings else 1)
            self._revision_room(1)
            binding = Binding(self._clock + 1, "literal", source, value, unit)
            self._replace(scope, name, binding, evidence)
            return {"status": "ok"}
        if action == "derive":
            op = command.get("op")
            if not isinstance(op, str) or op not in _ROLES:
                raise KernelError("invalid_operator", "operator is not in the executable language")
            args = command.get("args")
            if not isinstance(args, dict) or set(args) != set(_ROLES[op]):
                raise KernelError("invalid_command", "operator arguments must have the declared named roles")
            refs = tuple((role, _text(args[role], role)) for role in _ROLES[op])
            source = self._source(command)
            self._room(0 if name in bindings else 1)
            self._revision_room(1)
            binding = Binding(self._clock + 1, "derived", source, op=op, args=refs)
            self._replace(scope, name, binding, evidence)
            return {"status": "ok"}
        if action == "retract":
            source = self._source(command)
            if name not in bindings:
                raise KernelError("missing_reference", f"unknown binding: {scope}/{name}")
            self._revision_room(1)
            invalidated = self._invalidate(scope, name)
            old = bindings.pop(name)
            for _, ref in old.args:
                dependents = self._reverse[scope].get(ref)
                if dependents is not None:
                    dependents.discard(name)
                    if not dependents:
                        self._reverse[scope].pop(ref, None)
            self._active_nodes -= 1
            self._clock += 1
            evidence.append({
                "event": "retract", "scope": scope, "name": name,
                "previous_revision": old.revision, "revision": self._clock,
                "source": source, "invalidated": invalidated,
            })
            return {"status": "ok"}
        expected = None
        expected_unit = None
        if action == "assert":
            expected = _number(command.get("value"))
            expected_unit = _text(command.get("unit", "1"), "unit", limit=64)
        result = self._evaluate(scope, name, evidence)
        if action == "assert":
            passed = result.value == expected and result.unit == expected_unit
            evidence.append({
                "event": "assert", "scope": scope, "name": name,
                "expected": {"value": str(expected), "unit": expected_unit},
                "actual": {"value": str(result.value), "unit": result.unit}, "passed": passed,
            })
            if not passed:
                raise KernelError("conflict", "asserted quantity disagrees with the evaluated result")
        return {"status": "ok", "value": str(result.value), "unit": result.unit}

    def execute(self, command: dict[str, Any]) -> dict[str, Any]:
        """Execute and record one command; refusals preserve semantic state.

        Inputs must be JSON objects. Unknown JSON metadata is retained but is
        semantically inert. Non-JSON inputs are refused before execution and
        recorded as a deterministic invalid-input descriptor. Overlarge JSON
        commands retain only a hash and byte-count descriptor. Once receipt
        capacity is exhausted, further commands are refused before execution
        without appending a receipt; this terminal audit-capacity refusal is
        visible to the caller but cannot itself be added to the full log.
        """
        if len(self._trace) >= self.max_receipts:
            return {"status": "error", "error": "resource_limit", "message": "receipt capacity exhausted; command not executed or appended"}
        try:
            if not isinstance(command, dict):
                raise TypeError("command must be an object")
            encoded = canonical(command)
            normalized = json.loads(encoded)
            # JSON silently coerces non-string object keys; do not accept them.
            pending = [(command, 0)]
            while pending:
                item, depth = pending.pop()
                if depth > MAX_JSON_DEPTH:
                    raise TypeError("command nesting exceeds the JSON depth bound")
                if isinstance(item, dict):
                    if any(not isinstance(key, str) for key in item):
                        raise TypeError("JSON object keys must be strings")
                    pending.extend((value, depth + 1) for value in item.values())
                elif isinstance(item, (list, tuple)):
                    if isinstance(item, tuple):
                        raise TypeError("JSON arrays must be lists")
                    pending.extend((value, depth + 1) for value in item)
            descriptor = normalized == {"action": "__invalid_json_input__"} or (
                set(normalized) == {"action", "sha256", "byte_length"}
                and normalized.get("action") == "__oversize_command__"
                and isinstance(normalized.get("sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", normalized["sha256"]) is not None
                and isinstance(normalized.get("byte_length"), int)
                and not isinstance(normalized["byte_length"], bool)
                and 0 <= normalized["byte_length"] < 2**64
            )
            if len(encoded) > self.max_command_bytes and not descriptor:
                normalized = {"action": "__oversize_command__", "sha256": hashlib.sha256(encoded).hexdigest(), "byte_length": len(encoded)}
        except (TypeError, ValueError, UnicodeEncodeError, RecursionError):
            normalized = {"action": "__invalid_json_input__"}
        before = self.state_digest()
        evidence: list[dict[str, Any]] = []
        try:
            result = self._apply(normalized, evidence)
        except KernelError as exc:
            result = {"status": "error", "error": exc.code, "message": str(exc)}
        after = self.state_digest()
        receipt = {
            "schema": SCHEMA, "sequence": len(self._trace) + 1,
            "config": self.config, "previous_hash": self.head,
            "command": normalized, "result": result, "evidence": evidence,
            "state_before": before, "state_after": after,
        }
        receipt["hash"] = digest(receipt)
        self._trace.append(receipt)
        return copy.deepcopy(result)

    @classmethod
    def replay(cls, trace: list[dict[str, Any]], expected_head: str | None = None) -> Kernel:
        """Check receipt integrity and re-execute every recorded transition.

        ``expected_head`` is an optional independently retained anchor. Without
        it, a wholly rewritten internally consistent log cannot be detected.
        Replay verifies this executor's semantics, not the truth of sources.
        """
        if not isinstance(trace, list):
            raise ValueError("trace must be a list")
        if not trace:
            kernel = cls()
            if expected_head is not None and expected_head != ZERO_HASH:
                raise ValueError("trace head differs from trusted anchor")
            return kernel
        first = trace[0]
        if not isinstance(first, dict) or not isinstance(first.get("config"), dict):
            raise ValueError("trace is missing configuration")
        config = first["config"]
        if set(config) != set(LIMIT_DEFAULTS):
            raise ValueError("unknown trace configuration")
        kernel = cls(**config)
        previous = ZERO_HASH
        for index, recorded in enumerate(trace, 1):
            if not isinstance(recorded, dict):
                raise ValueError(f"receipt {index} is not an object")
            payload = {key: value for key, value in recorded.items() if key != "hash"}
            try:
                correct_hash = digest(payload)
            except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
                raise ValueError(f"receipt {index} is not canonical JSON") from exc
            if recorded.get("hash") != correct_hash:
                raise ValueError(f"receipt {index} hash mismatch")
            if recorded.get("previous_hash") != previous or recorded.get("sequence") != index:
                raise ValueError(f"receipt {index} chain mismatch")
            if recorded.get("schema") != SCHEMA or recorded.get("config") != config:
                raise ValueError(f"receipt {index} schema or configuration mismatch")
            kernel.execute(recorded.get("command"))
            if canonical(kernel._trace[-1]) != canonical(recorded):
                raise ValueError(f"receipt {index} semantic replay mismatch")
            previous = recorded["hash"]
        if expected_head is not None and previous != expected_head:
            raise ValueError("trace head differs from trusted anchor")
        return kernel
