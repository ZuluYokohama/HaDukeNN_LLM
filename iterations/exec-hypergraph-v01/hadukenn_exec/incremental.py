"""Incremental state checksum for the exec-hypergraph-v01 kernel.

``IncrementalKernel`` keeps the semantics and receipt chain of ``core.Kernel`` but
replaces the per-command full-state re-serialization in ``state_digest()`` with a
running sum of per-leaf hashes, updated only for leaves that change.

Scope of the guarantee: the state digest is an integrity checksum against
accidental drift. It is not cryptographic. Leaves are hashed with unkeyed
BLAKE2b and summed in four 64-bit lanes; that detects bugs and divergence, but
an adversary who controls inputs could search for colliding states. Do not use
it as a commitment, signature or proof of authorship.

Leaves are canonical JSON arrays, one element per field, so field boundaries are
unambiguous whatever characters names, units or sources contain.
"""

from __future__ import annotations

import copy
import hashlib
import struct
from fractions import Fraction
from typing import Any

from .core import (
    _ROLES,
    LIMIT_DEFAULTS,
    Binding,
    Cached,
    KernelError,
    _bounded,
    _number,
    _text,
    canonical,
    digest,
)

_MASK64 = (1 << 64) - 1
_STATE_DOMAIN = b"hadukenn.exec.incremental-checksum.v3\x00"
_LEAF_SRC_DOMAIN = b"hadukenn.exec.leaf.source.v3\x00"
_LEAF_BIND_DOMAIN = b"hadukenn.exec.leaf.binding.v3\x00"


def _leaf_limbs(domain: bytes, fields: list[Any]) -> tuple[int, int, int, int]:
    """Hash a domain-separated canonical-JSON leaf into four 64-bit lanes."""
    raw = hashlib.blake2b(domain + canonical(fields), digest_size=32).digest()
    return struct.unpack("<QQQQ", raw)


class IncrementalKernel:
    """Executable-relation kernel with an incremental state checksum.

    Same commands, refusals, resource limits, receipts and ``replay()`` contract as
    ``hadukenn_exec.core.Kernel``. The difference is ``state_digest()``: instead of
    re-serializing the whole state on every command, it keeps a running sum of
    per-leaf hashes (each live source and binding is one leaf) and updates it by
    subtracting the retiring leaf and adding the entering one.

    The state digest is an integrity checksum against accidental drift (bugs,
    lost or reordered updates, replay divergence). It is not cryptographic: the
    leaf hash is unkeyed and the accumulator is only 4 x 64 bits, so it does not
    resist an adversary who chooses inputs to force a collision. Tamper evidence
    for recorded traces comes from the receipt hash chain, not from this checksum.
    """

    SCHEMA = "hadukenn.exec.receipt.v3-incremental"
    ZERO_HASH = "0" * 64

    def __init__(
        self,
        max_nodes: int = 10_000,
        max_steps: int = 10_000,
        max_scopes: int = 256,
        max_revisions: int = 100_000,
        max_receipts: int = 100_000,
        max_source_bytes: int = 16_777_216,
        max_command_bytes: int = 1_100_000,
    ) -> None:
        self._Binding = Binding
        self._Cached = Cached
        self._KernelError = KernelError
        self._LIMIT_DEFAULTS = LIMIT_DEFAULTS
        self._ROLES = _ROLES
        self._bounded = _bounded
        self._number = _number
        self._text = _text
        self._canonical = canonical
        self._digest = digest

        limits = {
            "max_nodes": max_nodes,
            "max_steps": max_steps,
            "max_scopes": max_scopes,
            "max_revisions": max_revisions,
            "max_receipts": max_receipts,
            "max_source_bytes": max_source_bytes,
            "max_command_bytes": max_command_bytes,
        }
        for name, val in limits.items():
            if isinstance(val, bool) or not isinstance(val, int) or val < 1:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, val)

        self._scopes: dict[str, dict[str, Any]] = {"main": {}}
        self._bind_limbs: dict[str, dict[str, tuple[int, int, int, int]]] = {"main": {}}
        self._reverse: dict[str, dict[str, set[str]]] = {"main": {}}
        self._cache: dict[tuple[str, str], Any] = {}
        self._sources: dict[str, dict[str, Any]] = {}
        self._clock = 0
        self._trace: list[dict[str, Any]] = []
        self._active_nodes = 0
        self._source_bytes = 0

        # Running sum of leaf hashes, four independent 64-bit lanes (integrity checksum only).
        self._acc0 = 0
        self._acc1 = 0
        self._acc2 = 0
        self._acc3 = 0
        self._cached_state_digest: str | None = None

    @property
    def config(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self._LIMIT_DEFAULTS}

    @property
    def head(self) -> str:
        return self._trace[-1]["hash"] if self._trace else self.ZERO_HASH

    def _add_limbs(self, limbs: tuple[int, int, int, int]) -> None:
        self._acc0 = (self._acc0 + limbs[0]) & _MASK64
        self._acc1 = (self._acc1 + limbs[1]) & _MASK64
        self._acc2 = (self._acc2 + limbs[2]) & _MASK64
        self._acc3 = (self._acc3 + limbs[3]) & _MASK64
        self._cached_state_digest = None

    def _sub_limbs(self, limbs: tuple[int, int, int, int]) -> None:
        self._acc0 = (self._acc0 - limbs[0]) & _MASK64
        self._acc1 = (self._acc1 - limbs[1]) & _MASK64
        self._acc2 = (self._acc2 - limbs[2]) & _MASK64
        self._acc3 = (self._acc3 - limbs[3]) & _MASK64
        self._cached_state_digest = None

    def _binding_leaf(self, scope: str, name: str, binding: Any) -> tuple[int, int, int, int]:
        """Hash one binding as a canonical-JSON leaf.

        Every field is a separate JSON array element, so no string content (including
        control characters such as ``\x1f``) can move bytes from one field into another.
        """
        if binding.kind == "literal":
            fields: list[Any] = [
                "literal",
                scope,
                name,
                binding.revision,
                binding.source,
                str(binding.value),
                binding.unit,
            ]
        else:
            fields = [
                "derived",
                scope,
                name,
                binding.revision,
                binding.source,
                binding.op,
                [[role, ref] for role, ref in binding.args],
            ]
        return _leaf_limbs(_LEAF_BIND_DOMAIN, fields)

    def state_digest(self) -> str:
        """Return the state checksum: running leaf sum plus scope count, clock and node count.

        Integrity checksum for accidental drift, not a cryptographic commitment.
        """
        if self._cached_state_digest is None:
            buf = _STATE_DOMAIN + struct.pack(
                "<QQQQQQQ",
                len(self._scopes),
                self._clock,
                self._active_nodes,
                self._acc0,
                self._acc1,
                self._acc2,
                self._acc3,
            )
            self._cached_state_digest = hashlib.sha256(buf).hexdigest()
        return self._cached_state_digest

    def export_trace(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._trace)

    def _invalidate(self, scope: str, name: str) -> list[str]:
        pending = [name]
        visited: set[str] = set()
        removed = []
        rev_map = self._reverse[scope]
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            if self._cache.pop((scope, current), None) is not None:
                removed.append(current)
            deps = rev_map.get(current)
            if deps:
                pending.extend(sorted(deps, reverse=True))
        return sorted(removed)

    def _replace(self, scope: str, name: str, binding: Any, evidence: list[dict[str, Any]]) -> None:
        bindings = self._scopes[scope]
        limbs_map = self._bind_limbs[scope]
        old = bindings.get(name)
        invalidated = self._invalidate(scope, name)
        if old is not None:
            self._sub_limbs(limbs_map[name])
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
        new_limbs = self._binding_leaf(scope, name, binding)
        limbs_map[name] = new_limbs
        self._add_limbs(new_limbs)
        for _, ref in binding.args:
            self._reverse[scope].setdefault(ref, set()).add(name)
        evidence.append(
            {
                "event": "write",
                "scope": scope,
                "name": name,
                "previous_revision": old.revision if old else None,
                "binding": binding.record(),
                "invalidated": invalidated,
            }
        )

    def _arithmetic(self, binding: Any, values: dict[str, Any]) -> tuple[Fraction, str]:
        KernelError = self._KernelError
        _bounded = self._bounded
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

    def _evaluate(self, scope: str, name: str, evidence: list[dict[str, Any]], record_evidence: bool = True) -> Any:
        KernelError = self._KernelError
        Cached = self._Cached
        bindings = self._scopes[scope]
        local: dict[str, Any] = {}
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
                    if record_evidence:
                        evidence.append(
                            {
                                "event": "cache_reuse",
                                "scope": scope,
                                "name": current,
                                "revision": cached.revision,
                                "computed_at": cached.computed_at,
                                "value": str(cached.value),
                                "unit": cached.unit,
                            }
                        )
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
                value, unit = binding.value, binding.unit
            else:
                value, unit = self._arithmetic(binding, local)
            result = Cached(value, unit, binding.revision, len(self._trace) + 1)
            if record_evidence:
                self._cache[(scope, current)] = result
            local[current] = result
            active.remove(current)
            if record_evidence:
                evidence.append(
                    {
                        "event": "evaluate",
                        "scope": scope,
                        "name": current,
                        "revision": binding.revision,
                        "kind": binding.kind,
                        "operation": binding.op,
                        "source": binding.source,
                        "reads": {
                            role: {"name": ref, "revision": bindings[ref].revision} for role, ref in binding.args
                        },
                        "value": str(value),
                        "unit": unit,
                    }
                )
        return local[name]

    def _apply(self, command: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
        KernelError = self._KernelError
        Binding = self._Binding
        _text = self._text
        _number = self._number

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
            raw = text.encode("utf-8")
            if len(raw) > 1_048_576:
                raise KernelError("resource_limit", "source exceeds the byte bound")
            previous = self._sources.get(source_id)
            if previous:
                if previous["text"] != text:
                    raise KernelError("conflict", "source identifiers are immutable")
                evidence.append({"event": "source_reuse", "id": source_id})
                return {"status": "ok"}
            if self._active_nodes + 1 > self.max_nodes:
                raise KernelError("resource_limit", "active bindings and sources exceed max_nodes")
            if self._source_bytes + len(raw) > self.max_source_bytes:
                raise KernelError("resource_limit", "retained source bytes exceed max_source_bytes")
            sha = hashlib.sha256(raw).hexdigest()
            record = {"id": source_id, "text": text, "sha256": sha, "byte_length": len(raw)}
            self._sources[source_id] = record
            self._active_nodes += 1
            self._source_bytes += len(raw)
            self._add_limbs(_leaf_limbs(_LEAF_SRC_DOMAIN, ["source", source_id, sha]))
            evidence.append({"event": "source", **record})
            return {"status": "ok"}

        if action not in {"set", "derive", "retract", "fork", "query", "assert"}:
            raise KernelError("invalid_command", "unknown action")
        scope = _text(command.get("scope", "main"), "scope")
        if scope not in self._scopes:
            raise KernelError("missing_reference", f"unknown scope: {scope}")
        bindings = self._scopes[scope]

        if action == "fork":
            new_scope = _text(command.get("new_scope"), "new_scope")
            if new_scope in self._scopes:
                raise KernelError("conflict", "new scope already exists")
            if len(self._scopes) >= self.max_scopes:
                raise KernelError("resource_limit", "scope count exceeds max_scopes")
            if self._active_nodes + len(bindings) > self.max_nodes:
                raise KernelError("resource_limit", "active bindings and sources exceed max_nodes")
            if self._clock + len(bindings) > self.max_revisions:
                raise KernelError("resource_limit", "retained revision count exceeds max_revisions")
            copied = {}
            limbs_new = {}
            revisions = {}
            reverse: dict[str, set[str]] = {}
            for bname, bobj in sorted(bindings.items()):
                self._clock += 1
                cloned = Binding(self._clock, bobj.kind, bobj.source, bobj.value, bobj.unit, bobj.op, bobj.args)
                copied[bname] = cloned
                lb = self._binding_leaf(new_scope, bname, cloned)
                limbs_new[bname] = lb
                self._add_limbs(lb)
                revisions[bname] = {"from": bobj.revision, "to": cloned.revision}
                for _, ref in cloned.args:
                    reverse.setdefault(ref, set()).add(bname)
            self._scopes[new_scope] = copied
            self._bind_limbs[new_scope] = limbs_new
            self._reverse[new_scope] = reverse
            self._active_nodes += len(copied)
            self._cached_state_digest = None
            evidence.append({"event": "fork", "from": scope, "to": new_scope, "revisions": revisions})
            return {"status": "ok"}

        name = _text(command.get("name"), "name")
        source = None
        if "source" in command:
            source = _text(command["source"], "source")
            if source not in self._sources:
                raise KernelError("missing_reference", f"unknown source: {source}")

        if action == "set":
            value = _number(command.get("value"))
            unit = _text(command.get("unit", "1"), "unit", limit=64)
            if self._active_nodes + (0 if name in bindings else 1) > self.max_nodes:
                raise KernelError("resource_limit", "active bindings and sources exceed max_nodes")
            if self._clock + 1 > self.max_revisions:
                raise KernelError("resource_limit", "retained revision count exceeds max_revisions")
            binding = Binding(self._clock + 1, "literal", source, value, unit)
            self._replace(scope, name, binding, evidence)
            return {"status": "ok"}

        if action == "derive":
            op = command.get("op")
            if not isinstance(op, str) or op not in self._ROLES:
                raise KernelError("invalid_operator", "operator is not in the executable language")
            args = command.get("args")
            if not isinstance(args, dict) or set(args) != set(self._ROLES[op]):
                raise KernelError("invalid_command", "operator arguments must have the declared named roles")
            refs = tuple((role, _text(args[role], role)) for role in self._ROLES[op])
            if self._active_nodes + (0 if name in bindings else 1) > self.max_nodes:
                raise KernelError("resource_limit", "active bindings and sources exceed max_nodes")
            if self._clock + 1 > self.max_revisions:
                raise KernelError("resource_limit", "retained revision count exceeds max_revisions")
            binding = Binding(self._clock + 1, "derived", source, op=op, args=refs)
            self._replace(scope, name, binding, evidence)
            return {"status": "ok"}

        if action == "retract":
            if name not in bindings:
                raise KernelError("missing_reference", f"unknown binding: {scope}/{name}")
            if self._clock + 1 > self.max_revisions:
                raise KernelError("resource_limit", "retained revision count exceeds max_revisions")
            invalidated = self._invalidate(scope, name)
            old = bindings.pop(name)
            self._sub_limbs(self._bind_limbs[scope].pop(name))
            for _, ref in old.args:
                dependents = self._reverse[scope].get(ref)
                if dependents is not None:
                    dependents.discard(name)
                    if not dependents:
                        self._reverse[scope].pop(ref, None)
            self._active_nodes -= 1
            self._clock += 1
            self._cached_state_digest = None
            evidence.append(
                {
                    "event": "retract",
                    "scope": scope,
                    "name": name,
                    "previous_revision": old.revision,
                    "revision": self._clock,
                    "source": source,
                    "invalidated": invalidated,
                }
            )
            return {"status": "ok"}

        expected = None
        expected_unit = None
        if action == "assert":
            expected = _number(command.get("value"))
            expected_unit = _text(command.get("unit", "1"), "unit", limit=64)
        result = self._evaluate(scope, name, evidence)
        if action == "assert":
            passed = result.value == expected and result.unit == expected_unit
            evidence.append(
                {
                    "event": "assert",
                    "scope": scope,
                    "name": name,
                    "expected": {"value": str(expected), "unit": expected_unit},
                    "actual": {"value": str(result.value), "unit": result.unit},
                    "passed": passed,
                }
            )
            if not passed:
                raise KernelError("conflict", "asserted quantity disagrees with the evaluated result")
        return {"status": "ok", "value": str(result.value), "unit": result.unit}

    def execute(self, command: dict[str, Any]) -> dict[str, Any]:
        if len(self._trace) >= self.max_receipts:
            return {
                "status": "error",
                "error": "resource_limit",
                "message": "receipt capacity exhausted; command not executed or appended",
            }
        try:
            if not isinstance(command, dict):
                raise TypeError("command must be an object")
            pending: list[tuple[Any, int]] = [(command, 0)]
            while pending:
                item, depth = pending.pop()
                if depth > 64:
                    raise TypeError("command nesting exceeds the JSON depth bound")
                if isinstance(item, dict):
                    if any(not isinstance(k, str) for k in item):
                        raise TypeError("JSON object keys must be strings")
                    pending.extend((v, depth + 1) for v in item.values())
                elif isinstance(item, (list, tuple)):
                    if isinstance(item, tuple):
                        raise TypeError("JSON arrays must be lists")
                    pending.extend((v, depth + 1) for v in item)
            encoded = self._canonical(command)
            if len(encoded) > self.max_command_bytes:
                normalized: dict[str, Any] = {
                    "action": "__oversize_command__",
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    "byte_length": len(encoded),
                }
            else:
                normalized = command
        except (TypeError, ValueError, UnicodeEncodeError, RecursionError):
            normalized = {"action": "__invalid_json_input__"}

        before = self.state_digest()
        evidence: list[dict[str, Any]] = []
        try:
            result = self._apply(normalized, evidence)
        except self._KernelError as exc:
            result = {"status": "error", "error": exc.code, "message": str(exc)}
        after = self.state_digest()

        receipt = {
            "schema": self.SCHEMA,
            "sequence": len(self._trace) + 1,
            "config": self.config,
            "previous_hash": self.head,
            "command": normalized,
            "result": result,
            "evidence": evidence,
            "state_before": before,
            "state_after": after,
        }
        receipt["hash"] = self._digest(receipt)
        self._trace.append(receipt)
        return dict(result)

    @classmethod
    def replay(cls, trace: list[dict[str, Any]], expected_head: str | None = None) -> IncrementalKernel:
        if not isinstance(trace, list):
            raise ValueError("trace must be a list")
        if not trace:
            kernel = cls()
            if expected_head is not None and expected_head != cls.ZERO_HASH:
                raise ValueError("trace head differs from trusted anchor")
            return kernel
        first = trace[0]
        config = first["config"]
        kernel = cls(**config)
        previous = cls.ZERO_HASH
        for index, recorded in enumerate(trace, 1):
            payload = {k: v for k, v in recorded.items() if k != "hash"}
            if recorded.get("hash") != kernel._digest(payload):
                raise ValueError(f"receipt {index} hash mismatch")
            if recorded.get("previous_hash") != previous or recorded.get("sequence") != index:
                raise ValueError(f"receipt {index} chain mismatch")
            kernel.execute(copy.deepcopy(recorded["command"]))
            if kernel._canonical(kernel._trace[-1]) != kernel._canonical(recorded):
                raise ValueError(f"receipt {index} semantic replay mismatch")
            previous = recorded["hash"]
        if expected_head is not None and previous != expected_head:
            raise ValueError("trace head differs from trusted anchor")
        return kernel
