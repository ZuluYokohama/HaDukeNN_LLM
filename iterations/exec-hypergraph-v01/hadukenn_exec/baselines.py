"""Independent, exact conventional baselines for the executable-state pilot.

No code from the proposed kernel is imported here. Reference uses a fresh
evaluation memo for each query; CachedDAG uses ordinary dependency-directed
cache invalidation. Neither baseline supplies the kernel's audit receipts.
Their timings therefore describe implementation costs, not a matched-audit
architecture advantage or learned-model capability.
"""

import re
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction


class _Failure(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


_ROLES = {
    "add": ("left", "right"),
    "sub": ("left", "right"),
    "mul": ("left", "right"),
    "div": ("left", "right"),
    "neg": ("value",),
    "copy": ("value",),
}


def semantic_result(result):
    """Remove receipts/diagnostics, retaining the declared semantic outcome."""
    if result.get("status") == "error":
        return {"status": "error", "error": result.get("error")}
    output = {"status": result.get("status")}
    for key in ("value", "unit"):
        if key in result:
            output[key] = result[key]
    return output


def _string(value, limit=256):
    if not isinstance(value, str) or not value or len(value) > limit:
        raise _Failure("invalid_command")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise _Failure("invalid_command") from None
    return value


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise _Failure("invalid_command")
    if isinstance(value, str):
        if len(value) > 2048:
            raise _Failure("resource_limit")
        if re.fullmatch(r"[+-]?[0-9]+(?:/[+-]?[0-9]+)?", value) is None:
            raise _Failure("invalid_command")
    try:
        if isinstance(value, str) and "/" in value:
            numerator, denominator = value.split("/")
            result = Fraction(int(numerator), int(denominator))
        else:
            result = Fraction(value)
        return _bounded(result)
    except (ValueError, ZeroDivisionError, OverflowError):
        raise _Failure("invalid_command") from None


def _bounded(value):
    if value.numerator.bit_length() > 4096 or value.denominator.bit_length() > 4096:
        raise _Failure("resource_limit")
    return value


def _arithmetic(operator, values):
    if operator == "copy":
        return values[0]
    if operator == "neg":
        number, unit = values[0]
        return -number, unit
    (left, left_unit), (right, right_unit) = values
    if operator in ("add", "sub"):
        if left_unit != right_unit:
            raise _Failure("unit_mismatch")
        return (left + right if operator == "add" else left - right), left_unit
    if operator == "mul":
        if left_unit == "1":
            return left * right, right_unit
        if right_unit == "1":
            return left * right, left_unit
        raise _Failure("unit_mismatch")
    if operator == "div":
        if right_unit == "1":
            unit = left_unit
        elif left_unit == right_unit:
            unit = "1"
        else:
            raise _Failure("unit_mismatch")
        if right == 0:
            raise _Failure("division_by_zero")
        return left / right, unit
    raise _Failure("invalid_operator")


class Reference:
    """Exact scoped expression store with fresh per-query evaluation.

    A source record is preserved literally. Definitions refer to binding names
    within the query's scope. Forks copy the current definitions, so subsequent
    parent edits cannot affect a child scope. Values are fractions, never floats.
    """

    def __init__(self, max_nodes=10_000, max_steps=10_000):
        for limit in (max_nodes, max_steps):
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError("limits must be positive integers")
        self.max_nodes = max_nodes
        self.max_steps = max_steps
        self.sources = {}
        self.scopes = {"main": {}}

    def _size(self):
        return len(self.sources) + sum(len(definitions) for definitions in self.scopes.values())

    def _room(self, additional):
        if self._size() + additional > self.max_nodes:
            raise _Failure("resource_limit")

    def _scope(self, command):
        scope = _string(command.get("scope", "main"))
        if scope not in self.scopes:
            raise _Failure("missing_reference")
        return scope

    def _source(self, command):
        if "source" in command:
            source = _string(command["source"])
            if source not in self.sources:
                raise _Failure("missing_reference")
            return source
        return None

    def _replace(self, scope, name, definition):
        self.scopes[scope][name] = definition

    def _remove(self, scope, name):
        del self.scopes[scope][name]

    def _fork(self, scope, new_scope):
        self.scopes[new_scope] = deepcopy(self.scopes[scope])

    def _memo(self, scope):
        return {}

    def _evaluate(self, scope, target):
        definitions = self.scopes[scope]
        memo = self._memo(scope)
        visiting = set()
        stack = [(target, False)]
        steps = 0
        while stack:
            name, expanded = stack.pop()
            if name in memo:
                continue
            if not expanded:
                if name in visiting:
                    raise _Failure("cycle")
                if name not in definitions:
                    raise _Failure("missing_reference")
                steps += 1
                if steps > self.max_steps:
                    raise _Failure("resource_limit")
                definition = definitions[name]
                if definition["kind"] == "set":
                    memo[name] = (definition["value"], definition["unit"])
                    continue
                visiting.add(name)
                stack.append((name, True))
                roles = _ROLES[definition["op"]]
                for role in reversed(roles):
                    dependency = definition["args"][role]
                    if dependency not in memo:
                        stack.append((dependency, False))
            else:
                definition = definitions[name]
                values = [memo[definition["args"][role]] for role in _ROLES[definition["op"]]]
                number, unit = _arithmetic(definition["op"], values)
                memo[name] = (_bounded(number), unit)
                visiting.remove(name)
        return memo[target]

    def execute(self, command):
        """Execute one formal command; rejected commands do not mutate bindings."""
        try:
            if not isinstance(command, dict):
                raise _Failure("invalid_command")
            action = command.get("action")
            if not isinstance(action, str):
                raise _Failure("invalid_command")
            if action == "source":
                source_id = _string(command.get("id"))
                text = command.get("text")
                if not isinstance(text, str):
                    raise _Failure("invalid_command")
                try:
                    source_bytes = text.encode("utf-8")
                except UnicodeEncodeError:
                    raise _Failure("invalid_command") from None
                if len(source_bytes) > 1_048_576:
                    raise _Failure("resource_limit")
                if source_id in self.sources:
                    if self.sources[source_id] != text:
                        raise _Failure("conflict")
                else:
                    self._room(1)
                    self.sources[source_id] = text
                return {"status": "ok"}
            if action not in {"set", "derive", "retract", "fork", "query", "assert"}:
                raise _Failure("invalid_command")
            scope = self._scope(command)
            if action == "fork":
                new_scope = _string(command.get("new_scope"))
                if new_scope in self.scopes:
                    raise _Failure("conflict")
                self._room(len(self.scopes[scope]))
                self._fork(scope, new_scope)
                return {"status": "ok"}
            name = _string(command.get("name"))
            if action == "set":
                value = _number(command.get("value"))
                unit = _string(command.get("unit", "1"), 64)
                source = self._source(command)
                self._room(int(name not in self.scopes[scope]))
                self._replace(scope, name, {"kind": "set", "value": value, "unit": unit, "source": source})
                return {"status": "ok"}
            if action == "derive":
                operator = command.get("op")
                if not isinstance(operator, str) or operator not in _ROLES:
                    raise _Failure("invalid_operator")
                args = command.get("args")
                if not isinstance(args, dict) or set(args) != set(_ROLES[operator]):
                    raise _Failure("invalid_command")
                for argument in args.values():
                    _string(argument)
                source = self._source(command)
                self._room(int(name not in self.scopes[scope]))
                self._replace(scope, name, {"kind": "derive", "op": operator, "args": dict(args), "source": source})
                return {"status": "ok"}
            if action == "retract":
                self._source(command)
                if name not in self.scopes[scope]:
                    raise _Failure("missing_reference")
                self._remove(scope, name)
                return {"status": "ok"}
            if action == "assert":
                expected = _number(command.get("value"))
                expected_unit = _string(command.get("unit", "1"), 64)
            value, unit = self._evaluate(scope, name)
            if action == "assert" and (value != expected or unit != expected_unit):
                raise _Failure("conflict")
            return {"status": "ok", "value": str(value), "unit": unit}
        except _Failure as error:
            return {"status": "error", "error": error.code}


class CachedDAG(Reference):
    """Conventional memoized DAG with exact reverse-dependency invalidation.

    This is a competent storage/algorithm baseline, including sharing within a
    DAG, cache reuse, scoped snapshots, and invalidation after revisions. It is
    deliberately not a linear scan or a fixed context-window straw baseline.
    """

    def __init__(self, max_nodes=10_000, max_steps=10_000):
        super().__init__(max_nodes, max_steps)
        self._cache = {"main": {}}
        self._users = {"main": defaultdict(set)}

    def _memo(self, scope):
        return self._cache[scope]

    def _invalidate(self, scope, name):
        stack = [name]
        seen = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            self._cache[scope].pop(current, None)
            stack.extend(self._users[scope].get(current, ()))

    def _unlink(self, scope, name):
        previous = self.scopes[scope].get(name)
        if previous and previous["kind"] == "derive":
            for dependency in previous["args"].values():
                users = self._users[scope].get(dependency)
                if users is not None:
                    users.discard(name)
                    if not users:
                        del self._users[scope][dependency]

    def _replace(self, scope, name, definition):
        self._invalidate(scope, name)
        self._unlink(scope, name)
        super()._replace(scope, name, definition)
        if definition["kind"] == "derive":
            for dependency in definition["args"].values():
                self._users[scope][dependency].add(name)

    def _remove(self, scope, name):
        self._invalidate(scope, name)
        self._unlink(scope, name)
        super()._remove(scope, name)

    def _fork(self, scope, new_scope):
        super()._fork(scope, new_scope)
        self._cache[new_scope] = dict(self._cache[scope])
        self._users[new_scope] = deepcopy(self._users[scope])
