"""Deterministic formal fixtures and interventions, with explicit gold outcomes.

The supplied formal commands are oracle interpretations, not text understood by
a learned model. Generation and gold arithmetic do not import the kernel or its
baselines. Public fixture seeds are development/control data, not a hidden test
set and not a learned-model benchmark.
"""

from copy import deepcopy
from fractions import Fraction
from random import Random


def _ok(value=None, unit="1"):
    outcome = {"status": "ok"}
    if value is not None:
        outcome.update(value=str(Fraction(value)), unit=unit)
    return outcome


def _error(code):
    return {"status": "error", "error": code}


def _new(case_id, family, seed=0, depth=0):
    return {"id": case_id, "family": family, "seed": seed, "depth": depth, "commands": [], "expected": []}


def _put(case, command, expected=None):
    case["expected"].append({"index": len(case["commands"]), "result": _ok() if expected is None else expected})
    case["commands"].append(command)


def _set(name, value, unit="1", scope="main", source=None):
    result = {"action": "set", "scope": scope, "name": name, "value": str(Fraction(value)), "unit": unit}
    if source is not None:
        result["source"] = source
    return result


def _derive(name, operator, left, right=None, scope="main", source=None):
    args = {"value": left} if operator in ("neg", "copy") else {"left": left, "right": right}
    result = {"action": "derive", "scope": scope, "name": name, "op": operator, "args": args}
    if source is not None:
        result["source"] = source
    return result


def _query(case, name, value, unit="1", scope="main"):
    _put(case, {"action": "query", "scope": scope, "name": name}, _ok(value, unit))


def _source(case):
    _put(case, {"action": "source", "id": "fixture", "text": f"Formal fixture {case['id']}; seed={case['seed']}."})


def literal_cases():
    """Small hand-derived gold cases exercising roles and semantic failures."""
    result = []

    case = _new("literal-transfer", "literal", depth=4)
    _source(case)
    for name, value in (("a_initial", 20), ("b_initial", 5), ("outbound", 12), ("returned", 3)):
        _put(case, _set(name, value, "L", source="fixture"))
    for name, operator, left, right in (
        ("a_after", "sub", "a_initial", "outbound"),
        ("a_final", "add", "a_after", "returned"),
        ("b_after", "add", "b_initial", "outbound"),
        ("b_final", "sub", "b_after", "returned"),
        ("total", "add", "a_final", "b_final"),
    ):
        _put(case, _derive(name, operator, left, right, source="fixture"))
    _query(case, "a_final", 11, "L")
    _query(case, "b_final", 14, "L")
    _query(case, "total", 25, "L")
    result.append(case)

    case = _new("literal-rational-units", "literal", depth=2)
    for name, value, unit in (("a", "3/2", "m"), ("b", "1/2", "m"), ("scale", 2, "1"), ("clock", 1, "s"), ("zero", 0, "1")):
        _put(case, _set(name, value, unit))
    for name, operator, left, right, value, unit in (
        ("sum", "add", "a", "b", 2, "m"),
        ("difference", "sub", "a", "b", 1, "m"),
        ("ratio", "div", "a", "b", 3, "1"),
        ("scaled", "mul", "a", "scale", 3, "m"),
        ("rescaled", "div", "a", "scale", "3/4", "m"),
    ):
        _put(case, _derive(name, operator, left, right))
        _query(case, name, value, unit)
    _put(case, _derive("negative", "neg", "a"))
    _query(case, "negative", "-3/2", "m")
    _put(case, _derive("copied", "copy", "a"))
    _query(case, "copied", "3/2", "m")
    for name, operator, left, right, code in (
        ("bad_sum", "add", "a", "clock", "unit_mismatch"),
        ("bad_product", "mul", "a", "b", "unit_mismatch"),
        ("bad_quotient", "div", "a", "clock", "unit_mismatch"),
        ("undefined", "div", "a", "zero", "division_by_zero"),
    ):
        _put(case, _derive(name, operator, left, right))
        _put(case, {"action": "query", "name": name}, _error(code))
    result.append(case)

    case = _new("literal-revision-scope", "literal", depth=2)
    _put(case, _set("a", 8))
    _put(case, _set("b", 3))
    _put(case, _derive("answer", "sub", "a", "b"))
    _query(case, "answer", 5)
    _put(case, {"action": "fork", "new_scope": "hypothesis"})
    _put(case, _set("a", 20))
    _query(case, "answer", 17)
    _query(case, "answer", 5, scope="hypothesis")
    _put(case, _set("b", 2, scope="hypothesis"))
    _query(case, "answer", 6, scope="hypothesis")
    _query(case, "answer", 17)
    _put(case, {"action": "retract", "name": "a"})
    _put(case, {"action": "query", "name": "answer"}, _error("missing_reference"))
    _query(case, "answer", 6, scope="hypothesis")
    _put(case, _set("a", 9))
    _query(case, "answer", 6)
    result.append(case)

    case = _new("literal-forward-cycle", "literal", depth=2)
    _put(case, _derive("a", "copy", "b"))
    _put(case, {"action": "query", "name": "a"}, _error("missing_reference"))
    _put(case, _set("b", 7))
    _query(case, "a", 7)
    _put(case, _derive("b", "copy", "a"))
    _put(case, {"action": "query", "name": "a"}, _error("cycle"))
    _put(case, _set("b", 9))
    _query(case, "a", 9)
    result.append(case)

    case = _new("literal-source-conflict", "literal", depth=1)
    _put(case, {"action": "source", "id": "original", "text": "A = 4; café 🌍"})
    _put(case, {"action": "source", "id": "original", "text": "A = 4; café 🌍"})
    _put(case, {"action": "source", "id": "original", "text": "A = 400"}, _error("conflict"))
    _put(case, _set("a", 4, source="original"))
    _put(case, {"action": "assert", "name": "a", "value": "5"}, _error("conflict"))
    _query(case, "a", 4)
    _put(case, _set("a", 400, source="missing"), _error("missing_reference"))
    _query(case, "a", 4)
    _put(case, {"action": "source", "id": "revision", "text": "A is revised to 6"})
    _put(case, _set("a", 6, source="revision"))
    _query(case, "a", 6)
    result.append(case)

    case = _new("literal-invalid-atomicity", "literal", depth=1)
    _put(case, _set("a", 4))
    for command, code in (
        ({"action": "set", "name": "a", "value": True}, "invalid_command"),
        ({"action": "set", "name": "a", "value": 1.5}, "invalid_command"),
        ({"action": "set", "name": "a", "value": "1/0"}, "invalid_command"),
        ({"action": "derive", "name": "a", "op": "pow", "args": {"left": "a", "right": "a"}}, "invalid_operator"),
        ({"action": "derive", "name": "a", "op": "sub", "args": {"value": "a"}}, "invalid_command"),
        ({"action": "fork", "new_scope": "main"}, "conflict"),
        ({"action": "query", "scope": "absent", "name": "a"}, "missing_reference"),
        ({"action": "retract", "name": "absent"}, "missing_reference"),
    ):
        _put(case, command, _error(code))
        _query(case, "a", 4)
    result.append(case)
    return result


def _chain(seed, depth, family="chain"):
    """Gold is a direct signed-sum formula, independent of graph evaluation."""
    rng = Random(seed)
    case = _new(f"{family}-s{seed}-d{depth}", family, seed, depth)
    _source(case)
    initial = rng.randint(-20, 20)
    total_change = 0
    _put(case, _set("x0", initial, "L", source="fixture"))
    offsets = []
    for i in range(1, depth + 1):
        value = rng.randint(1, 9)
        sign = rng.choice((-1, 1))
        offsets.append((sign, value))
        total_change += sign * value
        _put(case, _set(f"c{i}", value, "L", source="fixture"))
        _put(case, _derive(f"x{i}", "add" if sign == 1 else "sub", f"x{i-1}", f"c{i}", source="fixture"))
    return case, initial, total_change, offsets


def _composition(seed, depth):
    case, initial, change, _ = _chain(seed, depth, "composition")
    _query(case, f"x{depth}", initial + change, "L")
    _put(case, _set("x0", initial + 7, "L", source="fixture"))
    _query(case, f"x{depth}", initial + change + 7, "L")
    _put(case, _set("irrelevant", 999, "s", source="fixture"))
    _query(case, f"x{depth}", initial + change + 7, "L")
    _put(case, {"action": "assert", "name": f"x{depth}", "value": str(initial + change + 8), "unit": "L"}, _error("conflict"))
    _query(case, f"x{depth}", initial + change + 7, "L")
    return case


def _snapshot(seed, depth):
    case, initial, change, offsets = _chain(seed, depth, "snapshot")
    _query(case, f"x{depth}", initial + change, "L")
    _put(case, {"action": "fork", "new_scope": "hypothesis"})
    _put(case, _set("x0", initial + 7, "L"))
    _query(case, f"x{depth}", initial + change + 7, "L")
    _query(case, f"x{depth}", initial + change, "L", "hypothesis")
    _put(case, _set("x0", initial - 3, "L", "hypothesis"))
    _query(case, f"x{depth}", initial + change - 3, "L", "hypothesis")
    _put(case, {"action": "retract", "name": "c1"})
    _put(case, {"action": "query", "name": f"x{depth}"}, _error("missing_reference"))
    _query(case, f"x{depth}", initial + change - 3, "L", "hypothesis")
    _put(case, _set("c1", offsets[0][1], "L"))
    _query(case, f"x{depth}", initial + change + 7, "L")
    return case


def _shared_dag(seed, depth):
    """Program gold computed as a sequence of exact scalar assignments."""
    rng = Random(seed + 100_003)
    case = _new(f"shared-dag-s{seed}-d{depth}", "shared_dag", seed, depth)
    _source(case)
    values = {"root": Fraction(rng.randint(1, 9)), "two": Fraction(2)}
    _put(case, _set("root", values["root"]))
    _put(case, _set("two", 2))
    previous = "root"
    for i in range(1, depth + 1):
        name = f"d{i}"
        if i % 5 == 0:
            _put(case, _derive(name, "mul", previous, "two"))
            values[name] = values[previous] * 2
        elif i % 7 == 0:
            _put(case, _derive(name, "neg", previous))
            values[name] = -values[previous]
        else:
            choices = ["root"] + [f"d{j}" for j in range(1, i)]
            shared = rng.choice(choices)
            operator = rng.choice(("add", "sub"))
            _put(case, _derive(name, operator, previous, shared))
            values[name] = values[previous] + values[shared] if operator == "add" else values[previous] - values[shared]
        previous = name
    _query(case, previous, values[previous])
    _query(case, previous, values[previous])
    _put(case, _set("noise", 901))
    _query(case, previous, values[previous])
    return case


def generate_suite(seeds=(11, 29, 47, 71, 101), depths=(4, 16, 64, 128)):
    """Return six literal cases plus three seeded families per seed/depth cell."""
    cases = literal_cases()
    for seed in seeds:
        for depth in depths:
            if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
                raise ValueError("depths must be positive integers")
            cases.extend((_composition(seed, depth), _snapshot(seed, depth), _shared_dag(seed, depth)))
    return cases


def _difference_case(case_id, left=8, right=3, swapped=False, rename=False):
    case = _new(case_id, "intervention", depth=1)
    a, b, out = ("unfamiliar_β", "entity_九", "out_7") if rename else ("a", "b", "answer")
    _put(case, _set(a, left))
    _put(case, _set(b, right))
    _put(case, _derive(out, "sub", b if swapped else a, a if swapped else b))
    return case, out, right - left if swapped else left - right


def intervention_pairs():
    """Return valid paired worlds and their prespecified final-answer relation.

    Each case retains per-command gold, so matching the final relation alone
    cannot pass a pair with incorrect answers or a failed mutation.
    """
    pairs = []
    for kind in ("role_swap", "relevant_value", "irrelevant_value", "entity_rename", "scope_isolation", "revision", "source_conflict", "assertion_conflict"):
        base, target, value = _difference_case(f"{kind}-base")
        _query(base, target, value)
        changed, changed_target, changed_value = _difference_case(
            f"{kind}-changed", left=12 if kind == "relevant_value" else 8,
            swapped=kind == "role_swap", rename=kind == "entity_rename",
        )
        if kind == "irrelevant_value":
            _put(changed, _set("unused", 10_000, "kg"))
        elif kind == "scope_isolation":
            _put(changed, {"action": "fork", "new_scope": "hypothesis"})
            _put(changed, _set("a", 100, scope="hypothesis"))
            _query(changed, "answer", 97, scope="hypothesis")
        elif kind == "revision":
            _put(changed, {"action": "source", "id": "revision", "text": "a is now 12"})
            _put(changed, _set("a", 12, source="revision"))
            changed_value = 9
        elif kind == "source_conflict":
            _put(changed, {"action": "source", "id": "evidence", "text": "a=8"})
            _put(changed, {"action": "source", "id": "evidence", "text": "a=80"}, _error("conflict"))
        elif kind == "assertion_conflict":
            _put(changed, {"action": "assert", "name": "answer", "value": "500"}, _error("conflict"))
        _query(changed, changed_target, changed_value)
        pairs.append({
            "id": kind, "kind": kind, "base": base, "changed": changed,
            "relation": "different" if kind in {"role_swap", "relevant_value", "revision"} else "equal",
        })
    return pairs


def benchmark_workloads(seed, depth, queries=40):
    """Construction-inclusive workloads; no hidden warmup or generated answers.

    Cold measures one query after construction, repeated_query reuses the graph,
    and revisions alternates relevant and irrelevant changes before each query.
    Benchmark runners must include construction and disclose absent audit parity.
    """
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
        raise ValueError("depth must be a positive integer")
    if not isinstance(queries, int) or isinstance(queries, bool) or queries < 1:
        raise ValueError("queries must be a positive integer")
    template, initial, change, _ = _chain(seed, depth, "benchmark")
    workloads = []
    for family in ("cold", "repeated_query", "revisions"):
        case = deepcopy(template)
        case["id"] = f"benchmark-{family}-s{seed}-d{depth}"
        case["family"] = family
        current = initial
        if family == "cold":
            _query(case, f"x{depth}", initial + change, "L")
        else:
            for i in range(queries):
                if family == "revisions":
                    if i % 2 == 0:
                        current = initial + i
                        _put(case, _set("x0", current, "L"))
                    else:
                        _put(case, _set("unrelated", i, "s"))
                _query(case, f"x{depth}", current + change, "L")
        workloads.append(case)
    return workloads
