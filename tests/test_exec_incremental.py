"""Incremental state checksum for the exec-hypergraph-v01 kernel.

The checksum guards against accidental drift; it is not cryptographic. These tests pin
leaf-encoding injectivity, add/retract inversion, agreement with a from-scratch
recomputation, replay parity, and command-by-command parity with ``core.Kernel``.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

EXEC_DIR = Path(__file__).resolve().parents[1] / "iterations" / "exec-hypergraph-v01"
if str(EXEC_DIR) not in sys.path:
    sys.path.insert(0, str(EXEC_DIR))

from hadukenn_exec.core import Kernel  # noqa: E402
from hadukenn_exec.incremental import (  # noqa: E402
    _LEAF_SRC_DOMAIN,
    _MASK64,
    IncrementalKernel,
    _leaf_limbs,
)

SEP = "\x1f"


def _acc(kernel: IncrementalKernel) -> tuple[int, int, int, int]:
    return (kernel._acc0, kernel._acc1, kernel._acc2, kernel._acc3)


def _recomputed_acc(kernel: IncrementalKernel) -> tuple[int, int, int, int]:
    acc = [0, 0, 0, 0]

    def add(limbs: tuple[int, int, int, int]) -> None:
        for i in range(4):
            acc[i] = (acc[i] + limbs[i]) & _MASK64

    for source_id, record in kernel._sources.items():
        add(_leaf_limbs(_LEAF_SRC_DOMAIN, ["source", source_id, record["sha256"]]))
    for scope, bindings in kernel._scopes.items():
        for name, binding in bindings.items():
            add(kernel._binding_leaf(scope, name, binding))
    return tuple(acc)


def _random_stream(rng: random.Random, length: int) -> list[dict]:
    names: list[str] = ["v0"]
    commands: list[dict] = [{"action": "set", "name": "v0", "value": "1", "unit": "L"}]
    for _ in range(length):
        roll = rng.random()
        if roll < 0.35:
            name = f"v{rng.randint(0, 9)}"
            value = str(rng.randint(-40, 40))
            if rng.random() < 0.2:
                value += f"/{rng.randint(1, 7)}"
            commands.append({"action": "set", "name": name, "value": value, "unit": rng.choice(["L", "s", "1"])})
            names.append(name)
        elif roll < 0.6:
            name = f"d{rng.randint(0, 6)}"
            op = rng.choice(["add", "sub", "mul", "div", "neg", "copy"])
            if op in ("neg", "copy"):
                args = {"value": rng.choice(names)}
            else:
                args = {"left": rng.choice(names), "right": rng.choice(names)}
            commands.append({"action": "derive", "name": name, "op": op, "args": args})
            names.append(name)
        elif roll < 0.72:
            commands.append({"action": "retract", "name": rng.choice(names)})
        elif roll < 0.8:
            commands.append({"action": "fork", "scope": "main", "new_scope": f"s{rng.randint(0, 4)}"})
        elif roll < 0.86:
            commands.append({"action": "source", "id": f"src{rng.randint(0, 2)}", "text": rng.choice(["a", "b"])})
        elif roll < 0.93:
            commands.append({"action": "query", "name": rng.choice(names)})
        else:
            commands.append({"action": "assert", "name": rng.choice(names), "value": "3", "unit": "L"})
    return commands


def test_separator_injection_pair_no_longer_collides():
    # Under the old "\x1f"-joined leaf encoding these two different states had one digest.
    a = IncrementalKernel()
    assert a.execute({"action": "set", "name": "x", "value": "7", "unit": f"1{SEP}L{SEP}{SEP}8{SEP}L"})["status"] == "ok"
    b = IncrementalKernel()
    assert b.execute({"action": "set", "name": f"x{SEP}1{SEP}L{SEP}{SEP}7", "value": "8", "unit": "L"})["status"] == "ok"
    assert a.execute({"action": "query", "name": "x"})["status"] == "ok"
    assert b.execute({"action": "query", "name": "x"})["error"] == "missing_reference"
    assert a.state_digest() != b.state_digest()
    # The full-hash reference kernel also tells them apart.
    ra, rb = Kernel(), Kernel()
    ra.execute({"action": "set", "name": "x", "value": "7", "unit": f"1{SEP}L{SEP}{SEP}8{SEP}L"})
    rb.execute({"action": "set", "name": f"x{SEP}1{SEP}L{SEP}{SEP}7", "value": "8", "unit": "L"})
    assert ra.state_digest() != rb.state_digest()


@pytest.mark.parametrize(
    ("first", "second"),
    [
        # name/unit boundary
        ({"name": "ab", "unit": "L"}, {"name": "a", "unit": "bL"}),
        # unit prefix
        ({"name": "y", "unit": "add"}, {"name": "y", "unit": "ad"}),
    ],
)
def test_field_boundaries_are_unambiguous(first, second):
    k1, k2 = IncrementalKernel(), IncrementalKernel()
    k1.execute({"action": "set", "value": "2", **first})
    k2.execute({"action": "set", "value": "2", **second})
    assert k1.state_digest() != k2.state_digest()


def test_source_leaf_is_unambiguous():
    k1, k2 = IncrementalKernel(), IncrementalKernel()
    k1.execute({"action": "source", "id": f"s{SEP}x", "text": "t"})
    k2.execute({"action": "source", "id": "s", "text": "t"})
    assert k1.state_digest() != k2.state_digest()


def test_add_then_retract_restores_the_accumulator():
    kernel = IncrementalKernel()
    kernel.execute({"action": "set", "name": "a", "value": "3", "unit": "L"})
    before = _acc(kernel)
    kernel.execute({"action": "set", "name": "b", "value": "4", "unit": "L"})
    kernel.execute({"action": "derive", "name": "c", "op": "add", "args": {"left": "a", "right": "b"}})
    assert _acc(kernel) != before
    kernel.execute({"action": "retract", "name": "c"})
    kernel.execute({"action": "retract", "name": "b"})
    assert _acc(kernel) == before
    # The full checksum also binds the clock, so it moves on even when the leaf sum returns.
    fresh = IncrementalKernel()
    fresh.execute({"action": "set", "name": "a", "value": "3", "unit": "L"})
    assert _acc(fresh) == before


def test_overwrite_changes_the_leaf_even_for_the_same_value():
    kernel = IncrementalKernel()
    kernel.execute({"action": "set", "name": "a", "value": "3", "unit": "L"})
    first = _acc(kernel)
    kernel.execute({"action": "set", "name": "a", "value": "3", "unit": "L"})
    assert _acc(kernel) != first  # revision is part of the leaf


def test_incremental_sum_matches_recomputation_and_replay():
    rng = random.Random(20261125)
    for _ in range(120):
        kernel = IncrementalKernel()
        for command in _random_stream(rng, rng.randint(5, 50)):
            kernel.execute(command)
            assert _acc(kernel) == _recomputed_acc(kernel)
        replayed = IncrementalKernel.replay(kernel.export_trace(), expected_head=kernel.head)
        assert replayed.state_digest() == kernel.state_digest()
        assert replayed.head == kernel.head


def test_results_match_the_reference_kernel_command_by_command():
    rng = random.Random(7)
    for _ in range(120):
        reference, incremental = Kernel(), IncrementalKernel()
        for command in _random_stream(rng, rng.randint(5, 50)):
            assert incremental.execute(command) == reference.execute(command), command


def test_replay_rejects_tampered_traces():
    kernel = IncrementalKernel()
    for command in (
        {"action": "set", "name": "a", "value": "3", "unit": "L"},
        {"action": "set", "name": "b", "value": "4", "unit": "L"},
        {"action": "derive", "name": "c", "op": "add", "args": {"left": "a", "right": "b"}},
        {"action": "query", "name": "c"},
    ):
        kernel.execute(command)
    trace = kernel.export_trace()
    trace[1]["command"]["value"] = "5"
    with pytest.raises(ValueError, match="hash mismatch"):
        IncrementalKernel.replay(trace)
    trace = kernel.export_trace()
    trace[2]["state_after"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        IncrementalKernel.replay(trace)
    with pytest.raises(ValueError, match="trusted anchor"):
        IncrementalKernel.replay(kernel.export_trace(), expected_head="f" * 64)


def test_empty_replay_and_receipt_capacity():
    assert IncrementalKernel.replay([]).head == IncrementalKernel.ZERO_HASH
    with pytest.raises(ValueError):
        IncrementalKernel.replay([], expected_head="f" * 64)
    kernel = IncrementalKernel(max_receipts=1)
    kernel.execute({"action": "set", "name": "a", "value": "1"})
    assert kernel.execute({"action": "query", "name": "a"})["error"] == "resource_limit"
