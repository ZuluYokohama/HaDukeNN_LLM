"""Optional record of gate decisions for the control-graph coverage check.

Recording is off unless a test run enables it. ``verify`` on a study does
not write a trace.
"""

from __future__ import annotations

_enabled = False
_test = "unscoped"
_events: list[dict] = []


def is_enabled() -> bool:
    return _enabled


def enable(*, clear: bool = True) -> None:
    global _enabled
    _enabled = True
    if clear:
        _events.clear()


def disable() -> None:
    global _enabled
    _enabled = False


def set_test(nodeid: str) -> None:
    global _test
    _test = nodeid


def emit(event: dict) -> None:
    if not _enabled:
        return
    row = dict(event)
    row["test"] = _test
    _events.append(row)


def events() -> list[dict]:
    return list(_events)


def trace_outcome(status: str, own_evaluated: bool, own_failed: bool, upstream: dict) -> str:
    """Tri-state trace label: ``pass``, ``fail``, or ``blocked``.

    The ledger keeps the binary ``status``. A gate whose own criteria never ran
    because a dependency had no passing record is ``blocked``, not ``fail``.
    """
    if status == "pass":
        return "pass"
    if not own_evaluated and any(value is False for value in upstream.values()):
        return "blocked"
    return "fail"
