"""Write a gate-decision trace when SHEAF_GATE_TRACE is set.

The gate-graph CI job turns this on, then
``python -m sheaf_verify.gate_coverage`` reads the file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from sheaf_verify import gate_trace


def pytest_configure(config):
    del config
    if os.environ.get("SHEAF_GATE_TRACE"):
        gate_trace.enable()


def pytest_runtest_setup(item):
    gate_trace.set_test(item.nodeid)


def pytest_sessionfinish(session, exitstatus):
    del session, exitstatus
    path = os.environ.get("SHEAF_GATE_TRACE")
    if not path:
        return
    lines = [json.dumps(event, sort_keys=True) for event in gate_trace.events()]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
