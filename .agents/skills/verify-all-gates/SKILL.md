---
name: verify-all-gates
description: Runs the full 8-stage HaDukeNN_LLM repository verification gate (lint, unit tests, >=85% sheaf_verify coverage, 22-gate pass/fail trace matrix, canonical priority anchor, 71-test executable hypergraph suite, 12/12 mutation score, and docs budget). Do not use when running external long-duration GPU pretraining sweeps.
---

# Verify All Repository Gates (`/verify-all-gates`)

Use this skill before and after modifying any code, schema, documentation, or iteration file in `HaDukeNN_LLM`.

## Workflow

1. Execute the bundled universal verification script from the repository root:
   ```bash
   python3 .agents/skills/verify-all-gates/scripts/verify_all.py
   ```
2. Inspect the emitted JSON summary and `ANTIGRAVITY_VERIFICATION_REPORT.json`:
   - Confirm `"ok": true`.
   - Confirm `canonical_fixture.anchor_sha256` equals `e247484042475d381eb454df95b718f55739a6cd0385006f142be257b8b86567`.
   - Confirm `pytest_suite.sheaf_verify_line_coverage_pct >= 85.0` and `pytest_suite.failed == 0`.
   - Confirm `gate_trace_coverage.ok == true` (`missing_gates == []`) and `mutation_adequacy.survived == 0`.
3. If `ruff` and `pytest` are installed in the active Python environment, also run:
   ```bash
   ruff check .
   pytest --cov=sheaf_verify --cov-report=term-missing --cov-fail-under=85
   ```

## Safety

- All commands executed by `scripts/verify_all.py` are strictly read-only against `fixtures/passing/` and write temporary stores only inside isolated `tempfile.TemporaryDirectory` folders.
- Never modify `fixtures/passing/` or `fixtures/test_seal.key` to make a failing check pass.
- Do not commit temporary `gate-trace.jsonl` or `coverage.xml` artifacts to git.
