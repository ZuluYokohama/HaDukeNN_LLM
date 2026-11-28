---
trigger: glob
globs: "src/sheaf_verify/**/*.py, tests/**/*.py"
description: Enforces Python 3.11+ pure-stdlib constraints, >=85% coverage, 22-gate traces, and 12/12 mutant adequacy on sheaf_verify.
---

# `sheaf_verify` Pure-Stdlib & Gate-Coverage Rules

- **Strict Standard-Library Purity**: Every module under `src/sheaf_verify/` must run on Python 3.11+ using **only** standard-library imports (`dependencies = []` in `pyproject.toml`). Place any `numpy`, `scipy`, or `torch` code under `src/hadukenn_core/`.
- **Ruff `E / F / I / UP` Compliance**:
  - Include `from __future__ import annotations` at the top of new modules.
  - Use Python 3.11+ built-in generics (`list[...]`, `dict[...]`, `tuple[...]`, `X | None`) rather than legacy `typing.List`, `typing.Dict`, or `typing.Optional`.
  - Sort imports alphabetically within standard-library, third-party, and first-party (`hadukenn_core`, `sheaf_verify`) groups with zero unused imports.
- **Verification Floor**: Any edit to `src/sheaf_verify/` or `tests/` must keep `sheaf_verify` statement coverage `>= 85.0%`, keep all 22 control-graph gates covered by both a `pass` test and an own-criteria `fail` test, and keep `sheaf-verify mutation-score` at `12/12` (`1.0`).
