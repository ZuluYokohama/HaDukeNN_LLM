# Contributing

HaDukeN //\\\\// HGNLLM Experiments. The package name is `sheaf-verify`.

## Quality gate

main is protected. An iteration branch merges only by pull request, and only with green CI. The required checks are:

| Check | What it refuses |
| --- | --- |
| `lint` | `ruff check .` |
| `unit` | the pytest suite, or line coverage of `src/sheaf_verify` under 85% |
| `verify` | `sheaf-verify verify` on `fixtures/passing` |
| `gate-graph` | a control-graph node with no pass test or no own-criteria fail test, a dependency edge missing a satisfied run or a blocked run, or `docs/control-graph.mmd` out of step with `gates.GATES` |

The `required` check is green only when those four are green. Protect `main` with that check. Do not push iteration results straight onto `main`.

Run the same four commands locally before opening the pull request:

```bash
pip install -e ".[dev]"
ruff check .
pytest --cov=sheaf_verify --cov-report=term-missing --cov-fail-under=85
sheaf-verify verify --store fixtures/passing --seal-key-file fixtures/test_seal.key
SHEAF_GATE_TRACE=gate-trace.jsonl pytest
python -m sheaf_verify.gate_coverage gate-trace.jsonl --matrix gate-coverage.md
```

`gate-coverage.md` is the coverage matrix: each gate against its pass path, its fail path, and the test file. A fail counts only when that gate's own criteria ran and one of them failed. An upstream block is recorded as `blocked`, a third outcome in the trace and its own matrix column. It counts for the dependency edge, not as the downstream gate's fail path. The matrix is a CI artifact. It is not a research result.

## Iteration branches

Cut `iter/<id>` from `main`. Put `iterations/<id>/BRIEFING.md`, `MODE_METHOD.md`, and `results/` on that branch. The briefing keeps the OODA block and the op-order sections Situation, Mission, Execution, Sustainment, and Command & Signal.

Merging that branch back to `main` is a pull request. CI has to be green. Fill in the pull-request template: gates touched, evidence files, the ledger head before and after, and either an unchanged preregistration or an amendment that is dated and justified. The diff must not add data, weights, a tokenizer table, a secret, or a machine path.

`iterations/TEMPLATE/` is the blank form. It is not an iteration.
