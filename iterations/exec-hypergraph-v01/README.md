# HaDukeNN executable hypergraph prototype

An isolated engineering iteration for the proposed graph-native redesign. It provides a reversible byte/span interface, scoped exact relational execution, provenance, intervention controls, and semantic replay. **It is a reference environment, not a trained language model or a tenfold capability result.**

Python 3.11 or newer; the runtime and prototype tests use only the standard library. The parent repository's license and notice apply.

## Run

From this iteration directory:

```bash
python -m unittest discover -s tests -v
python -m hadukenn_exec evaluate --out results/local-run
python -m hadukenn_exec run examples/transfers.json --out results/local-demo
python -m hadukenn_exec replay results/local-demo/trace.json
python -m hadukenn_exec spans examples/source.txt --out results/local-demo/spans.json
```

The evaluation runs independent exact and cached-DAG baselines against hand-computed and seeded expected results. It tests interventions and replays actual kernel receipts. It returns a failing exit code for semantic, intervention, replay, or timed-workload correctness failures. Timing costs include construction and commands; audit obligations differ and are disclosed.

The demonstration includes a conservation check, a hypothetical branch, a later revision, a deliberately contradictory assertion, and a retraction. Expected refusals are visible in its outcomes and trace. The `run` command reports execution; use `evaluate` and the tests for pass/fail acceptance.

## Inspect

- `spec/IMPLEMENTATION.md`: semantics, architecture boundary, resource limits, next learned-interface experiment.
- `spec/PREREGISTRATION.json`: original pre-run engineering specification and fixed task grid.
- `spec/AMENDMENTS.md`: controls added during implementation and limitations of the design.
- `spec/REVIEW.md`: independent review and remaining limits.
- `results/confirmation/REPORT.md`: recorded engineering findings after the implementation stabilized.
- `results/confirmation/run_manifest.json`: code, input, and specification hashes.
- `hadukenn_exec/incremental.py`: post-confirmation `IncrementalKernel` from `origin/main` (integrity checksum, not a cryptographic commitment). Not part of the sealed confirmation `code_sha256` set.
- `results/demo/trace.json`: replayable demonstration receipts.
- `results/demo/graph.json`: named-port relation graph and source provenance.

## Boundaries

The supplied formal commands stand in for an oracle interpreter. The byte/span encoder does not yet infer these commands. No neural policy has been trained. Synthetic success tests correct execution of specified operations; it is not evidence of learned extrapolation or superiority to a conventional program. The independent cached-DAG control is expected to compute the same answers.

Trace hashes detect changes relative to a retained head. They do not provide signatures, independent timestamps, or guarantees about source truth. The kernel is a Python reference implementation and does not benchmark the user's GPU.

Frozen T-38 and the existing campaign ledger are outside this iteration.

`IncrementalKernel` is a later code-only addition. Adding it does not change the confirmation run. Tests in `tests/test_exec_incremental.py` pin leaf-encoding injectivity, add/retract inversion, and command-by-command parity with `core.Kernel`.
