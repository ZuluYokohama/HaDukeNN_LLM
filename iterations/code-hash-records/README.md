# Code hash records (hashes only, no code)

The trainer and its supporting modules are not in this repository (see `README.md`). This folder publishes the SHA-256 values of the trainer-side files that changed between Nov 27, 7:20 PM and Nov 28, 5:03 AM CT, 2026, so a reader holding the same files can check them against the results in `iterations/`. Each record in `RECORDS.json` was written before any run using the changed code produced data. Blake Jones approved each change (rulings t250u and t251u).

Why this exists: results from the r2 arms (muP, K_neg, shrunk residual, StochasTok schedule, OE), the corpus-hypergraph kvreg arms, and the GPU-1 fp32 evaluation runs depend on code that is not published. The hashes pin which code produced them.

What is in `RECORDS.json`:

| Record | Covers |
|---|---|
| `t250u-r2-trainer-promotion` | The r2 trainer, `struct_model.py`, `stochastok.py`: before and after hashes, the unchanged modules, the post-fix smoke results, and the plans covered. The main script was hashed twice: an intermediate version that never produced data, and the final `5796c6e5…` (a variable-shadowing fix in the muP loop, no math change) |
| `t250u-corpus-hypergraph-kvreg-and-gpu1-fp32` | The kvreg eval fix (the cause was the caller leaving stale registers set, not the kvreg module), the GPU-1 fp32 eval-only path for E2 and e1v3, and the raw-data path fallback |
| `t251u-single-arm-r2-and-flop-probe` | Single-arm launch for r2-kneg3 and r2-shrunk-resid (the tokenizer preflight rule is unchanged), and the FLOP cross-check tool fix |
| `trainer-hash-equivalence-r2` | The reviewed claim that the r2 trainer is identical to the previous one on flags-off paths, with its bit-identical CPU check |

These are records of what was changed and approved. They are not an independent verification: the code is not published and the checks ran on the author's machine. Smoke runs are not results and are not used anywhere in `iterations/`.
