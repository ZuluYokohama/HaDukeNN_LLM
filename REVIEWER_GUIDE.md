# Reviewer guide

Each objection is a question a skeptical reader should be able to settle from the ledger without rerunning training. `sheaf-verify checklist` prints whether the current store answers it. An objection is answered when its gate's latest record passed, or, for the two integrity objections, when `verify` reported no problem of that kind.

An open objection does not by itself fail `verify`. `bundle` exits non-zero while any objection is open, any gate is not `pass`, or verification found a hash or ledger problem. Read the criterion `detail` in the ledger before treating a pass as a scientific claim. H7 can pass while recording that the two controls disagree, as long as superiority was not claimed.

The precedents for these checks are described in `ARCHITECTURE.md`. The pass conditions are in `GATES.md`.

| Id | Settled by | Objection |
| --- | --- | --- |
| OBJ-TEST-IN-GRAPH | H0 | Train hyperedges cite validation or test documents. |
| OBJ-SPLIT-OVERLAP | G0 | Train, validation, and test document ids overlap or the byte hashes are not distinct. |
| OBJ-CHRONOLOGY | G0 | The holdout is not strictly after the training split. |
| OBJ-FUTURE-LEAK | H2 | Perturbing a future position changes an earlier output. |
| OBJ-CAUSAL-VACUOUS | H2 | The causality probe does not change the future positions it claims to perturb. |
| OBJ-LOOP | H2 | Attached sheaf restriction maps do not agree on converging paths. |
| OBJ-BOUNDARY | H0 | An attached 2-complex fails the boundary identity. |
| OBJ-UNEQUAL-VOCAB | H1 | The arms do not share one unit inventory size. |
| OBJ-NO-BPE-CONTROL | H1 | There is no equal-size superword-BPE control. |
| OBJ-NO-RANDOM-GRAPH | H3 | There is no randomized hypergraph of the same order and cardinality histogram. |
| OBJ-FLOPS-OMIT-GRAPH | G1 | Reported FLOPs omit graph construction. |
| OBJ-BYTE-MISMATCH | G1 | The arms were not trained on the same number of bytes. |
| OBJ-BPB-TOKEN-DENOM | H6 | Bits per byte uses a token denominator instead of raw corpus bytes. |
| OBJ-SCORED-BYTES | H6 | The arms were scored on different numbers of bytes. |
| OBJ-FEW-SEEDS | H4 | The study uses fewer than the minimum number of seeds. |
| OBJ-STATS | H7 | The reported test, interval, or effect size does not match a recomputation. |
| OBJ-CONTROLS-DISAGREE | H7 | A superiority claim is carried by only one of the two controls. |
| OBJ-ENV | H5 | Python, CUDA, driver, GPU, or platform is missing. |
| OBJ-CODE | H5 | The code snapshot does not match its listing or git tree id. |
| OBJ-CHECKPOINT-IN-GIT | H5 | Checkpoint bytes were stored in the harness instead of by hash. |
| OBJ-KERNEL-HOST | H5 | A device kernel has no distinct host-reference payload. |
| OBJ-FIGURE | P4 | The figure is not bound to the metrics payload. |
| OBJ-PREREG | P0 | The claim is not a frozen preregistration. |
| OBJ-CLAIM-AFTER | P1 | The claim was registered after the metrics or quotes their hash. |
| OBJ-LEDGER | ledger verification | The ledger was edited, reordered, truncated, or signed with the wrong key. |
| OBJ-PROVENANCE | manifest and result verification | A manifest hash, payload hash, or result link does not recompute. |

## How to read a bundle

`reviewer-bundle/REPORT.md` is a rendering of the same records. The copies of `manifests/`, `payloads/`, `snapshots/`, and `ledger/` are what `verify` hashed. Checkpoint bytes are not in the bundle. The notes name their `external://` locators.

Confirm three things before reading the prose around a figure.

1. `verify` prints no problems. Integrity is not the same sentence as "the model is better."
2. H7's `stats.control_agreement` detail. If it says the controls disagree, the study did not earn a superiority claim.
3. The code snapshot's git tree id, and the kernel's host-reference hash when the backend is not `cpu`. Neither value is a substitute for running the code.

## What this does not certify

The causality trace is a submitted record. The per-seed BPB values are submitted records. A passer of H2 and H7 has a consistent story. An independent rerun on the named code, data, and seeds is still the check that the story was produced by that code.

HMAC verification shows that a holder of the seal key produced the ledger. It does not show which person that was, and it does not stop that holder from writing a different ledger. Compare `ledger/head.json` with a copy that was stored away from the key.
