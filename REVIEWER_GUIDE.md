# Reviewer guide

Each objection is a question a skeptical reader should be able to settle from the ledger without rerunning training. `sheaf-verify checklist` prints whether the current store answers it. An objection is answered when its gate's latest record passed, or, for the two integrity objections, when `verify` reported no problem of that kind.

An open objection does not by itself fail `verify`. `bundle` exits non-zero while any objection is open, any gate is not `pass`, or verification found a hash or ledger problem. Read the criterion `detail` in the ledger before treating a pass as a scientific claim. G4 can pass while recording that the two methodology controls disagree, as long as superiority was not claimed. G0's recorded feasibility example fails `g0.vocab_cap` on purpose: `V = 65,536` is flagged against the written cap of 64,000.

The precedents for these checks are described in `ARCHITECTURE.md`. The pass conditions are in `GATES.md`.

| Id | Settled by | Objection |
| --- | --- | --- |
| OBJ-G0-UNITS | G0 | Canonical units are not at least 15% fewer than GPT-2 BPE on the valid split. |
| OBJ-G0-FALLBACK | G0 | Byte fallback is above 1%. |
| OBJ-G0-VOCAB | G0 | Vocabulary size exceeds the written cap of 64,000. V=65,536 is the flagged inconsistency. |
| OBJ-G0-BUILD | G0 | The build exceeds 2 CPU-hours or 16 GB peak RAM. |
| OBJ-G1-BPB | G1 | hnt64k test BPB_corpus is above struct32k plus one standard deviation. |
| OBJ-G1-TRAIN-FLOPS | G1 | hnt64k train FLOPs per byte are not lower than struct32k and sw64k. |
| OBJ-PILOT-SEEDS | G1 | A pilot arm has fewer than 3 seeds. |
| OBJ-TEST-IN-GRAPH | H0 | Train hyperedges cite validation or test documents. |
| OBJ-SPLIT-OVERLAP | H0 | Train, validation, and test document ids overlap or the byte hashes are not distinct. |
| OBJ-CHRONOLOGY | H0 | The holdout is not strictly after the training split. |
| OBJ-FUTURE-LEAK | H4 | Perturbing a future position changes an earlier output. |
| OBJ-CAUSAL-VACUOUS | H4 | The causality probe does not change the future positions it claims to perturb. |
| OBJ-LOOP | H4 | Attached sheaf restriction maps do not agree on converging paths. |
| OBJ-BOUNDARY | H0 | An attached 2-complex fails the boundary identity. |
| OBJ-UNEQUAL-VOCAB | G1 | The arms do not share one unit inventory size, or sw64k does not match hnt64k. |
| OBJ-NO-BPE-CONTROL | G1 | The equal-size sw64k superword-BPE control is missing. |
| OBJ-NO-RANDOM-GRAPH | G4 | There is no randomized hypergraph of the same order and cardinality histogram. |
| OBJ-FLOPS-OMIT-GRAPH | G1 | Reported FLOPs omit graph construction. |
| OBJ-BYTE-MISMATCH | G1 | The arms were not trained on the same number of bytes. |
| OBJ-BPB-TOKEN-DENOM | G1 | Bits per byte uses a token denominator instead of raw corpus bytes. |
| OBJ-SCORED-BYTES | G1 | The arms were scored on different numbers of bytes. |
| OBJ-FEW-SEEDS | G4 | A headline arm has fewer than 5 seeds. |
| OBJ-STATS | G4 | The reported test, interval, or effect size does not match a recomputation. |
| OBJ-CONTROLS-DISAGREE | G4 | A superiority claim is carried by only one of the two controls. |
| OBJ-ENV | P0 | Python, CUDA, driver, GPU, or platform is missing. |
| OBJ-CODE | P0 | The code snapshot does not match its listing or git tree id. |
| OBJ-CHECKPOINT-IN-GIT | P0 | Checkpoint bytes were stored in the harness instead of by hash. |
| OBJ-KERNEL-HOST | G3 | A device kernel has no distinct host-reference payload. |
| OBJ-FIGURE | H7 | The figure is not bound to the metrics payload. |
| OBJ-PREREG | P0 | The claim is not a frozen preregistration. |
| OBJ-CLAIM-AFTER | P0 | The claim was registered after the metrics or quotes their hash. |
| OBJ-LEDGER | ledger verification | The ledger was edited, reordered, truncated, or signed with the wrong key. |
| OBJ-PROVENANCE | manifest and result verification | A manifest hash, payload hash, or result link does not recompute. |

## How to read a bundle

`reviewer-bundle/REPORT.md` is a rendering of the same records. The copies of `manifests/`, `payloads/`, `snapshots/`, and `ledger/` are what `verify` hashed. Checkpoint bytes are not in the bundle. The notes name their `external://` locators.

Confirm three things before reading the prose around a figure.

1. `verify` prints no problems. Integrity is not the same sentence as "the model is better."
2. G4's `stats.control_agreement` detail. If it says the controls disagree, the study did not earn a superiority claim. Also read `g4.baseline_sd` and the component `g4.p.*` lines: the spec's headline test is a gain of more than 2 baseline standard deviations at matched all-in FLOPs, and every named component has to beat its random-hypergraph control.
3. The code snapshot's git tree id, and the kernel's host-reference hash when the backend is not `cpu`. Neither value is a substitute for running the code.

## What this does not certify

The causality trace is a submitted record. The per-seed BPB values are submitted records. A passer of H4 and G4 has a consistent story. An independent rerun on the named code, data, and seeds is still the check that the story was produced by that code.

HMAC verification shows that a holder of the seal key produced the ledger. It does not show which person that was, and it does not stop that holder from writing a different ledger. After `verify` succeeds, `sheaf-verify anchor` prints the head hash. Copy that line by hand, or pass `--out` to append it to a file outside the store, and keep the copy where the seal key cannot rewrite it. The command does not contact an external service. A later ledger that verifies under the same key but has a different head hash is not the ledger that was anchored.
