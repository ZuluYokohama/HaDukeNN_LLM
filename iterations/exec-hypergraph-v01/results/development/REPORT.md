# HaDukeNN executable hypergraph — engineering pilot

**Status: PASS.** This status applies to the engineering checks below.

This is an executable symbolic reference environment for the proposed redesign. It is not a trained HaDukeNN model or a demonstration of a tenfold capability gain.

## Recorded scope

- Scenarios: 66
- Exact replay: 66 passed; 0 failed
- Intervention evaluations: 24/24
- Preregistration SHA-256: `47626de7927c8fd929b30642235cee2ef2a1ad003dec25563bb4e028010b7d37`
- Environment: session CPU container; no GPU run or user-workstation measurement

## Exact expected outcomes

| Arm | Cases | Checked commands | Correct | Queries |
| --- | ---: | ---: | ---: | ---: |
| reference | 66 | 5992 | 5992 | 318 |
| cached_dag | 66 | 5992 | 5992 | 318 |
| hypergraph_kernel | 66 | 5992 | 5992 | 318 |

## Descriptive end-to-end workload timings

Costs include engine construction and all supplied commands. Input generator time is excluded identically. Timing blocks first average the repeats for one workload/seed. SE is across workload blocks, not training runs. The kernel emits richer audit records than the ordinary baselines, so this table cannot establish an architectural speedup.

| Arm | Depth | Mean ms | SE ms | Workload blocks |
| --- | ---: | ---: | ---: | ---: |

## Claim boundaries

- No neural model, learned tokenizer, learned policy, or natural-language extraction is present.
- Synthetic programs use supplied exact operations; success does not establish learned generalization.
- The kernel maintains revision/provenance and full deterministic receipts; ordinary baselines do not supply equivalent audit records. Timing is diagnostic, not a matched-audit superiority comparison.
- Timing repetitions and synthetic seeds are not independent training runs; no capability p-value is reported.
- Python allocator peaks exclude native allocations and do not measure process RSS or VRAM.
- Trace hashes are integrity checks, not signatures or independent execution attestations.
- A graph runtime does not establish hypergraph tokenizer gains or elimination of neural opacity.

## Decision

A passing run qualifies the supplied symbolic semantics and reproducibility on these inputs. It does not authorize a capability claim, a tokenizer replacement, or a large training campaign. The next scientific comparison must match audit obligations and add a learned action-selection task with held-out compositions and a strong conventional controller.

The existing T-38 design and campaign ledger are outside this iteration.
