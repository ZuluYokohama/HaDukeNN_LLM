# Briefing — exec-digest-v01

Code only. This iteration records no experiment, no measurement and no result. It does not change any gate status.

## OODA

### Observe

The exec-hypergraph-v01 reference kernel (`hadukenn_exec/core.py`) re-serializes the whole state with sorted-key JSON for `state_digest()` before and after every command, so the per-command cost grows with state size. An earlier incremental draft summed per-binding hashes instead, but it built each leaf by joining fields with `\x1f`. Names, units and source ids may contain `\x1f`, so two different states could share one digest. A concrete colliding pair was found by audit.

### Orient

No G-gate and no prior iteration result is involved. The only reference is `core.Kernel`, which the new kernel must match command for command.

### Decide

Add `IncrementalKernel` with canonical-JSON leaves, and label its state digest as an integrity checksum against accidental drift, not a cryptographic commitment. Keep the colliding pair as a regression test.

### Act

Merge as code with tests. Any later use (for example as a faster checksum inside a larger run) is a separate iteration with its own preregistration.

## 1. Situation

`core.py` is added unchanged as the base. Nothing else from the v01 prototype is included: no baselines, cases, runner, spans, results or timing tables. No corpus text, weights, tokenizer table or trainer code.

## 2. Mission

Provide an incremental state checksum that returns the same command results, receipts contract and `replay()` behaviour as `core.Kernel`, with field boundaries that cannot be forged by string content.

## 3. Execution

See `MODE_METHOD.md`. In short: each live source and binding is one leaf, hashed as a domain-separated canonical-JSON array, one element per field. The leaf hashes are summed in four 64-bit lanes and updated only for leaves a command adds, replaces or retracts. The summary digest also binds the scope count, revision clock and live node count.

## 4. Sustainment

CPU-only unit tests, about one second. No GPU, no data and no stored artifacts.

The checksum is not cryptographic. Leaf hashes are unkeyed BLAKE2b and the sum has only 4 × 64 bits, so it detects bugs and divergence but does not resist someone who chooses inputs to force a collision. Tamper evidence for a recorded trace comes from the receipt hash chain, checked by `replay()` against a trusted head.

## 5. Command & Signal

Branch `iter/exec-digest-v01`, cut from `main` at `d4f6d5d`. Ledger head anchored at `307711a4d841c2c8fa0bbd954d15b2a585012608528c9f53b9ea08ad717bd7c4` (count 22, `fixtures/passing`), unchanged by this branch.
