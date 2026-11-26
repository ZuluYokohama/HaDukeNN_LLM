# exec-hypergraph-v01: incremental state checksum

This folder holds code only. It records no experiment result.

- `hadukenn_exec/core.py` is the exec-hypergraph-v01 reference kernel: exact, scoped executable relations with a hash-chained receipt log and `replay()`. It is a symbolic runtime, not a trained model. `state_digest()` re-serializes the whole state on every command.
- `hadukenn_exec/incremental.py` adds `IncrementalKernel`. It has the same commands, refusals, limits, receipts and `replay()` contract. Its `state_digest()` keeps a running sum of per-leaf hashes, one leaf per live source or binding, and updates only the leaves a command changes.

## What the state digest is and is not

The incremental state digest is an **integrity checksum against accidental drift**: bugs, lost or reordered updates, replay divergence. It is **not cryptographic**. Leaves are hashed with unkeyed BLAKE2b and summed in four 64-bit lanes. Someone who controls the inputs could search for two states with the same digest. Do not treat it as a commitment, a signature or proof of authorship. Tamper evidence for a recorded trace comes from the receipt hash chain checked by `replay()` against a trusted head.

Each leaf is a domain-separated canonical-JSON array with one element per field, so a name, unit or source id cannot shift bytes across a field boundary. An earlier draft joined fields with `\x1f`, which let two different states share a digest. `tests/test_exec_incremental.py` keeps that pair as a regression test.

## Tests

`tests/test_exec_incremental.py` runs in the main pytest suite. It checks the regression pair and other field-boundary cases, that adding then retracting bindings restores the leaf sum, that the running sum equals a from-scratch recomputation on random command streams, replay parity and tamper rejection, and that every command returns the same result as `core.Kernel`.
