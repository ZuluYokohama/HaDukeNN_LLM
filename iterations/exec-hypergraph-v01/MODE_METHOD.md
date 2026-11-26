# Mode and method — exec-digest-v01

Code only. This file describes the implementation and its tests. It is not a result.

## Mode

Symbolic executable-relation kernel with exact fractions. No text units, no vocabulary and no split are involved.

## Method

- Inputs: the same JSON commands as `core.Kernel` (`source`, `set`, `derive`, `retract`, `fork`, `query`, `assert`).
- Leaf encoding: a source leaf is `["source", id, sha256]`; a binding leaf is `["literal", scope, name, revision, source, value, unit]` or `["derived", scope, name, revision, source, op, [[role, ref], ...]]`. Each leaf is canonical JSON with a domain-separation prefix, hashed with BLAKE2b-256 and split into four 64-bit lanes.
- Update: replacing or retracting a binding subtracts its old leaf; writing one adds its new leaf; a fork adds one leaf per copied binding.
- Output: `state_digest()` is SHA-256 over a domain prefix, the scope count, the revision clock, the live node count and the four lanes. It is an integrity checksum against accidental drift, not a cryptographic commitment.

## Controls

`core.Kernel` is the only comparison: identical results for every command on random command streams.

## Gates

None. This iteration does not change the status of any gate in `GATES.md`.

## Out of scope

Speed claims, learned policies, source-contradiction gating and any language-model result. Adversarial collision resistance is explicitly not claimed.

## Checks

`tests/test_exec_incremental.py`: the former `\x1f` collision pair and other field-boundary cases now differ; adding then retracting bindings restores the leaf sum; the running sum equals a from-scratch recomputation on random streams; replay reproduces the head and digest; tampered traces and wrong anchors are rejected; every command returns the same result as `core.Kernel`.
