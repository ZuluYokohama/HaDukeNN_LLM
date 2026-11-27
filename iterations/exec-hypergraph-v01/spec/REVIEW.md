# Validation review

## Scope

This iteration implements an exact, structured-command engineering prototype. It
does not train a neural model or measure language understanding. Its computation
trace records what the executable system actually did; it does not explain the
internal mechanism of a neural encoder or controller.

The local pre-run specification is `PREREGISTRATION.json`. It is not an external
timestamp, a signed priority record, or a claim that this design was independently
registered before all development.

## Independent checks

Validation uses hand-computed expected results in addition to comparison with a
separately implemented reference evaluator. Error handling, role reversal,
scope separation, cache invalidation, source identity, and replay rejection are
scientific integrity controls, not model-capability scores.

## Observed validation

On 2026-11-25, the following command passed all **66 unittest methods** in the
prototype directory:

```bash
python3 -m unittest discover -s tests -q
```

The suite contains 56 core/control methods and 10 byte/span methods. A method can
contain multiple explicit cases; this count is not a count of independent model
tasks or training runs. The observed run used Python 3.12 in the session CPU
environment. Repository-wide CI and the separately generated pilot results are
reported by their own checks.

The core checks include:

- Hand-computed exact fractions, noncommutative roles, dimensions, and explicit
  divide-by-zero or conflicting-assertion refusals.
- Transitive invalidation after replacement, retraction, recreation, and changing
  dependency edges; snapshot branches do not inherit subsequent parent changes.
- Actual query receipts with named operand reads and revision ids, plus cache
  reuse receipts pointing to the earlier computation.
- Immutable UTF-8 source identity and byte count, detached command/trace/graph
  copies, and atomic rejection of malformed operations.
- Scope, revision, live-node, query-step, retained-source-byte, command-size,
  numeric-size, metadata-depth, and receipt-capacity controls.
- Exact deterministic semantic replay; changed commands, changed answers,
  reordered/interior-deleted receipts, and rehashed fabricated answers are
  refused. A retained final head detects otherwise-valid truncation.
- A supplemental adversarial regression with **2,400 mixed commands** across
  30 streams, compared with both independent conventional evaluators, followed
  by replay of each stream. This regression was added during implementation
  review and is separate from the prespecified pilot grid.

The byte/span tests exercise lossless arbitrary-byte reconstruction, overlapping
span candidates, ordered ports, prefix causality, candidate limits, and tamper
refusal. They establish properties of the deterministic span substrate, not
learned tokenization quality.

## Findings and repairs

1. **Malformed action values.** Review identified an uncaught-type-error risk
   when an action discriminator was a list or object. The executor now refuses
   non-string actions before set-membership dispatch; a regression covers it.
2. **Error evaluation order.** Mixed-mutation testing found that a conventional
   evaluator could report a right-hand cycle before inspecting a missing
   left-hand operand. The baseline's premature cycle check was removed so both
   implementations follow the declared left-to-right role order. The minimal
   example and the randomized stream remain as regressions. This was an error
   classification discrepancy, not a falsely accepted numeric answer.
3. **Input-contract differences.** Unit-validation priority for division and
   identifier, unit, source, and number bounds were aligned between the kernel
   and conventional evaluators where those bounds are shared input semantics.
4. **Finite audit and scope capacities.** Empty scopes, repeated revisions,
   retained source bytes, oversized commands, and receipt growth received
   explicit limits. Once receipt capacity is full, the next command is refused
   before execution and no further receipt is appended. That terminal refusal
   is returned to the caller; it is deliberately not described as a newly logged
   event in an already-full log.
5. **Nested metadata.** Deep JSON metadata is refused before state mutation to
   avoid a successful semantic write followed by a serialization failure.

These repairs are engineering development within the locally recorded integrity
scope. They are not retroactively described as untouched confirmation tests or
as evidence of a tenfold capability effect. The original preregistration file
was not rewritten to hide the discoveries.

## Interpretation limits

- Exact input structures are supplied. Parsing errors remain unmeasured.
- A hash chain detects changes relative to a retained anchor. An attacker able to
  rewrite the entire chain can produce another internally consistent history.
- A rejected assertion demonstrates conflict with the current formal state; it
  does not establish which real-world observation is true.
- A typed hyperedge can be represented by an ordinary relation node and named
  ports. This prototype must not claim a unique expressivity advantage from its
  label.
- Complete receipts add time and storage. A baseline without comparable receipts
  is useful for measuring overhead but is not a matched performance comparison.
- Determinism excludes wall-clock timings and host-dependent resource measures.
- The runtime's configurable capacities are acceptance limits, not a measured
  process-RSS guarantee. Canonicalization and receipt construction allocate
  temporary objects, and the default theoretical maximum is not a recommended
  operating memory budget for a workstation.
- Baselines intentionally retain conventional cache optimizations. Their
  performance and cache-hit behavior should not be weakened to favor the new
  kernel. Additional audit capacities are irrelevant to ordinary pilot cases
  that remain below them, but need explicit matching in future capacity studies.
- Recording a hash of an oversized refused command does not retain its original
  bytes. Replay reproduces the refusal descriptor; it cannot reconstruct omitted
  input bytes from a hash.

## Follow-up boundary

Learned semantic extraction, policy learning, held-out composition generalization,
spectral/sheaf effects, and tenfold capability hypotheses require separate
experiments with their own untouched confirmation sets.
