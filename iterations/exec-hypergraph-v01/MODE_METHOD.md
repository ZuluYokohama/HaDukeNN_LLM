# Mode and method — exec-hypergraph-v01

## Mode

CPU-only engineering pilot for exact scoped relational operations and reversible byte-span incidence representation. No language-model vocabulary size, train split, or BPB result applies. Inputs are supplied formal commands and explicitly synthetic source examples.

## Method

1. Record the original engineering specification and hash before the relational pilot results.
2. Implement exact typed operations, snapshot scopes, revision invalidation, provenance, finite capacities, and actual execution receipts.
3. Implement two independent conventional evaluators and literal/seeded expected answers without importing kernel calculations.
4. Run direct semantic, negative, intervention, replay, and corrupted-trace controls.
5. Construct bounded overlapping byte spans and check exact reconstruction and prefix causality separately.
6. Run the fixed 5-seed by 4-depth relational grid and record code/input hashes, all failures, and exact matches.
7. Measure construction-inclusive workload timing in rotated arm orders, three repeats per workload. Average repeats within workload/seed before descriptive summaries. Measure Python allocation separately.
8. Profile one revision workload as an instrumented diagnostic. Do not merge its times into benchmark samples.

## Controls

- Independent fresh evaluator with per-query memoization.
- Conventional cached DAG with reverse-dependency invalidation and snapshot caches.
- Hand-derived expected outcomes and seeded generator arithmetic independent of either engine.
- Relevant and irrelevant input changes, named-role swaps, renaming, scope changes, revisions, retractions, and assertion conflicts.
- Rehashed forged answers, changed commands, reordered receipts, and truncation against a retained head.
- Arbitrary bytes, malformed UTF-8, every prefix boundary of Unicode examples, unseen suffix changes, span/byte limits, and ordered incidence corruption.
- Evaluator negative controls: a false expected answer, false intervention relation, or incorrect timed workload must produce a failing verdict.

## Gates

No existing gate in `GATES.md` changes status. This iteration's acceptance conditions are engineering checks defined in its own pre-run specification. They cannot be interpreted as passing H4 causal-model behavior, G4 language-model superiority, G5 scaling, or any tenfold capability threshold.

The original parent repository checks remain required. The added root pytest wrapper executes the iteration-local unittest suite and its no-timing relational pilot, so normal repository CI cannot silently omit the new experiment.

## Out of scope

Training, natural-language extraction, learned segmentation, spectral/sheaf advantage, general reasoning benchmarks, a capability comparison with frontier models, an audit-matched speed comparison, user-hardware throughput, cryptographic authorship, external truth, and any change to T-38 or packing status.

The original specification remains unchanged. Development amendments and repairs are explicit. The same public deterministic fixtures are reusable regression cases; calling the stabilized run `confirmation` is an engineering reproducibility check, not an untouched model-generalization holdout.

## Post-confirmation addition

`IncrementalKernel` (`hadukenn_exec/incremental.py`) was added from `origin/main` after the confirmation run recorded in `results/confirmation/`. It is covered by repository tests, not by the sealed confirmation manifest. The confirmation `code_sha256` set is unchanged and remains the canonical hash record of that run.
