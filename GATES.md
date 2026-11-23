# Gates

A gate passes only when every criterion it writes has passed, and every gate it depends on has a passing ledger record bound to the same manifest and payload hashes. `run` still writes the failures. Thresholds below are harness constants. A study cannot loosen them by writing a smaller number into its metrics.

```
MIN_SEEDS = 5
MAX_FLOP_RELATIVE_TOLERANCE = 0.05
ALPHA = 0.05
MIN_BOOTSTRAP_DRAWS = 1000
MAX_BOOTSTRAP_DRAWS = 100000
```

The order is the topological order of the graph.

## P0 Claim freeze

Binds `claim`. No dependencies.

The payload is either a `sheaf-verify.claim.v1` object or plain UTF-8 of at least 20 characters. A structured claim needs a hypothesis id, a description of at least 20 characters, metric `bpb`, direction `less`, alpha equal to `ALPHA`, and a primary contrast that is one of the control arms. The payload must not contain a 64-character hex digest.

## G0 Corpus identity and disjoint splits

Binds `corpus.train`, `corpus.valid`, `corpus.test`. No dependencies.

The three payload hashes differ. Document ids are unique inside a split and pairwise disjoint across splits. Each split is non-empty. Metadata orders are integers with train < valid < test.

## H4 Seeds

Binds `seeds`, `metrics`. No dependencies.

The design is `independent` (pairwise disjoint seed lists) or `paired` (one shared list). Each arm has at least `MIN_SEEDS` distinct integer seeds. `bpb_per_seed` has the same length. Metrics cite the seeds manifest and repeat the design.

## H5 Code, kernel, environment, config, checkpoints, and logs

Binds `code`, `environment`, `config`, `kernel`, `metrics`, `checkpoint.hypergraph`, `checkpoint.superword_bpe`, `checkpoint.randomized_hypergraph`, and the three `log.*` roles. No dependencies.

The code snapshot matches the listing, and the recomputed git tree SHA-1 matches the manifest. Environment fields `python`, `cuda`, `driver`, `gpu`, and `platform` are non-blank and are not `unknown`, `none`, or `n/a`. Config is a non-empty JSON object. Each checkpoint is external, is not also stored as a payload, and cites the code, config, seeds, and kernel manifests. Each log is non-empty and cites its checkpoint.

Kernel metadata needs `backend` (`cpu`, `cuda`, `opencl`, or `c`) and `entry`. A device backend must set `host_reference_sha256` to a different payload hash that belongs to one of the kernel manifest's inputs.

## G1 Matched FLOPs and train bytes

Binds `corpus.train`, `metrics`. Depends on G0.

For every arm, `total = train + graph_build`, all non-negative. Graph arms have `graph_build > 0`. The superword-BPE arm has `graph_build = 0`. The relative span of the three totals is at most the declared tolerance, which itself must sit in `[0, 0.05]`. Every `train_bytes` equals the train corpus size. Metrics cite all three corpus manifests.

## H0 Train-only hypergraph

Binds `hypergraph`, `hypergraph.random`, and the three corpora. Depends on G0.

Both graphs are `sheaf-verify.hypergraph.v1`, `built_from_split` is `train`, document ids are a subset of the train split, and inputs cite the train manifest and not the valid or test manifests. Vertices and hyperedge positions agree. If a `complex` block is present, the boundary identity holds: the maximum absolute entry of B1 B2 is 0. A missing complex is recorded and does not fail this criterion.

## H1 Equal-size vocabulary and superword-BPE

Binds the three corpora, `vocabulary`, and `metrics`. Depends on G0.

Unit strings are unique. Every arm's `vocab_size` equals that count. The superword-BPE arm is present. Vocabulary inputs cite the train corpus and not valid or test.

## H6 Raw-byte BPB and identical scored bytes

Binds `corpus.test`, `metrics`. Depends on G0.

Each arm's denominator is `raw_corpus_bytes`. `raw_bytes` and `scored_bytes` both equal the test corpus size. The mean of `bpb_per_seed` equals `total_bits / raw_bytes` within `1e-9`. Scored-byte counts are identical across arms.

## H2 Causality

Binds `hypergraph`, `trace.causality`. Depends on H0.

The trace leaves a future position after `horizon`. Perturbations are strictly after the horizon. Outputs at and before the horizon are unchanged. Every probed future position changes. Consulted hyperedges end at or before `predict_at`, and there is one entry per position. The trace cites the study hypergraph. If a `sheaf` block is present, every listed pair of paths composes to the same restriction map. A missing sheaf is recorded and does not fail this criterion.

## H3 Randomized hypergraph control

Binds both graphs and `metrics`. Depends on H0.

`randomized` is false on the study graph and true on the control. Incidence hashes differ. Vertex counts, hyperedge counts, and cardinality histograms match. The metrics arm repeats the control's incidence hash, order, histogram, and `randomization_seed`.

## H7 Statistics

Binds `metrics`, `seeds`, `claim`. Depends on H4 and H6.

`bootstrap_draws` is inside the declared range. For independent seeds the harness computes Welch's test; for paired seeds, the paired t test. Both contrasts are computed: hypergraph against superword-BPE, and hypergraph against the randomized hypergraph. The bootstrap is a percentile interval of the mean difference with linear interpolation at `(n - 1) * p`. Hedges' g uses the small-sample correction `1 - 3 / (4(n1 + n2) - 9)`. The sign is mean(primary) minus mean(control). A negative BPB difference favors the hypergraph.

If the metrics contain a `reported` block, its `t`, `df`, `p`, `low`, `high`, and `g` must match the recomputation within `1e-8` times the scale. If the block is absent, the ledger values are the report.

`superiority.claimed` may be false, and the gate can still pass. If it is true, the named contrast must have `p < ALPHA`, a bootstrap high end below 0, and Hedges' g below 0, and the other control must agree. Disagreement fails the gate. When the claim is structured, `superiority.against` must equal the preregistered contrast.

## P1 Data protocol

Binds `claim`, `metrics`. Depends on P0 and G0.

The claim's `registered_at` is strictly earlier than the metrics manifest. The claim payload does not contain the metrics payload hash.

## P2 Units and controls

Depends on H0, H1, H2, and H3.

The metrics arms are exactly `hypergraph`, `superword_bpe`, and `randomized_hypergraph`.

## P3 Matched training

Depends on G1, H4, and H5.

All three checkpoints are external.

## P4 Evaluation, statistics, and figure

Binds `metrics`, `seeds`, `claim`, `figure`. Depends on H6, H7, P0, P1, P2, and P3.

Figure metadata has `plots` equal to `bpb` and `metrics_payload_sha256` equal to the metrics payload hash. The figure cites the metrics manifest. The payload starts with `<svg` or `<?xml`.
